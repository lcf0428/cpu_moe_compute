from pathlib import Path
p=Path('vllm-ascend/vllm_ascend/expert_offload/expert_offload_manager.py')
s=p.read_text()
s=s.replace('from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy','''from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.expert_offload.native_mxfp4_h2d import NativeExpertUploadTask, NativeMXFP4H2DTransport''')
anchor='        self.offload_config = get_ascend_config().expert_offload_config\n'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'''        cpu_config = self.offload_config.cpu_moe
        self._native_shared_weights = cpu_config.enabled and cpu_config.weight_format == "native_mxfp4"
        self._native_owners = {}
        self._native_loaded = {}
        self._native_tp_size = vllm_config.parallel_config.tensor_parallel_size
        if self._native_shared_weights and (
            self.offload_config.enable_multi_card or self.offload_config.shard_per_rank
            or self._native_tp_size != 1 or cpu_config.threadpool_count != 1
        ):
            raise ValueError("Shared native MXFP4 requires single-card offload and one NUMA pool")
''')
anchor='    def init_layer_cpu_buffers(self, layer, layer_moe_idx: int):'
methods='''    @property
    def native_mxfp4_enabled(self) -> bool:
        return getattr(self, "_native_shared_weights", False)

    def _init_native_layer_buffers(self, layer, layer_idx, experts):
        # Optional KT dependency: only native CPU/NPU workers import this owner.
        from kt_kernel.utils.native_mxfp4 import NativeMXFP4Weights

        w13 = _expert_weight(layer, "w13_weight")
        w2 = _expert_weight(layer, "w2_weight")
        if w13.dtype != torch.uint8 or w2.dtype != torch.uint8 or w13.ndim != 3 or w2.ndim != 3:
            raise ValueError("Shared native storage requires unpacked uint8 MXFP4 expert parameters")
        hidden, intermediate = int(w13.shape[2]) * 2, int(w2.shape[2]) * 2
        if w13.shape[1] != 2 * intermediate or w2.shape[1] != hidden:
            raise ValueError("Unsupported native MXFP4 expert weight shapes")
        if not hasattr(layer, "w13_weight_scale") or not hasattr(layer, "w2_weight_scale"):
            raise ValueError("Native MXFP4 requires both E8M0 weight scales")
        # Allocate once through the selected pinned-host allocator. These six
        # buffers become the only routed-expert DRAM weight/scale owner.
        shapes = (
            (experts, intermediate, hidden // 2),
            (experts, intermediate, hidden // 2),
            (experts, hidden, intermediate // 2),
            (experts, intermediate, hidden // 32),
            (experts, intermediate, hidden // 32),
            (experts, hidden, intermediate // 32),
        )
        owner = NativeMXFP4Weights(*[
            self._allocate_expert_host_tensor(shape, torch.uint8) for shape in shapes
        ]).validate(experts, hidden, intermediate)
        self._native_owners[layer_idx] = owner
        self._native_loaded[layer_idx] = set()
        # Compatibility views for expert-count bookkeeping. Native transfers
        # never interpret the gate view as combined w13 storage.
        self.w13_weights_cpu.append(list(owner.gate.unbind(0)))
        self.w2_weights_cpu.append(list(owner.down.unbind(0)))
        self.scale_cpu_buffers.setdefault("w13_weight_scale", []).append(list(owner.gate_scale.unbind(0)))
        self.scale_cpu_buffers.setdefault("w2_weight_scale", []).append(list(owner.down_scale.unbind(0)))
        self.w13_expert_size_bytes = owner.gate[0].numel() + owner.up[0].numel()
        self.w2_expert_size_bytes = owner.down[0].numel()

    def _require_native_weights_loaded(self):
        for layer_idx, owner in self._native_owners.items():
            expected = {
                (expert, projection, kind)
                for expert in range(owner.gate.shape[0])
                for projection in ("w1", "w3", "w2")
                for kind in ("weight", "scale")
            }
            missing = expected - self._native_loaded[layer_idx]
            if missing:
                raise RuntimeError(f"Native MXFP4 layer {layer_idx} has unloaded expert fields: {sorted(missing)[:8]}")

    def _attach_native_cpu_backends(self):
        from vllm_ascend.expert_offload.cpu_backend.wiring import maybe_attach_cpu_expert_backend

        self._require_native_weights_loaded()
        for layer_idx, layer in enumerate(self.moe_layers):
            maybe_attach_cpu_expert_backend(
                routed_experts=layer,
                layer_name=layer.layer_name,
                moe_config=layer.moe_config,
                cpu_moe_config=self.offload_config.cpu_moe,
                enable_multi_card=self.enable_multi_card,
                tp_size=self._native_tp_size,
                native_weights=self._native_owners[layer_idx],
            )

    @staticmethod
    def _copy_native_field(target, owned):
        source = owned.view(torch.uint8)
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(f"Native MXFP4 checkpoint field shape mismatch: {source.shape} != {target.shape}")
        target.copy_(source)

    def _load_native_field(self, layer_idx, expert, projection, kind, loaded):
        owner = self._native_owners[layer_idx]
        field = {"w1": "gate", "w3": "up", "w2": "down"}[projection]
        if kind == "scale":
            field += "_scale"
        target = getattr(owner, field)[expert]
        # The mmap can close before background loading finishes. Keep only a
        # bounded transient clone, using the existing load-pool drain policy.
        owned = loaded.cpu().clone()
        future = self._get_load_pool().submit(self._copy_native_field, target, owned)
        self._track_load_future(future)
        self._native_loaded[layer_idx].add((expert, projection, kind))

    def _native_upload_task(self, layer_idx, source_eid, w13, w2, scale13, scale2, destination_eid, name):
        owner = self._native_owners[layer_idx]
        size13 = w13.untyped_storage().nbytes() // w13.shape[0]
        size2 = w2.untyped_storage().nbytes() // w2.shape[0]
        start13, start2 = destination_eid * size13, destination_eid * size2
        return NativeExpertUploadTask(
            owner=owner, expert=source_eid,
            w13_destination=w13.untyped_storage()[start13:start13 + size13],
            w2_destination=w2.untyped_storage()[start2:start2 + size2],
            scale13_destination=scale13[destination_eid],
            scale2_destination=scale2[destination_eid], name=name,
        )

'''
assert anchor in s;s=s.replace(anchor,methods+anchor,1)
anchor='        _w13 = _expert_weight(layer, "w13_weight")\n'
# Insert native allocation before the first normal host allocation in init_layer only.
a=s.index('    def init_layer_cpu_buffers(');b=s.index(anchor,a)
s=s[:b]+'''        if self.native_mxfp4_enabled:
            self._init_native_layer_buffers(layer, layer_moe_idx, ntotal)
            self.moe_layers.append(layer)
            self._extend_cache_for_layer()
            self._register_layer_gate(layer)
            return

'''+s[b:]
anchor='        self.drain_load_pool()\n';a=s.index('    def _finalize_offload(');b=s.index(anchor,a)
s=s[:b]+s[b:].replace(anchor,anchor+'''        if self.native_mxfp4_enabled:
            self._attach_native_cpu_backends()
''',1)
anchor='        first_w13 = self.w13_weights_cpu[0][0]\n';a=s.index('    def process_weights_after_loading(');b=s.index(anchor,a)
s=s[:b]+'''        if self.native_mxfp4_enabled:
            self._require_native_weights_loaded()
            # Keep native bytes/scales on the host. Only device slot metadata
            # is prepared here; every actual miss converts in HBM.
            self._cast_device_slots_to_mxfp4_nz()
            return

'''+s[b:]
# Route native checkpoint fields directly into the shared owner; never transpose or relayout on CPU.
for fn,kind,proj in [('load_w13','weight','shard_id'),('load_w2','weight','"w2"'),('_load_scale_shard','scale','shard_id')]:
 a=s.index('    def '+fn+'(');b=s.index('        self._weight_load_calls += 1\n',a)+len('        self._weight_load_calls += 1\n')
 extra='''        if self.native_mxfp4_enabled:
'''
 if kind=='scale':extra+='''            if attr_name not in ("w13_weight_scale", "w2_weight_scale"):
                raise ValueError(f"Unsupported native MXFP4 quantization attribute {attr_name}")
'''
 extra+=f'''            self._load_native_field(layer_moe_idx, expert_id, {proj}, "{kind}", loaded_weight)
            return
'''
 s=s[:b]+extra+s[b:]
anchor='        w13_start = destination_eid * self.w13_expert_size_bytes\n';a=s.index('    def _build_prefill_h2d_tasks(');b=s.index(anchor,a)
s=s[:b]+'''        if self.native_mxfp4_enabled:
            return [self._native_upload_task(
                layer_idx, source_eid, self._prefill_w13[pool_slot], self._prefill_w2[pool_slot],
                self._prefill_w13_scale[pool_slot], self._prefill_w2_scale[pool_slot],
                destination_eid, f"native-prefill[L{layer_idx},E{source_eid}->P{pool_slot}:E{destination_eid}]",
            )]
'''+s[b:]
anchor='        w13_start = slot * self.w13_expert_size_bytes\n';a=s.index('    def _build_expert_h2d_tasks(');b=s.index(anchor,a)
s=s[:b]+'''        if self.native_mxfp4_enabled:
            return [self._native_upload_task(
                layer_idx, eid, _expert_weight(layer, "w13_weight"), _expert_weight(layer, "w2_weight"),
                layer.w13_weight_scale, layer.w2_weight_scale, slot,
                f"native-decode[L{layer_idx},E{eid}->S{slot}]",
            )]
'''+s[b:]
# All existing copy_batch sites, including prefetch and hot preload, now pass
# native tasks through this wrapper on the same load stream.
anchor='        return self.h2d_transport\n';a=s.index('    def _get_h2d_transport(');b=s.index(anchor,a)
s=s[:b]+'''        if self.native_mxfp4_enabled and not isinstance(self.h2d_transport, NativeMXFP4H2DTransport):
            self.h2d_transport = NativeMXFP4H2DTransport(self.h2d_transport)
'''+s[b:]
# Prevent accidental use of the legacy raw combined-w13 source with native views.
for fn in ['_expert_src_storage','_expert_dst_storage']:
 a=s.index('    def '+fn+'(');b=s.index('        cpu_buf = ',a)
 s=s[:b]+'''        if self.native_mxfp4_enabled:
            raise RuntimeError("Native MXFP4 sources must use the shared owner and native upload tasks")
'''+s[b:]
# Correct unique-storage memory reporting: compatibility views omit the up buffer.
anchor='        expert_bytes = weights_bytes + quant_bytes\n';s=s.replace(anchor,'''        if self.native_mxfp4_enabled:
            weights_bytes = self._cpu_tensor_storage_bytes([owner.tensors[:3] for owner in self._native_owners.values()])
            quant_bytes = self._cpu_tensor_storage_bytes([owner.tensors[3:] for owner in self._native_owners.values()])
'''+anchor,1)
p.write_text(s)
# Delay native KT construction until checkpoint copies are drained. Never read a second checkpoint in wiring.
p=Path('vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py');s=p.read_text();a=s.index('        maybe_attach_cpu_expert_backend(\n');b=s.index('\n    def _wrap_weight_loader_for_offload',a)
block=s[a:b];s=s[:a]+'''        if not mgr.native_mxfp4_enabled:
'''+''.join('    '+line+'\n' for line in block.rstrip().splitlines())+s[b:];p.write_text(s)
p=Path('vllm-ascend/vllm_ascend/expert_offload/cpu_backend/wiring.py');s=p.read_text();s=s.replace('    tp_size: int,\n','    tp_size: int,\n    native_weights=None,\n',1)
anchor='    backend = KtKernelCPUExpertBackend(config)\n';assert anchor in s
s=s.replace(anchor,'''    if config.weight_format == "native_mxfp4":
        if native_weights is None:
            raise RuntimeError("Native CPU backend must borrow the offload manager's loaded weight owner")
        backend = KtKernelCPUExpertBackend(config, native_weights=native_weights)
    else:
        backend = KtKernelCPUExpertBackend(config)
''',1);p.write_text(s)
