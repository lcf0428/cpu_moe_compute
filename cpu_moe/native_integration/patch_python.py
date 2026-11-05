from pathlib import Path
import shutil
r=Path('cpu_moe/dependencies/ktransformers-AK/kt-kernel')
p=r/'python/utils/llamafile.py';s=p.read_text()
s=s.replace('        numa_nodes: Optional[List[int]] = None,\n','        numa_nodes: Optional[List[int]] = None,\n        native_weights=None,\n',1)
a=s.index('        if not os.path.exists(weight_path):');b=s.index('        # Validate TP configuration',a)
old=s[a:b]
s=s[:a]+'''        self.native_weights = native_weights
        if native_weights is not None:
            if threadpool_count != 1:
                raise ValueError("Native MXFP4 currently requires one NUMA pool")
            native_weights.validate(num_experts, hidden_size, moe_intermediate_size)
            self.gguf_loader = None
        else:
'''+''.join('    '+line+'\n' for line in old.splitlines())+s[b:]
a=s.index('        _load_tensor = self.gguf_loader.get_undequanted_tensor_and_ggml_type');b=s.index('        pin_weights = _pin_weights()',a)
old=s[a:b]
s=s[:a]+'''        if self.native_weights is not None:
            if not torch.equal(physical_to_logical_map_cpu, torch.arange(self.num_experts, dtype=torch.int32)):
                raise ValueError("Native MXFP4 requires identity expert mapping")
            gate_data, up_data, down_data = self.native_weights.tensors[:3]
            gate_type = up_type = down_type = 39  # GGML_TYPE_MXFP4
        else:
'''+''.join('    '+line+'\n' for line in old.splitlines())+s[b:]
anchor='        # Set quantization types\n'
s=s.replace(anchor,'''        if self.native_weights is not None:
            if not hasattr(moe_config, "native_mxfp4"):
                raise RuntimeError("Native MXFP4 requires rebuilding kt-kernel")
            moe_config.native_mxfp4 = True
            moe_config.gate_scale = self.native_weights.gate_scale.data_ptr()
            moe_config.up_scale = self.native_weights.up_scale.data_ptr()
            moe_config.down_scale = self.native_weights.down_scale.data_ptr()

'''+anchor,1)
p.write_text(s)
# Keep backup copies of Ascend edits alongside the already archived source.
base=Path('vllm-ascend');before=Path('cpu_moe/native_integration/before/vllm-ascend')
paths=['vllm_ascend/ascend_config.py','vllm_ascend/expert_offload/cpu_backend/kt_kernel_backend.py','vllm_ascend/expert_offload/cpu_backend/wiring.py']
for name in paths:
 dst=before/name;dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():raise RuntimeError('Backup already exists')
 shutil.copy2(base/name,dst)
p=base/paths[0];s=p.read_text();i=s.index('class CpuMoEConfig') if 'class CpuMoEConfig' in s else s.index('        "backend": "kt_kernel",');anchor='        "backend": "kt_kernel",';assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n        "weight_format": "gguf",  # or native_mxfp4 checkpoint directory')
anchor='        if not isinstance(self.config["weight_path"], str):';s=s.replace(anchor,'''        if self.config["weight_format"] not in ("gguf", "native_mxfp4"):
            raise ValueError("cpu_moe.weight_format must be gguf or native_mxfp4")
        if self.config["weight_format"] == "native_mxfp4" and self.config["threadpool_count"] != 1:
            raise ValueError("Native MXFP4 currently requires one NUMA pool")
'''+anchor,1)
s=s.replace('if multiple and "{layer_idx}" not in weight_path:', 'if multiple and self.config["weight_format"] == "gguf" and "{layer_idx}" not in weight_path:')
p.write_text(s)
p=base/paths[1];s=p.read_text();s=s.replace('    numa_nodes: tuple[int, ...] | None = None\n','    numa_nodes: tuple[int, ...] | None = None\n    weight_format: str = "gguf"\n',1)
s=s.replace('def _default_wrapper_factory(**kwargs: Any) -> Any:', '''def _default_wrapper_factory(weight_format="gguf", native_weights=None, **kwargs: Any) -> Any:
    if weight_format == "native_mxfp4":
        from kt_kernel.utils.llamafile import LlamafileMoEWrapper
        from kt_kernel.utils.native_mxfp4 import NativeMXFP4Weights
        if native_weights is None:
            nodes = kwargs.get("numa_nodes")
            native_weights = NativeMXFP4Weights.from_checkpoint(
                kwargs["weight_path"], kwargs["layer_idx"], kwargs["num_experts"],
                kwargs["hidden_size"], kwargs["moe_intermediate_size"],
                numa_node=nodes[0] if nodes else 0)
        return LlamafileMoEWrapper(native_weights=native_weights, **kwargs)''')
s=s.replace('        wrapper_factory: Callable[..., Any] = _default_wrapper_factory,','        wrapper_factory: Callable[..., Any] = _default_wrapper_factory,\n        native_weights=None,',1)
anchor='        self._wrapper = wrapper_factory(\n'
s=s.replace(anchor,'''        native_options = {}
        if config.weight_format == "native_mxfp4":
            native_options = {"weight_format": config.weight_format, "native_weights": native_weights}
        elif native_weights is not None:
            raise ValueError("native_weights requires weight_format=native_mxfp4")
        self._wrapper = wrapper_factory(
            **native_options,
''',1)
p.write_text(s)
p=base/paths[2];s=p.read_text();s=s.replace('        weight_path=cpu_moe_config.weight_path,','        weight_path=cpu_moe_config.weight_path,\n        weight_format=cpu_moe_config.weight_format,',1);p.write_text(s)
