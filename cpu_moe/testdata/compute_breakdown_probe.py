#!/usr/bin/env python3
"""Codex: CPU-only measurement with real GGUF weights and synthetic 8-token routes."""
import json
import os
import pathlib
import random
import sys
import time

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
os.environ['KT_MOE_CPU_TRACE'] = str(out / 'compute_%p.jsonl')
os.environ['KT_MOE_LIFECYCLE_TRACE'] = '1'
os.environ['KT_MOE_COMPUTE_BREAKDOWN'] = '1'
os.environ['KT_MOE_PIN_WEIGHTS'] = '1'
for key in ['KT_MOE_LIFECYCLE_WORKERS','KT_MOE_TASK_TRACE','KT_MOE_PHASE_TIMING','KT_MOE_PAGE_TRACE']:
    os.environ.pop(key, None)
sys.path.insert(0, '/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/dependencies/ktransformers-AK/kt-kernel/build/lib.linux-aarch64-cpython-311')
import torch
torch.set_num_threads(1)
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from kt_kernel import kt_kernel_ext

print('Codex: CPU-only probe, no NPU initialization', flush=True)
wrapper = LlamafileMoEWrapper(layer_idx=3,num_experts=256,num_experts_per_tok=6,
    hidden_size=4096,moe_intermediate_size=2048,gpu_experts_mask=None,
    cpuinfer_threads=32,threadpool_count=1,
    weight_path='/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/DeepSeek-V4-Flash-0731-GGUF/dsv4_layer3_mxfp4.gguf',
    chunked_prefill_size=8,numa_nodes=[1])
wrapper.load_weights()
torch.manual_seed(1024)
x = (torch.randn(8,4096)*.01).to(torch.bfloat16).contiguous()
y = torch.empty_like(x)
bsz = torch.tensor([8],dtype=torch.int32)
samples = []
def run(c, measured):
    ids = torch.full((8,6),-1,dtype=torch.int64)
    weights = torch.zeros((8,6),dtype=torch.float32)
    for token in range(8):
        count = c//8 + int(token<c%8)
        for slot in range(count):
            ids[token,slot] = (token*3+slot)%16
            weights[token,slot] = 1/6
    # Codex: use the actual TaskQueue, NUMA distributor and LLAMAFILE forward path.
    tid = kt_kernel_ext.lifecycle_begin(3)
    wrapper.cpu_infer.submit(wrapper.moe.forward_task(bsz.data_ptr(),6,ids.data_ptr(),weights.data_ptr(),x.data_ptr(),y.data_ptr(),False))
    wrapper.cpu_infer.sync()
    if measured:
        samples.append({'task_id':tid,'C':c})
        assert torch.isfinite(y).all()
        if c == 0:
            assert torch.count_nonzero(y) == 0

cases = [0,8,12,16,20,24,28,32]
for c in cases:
    for _ in range(8):
        run(c,False)
schedule = cases*40
random.Random(1024).shuffle(schedule)
for c in schedule:
    run(c,True)
(out/'samples.json').write_text(json.dumps(samples,indent=2))
(out/'probe_config.json').write_text(json.dumps({'tokens':8,'threads':32,'numa':[1],'layer':4,'cases':cases,'samples_per_case':40,'synthetic_routes':True,'real_gguf_weights':True,'extension':kt_kernel_ext.__file__},indent=2))
time.sleep(1)
print('Codex: probe passed, 320 measured tasks; outputs finite, zero-route output zero',flush=True)
