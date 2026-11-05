import gc,json,weakref
from pathlib import Path
import torch,torch_npu
from kt_kernel.utils.native_mxfp4 import NativeMXFP4Weights
from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import KtKernelBackendConfig,KtKernelCPUExpertBackend
ROOT=Path('/mnt/share/y50063808/vllm-ascend-workspace')
torch.set_num_threads(1);torch.npu.set_device(7)
owner=NativeMXFP4Weights.from_checkpoint('/mnt/share/DeepSeek-V4-Flash-0731',16,3,4096,2048,numa_node=1)
ptrs=[t.data_ptr() for t in owner.tensors];nbytes=owner.nbytes;weak=weakref.ref(owner)
cfg=KtKernelBackendConfig(layer_idx=16,num_experts=3,top_k=3,hidden_size=4096,intermediate_size=2048,weight_path='/mnt/share/DeepSeek-V4-Flash-0731',weight_format='native_mxfp4',cpuinfer_threads=2,threadpool_count=1,max_num_tokens=17,numa_nodes=(1,))
backend=KtKernelCPUExpertBackend(cfg,native_weights=owner)
assert backend._wrapper.moe.native_weight_pointers()==ptrs
assert backend._wrapper.native_weights is owner
del owner;gc.collect();assert weak() is not None
results={'native_weight_bytes':nbytes,'cpp_pointers_equal_owner':True,'owner_retained':True,'eager_cases':[],'graph_replays':0}
def reference(x,ids,weights):
    # Same owner, independent native full-compute reference from previous validation.
    import ctypes as C
    import numpy as np
    lib=C.CDLL(str(ROOT/'cpu_moe/native_mxfp4_dot_prototype/libnative_moe.so'))
    ref=C.CDLL(str(ROOT/'cpu_moe/native_integration/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'))
    lib.native_moe.argtypes=[C.c_int]*5+[C.c_void_p]*12;lib.native_moe.restype=C.c_int
    buffers=[];o=weak()
    for tensor in o.tensors:buffers.append((C.c_void_p*3)(*[tensor[i].data_ptr() for i in range(3)]))
    # Native C API pointer list order gate,scale,up,scale,down,scale.
    order=[0,3,1,4,2,5]
    a=x.cpu().float().contiguous();r=weights.cpu().float().contiguous();i=ids.cpu().long().contiguous();out=torch.empty(a.shape,dtype=torch.float32)
    mask=torch.zeros(3,dtype=torch.uint8)
    status=lib.native_moe(len(a),3,3,4096,2048,*[C.cast(buffers[j],C.c_void_p) for j in order],mask.data_ptr(),i.data_ptr(),r.data_ptr(),a.data_ptr(),out.data_ptr(),C.cast(ref.quantize_row_q8_0,C.c_void_p))
    assert status==0;return out.to(torch.bfloat16)
for tokens in [1,4,10,17]:
    x=torch.randn(tokens,4096,device='npu',dtype=torch.bfloat16)*.1
    ids=torch.tensor([[0,1,2]],device='npu',dtype=torch.int64).repeat(tokens,1);ids[::2,0]=-1
    weights=torch.tensor([[.2,.3,.5]],device='npu').repeat(tokens,1)
    expected=reference(x,ids,weights)
    task=backend.submit(layer_idx=16,hidden_states=x,topk_ids=ids,topk_weights=weights)
    # Work continues in the original KT queue until wait joins it.
    actual=task.wait().cpu();torch.npu.synchronize()
    diff=(actual.float()-expected.float()).abs().max().item();scale=max(expected.float().abs().max().item(),1e-30)
    assert diff/scale<=.01,(tokens,diff,scale)
    assert backend._wrapper.moe.native_weight_pointers()==ptrs
    results['eager_cases'].append({'tokens':tokens,'max_abs':diff,'normalized_peak':diff/scale})
    print('PASS adapter eager',tokens,diff,flush=True)
# Capture a real CPU callback graph; replay same addresses with new inputs.
for graph_tokens in [1,3]:
    x=torch.randn(graph_tokens,4096,device='npu',dtype=torch.bfloat16)*.1
    ids=torch.tensor([[0,1,2]],device='npu',dtype=torch.int64).repeat(graph_tokens,1);weights=torch.tensor([[.2,.3,.5]],device='npu').repeat(graph_tokens,1)
    for _ in range(3):backend.submit(layer_idx=16,hidden_states=x,topk_ids=ids,topk_weights=weights).wait()
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out=backend.submit(layer_idx=16,hidden_states=x,topk_ids=ids,topk_weights=weights).wait()
    for step in range(5):
        x.copy_(torch.randn_like(x)*.1)
        expected=reference(x,ids,weights)
        graph.replay();torch.npu.synchronize();actual=out.cpu()
        diff=(actual.float()-expected.float()).abs().max().item();scale=max(expected.float().abs().max().item(),1e-30)
        assert diff/scale<=.01,(step,diff,scale)
        results['graph_replays']+=1
        print('PASS adapter graph replay',step,diff,flush=True)
results['status']='PASS'
(ROOT/'cpu_moe/native_integration/adapter_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
