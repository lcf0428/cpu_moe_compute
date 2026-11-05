import torch,torch_npu,acl
from pathlib import Path
from kt_kernel.utils.native_mxfp4 import NativeMXFP4Weights
from vllm_ascend.expert_offload.native_mxfp4_h2d import NativeMXFP4H2DTransport,NativeExpertUploadTask
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import apply_mxfp4_weight_scale_layout as layout
torch.set_num_threads(1);torch.npu.set_device(3)
o=NativeMXFP4Weights.from_checkpoint('/mnt/share/DeepSeek-V4-Flash-0731',16,3,4096,2048,numa_node=1)
def cast(x):return torch_npu.npu_format_cast(x,29,customize_dtype=torch.float8_e4m3fn,input_dtype=torch_npu.float4_e2m1fn_x2)
def read(x):
 s=x.untyped_storage();h=torch.empty(s.nbytes(),dtype=torch.uint8,pin_memory=True);torch.npu.synchronize();assert acl.rt.memcpy(h.data_ptr(),h.numel(),s.data_ptr(),s.nbytes(),2)==0;return h
class Raw:
 supports_remote_sources=False
 def allocate_host_tensor(self,shape,dtype):return torch.empty(shape,dtype=dtype,pin_memory=True)
 def copy_batch(self,tasks):
  for t in tasks:assert acl.rt.memcpy_async(t.destination.data_ptr(),t.nbytes,t.source.data_ptr(),t.nbytes,1,int(torch.npu.current_stream().npu_stream))==0
 def synchronize(self,stream):stream.synchronize()
 def close(self):pass
w13=torch.cat((o.gate,o.up),dim=1).to('npu');w2=o.down.to('npu')
expected13=cast(w13).transpose(1,2);expected2=cast(w2).transpose(1,2)
for mode,base in [('torch',TorchCopyH2DTransport()),('raw',Raw())]:
 d13=cast(torch.zeros_like(w13)).transpose(1,2);d2=cast(torch.zeros_like(w2)).transpose(1,2)
 s13=layout(torch.zeros((3,4096,128),dtype=torch.uint8,device='npu'));s2=layout(torch.zeros((3,4096,64),dtype=torch.uint8,device='npu'))
 t=NativeMXFP4H2DTransport(base)
 tasks=[]
 for e in range(3):
  n13=d13.untyped_storage().nbytes()//3;n2=d2.untyped_storage().nbytes()//3
  tasks.append(NativeExpertUploadTask(o,e,d13.untyped_storage()[e*n13:(e+1)*n13],d2.untyped_storage()[e*n2:(e+1)*n2],s13[e],s2[e]))
 load=torch.npu.Stream()
 main=torch.npu.current_stream()
 torch.npu._subscribe_report(main)
 def callback(args):
  transport,items,stream=args
  with torch.npu.stream(stream):transport.copy_batch(items)
  transport.synchronize(stream)
 graph=torch.npu.NPUGraph()
 with torch.npu.graph(graph):torch.npu._launch_host_func(main,callback,(t,tasks,load))
 for _ in range(3):graph.replay();main.synchronize()
 torch.npu._unsubscribe_report(main)
 print(mode,'weight13 byte equal',torch.equal(read(d13),read(expected13)),'weight2',torch.equal(read(d2),read(expected2)),flush=True)
 es13=layout(torch.zeros((3,4096,128),dtype=torch.uint8,device='npu'));es2=layout(torch.zeros((3,4096,64),dtype=torch.uint8,device='npu'))
 a=layout(torch.cat((o.gate_scale,o.up_scale),dim=1)).contiguous();b=layout(o.down_scale).contiguous()
 for src,dst in [(a,es13),(b,es2)]:
  if mode=='torch':dst.copy_(src)
  else:base.copy_batch([type('T',(),{'source':src,'destination':dst,'nbytes':src.numel()})()])
 torch.npu.synchronize()
 print(mode,'scale13 bytes',torch.equal(read(s13),read(es13)),'scale2',torch.equal(read(s2),read(es2)),flush=True)
