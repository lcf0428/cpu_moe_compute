"""Single real MXFP4 expert: prepared NZ copy vs native H2D+NZ+fixed slot+compute."""
import json, mmap, struct, time
from pathlib import Path
import numpy as np
import torch
import torch_npu
import acl
ROOT=Path('/mnt/share/y50063808/vllm-ascend-workspace')
MODEL=Path('/mnt/share/DeepSeek-V4-Flash-0731')
torch.set_num_threads(1)
torch.npu.set_device(7)
# NZ raw storage is copied through ACL, matching production byte transfers.
wm=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
def load(key):
    with (MODEL/wm[key]).open('rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        n=struct.unpack('<Q',mm[:8])[0];h=json.loads(mm[8:8+n])[key]
        a=np.ndarray(h['shape'],np.uint8,buffer=mm,offset=8+n+h['data_offsets'][0]).copy()
        mm.close()
    return torch.from_numpy(a)
def host(x):
    out=torch.empty(x.shape,dtype=torch.uint8,pin_memory=True);out.copy_(x);return out
prefix='layers.16.ffn.experts.0.'
w13=host(torch.cat([load(prefix+'w1.weight'),load(prefix+'w3.weight')]))
w2=host(load(prefix+'w2.weight'))
s13=host(torch.cat([load(prefix+'w1.scale'),load(prefix+'w3.scale')]))
s2=host(load(prefix+'w2.scale'))
def layout(x):
    n,k=x.shape
    return x.reshape(n,k//2,2).transpose(0,1)
def cast(w):
    return torch_npu.npu_format_cast(w,29,customize_dtype=torch.float8_e4m3fn,input_dtype=torch_npu.float4_e2m1fn_x2).transpose(-1,-2)
def raw(x):
    n=x.untyped_storage().nbytes()
    return torch.empty(0,dtype=torch.uint8,device=x.device).set_(x.untyped_storage(),0,(n,),(1,))
d13=torch.empty(w13.shape,dtype=torch.uint8,device='npu');d2=torch.empty(w2.shape,dtype=torch.uint8,device='npu')
d13.copy_(w13);d2.copy_(w2)
slot13=cast(d13);slot2=cast(d2)
slot_s13=layout(s13.to('npu'));slot_s2=layout(s2.to('npu'))
raw13=raw(slot13);raw2=raw(slot2);rs13=raw(slot_s13);rs2=raw(slot_s2)
def download(x):
    y=torch.empty((x.numel(),),dtype=torch.uint8,pin_memory=True)
    torch.npu.synchronize()
    ret=acl.rt.memcpy(y.data_ptr(),y.numel(),x.data_ptr(),x.numel(),2)
    assert ret==0,ret
    return y
def bytecopy(dst,src,kind):
    ret=acl.rt.memcpy_async(dst.data_ptr(),dst.numel(),src.data_ptr(),src.numel(),kind,int(torch.npu.current_stream().npu_stream))
    assert ret==0,ret
h13=download(raw13);h2=download(raw2);hs13=download(rs13);hs2=download(rs2)
ds13=torch.empty(s13.shape,dtype=torch.uint8,device='npu');ds2=torch.empty(s2.shape,dtype=torch.uint8,device='npu')
held=[]
def upload(mode):
    if mode=='nz':
        bytecopy(raw13,h13,1);bytecopy(raw2,h2,1)
        bytecopy(rs13,hs13,1);bytecopy(rs2,hs2,1)
    else:
        d13.copy_(w13,non_blocking=True);d2.copy_(w2,non_blocking=True)
        ds13.copy_(s13,non_blocking=True);ds2.copy_(s2,non_blocking=True)

def convert():
    a=cast(d13);b=cast(d2);sa=layout(ds13);sb=layout(ds2)
    bytecopy(raw13,raw(a),3);bytecopy(raw2,raw(b),3);bytecopy(rs13,raw(sa),3);bytecopy(rs2,raw(sb),3)
    # Keep outputs alive until completion; caching allocator follows current stream.
    return a,b,sa,sb

def compute(x):
    q,s=torch_npu.npu_dynamic_mx_quant(x,dst_type=torch.float8_e4m3fn)
    y=torch_npu.npu_quant_matmul(q,slot13,slot_s13,scale_dtype=torch_npu.float8_e8m0fnu,pertoken_scale=s,pertoken_scale_dtype=torch_npu.float8_e8m0fnu,output_dtype=torch.bfloat16,x2_dtype=torch_npu.float4_e2m1fn_x2,group_sizes=[0,0,32])
    gate,up=y.chunk(2,dim=-1)
    a=torch.nn.functional.silu(gate)*up
    q,s=torch_npu.npu_dynamic_mx_quant(a,dst_type=torch.float8_e4m3fn)
    return torch_npu.npu_quant_matmul(q,slot2,slot_s2,scale_dtype=torch_npu.float8_e8m0fnu,pertoken_scale=s,pertoken_scale_dtype=torch_npu.float8_e8m0fnu,output_dtype=torch.bfloat16,x2_dtype=torch_npu.float4_e2m1fn_x2,group_sizes=[0,0,32])
def summary(x):
    return {'p50_ms':float(np.percentile(x,50)),'p95_ms':float(np.percentile(x,95)),'min_ms':float(min(x))}
report={'device':7,'layer':16,'expert':0,'host_bytes':sum(x.numel() for x in [w13,w2,s13,s2]),'cases':[],'note':'Real expert, fixed NZ destination; target includes conversion allocations and copies into fixed slots. Standalone quant_matmul pipeline, not production fused GMM/ACL graph.'}
for tokens in [1,16]:
    x=torch.randn(tokens,4096,dtype=torch.bfloat16,device='npu')*.1
    upload('nz');gold=compute(x);torch.npu.synchronize()
    upload('native');held=convert();test=compute(x);torch.npu.synchronize()
    assert torch.equal(gold,test),'target result differs from baseline'
    for mode in ['nz','native']:
        for _ in range(20):
            upload(mode)
            if mode=='native':held=convert()
            y=compute(x)
        torch.npu.synchronize()
        records={'h2d':[],'convert_and_slot_copy':[],'compute':[],'total_device':[],'total_wall':[]}
        for _ in range(100):
            events=[torch.npu.Event(enable_timing=True) for i in range(4)]
            t=time.perf_counter();events[0].record();upload(mode);events[1].record()
            if mode=='native':held=convert()
            events[2].record();y=compute(x);events[3].record();events[3].synchronize()
            records['total_wall'].append((time.perf_counter()-t)*1000)
            for label,a,b in [('h2d',0,1),('convert_and_slot_copy',1,2),('compute',2,3),('total_device',0,3)]:records[label].append(events[a].elapsed_time(events[b]))
        report['cases'].append({'tokens':tokens,'mode':mode,'correctness':'bit_exact',**{k:summary(v) for k,v in records.items()}})
        print('PASS',tokens,mode,report['cases'][-1],flush=True)
(ROOT/'cpu_moe/native_integration/npu_swap_results.json').write_text(json.dumps(report,indent=2)+'\n')
