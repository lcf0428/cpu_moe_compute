"""Full native MoE vs existing compiled KT MOE, CPU only, immutable native pointers."""
import ctypes as C
import hashlib
import json
import os
import sys
import time
import numpy as np
import check_dot as d
os.environ.pop('KT_MOE_PIN_WEIGHTS',None) # This isolated reference process borrows test GGUF buffers.
sys.path.insert(0,str(d.KT/'kt-kernel/build/lib.linux-aarch64-cpython-311/kt_kernel'))
import kt_kernel_ext as ext
lib=C.CDLL(str(d.HERE/'libnative_moe.so'))
P=C.c_void_p
lib.native_moe.argtypes=[C.c_int]*5+[P]*12
lib.native_moe.restype=C.c_int
for name in ['ggml_fp32_to_bf16_row','ggml_bf16_to_fp32_row']:
    getattr(d.ref,name).argtypes=[P,P,C.c_int64]
wp=ext.WorkerPoolConfig(); wp.subpool_count=1; wp.subpool_numa_map=[1]; wp.subpool_thread_count=[2]
pool=ext.CPUInfer(wp)
rng=np.random.default_rng(20261013)
report={'status':'RUNNING','cases':[],'elements':0,'max_abs_diff':0.,'max_normalized_diff':0.}

def bf16(x):
    x=np.ascontiguousarray(x,np.float32); b=np.empty(x.shape,np.uint16)
    d.ref.ggml_fp32_to_bf16_row(d.ptr(x),d.ptr(b),x.size)
    return b

def expand(b):
    x=np.empty(b.shape,np.float32); d.ref.ggml_bf16_to_fp32_row(d.ptr(b),d.ptr(x),b.size); return x

def run_set(label,weights,packed):
    # weights = [w1,s1,w3,s3,w2,s2], each a list of immutable per-expert arrays.
    e=len(weights[0]); inter,h2=weights[0][0].shape; h=h2*2
    owners=[(C.c_void_p*e)(*[d.ptr(x) for x in proj]) for proj in weights]
    before=[hashlib.sha256(x.tobytes()).hexdigest() for proj in weights for x in proj]
    for hidden_type in [0,30]:
        mask=np.array([0,0,1],np.uint8) if e==3 else np.zeros(e,np.uint8)
        config=ext.moe.MOEConfig(e,3,h,inter,d.ptr(mask))
        config.pool=pool.backend_; config.layer_idx=0
        config.m_block=32; config.group_min_len=10; config.group_max_len=16; config.max_len=32
        config.gate_proj=d.ptr(packed[0]); config.up_proj=d.ptr(packed[1]); config.down_proj=d.ptr(packed[2])
        config.gate_type=config.up_type=config.down_type=39; config.hidden_type=hidden_type
        moe=ext.moe.MOE(config); pool.submit(moe.load_weights_task()); pool.sync()
        for tokens in [1,4,9,10,16,17]:
            for mode in ['mixed','all_skipped']:
                x=(rng.standard_normal((tokens,h))*.1).astype(np.float32)
                ids=np.tile(np.arange(3,dtype=np.int64),(tokens,1))
                ids[::2,0]=-1
                if mode=='all_skipped': ids[:]=-1
                elif mode=='duplicate': ids[:]=0
                router=rng.random((tokens,3),dtype=np.float32); router/=router.sum(axis=1,keepdims=True)
                if tokens>1: x[0]=0
                native_input=x if hidden_type==0 else expand(bf16(x))
                ref_input=x if hidden_type==0 else bf16(x)
                actual=np.full((tokens,h),np.nan,np.float32)
                status=lib.native_moe(tokens,3,e,h,inter,*[C.cast(a,P) for a in owners],d.ptr(mask),d.ptr(ids),d.ptr(router),d.ptr(native_input),d.ptr(actual),C.cast(d.ref.quantize_row_q8_0,P))
                assert status==0,status
                expected=np.zeros((tokens,h),np.float32 if hidden_type==0 else np.uint16)
                count=np.array([tokens],np.int32)
                pool.submit(moe.forward_task(d.ptr(count),3,d.ptr(ids),d.ptr(router),d.ptr(ref_input),d.ptr(expected),False)); pool.sync()
                expected=expected if hidden_type==0 else expand(expected)
                actual=actual if hidden_type==0 else expand(bf16(actual))
                assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                diff=float(np.max(np.abs(actual-expected)))
                scale=max(float(np.max(np.abs(expected))),1e-30)
                normalized=diff/scale
                # F32: expf/fast-math and routing FMA may differ. BF16: <=1% peak allows rounding at adjacent bins.
                tolerance=2e-5 if hidden_type==0 else .01
                assert normalized<=tolerance,(label,hidden_type,tokens,mode,diff,normalized)
                if mode=='all_skipped': assert np.count_nonzero(actual)==0
                report['elements']+=actual.size
                report['max_abs_diff']=max(report['max_abs_diff'],diff)
                report['max_normalized_diff']=max(report['max_normalized_diff'],normalized)
                report['cases'].append({'set':label,'dtype':'F32' if hidden_type==0 else 'BF16','tokens':tokens,'mode':mode,'max_abs':diff,'normalized_peak':normalized,'bit_exact':bool(np.array_equal(actual,expected))})
        del moe
    after=[hashlib.sha256(x.tobytes()).hexdigest() for proj in weights for x in proj]
    assert before==after,'native weight buffers mutated'
    print('PASS',label,flush=True)

def synthetic():
    e,h,i=3,256,256
    arrays=[]
    for rows,k in [(i,h),(i,h),(h,i)]:
        w=[rng.integers(0,256,(rows,k//2),np.uint8) for _ in range(e)]
        s=[rng.integers(119,125,(rows,k//32),np.uint8) for _ in range(e)]
        arrays.extend([w,s])
    packed=[]
    for j in [0,2,4]:
        packed.append(np.stack([np.stack([d.repack(w,s) for w,s in zip(arrays[j][a],arrays[j+1][a])]) for a in range(e)]))
    run_set('synthetic',arrays,packed)

def real():
    files=d.NativeFiles(d.Path('/mnt/share/DeepSeek-V4-Flash-0731'))
    for layer in [0,16,42]:
        reader=d.GGUFReader(str(d.ROOT/f'cpu_moe/DeepSeek-V4-Flash-0731-GGUF/dsv4_layer{layer}_mxfp4.gguf'))
        tm={t.name:t for t in reader.tensors}; arrays=[]; packed=[]
        for proj,name in [('w1','gate'),('w3','up'),('w2','down')]:
            w=[];s=[]
            for expert in [0,127,255]:
                key=f'layers.{layer}.ffn.experts.{expert}.{proj}'
                w.append(files.get(key+'.weight'));s.append(files.get(key+'.scale'))
            arrays.extend([w,s])
            tensor=tm[f'blk.{layer}.ffn_{name}_exps.weight']; k,rows,e=map(int,tensor.shape)
            view=np.asarray(tensor.data).reshape(e,rows,k//32*17)
            packed.append(np.ascontiguousarray(view[[0,127,255]]))
        run_set(f'real_L{layer}',arrays,packed)

if __name__=='__main__':
    suite=sys.argv[1] if len(sys.argv)>1 else 'real'
    assert suite in ('real','synthetic')
    start=time.perf_counter()
    (real if suite=='real' else synthetic)()
    report['status']='PASS';report['elapsed_seconds']=time.perf_counter()-start
    report['case_count']=len(report['cases']);report['bit_exact_cases']=sum(c['bit_exact'] for c in report['cases'])
    (d.HERE/f'moe_{suite}_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)
