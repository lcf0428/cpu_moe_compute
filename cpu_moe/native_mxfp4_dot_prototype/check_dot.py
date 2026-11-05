"""Native NEON vs deployed GGML .so vs independent float64 oracle. CPU only."""
import ctypes as C
import json
import mmap
from pathlib import Path
import struct
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
KT=ROOT/'cpu_moe/dependencies/ktransformers-AK'
sys.path.insert(0,str(KT/'third_party/llama.cpp/gguf-py'))
# Load only the reader, without the package tokenizer imports.
import types
pkg=types.ModuleType('gguf')
pkg.__path__=[str(KT/'third_party/llama.cpp/gguf-py/gguf')]
sys.modules['gguf']=pkg
from gguf.gguf_reader import GGUFReader
P=C.c_void_p
native=C.CDLL(str(HERE/'libnative_dot.so'))
native.native_dot.argtypes=[C.c_int,P,P,P,P]; native.native_dot.restype=C.c_int
ref=C.CDLL(str(KT/'kt-kernel/build/lib.linux-aarch64-cpython-311/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'))
ref.ggml_vec_dot_mxfp4_q8_0.argtypes=[C.c_int,P,C.c_size_t,P,C.c_size_t,P,C.c_size_t,C.c_int]
ref.quantize_row_q8_0.argtypes=[P,P,C.c_int64]
rng=np.random.default_rng(20261013)
stats={'comparisons':0,'bit_exact':0,'max_abs_diff':0.,'max_oracle_scaled_error':0.,'invalid_checks':0,'real_weight_rows':0,'cases':{}}
def ptr(a): return a.ctypes.data

def repack(w,s):
    # Test fixture only. Native kernel never calls this or reads these bytes.
    blocks=w.reshape(-1,16)
    n=np.empty((len(blocks),32),dtype=np.uint8)
    n[:,0::2]=blocks&15; n[:,1::2]=blocks>>4
    out=np.empty((len(blocks),17),dtype=np.uint8)
    out[:,0]=s; out[:,1:]=n[:,:16]|(n[:,16:]<<4)
    return out.reshape(-1)

def quantize(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    q=np.empty(a.size//32*34,dtype=np.uint8)
    ref.quantize_row_q8_0(ptr(a),ptr(q),a.size)
    return q

def oracle(w,s,q):
    table=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float64)
    # Independent mathematical E8M0 definition (valid finite exponents 0..254).
    values=np.empty(w.size*2,dtype=np.float64)
    values[0::2]=table[w&15]; values[1::2]=table[w>>4]
    wb=values.reshape(-1,32)*np.exp2(s.astype(np.float64)-127)[:,None]
    qb=q.reshape(-1,34)
    d=np.ascontiguousarray(qb[:,:2]).reshape(-1).view(np.float16).astype(np.float64)
    a=qb[:,2:].view(np.int8).astype(np.float64)*d[:,None]
    product=wb*a
    return float(product.sum(dtype=np.float64)),float(np.abs(product).sum(dtype=np.float64))

def check(w,s,q,packed=None,case='synthetic',use_oracle=True):
    w=np.ascontiguousarray(w,dtype=np.uint8); s=np.ascontiguousarray(s,dtype=np.uint8)
    q=np.ascontiguousarray(q,dtype=np.uint8)
    k=w.size*2
    packed=repack(w,s) if packed is None else np.ascontiguousarray(packed,dtype=np.uint8)
    x=C.c_float(); y=C.c_float()
    status=native.native_dot(k,ptr(w),ptr(s),ptr(q),C.byref(x))
    assert status==0,(case,status)
    ref.ggml_vec_dot_mxfp4_q8_0(k,C.byref(y),0,ptr(packed),packed.size,ptr(q),q.size,1)
    assert np.isfinite(x.value) and np.isfinite(y.value),(case,x.value,y.value)
    stats['comparisons']+=1
    stats['cases'][case]=stats['cases'].get(case,0)+1
    diff=abs(x.value-y.value)
    stats['max_abs_diff']=max(stats['max_abs_diff'],diff)
    if struct.pack('f',x.value)==struct.pack('f',y.value): stats['bit_exact']+=1
    # Require exact equality against the actual deployed kernel for this prototype.
    assert struct.pack('f',x.value)==struct.pack('f',y.value),(case,k,x.value,y.value)
    if use_oracle:
        truth,magnitude=oracle(w,s,q)
        err=abs(x.value-truth)/max(magnitude,1e-38)
        stats['max_oracle_scaled_error']=max(stats['max_oracle_scaled_error'],err)
        # Underflow in float32 is bounded separately for the lowest E8M0 exponents.
        assert abs(x.value-truth)<=3e-6*magnitude+2e-43,(case,k,x.value,truth,err)

def synthetic():
    # Cover every byte: each low/high nibble pairing and position-specific one-hot activations.
    for byte in range(256):
        w=np.full(16,byte,dtype=np.uint8); s=np.array([127],np.uint8)
        for pos in range(32):
            a=np.zeros(32,np.float32); a[pos]=1
            check(w,s,quantize(a),case='all_nibbles_onehot')
    for exponent in range(255):
        w=np.arange(16,dtype=np.uint8)*17
        q=np.zeros(34,np.uint8)
        # At high exponents use a small activation scale to avoid FP32 overflow.
        q[:2]=np.array([2.**(-14 if exponent>240 else 0)],np.float16).view(np.uint8)
        q[2:]=np.tile(np.array([1,-1],np.int8),16).view(np.uint8)
        check(w,np.array([exponent],np.uint8),q,case='all_finite_e8m0')
    for k in [32,64,96,128,160,256,512,2048,4096,8192]:
        for seed in range(24):
            w=rng.integers(0,256,k//2,dtype=np.uint8)
            s=rng.integers(110,137,k//32,dtype=np.uint8)
            a=rng.standard_normal(k).astype(np.float32)
            if seed%6==0: a.fill(0)
            elif seed%6==1: a[1::2]*=-10
            elif seed%6==2: a.fill(1)
            check(w,s,quantize(a),case='random_odd_even_blocks')
    out=C.c_float(123); w=np.zeros(16,np.uint8); s=np.array([127],np.uint8); q=np.zeros(34,np.uint8)
    for k in [-32,0,1,31,33]:
        assert native.native_dot(k,ptr(w),ptr(s),ptr(q),C.byref(out))==-1
        assert out.value==123; stats['invalid_checks']+=1
    s[0]=255
    assert native.native_dot(32,ptr(w),ptr(s),ptr(q),C.byref(out))==-2
    assert out.value==123; stats['invalid_checks']+=1
    for index in [1,2,3,4]:
        args=[32,ptr(w),ptr(s),ptr(q),C.byref(out)]; args[index]=None
        assert native.native_dot(*args)==-1; stats['invalid_checks']+=1
    # Deliberately unaligned input addresses with guard bytes.
    s[0]=127
    ww=np.full(18,199,np.uint8); ww[1:17]=rng.integers(0,256,16,dtype=np.uint8)
    qq=np.full(36,199,np.uint8); qq[1:35]=quantize(rng.standard_normal(32))
    check(ww[1:17],s,qq[1:35],case='unaligned')
    assert ww[0]==ww[-1]==qq[0]==qq[-1]==199

class NativeFiles:
    def __init__(self,root):
        self.root=root; self.weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']; self.cache={}
    def get(self,key):
        name=self.weight_map[key]
        if name not in self.cache:
            f=(self.root/name).open('rb'); mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
            size=struct.unpack('<Q',mm[:8])[0]; self.cache[name]=(f,mm,json.loads(mm[8:8+size]),8+size)
        f,mm,hdr,base=self.cache[name]; h=hdr[key]
        return np.ndarray(h['shape'],dtype=np.uint8,buffer=mm,offset=base+h['data_offsets'][0])

def real():
    files=NativeFiles(Path('/mnt/share/DeepSeek-V4-Flash-0731'))
    for layer in [0,16,42]:
        path=ROOT/f'cpu_moe/DeepSeek-V4-Flash-0731-GGUF/dsv4_layer{layer}_mxfp4.gguf'
        reader=GGUFReader(str(path)); tensors={t.name:t for t in reader.tensors}
        for proj,name in [('w1','gate'),('w3','up'),('w2','down')]:
            t=tensors[f'blk.{layer}.ffn_{name}_exps.weight']
            k,rows,experts=map(int,t.shape)
            packed=np.asarray(t.data).reshape(experts,rows,k//32*17)
            for eid in [0,127,255]:
                key=f'layers.{layer}.ffn.experts.{eid}.{proj}'
                w=files.get(key+'.weight'); s=files.get(key+'.scale')
                assert w.shape==(rows,k//2) and s.shape==(rows,k//32)
                assert not np.any(s==255),key
                # All rows for expert 0; boundary and scattered rows for others.
                selected=range(rows) if eid==0 else sorted(set([0,1,rows//2,rows-2,rows-1]+rng.integers(0,rows,27).tolist()))
                activations=[quantize(rng.standard_normal(k)),quantize(np.ones(k)),quantize(np.zeros(k)),quantize(np.tile([1.,-1.],k//2))]
                for row in selected:
                    fixture=repack(w[row],s[row])
                    assert np.array_equal(fixture,packed[eid,row]),(key,row,'GGUF bytes mismatch')
                    stats['real_weight_rows']+=1
                    for q in activations:
                        check(w[row],s[row],q,packed[eid,row],case=f'real_L{layer}_{proj}',use_oracle=(row%127==0))
            print(f'PASS real layer={layer} projection={proj}',flush=True)

if __name__=='__main__':
    start=time.perf_counter()
    synthetic(); print('PASS synthetic',flush=True)
    real()
    stats['elapsed_seconds']=time.perf_counter()-start
    stats['status']='PASS'
    (HERE/'results.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2),flush=True)
