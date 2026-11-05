from pathlib import Path
root=Path('cpu_moe/dependencies/ktransformers-AK/kt-kernel')
p=root/'operators/common.hpp';s=p.read_text();i=s.index('struct GeneralMOEConfig');a=s[:i];b=s[i:];anchor='  // for llamafile\n';assert anchor in b;b=b.replace(anchor,anchor+'  bool native_mxfp4 = false;\n',1);p.write_text(a+b)
p=root/'ext_bindings.cpp';s=p.read_text();anchor='      .def_readwrite("gate_type", &GeneralMOEConfig::gate_type)';assert anchor in s;s=s.replace(anchor,'      .def_readwrite("native_mxfp4", &GeneralMOEConfig::native_mxfp4)\n'+anchor);p.write_text(s)
p=root/'operators/llamafile/moe.hpp';s=p.read_text();s=s.replace('#include "conversion.h"','#include "conversion.h"\n#include "native_mxfp4.h"')
start=s.index('class LLAMA_MOE_TP {');head=s[:start];body=s[start:]
for proj in ['gate','up','down']:
 for obj in ['config','config_']:
  body=body.replace(f'ggml_type_size((ggml_type){obj}.{proj}_type)',f'weight_type_size((ggml_type){obj}.{proj}_type)')
anchor='  GeneralMOEConfig config_;'
helper='''
  size_t weight_type_size(ggml_type type) const {
    return config_.native_mxfp4 ? 16 : ggml_type_size(type);
  }
  bool matmul(const void* scale_base, const uint8_t* weight_base,
              long m, long n, long k, const void* A, long lda, const void* B, long ldb,
              void* C, long ldc, int ith, int nth, int task, ggml_type at, ggml_type bt,
              ggml_type ct, int precision) {
    if (!config_.native_mxfp4) return kt_llamafile_sgemm(m,n,k,A,lda,B,ldb,C,ldc,ith,nth,task,at,bt,ct,precision);
    const auto* w = static_cast<const uint8_t*>(A);
    const auto* scales = static_cast<const uint8_t*>(scale_base) + (w-weight_base)/16;
    const auto* q = static_cast<const uint8_t*>(B);
    auto* out = static_cast<float*>(C);
    for (long col=0;col<n;++col) for (long row=0;row<m;++row) {
      float sum=0;
      kt_native_mxfp4_dot(k*32,w+row*lda*16,scales+row*lda,q+col*ldb*34,&sum);
      out[col*ldc+row]=sum;
    }
    return true;
  }
'''
assert body.count(anchor)==1;body=body.replace(anchor,anchor+helper)
pos=0;calls=0
while True:
 a=body.find('kt_llamafile_sgemm(',pos)
 if a<0:break
 level=1;b=a+len('kt_llamafile_sgemm(')
 while level:
  if body[b]=='(':level+=1
  elif body[b]==')':level-=1
  b+=1
 text=body[a:b];proj=next((x for x in ['gate','up','down'] if x+'_proj_ptr' in text),None)
 if proj:
  repl=f'matmul(config_.{proj}_scale, m_local_{proj}_proj_, '
  body=body[:a]+repl+body[a+len('kt_llamafile_sgemm('):];pos=a+len(repl);calls+=1
 else:pos=b
assert calls==6,calls
anchor='    MemoryRequest mem_requests;'
validation='''    if (config_.native_mxfp4) {
      if (config_.pool->config.subpool_count != 1 || config_.gate_type != GGML_TYPE_MXFP4 ||
          config_.up_type != GGML_TYPE_MXFP4 || config_.down_type != GGML_TYPE_MXFP4 ||
          !config_.gate_proj || !config_.up_proj || !config_.down_proj ||
          !config_.gate_scale || !config_.up_scale || !config_.down_scale ||
          config_.hidden_size % 256 || config_.intermediate_size % 256)
        throw std::invalid_argument("Native MXFP4 requires one NUMA pool, aligned dimensions and all weight/scale pointers");
      const size_t count=(size_t)config_.expert_num*config_.hidden_size*config_.intermediate_size/32;
      for (const void* src : {config_.gate_scale,config_.up_scale,config_.down_scale})
        for (size_t j=0;j<count;++j) if (static_cast<const uint8_t*>(src)[j]==255)
          throw std::invalid_argument("Reserved E8M0 scale 255 in native MXFP4 weights");
    }
'''
body=body.replace(anchor,validation+anchor,1)
anchor='    auto size = 1ll * config.expert_num * config.intermediate_size * config.hidden_size;'
assert anchor in body;body=body.replace(anchor,'''    if (config_.native_mxfp4) {
      m_local_gate_proj_ = static_cast<uint8_t*>(config_.gate_proj);
      m_local_up_proj_ = static_cast<uint8_t*>(config_.up_proj);
      m_local_down_proj_ = static_cast<uint8_t*>(config_.down_proj);
      m_weights_aliased_ = true;
      return;
    }
'''+anchor)
anchor='    auto& config = config_;\n';assert anchor in body
body=body.replace(anchor,anchor+'''    if (config_.native_mxfp4) {
      if (offset != 0 || complete_intermediate_size != config_.intermediate_size)
        throw std::invalid_argument("Native MXFP4 cannot copy or repartition borrowed weights");
      return;
    }
''',1)
anchor='      for (int i = 0; i < 3; ++i) {'
a=body.index(anchor,body.index('  void pin_weights()'))
body=body[:a]+body[a:].replace(anchor,'''      if (config_.native_mxfp4) {
        for (void* scale : {config_.gate_scale,config_.up_scale,config_.down_scale}) {
          const size_t bytes=elements/32;
          if (mlock(scale,bytes)!=0) throw std::runtime_error("mlock failed for native MXFP4 scale");
          locked_weight_ranges_.emplace_back(scale,bytes);
        }
      }
'''+anchor,1)
# pin_weights types[] path must also use native physical block bytes.
body=body.replace('elements * ggml_type_size((ggml_type)types[i])','elements * weight_type_size((ggml_type)types[i])')
p.write_text(head+body)
print('Patched existing KT native matmul calls:',calls)
