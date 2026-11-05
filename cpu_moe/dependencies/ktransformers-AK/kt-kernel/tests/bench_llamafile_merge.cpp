// CPU-only differential check and conversion crossover benchmark.
// Build twice around the production-header change; --check hashes must match.
#include "operators/llamafile/moe.hpp"
#include <bit>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>

using Clock=std::chrono::steady_clock;
static constexpr size_t MaxElements=2*1024*1024;
static GeneralMOEConfig fixture_config(WorkerPool* pool) {
  GeneralMOEConfig c;
  c.expert_num=1;c.num_experts_per_tok=1;c.hidden_size=16384;c.intermediate_size=256;
  c.pool=pool;c.group_min_len=2;c.group_max_len=1;
  c.gate_type=GGML_TYPE_F32;c.up_type=GGML_TYPE_F32;c.down_type=GGML_TYPE_F32;c.hidden_type=GGML_TYPE_BF16;
  return c;
}
struct Fixture: TP_MOE<LLAMA_MOE_TP> {
  std::vector<float> a,b,scratch,pattern;
  std::vector<uint8_t> output;
  explicit Fixture(WorkerPool* pool):TP_MOE<LLAMA_MOE_TP>(fixture_config(pool)),
      a(MaxElements),b(MaxElements),scratch(MaxElements),pattern(MaxElements),output(MaxElements*4) {
    for(size_t i=0;i<MaxElements;++i)pattern[i]=float(int((i*48271+17)%20011)-10005)/97.0f;
  }
  void configure(int hidden,ggml_type type,int parts) {
    config.hidden_size=hidden;config.hidden_type=type;tp_count=parts;
    local_output_numa={a.data()};if(parts==2)local_output_numa.push_back(b.data());local_output=scratch.data();
  }
  void reset(size_t n,bool special) {
    std::copy_n(pattern.data(),n,a.data());
    for(size_t i=0;i<n;++i)b[i]=pattern[(i+37)%MaxElements]*0.125f;
    if(special && n>32) {
      const uint32_t bits[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f808000,
        0x3f818000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc01234};
      for(size_t i=0;i<std::size(bits);++i)a[i]=std::bit_cast<float>(bits[i]);
    }
    std::fill(output.begin(),output.end(),uint8_t(0x5a));
  }
  void prepare_hot(size_t n) {
    config.pool->do_work_stealing_job(32,[this,n](int i){
      const size_t first=n*size_t(i)/32,last=n*size_t(i+1)/32;
      std::copy(pattern.begin()+first,pattern.begin()+last,a.begin()+first);
    });
  }
  void serial_conversion(int tokens) {
    const int h=config.hidden_size;const auto type=ggml_type(config.hidden_type);
    const int chunks=std::clamp(h/256,1,32),chunk=(h+chunks-1)/chunks;
    const size_t bytes=ggml_type_size(type);
    for(int task=0;task<tokens*chunks;++task) {
      const int token=task/chunks,first=(task%chunks)*chunk,last=std::min(h,first+chunk);
      if(first<last)from_float(a.data()+size_t(token)*h+first,
          output.data()+(size_t(token)*h+first)*bytes,last-first,type);
    }
  }
};
static std::string hash_bytes(const uint8_t* p,size_t n) {
  uint64_t h=1469598103934665603ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}
  std::ostringstream out;out<<std::hex<<h;return out.str();
}
static double median(std::vector<double> a) {std::sort(a.begin(),a.end());return a[a.size()/2];}
static double p90(std::vector<double> a) {std::sort(a.begin(),a.end());return a[(a.size()-1)*9/10];}
int main(int argc,char** argv) {
  unsetenv("KT_MOE_CPU_TRACE");unsetenv("KT_MOE_LIFECYCLE_TRACE");unsetenv("KT_MOE_LIFECYCLE_WORKERS");
  unsetenv("KT_MOE_PHASE_TIMING");unsetenv("KT_MOE_NATIVE_POD");unsetenv("KT_MOE_RT_PRIORITY");
  WorkerPoolConfig pc;pc.subpool_count=1;pc.subpool_numa_map={0};pc.subpool_thread_count={32};
  WorkerPool pool(pc);Fixture f(&pool);
  const bool check=argc>1 && std::string(argv[1])=="--check";
  if(check) {
    int cases=0;
    for(int h:{256,257,512,513,4096,7168,8192,16384})for(int tokens:{0,1,2,8,32,128,256})
      for(auto type:{GGML_TYPE_F32,GGML_TYPE_BF16,GGML_TYPE_F16,GGML_TYPE_Q8_0})
        for(int parts:{1,2})for(bool incremental:{false,true}) {
          const size_t n=size_t(h)*tokens;if(n>MaxElements || h%ggml_blck_size(type))continue;
          f.configure(h,type,parts);f.reset(n,type!=GGML_TYPE_Q8_0);
          if(incremental && n)from_float(f.pattern.data(),f.output.data(),int(n),type);
          f.merge_results(tokens,f.output.data(),incremental);
          const size_t bytes=n*ggml_type_size(type)/ggml_blck_size(type);
          std::cout<<"MERGE_JSON {\"kind\":\"check\",\"h\":"<<h<<",\"tokens\":"<<tokens
            <<",\"type\":"<<int(type)<<",\"parts\":"<<parts<<",\"incremental\":"<<incremental
            <<",\"hash\":\""<<hash_bytes(f.output.data(),bytes)<<"\"}\n";
          ++cases;
        }
    std::cout<<"MERGE_JSON {\"kind\":\"check_count\",\"cases\":"<<cases<<"}\n";
  } else {
    const bool one = argc>1 && std::string(argv[1])=="--one";
    for(auto type:{GGML_TYPE_BF16,GGML_TYPE_F16,GGML_TYPE_F32})
      for(int h:{4096,7168,8192,16384})for(int tokens:{1,2,4,8,16,32,64,128,256}) {
        const size_t n=size_t(h)*tokens;if(n>MaxElements)continue;
        if(one && (type!=GGML_TYPE_BF16 || h!=7168 || tokens!=8))continue;
        f.configure(h,type,1);std::vector<double> parallel,serial;
        for(int round=0;round<72;++round) {
          for(int which=0;which<2;++which) {
            const bool serial_run=bool((round+which)&1);
            f.prepare_hot(n);const auto start=Clock::now();
            if(serial_run)f.serial_conversion(tokens);else f.merge_results(tokens,f.output.data(),false);
            const double time=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
            if(round>=8)(serial_run?serial:parallel).push_back(time);
          }
        }
        std::cout<<"MERGE_JSON {\"kind\":\"bench\",\"h\":"<<h<<",\"tokens\":"<<tokens
          <<",\"elements\":"<<n<<",\"type\":"<<int(type)<<",\"production_median_us\":"<<median(parallel)
          <<",\"serial_median_us\":"<<median(serial)<<",\"production_p90_us\":"<<p90(parallel)
          <<",\"serial_p90_us\":"<<p90(serial)<<"}\n";
      }
  }
}
