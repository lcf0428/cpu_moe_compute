// Standalone regression check for architecture-specific MXFP8 dispatch.
#include "../operators/llamafile/mxfp8_match.h"
#include <cstdio>
int main() {
 const bool present = std::getenv("KT_MOE_MATCH_NPU_MXFP8") != nullptr;
#if defined(__aarch64__)
 const bool expected = false;
#else
 const bool expected = present;
#endif
 if (kt_match_npu::enabled() != expected) return 1;
 float input[32];
 for (int i=0;i<32;++i) input[i]=0.12345f*(i+1);
 float original[32];
 std::memcpy(original,input,sizeof(input));
 kt_match_npu::quant_dequant(input,32);
 if (!expected && std::memcmp(original,input,sizeof(input)) != 0) return 2;
 std::printf("PASS matching=%d env_present=%d\n",int(expected),int(present));
 return 0;
}
