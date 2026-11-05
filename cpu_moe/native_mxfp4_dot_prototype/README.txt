Native MXFP4 CPU dot prototype - 2026-10-13 (client date)

Result: PASS. 109280/109280 dot products bit-exact against ggml_vec_dot_mxfp4_q8_0 exported from the CURRENT built kt_kernel_ext.so; max absolute difference 0.
Independent float64 mathematical oracle sampled on synthetic and real rows: maximum error / sum(abs(products)) = 4.2971155942437274e-08. This metric is not pointwise relative error near cancellation.

Coverage:
- All 256 packed byte values x all 32 one-hot K positions (8192 checks).
- Every finite E8M0 exponent 0..254; reserved 255 explicitly rejected.
- K=32,64,96,128,160,256,512,2048,4096,8192, including odd/even block counts, random inputs, zeros, constants and sign patterns.
- Real model DeepSeek-V4-Flash-0731, layers 0/16/42, expert 0 all rows and experts 127/255 sampled boundary/random rows, all w1/w3/w2 projections.
- 25148 real weight rows: native bytes repacked by the test fixture equal bytes from actual existing GGUF files. Each row tested with 4 activation patterns.
- 10 invalid argument checks, unaligned addresses and guard-byte checks.
- ASan+UBSan: 1000 exact-size allocation tests passed; no errors or leaks reported.

Kernel behavior:
Native weights and E8M0 scales are separate pointers. NEON loads 16 consecutive-packed bytes, performs nibble LUT, zip1/zip2 to restore K order in registers, SDOT with existing Q8_0 activations, and the same two FMA accumulator chains as the deployed reference. No weight allocation, full-row repack or BF16 expert expansion occurs inside native_dot. The harness creates a GGML repack only to verify reference bytes.

Files:
native_dot.c: prototype NEON kernel, API returns 0 / -1 invalid argument / -2 reserved E8M0 scale.
check_dot.py: actual deployed GGML ctypes calls, existing Q8_0 quantization, independent oracle, native/safetensors and existing GGUF tests. Loads only GGUF reader to avoid optional tokenizer dependency.
check_bounds.c: sanitizer harness.
results.json, check.log, bounds.log: evidence.
source.sha256: source and reference binary identities.

Reproduce on 105:
cd /mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/native_mxfp4_dot_prototype
gcc -O3 -std=c11 -shared -fPIC -march=armv8.2-a+fp16+dotprod -Wall -Wextra -Werror native_dot.c -o libnative_dot.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 check_dot.py
gcc -O1 -g -std=c11 -march=armv8.2-a+fp16+dotprod -fsanitize=address,undefined -fno-omit-frame-pointer native_dot.c check_bounds.c -o check_bounds
ASAN_OPTIONS=detect_leaks=1 ./check_bounds

Scope and limits:
Independent CPU-only dot prototype; installed backend and serving scripts unchanged. No NPU workload launched. Results establish tested native layout/scale/dot equivalence with the existing Q8_0 activation path, not complete MoE backend correctness or performance. Full gate/up/down integration, grouped tokens, activation, routing reduction, ownership/NUMA/pinning and graph/NPU transfers remain separate acceptance steps. Harness duration is not an inference latency benchmark.

Pre-existing full code backup:
/mnt/share/y50063808/vllm-ascend-workspace/moe_offload/backup_single_native_20261013_012709/source.tar
SHA256 verified in previous inspection. Includes vllm-ascend and cpu_moe/dependencies with uncommitted modifications, submodules and build artifacts.
