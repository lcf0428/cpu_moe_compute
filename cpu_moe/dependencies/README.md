# CPU MoE dependencies

This directory contains the external sources used by the vLLM-Ascend hybrid
CPU/NPU MoE prototype.

## Layout

- `cann-recipes-infer/`: reference delivery, branch
  `dsv4-npu-fullfeature-a3`.
- `ktransformers-AK/`: ktransformers baseline `d7b5b49`, local branch
  `dsv4-npu-release`.

The ktransformers working tree has these reference patches applied but not
committed:

1. `0001-kt-kernel-ascend-npu-backend.patch`
2. `0002-kt-kernel-cpu-moe-mxfp4-kernel.patch`

Patch sources are under:

`cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload/main_repo/`

Required upstream submodules already initialized:

- `third_party/pybind11`
- `third_party/custom_flashinfer`

No build or installation has been performed yet.

The CPU backend source is:

`ktransformers-AK/kt-kernel/`

The vLLM-Ascend adapter imports `KTMoEWrapper` from the resulting `kt_kernel`
Python package after it is built and made available on `PYTHONPATH`.
