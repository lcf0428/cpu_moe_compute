# Local patches applied to `third_party/llama.cpp`

`mxfp4-ggml.patch` adds MXFP4 support — including the `ggml_vec_dot_mxfp4_q8_0`
kernel — to the vendored ggml that `kt-kernel/operators/llamafile/` links against.

- Target: `third_party/llama.cpp` at `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`
  (`update: support Qwen2-57B-A14B (#7835)`)
- Size: 150 insertions across 7 files — `ggml-quants.c`, `ggml-quants.h`,
  `ggml-common.h`, `ggml.c`, `ggml.h`, `gguf-py/gguf/constants.py`,
  `gguf-py/gguf/gguf_reader.py`

Apply:

    cd third_party/llama.cpp
    git apply /path/to/mxfp4-ggml.patch

The patch is carried here as a file because the submodule's origin is the
2024-era upstream `ggerganov/llama.cpp`, which cannot host local work.
