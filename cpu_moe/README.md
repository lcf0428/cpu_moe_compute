# CPU 计算后端

本目录保存 CPU MoE 后端及配套源码。与 [vllm-ascend 的 CPU分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928) 配套：Ascend 侧负责专家分工、信号、搬运和同步；这里的 kt-kernel / CPUInfer 负责实际 CPU 专家计算。

| 路径 | 内容 |
| --- | --- |
| [dependencies/ktransformers-AK/kt-kernel/](dependencies/ktransformers-AK/kt-kernel/) | C++ CPU专家算子、CPUInfer线程池、GGUF加载及构建入口。 |
| [kt_async_runtime/kt_kernel/](kt_async_runtime/kt_kernel/) | Python运行时、后端适配与poll源码。 |
| [dependencies/ktransformers-AK/tools/](dependencies/ktransformers-AK/tools/) | MXFP4逐层GGUF转换及校验工具。 |
| [dependencies/ktransformers-AK/third_party/](dependencies/ktransformers-AK/third_party/) | 后端构建所用第三方源码和许可证。 |
| `merge_cpu_compute_latest.sh` / `merge_cpu_compute_trace.py` | CPU/NPU日志合成参考实现。当前使用入口见根目录指南。 |

## 与153已验证环境的对应关系

153已完成整模型启动、推理和性能测试；CPU运行包为 `kt-native-profile-package`，计算采用GGUF，参考成本 **93.16 μs / expert-token（约90 μs）**。

本目录核心备份尚未完整同步该运行包：当前 `ext_bindings.cpp` 不含 `native_dispatch_api`、`native_cpu_handle` 及动态profile开关；根目录 [原生接口补丁](../patches/kt_native_dispatch_api.patch) 补充派发接口，不能替代整个已验证运行包，也不包含全部profile改动。当前复现优先使用该运行包，不应将本目录直接标为153最新源码副本。

`.so`、GGUF和NPU模型权重不在本目录中。历史脚本保留作参考，实际启动使用 [脚本设定指南](../docs/脚本设定指南.md)，日志与权重准备分别见 [日志合成指南](../docs/日志合成指南.md) 和 [权重准备指南](../docs/权重准备指南.md)。
