# CPU 计算后端

本目录保存 CPU MoE 后端及配套源码。与 [vllm-ascend 的 CPU分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928) 配套：Ascend 侧负责专家分工、信号、搬运和同步；这里的 kt-kernel / CPUInfer 负责实际 CPU 专家计算。

| 路径 | 内容 |
| --- | --- |
| [dependencies/ktransformers-AK/kt-kernel/](dependencies/ktransformers-AK/kt-kernel) | C++ CPU专家算子、CPUInfer线程池、GGUF加载及构建入口。 |
| [kt_async_runtime/kt_kernel/](kt_async_runtime/kt_kernel) | Python运行时、后端适配与poll源码。 |
| [dependencies/ktransformers-AK/tools/](dependencies/ktransformers-AK/tools) | MXFP4逐层GGUF转换及校验工具。 |
| [dependencies/ktransformers-AK/third_party/](dependencies/ktransformers-AK/third_party) | 后端构建所用第三方源码和许可证。 |
| `merge_cpu_compute_latest.sh` / `merge_cpu_compute_trace.py` | CPU/NPU日志合成参考实现。当前使用入口见独立的profile合成指南。 |

## 与153已验证环境的对应关系

153已完成整模型启动、推理和性能测试；CPU运行包为 `kt-native-profile-package`，计算采用GGUF，参考成本 **93.16 μs / expert-token（约90 μs）**。

本目录kt-kernel和Python/C运行时源码已从153实测工作区直接同步，包含原生派发接口、动态profile开关、MXFP4复用及MXFP8匹配相关实现。源码文件保持服务器字节内容；276个纳入范围的文件校验值保存在 [version.json](version.json)，运行库、权重、日志、测试/benchmark及依赖自带Markdown不在该校验范围，第三方依赖沿用仓库已有源码。

固定配对：本仓库标签 [`cpu-moe-153-20261109`](https://github.com/yxk-fdu/cpu_moe_compute/tree/cpu-moe-153-20261109/cpu_moe) ↔ Ascend提交 [`7cf63ca`](https://github.com/yxk-fdu/vllm-ascend/commit/7cf63ca477bf6045ad5a262ff6d59b66ace463bd)，vLLM `0.30.0+empty`。**两边都要准备，缺一不可。** 本版本已集成原生派发接口，无需额外接口补丁。

`.so`、GGUF和NPU模型权重不在本目录中。历史脚本保留作参考，实际启动使用 [脚本设定指南](../部署指南及参考脚本/脚本设定指南.md)，日志与权重准备分别见 [profile合成指南](../profile合成指南/README.md) 和 [权重准备指南](../部署指南及参考脚本/权重准备指南.md)。
