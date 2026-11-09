# CPU MoE

本仓库保存 **CPU 专家计算后端、使用指南和日志合成工具**，配合下面的 Ascend 分支使用。

## 两个仓库的关系

| 位置 | 职责 |
| --- | --- |
| [vllm-ascend · moe_cpu_compute_v1-on-0928](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928) | 实际运行的 Ascend 代码：CPU/NPU 专家分工、LRC、专家搬运、预取与同步，以及共享信号和原生任务派发。 |
| [本仓库 cpu_moe/](cpu_moe/) | CPU 计算后端及配套源码。核心 C++ 算子、CPUInfer 线程池和 GGUF 加载代码在 [kt-kernel](cpu_moe/dependencies/ktransformers-AK/kt-kernel/)，Python 运行时在 [kt_async_runtime](cpu_moe/kt_async_runtime/kt_kernel/)。 |
| [scripts/](scripts/) · [patches/](patches/) | 启动入口、环境变量模板、CPU/NPU 日志合成工具及后端接口补丁。 |

Ascend 分支决定**哪些专家在 CPU 算、何时提交计算/搬运、何时同步**；本仓库的 CPU 后端执行**GGUF 专家计算**。运行时需要两边配套，不是只安装本仓库就能启动完整模型。GGUF 是 CPU 权重格式，不是计算后端。

## 使用指南

- [脚本设定指南](docs/脚本设定指南.md)：安装、启动、NUMA/线程配置、DSpark 与预取开关、重要参数。
- [日志合成指南](docs/日志合成指南.md)：采集 CPU/NPU 日志、同次配对、时钟对齐及合成。
- [权重准备指南](docs/权重准备指南.md)：复用已有 GGUF，或转换与校验逐层权重。

## 已验证配置

这套 CPU MoE 配置已在 **153服务器完成整模型启动、推理和性能测试**：单卡、CPU 计算开启、BS1，覆盖 DSpark 关闭/7、下一层预取关闭/开启。

- vLLM：**0.30.0**，实测安装版本 `0.30.0+empty`，保留同一实验构建。
- Ascend 固定源码：[7cf63ca](https://github.com/yxk-fdu/vllm-ascend/commit/7cf63ca477bf6045ad5a262ff6d59b66ace463bd)；后续开发见上面的分支链接。
- CPU 参考计算成本：**93.16 μs / expert-token，约90 μs**。一个专家处理两个 token 按两份计算量估计，不是按一个唯一专家只算一次。
- 权重搬运参考成本：275.828 μs / 专家；NPU 计算固定估计：200 μs / 层。以上是调度校准值，不是每次执行的固定耗时。

`cpu_moe/` 当前仍是后端源码备份，近期提交主要补充了使用说明和根目录工具；其中的 `ext_bindings.cpp` 尚未内置本次原生派发接口，接口补丁见 [patches/kt_native_dispatch_api.patch](patches/kt_native_dispatch_api.patch)。153实测使用的是已构建的 `kt-native-profile-package`，不能把备份目录直接当作该运行包的完整同步副本。源码与运行包状态见 [cpu_moe/README.md](cpu_moe/README.md)。

模型权重、编译产物、运行日志和本次性能统计不随使用指南上传。后端依赖及其许可证保留在 `cpu_moe/`，历史实现参考在 `reference/`。
