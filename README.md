# CPU MoE

本仓库分为三个部分：

| 目录 | 内容 |
| --- | --- |
| [cpu_moe/](cpu_moe/) | **CPU计算配套实现**：计算后端、Python运行时、依赖源码和GGUF工具。使用CPU计算时下载完整目录，保持内部结构。 |
| [部署指南及参考脚本/](部署指南及参考脚本/) | 仅保留[参考启动脚本](部署指南及参考脚本/参考启动脚本.sh)、[脚本设定指南](部署指南及参考脚本/脚本设定指南.md)、[权重准备指南](部署指南及参考脚本/权重准备指南.md)。 |
| [profile合成指南/](profile合成指南/) | CPU/NPU日志采集、时钟对齐、合成脚本和样例。需要分析profile时使用；[复制agent合成指令](profile合成指南/如何让agent合成profile.md)，默认只下载最终JSON到本地桌面。 |

## 下载什么

**准备CPU计算后端，下载完整的 [`cpu_moe/`](cpu_moe/) 目录即可。** 不要只取其中的 `kt-kernel/`，它还需要同级依赖与运行时文件。另从 [部署指南及参考脚本](部署指南及参考脚本/) 取使用说明、启动脚本；日志合成工具在独立的 [profile合成指南](profile合成指南/) 中。无需为了准备后端下载其他说明或日志样例。

## 两个仓库的关系

还必须单独准备 [vllm-ascend 的 `moe_cpu_compute_v1-on-0928` 分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928)。**Ascend侧代码与CPU计算后端缺一不可**：前者负责CPU/NPU专家分工、LRC、搬运、预取和同步；本仓库 `cpu_moe/` 负责CPU专家计算。

| 组件 | 配套版本 |
| --- | --- |
| vLLM | **0.30.0**，实装 `0.30.0+empty`，使用同一实验构建。 |
| vllm-ascend | 固定提交 [`7cf63ca`](https://github.com/yxk-fdu/vllm-ascend/commit/7cf63ca477bf6045ad5a262ff6d59b66ace463bd)。 |
| CPU计算后端 | 固定标签 [`cpu-moe-153-20261109`](https://github.com/yxk-fdu/cpu_moe_compute/tree/cpu-moe-153-20261109/cpu_moe)，校验清单见 [cpu_moe/version.json](cpu_moe/version.json)。 |

这套配对已在 **153服务器完成整模型启动、推理和性能测试**：单卡、BS1、CPU计算开启，覆盖DSpark关闭/7及下一层预取关闭/开启。CPU参考成本为 **93.16 μs / expert-token，约90 μs**；同一专家处理两个token按两份计算量估计。

后端源码已从153实测工作区同步。模型权重、GGUF和编译后的运行库另行准备。以后更新GitHub时同时检查两边源码，有改动的一边同步更新，并记录新的配套版本。
