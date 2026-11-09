# CPU MoE

本仓库保存 **CPU 专家计算后端、使用指南和日志合成工具**，配合下面的 Ascend 分支使用。

## 需要下载哪些目录

**CPU 后端要下载完整的 [`cpu_moe/`](cpu_moe/) 目录，保留内部目录结构。推荐直接克隆整个本仓库，这样后端、第三方依赖源码、启动脚本和指南会一起下载。** 只下载其中的 `kt-kernel/` 会漏掉同级依赖；只下载 `cpu_moe/` 则没有根目录的新版启动与合成工具。

| 下载内容 | 用途 |
| --- | --- |
| 本仓库完整 `cpu_moe/` | CPU 后端与配套源码，包含 `dependencies/`、第三方源码及 `kt_async_runtime/`，保持原有相对路径。 |
| 本仓库 `scripts/` | 当前启动脚本、环境变量模板和日志合成工具。 |
| 本仓库 `docs/`、`README.md` | 三份使用指南和配套版本说明。 |
| 另一仓库完整的 `vllm-ascend` 指定版本 | Ascend 侧执行与调度代码；必须另外准备，与CPU后端配套。 |

推荐按以下方式下载到两个新目录：

```bash
# CPU后端及配套脚本/指南：整个仓库一起下载。
git clone --branch cpu-moe-153-20261109 https://github.com/yxk-fdu/cpu_moe_compute.git cpu_moe_compute

# Ascend：下载对应分支，并固定到配套提交。
git clone --branch moe_cpu_compute_v1-on-0928 https://github.com/yxk-fdu/vllm-ascend.git vllm-ascend
git -C vllm-ascend checkout 7cf63ca477bf6045ad5a262ff6d59b66ace463bd
```

下载后按 [脚本设定指南](docs/脚本设定指南.md) 编译/安装。还需要已准备好的 **vLLM 0.30.0 实验环境、NPU模型权重和CPU逐层GGUF权重**；这些环境、权重及编译后的运行库不包含在源码下载中，已有匹配权重可直接复用。

## 两个仓库的关系

| 位置 | 职责 |
| --- | --- |
| [vllm-ascend · moe_cpu_compute_v1-on-0928](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928) | 实际运行的 Ascend 代码：CPU/NPU 专家分工、LRC、专家搬运、预取与同步，以及共享信号和原生任务派发。 |
| [本仓库 cpu_moe/](cpu_moe/) | CPU 计算后端及配套源码。核心 C++ 算子、CPUInfer 线程池和 GGUF 加载代码在 [kt-kernel](cpu_moe/dependencies/ktransformers-AK/kt-kernel/)，Python 运行时在 [kt_async_runtime](cpu_moe/kt_async_runtime/kt_kernel/)。 |
| [scripts/](scripts/) · [patches/](patches/) | 启动入口、环境变量模板、CPU/NPU 日志合成工具及后端接口补丁。 |

Ascend 分支决定**哪些专家在 CPU 算、何时提交计算/搬运、何时同步**；本仓库的 CPU 后端执行**GGUF 专家计算**。运行时需要两边配套，不是只安装本仓库就能启动完整模型。GGUF 是 CPU 权重格式，不是计算后端。

## 配套版本（两边缺一不可）

| 项目 | 本次固定版本 |
| --- | --- |
| vLLM | `0.30.0`，153实际安装 `0.30.0+empty`。 |
| vllm-ascend | [`moe_cpu_compute_v1-on-0928` 分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928)，固定提交 [`7cf63ca477bf6045ad5a262ff6d59b66ace463bd`](https://github.com/yxk-fdu/vllm-ascend/commit/7cf63ca477bf6045ad5a262ff6d59b66ace463bd)。 |
| CPU MoE 后端 | 本仓库固定标签 [`cpu-moe-153-20261109`](https://github.com/yxk-fdu/cpu_moe_compute/tree/cpu-moe-153-20261109/cpu_moe)，源码对应关系及SHA256见 [`cpu_moe/version.json`](cpu_moe/version.json)。 |

这是一组配套版本：准备对应 Ascend 源码，并构建/加载这里的 CPU 后端，才能使用本次 CPU 计算路径。单独更新一边可能产生接口或同步不兼容。

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

`cpu_moe/` 的kt-kernel及运行时源码已直接从153实测工作区同步，276个纳入范围的源码文件逐文件核对SHA256；同步范围排除二进制、权重、日志、测试/benchmark及依赖自带Markdown。原生派发接口和动态profile开关已在源码中，不需再打根目录历史接口补丁。运行包需要使用匹配环境编译，详细入口见 [后端目录说明](cpu_moe/README.md)。

模型权重、编译产物、运行日志和本次性能统计不随使用指南上传。后端依赖及其许可证保留在 `cpu_moe/`，历史实现参考在 `reference/`。

## 后续更新约定

更新GitHub时同时检查Ascend源码、CPU后端与实际运行包的Python/C源码。哪边有变化就同步哪边，不能只上传Ascend改动或把后端改动长期留在补丁里。以实际验证过的源码为准，不在发布前重新格式化；同步后更新配套提交/标签和源码校验清单。没有改动的一边保留原版本引用。
