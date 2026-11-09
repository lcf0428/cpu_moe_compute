# CPU MoE 使用说明

本仓库是 **CPU MoE 的使用入口**：说明组件关系、依赖、安装、启动、NUMA 配置和 profiler 采集，并保留 CPU 计算后端参考源码。

**实际运行的 Ascend 插件源码在 [yxk-fdu/vllm-ascend · moe_cpu_compute_v1-on-0928](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928)。** 本次对应的固定源码版本为 [7cf63ca](https://github.com/yxk-fdu/vllm-ascend/commit/7cf63ca477bf6045ad5a262ff6d59b66ace463bd)，提交文件已与保存的实测源码快照逐文件核对。后续跟进开发看分支，复现本次配置使用固定提交。

## 组件各自负责什么

| 组件 | 作用与来源 |
| --- | --- |
| vLLM 0.30.0 实验构建 | API 服务、请求调度、模型执行框架及 DSpark 集成。153 使用 `0.30.0+empty`，需保留包含 DeepSeek V4/DSpark 支持的实验源码与构建。 |
| 上面的 vllm-ascend 分支 | Ascend/NPU 适配、专家驻留与搬运、CPU/NPU 负载分配、C++ LRC、原生信号监测及预取同步。真实执行逻辑以该仓库为准。 |
| kt-kernel / CPUInfer | CPU 专家数学计算、GGUF 加载、计算线程池；通过原生 API 接收 Ascend 调度器提交的任务。本仓库 `cpu_moe/` 保留后端参考源码，`patches/` 提供原生接口补丁。 |
| 本仓库的 scripts / docs | 运行脚本和使用说明，连接以上组件；不替代完整的 vLLM 或 vllm-ascend。 |
| NPU 模型权重与 CPU GGUF | 分别供 NPU 和 CPU 专家计算使用。GGUF 是 CPU 权重数据格式，kt-kernel 才是 CPU 计算后端。两份权重均需另行准备。 |

## 按 153 跑通的配置使用

1. 阅读 [vLLM 0.30 使用流程与重要参数](docs/05_vllm030_native_dispatch.md)。
2. 将 [环境变量模板](scripts/cpu_moe.env.example) 复制到本机并填写已有路径。
3. 在已验证容器中执行 [单卡启动入口](scripts/run_cpu_moe_vllm030.sh)，选择 DSpark 和预取开关。
4. 按 [验证与短 profile](docs/05_vllm030_native_dispatch.md#验证与短-profile) 确认接口、生成和采集。

本次验证对象为单卡、CPU 计算开启、BS1，DSpark 0/7 × 下一层预取关/开。vLLM 固定 0.30.0，Torch `2.10.0+cpu`、Torch NPU `2.10.0.post4`、CANN 9.2.0。当前入口按实测参数整理；原始逐轮 `run.sh` 尚未重新从服务器下载核对，不标为原件，也不声称已完成全新容器从零重建验证。优先复用153已验证容器中的 vLLM 和 kt 运行包。

性能统计表和本次测试日志保存在本地结果目录，使用说明仓库不收录这些统计结果。

## 参考资料与历史备份

- [GGUF 转换](docs/04_gguf_generation.md)
- [CPU / NPU 日志合成](docs/01_profile_merge_guide.md)
- [NUMA 与线程设计](docs/02_script_design.md)
- [原生接口补丁](patches/kt_native_dispatch_api.patch)

以下目录说明属于历史源码备份。当前启动入口以前面的 vLLM 0.30 使用流程为准；旧脚本中的参数和 callback 实现不能直接当作当前配置。

## 路径约定

**操作说明中的核心工具路径以 `cpu_moe/` 目录为基准。目录内部相对位置固定，`cpu_moe` 的绝对路径根据实际服务器或本地工作区确定，不绑定某台服务器。**

例如，GGUF 工具相对位置是 `dependencies/ktransformers-AK/tools/`，参考启动脚本是 `run_dpark_26.sh`，原始合成入口是 `merge_cpu_compute_latest.sh`。文件名中的 26 是原有命名，不表示只能在该服务器使用。

本仓库另有位于 `cpu_moe/` 外面的 `scripts/`、`docs/` 和 `reference/`，属于仓库根目录，文档会明确区分。下面的可点击链接使用仓库相对路径。

## 核心代码位置

| 路径 | 内容 |
|---|---|
| [cpu_moe/run_dpark_26.sh](cpu_moe/run_dpark_26.sh) | 原始参考启动脚本，使用前按参数建议调整。 |
| [cpu_moe/kt_async_runtime/kt_kernel](cpu_moe/kt_async_runtime/kt_kernel/) | 实际运行时 Python 代码、poll C 源码等；不含编译后的扩展和 .so。 |
| [cpu_moe/dependencies/ktransformers-AK/kt-kernel](cpu_moe/dependencies/ktransformers-AK/kt-kernel/) | kt 后端 C++ 算子、线程池、CPU 计时、Python 包及构建入口。 |
| [cpu_moe/dependencies/ktransformers-AK/third_party](cpu_moe/dependencies/ktransformers-AK/third_party/) | 当前 CMake 引用的 pybind11、llama.cpp、llamafile 源码。 |
| [cpu_moe/dependencies/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload/main_repo](cpu_moe/dependencies/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload/main_repo/) | CPU MoE 交付参考补丁；未复制整套 CANN 参考工程。 |
| [cpu_moe/dependencies/ktransformers-AK/tools](cpu_moe/dependencies/ktransformers-AK/tools/) | 原生 MXFP4 到逐层 GGUF 的转换、批处理和校验工具。 |
| [cpu_moe/expert_rank_gsm8k.json](cpu_moe/expert_rank_gsm8k.json) | 原环境的热专家配置，移植时需核对模型编号与数据分布。 |
| [cpu_moe/merge_cpu_compute_latest.sh](cpu_moe/merge_cpu_compute_latest.sh) | 服务器原始合成入口；原始版本没有应用本仓库新增的强制校时逻辑。 |

`cpu_moe/` 保留筛选范围内的服务器原始文件，包含当前方案与实验性 native 源码；源码存在不代表统一权重方案已成熟。当前文档只支持 CPU GGUF 与 NPU 双份权重方案。GGUF 转换目前仅在 DeepSeek V4 Flash 和 DeepSeek V4 Flash 0731 上试过，其他模型未测试。

`scripts/` 是带真实 T 计数、强制校时、精确窗口裁剪的改进合成参考，以及待集成补丁；不要与 `cpu_moe/` 中原始同名脚本混淆。`reference/` 保留说明所引用的历史源码快照。CPU/NPU 时钟对齐仍需真实采集验证，不能根据截图证明模型正确或性能收益。

## 备份范围与运行依赖

备份保留实际工作区文件，未复制 Git 历史；来源信息记录在 `backup_provenance.json`。保留所含依赖的 LICENSE 和第三方声明。误传的通用 `benchmark/` 已从当前分支删除。

排除超过 5 MiB 的单文件、权重、编译二进制、缓存、日志、旧归档、备份及实验输出；未复制整套 cann-recipes-infer、ktransformers 非核心目录和未被当前 kt-kernel CMake 引用的第三方工程。详细排除项含目录级规则，见 [backup_exclusions.json](backup_exclusions.json)；源文件 SHA256 和来源见 [backup_manifest.json](backup_manifest.json)、[backup_provenance.json](backup_provenance.json)。内部普通文件链接按内容备份，外部或损坏链接排除。

vLLM 与 vLLM Ascend 主工作区仍需另行准备；本仓库中的适配器片段不是完整 vLLM Ascend。`.so` 和 kt 后端扩展必须按目标机器架构、CANN、torch/torch_npu 等环境准备或重新编译。首次复现建议关闭专家预取，并显式设置 `enable_cpu_binding=false`，由 kt 后端完成绑核。

以上 `cpu_moe/` 是原始备份说明；2026-11-09 的当前CPU计算运行入口和使用说明见前面的 vLLM 0.30 章节。
