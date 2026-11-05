# CPU MoE Compute

26 服务器 `/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe` 的核心源码、运行脚本与配置备份，以及 profiler 合成参考。核心备份位于 [cpu_moe](cpu_moe/)，共 1621 个文件，22,337,873 字节。核心源码备份排除权重、动态库、构建产物和运行日志；另按用户要求在 `examples/` 保留一份完整 trace 参考。

- [profile 合成指南](docs/01_profile_merge_guide.md)
- [可下载的 CPU 向 NPU 传输权重 trace 样例](examples/traces/first_miss_cpu2npu_weight/README.md)
- [脚本设计与 NUMA 绑核](docs/02_script_design.md)
- [参数设置建议](docs/03_parameters.md)

## 核心代码位置

| 路径 | 内容 |
|---|---|
| [cpu_moe/run_dpark_26.sh](cpu_moe/run_dpark_26.sh) | 26 服务器原始启动脚本，使用前按参数建议调整。 |
| [cpu_moe/kt_async_runtime/kt_kernel](cpu_moe/kt_async_runtime/kt_kernel/) | 实际运行时 Python 代码、poll C 源码等；不含编译后的扩展和 .so。 |
| [cpu_moe/dependencies/ktransformers-AK/kt-kernel](cpu_moe/dependencies/ktransformers-AK/kt-kernel/) | kt 后端 C++ 算子、线程池、CPU 计时、Python 包及构建入口。 |
| [cpu_moe/dependencies/ktransformers-AK/third_party](cpu_moe/dependencies/ktransformers-AK/third_party/) | 当前 CMake 引用的 pybind11、llama.cpp、llamafile 源码。 |
| [cpu_moe/dependencies/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload/main_repo](cpu_moe/dependencies/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload/main_repo/) | CPU MoE 交付参考补丁；未复制整套 CANN 参考工程。 |
| [cpu_moe/expert_rank_gsm8k.json](cpu_moe/expert_rank_gsm8k.json) | 原环境的热专家配置，移植时需核对模型编号与数据分布。 |
| [cpu_moe/merge_cpu_compute_latest.sh](cpu_moe/merge_cpu_compute_latest.sh) | 服务器原始合成入口；原始版本没有应用本仓库新增的强制校时逻辑。 |

`cpu_moe/` 保留筛选范围内的服务器原始文件，包含当前方案与实验性 native 源码；源码存在不代表统一权重方案已成熟。当前文档只支持 CPU GGUF 与 NPU 双份权重方案。

`scripts/` 是带真实 T 计数、强制校时、精确窗口裁剪的改进合成参考，以及待集成补丁；不要与 `cpu_moe/` 中原始同名脚本混淆。`reference/` 保留说明所引用的历史源码快照。CPU/NPU 时钟对齐仍需真实采集验证，不能根据截图证明模型正确或性能收益。

## 备份范围与运行依赖

本次从 26 服务器备份实际工作区文件，未复制 Git 历史。保留所含依赖的 LICENSE 和第三方声明。误传的通用 `benchmark/` 已从当前分支删除。

排除超过 5 MiB 的单文件、权重、编译二进制、缓存、日志、旧归档、备份及实验输出；未复制整套 cann-recipes-infer、ktransformers 非核心目录和未被当前 kt-kernel CMake 引用的第三方工程。详细排除项含目录级规则，见 [backup_exclusions.json](backup_exclusions.json)；源文件 SHA256 和来源见 [backup_manifest.json](backup_manifest.json)、[backup_provenance.json](backup_provenance.json)。内部普通文件链接按内容备份，外部或损坏链接排除。

vLLM 与 vLLM Ascend 主工作区仍需另行准备；本仓库中的适配器片段不是完整 vLLM Ascend。`.so` 和 kt 后端扩展必须按目标机器架构、CANN、torch/torch_npu 等环境准备或重新编译。首次复现建议关闭专家预取，并显式设置 `enable_cpu_binding=false`，由 kt 后端完成绑核。

此次仅备份和更新 GitHub，不修改服务器源码或正在运行的服务。
