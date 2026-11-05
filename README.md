# CPU MoE Compute

22 服务器 benchmark 的轻量备份，以及 CPU MoE 运行和 profiler 合成参考。大文件、权重、构建产物和历史 Git 数据不上传。

- [profile 合成指南](docs/01_profile_merge_guide.md)
- [脚本设计与 NUMA 绑核](docs/02_script_design.md)
- [参考脚本参数说明](docs/03_parameters.md)

`benchmark/` 是 2026-11-05 从 `/mnt/share/y50063808/vllm-ascend-workspace/benchmark` 复制的 2637 个文件，总原始字节 62,784,026。保留上游 LICENSE、NOTICE 和第三方声明。它不是包含原始 Git 历史的镜像；缺失大数据集后部分基准需另行准备数据。

`reference/` 保留运行脚本和部分采集实现快照；`scripts/` 包含带真实 T 计数、强制校时和精确窗口裁剪的合成参考版本。模型权重、自定义 .so、完整 vLLM Ascend 和 kt 运行时不在此备份内；它不是可以直接 pip 安装的完整 CPU MoE 后端。

CPU GGUF 与 NPU 模型使用双份权重。统一权重方案尚不成熟。profile 图用于观察时序，不能代替模型正确性验证。

## 排除规则

单文件超过 5 MiB 不上传；排除 `.git`、`__pycache__`、`.eggs`、`.pytest_cache`、`outputs`、`tmp`、`node_modules`、`.venv` 目录，排除符号链接及 `.pyc/.so/.bin/.safetensors/.gguf/.pt/.pth/.log` 和 `.env*` 文件。逐文件排除记录见 `backup_exclusions.json`；目录级规则不逐项展开。文件 SHA256 清单见 `backup_manifest.json`。

保留原始服务器文件和正在运行的服务，本次仅修改备份副本。原始脚本存在旧绑核和预取注释与实际 export/JSON 不一致的情况，已在设计文档明确指出。

源 benchmark HEAD 为 `29c363e38e9d1560e6f19eff582f6117943b6a77`。源目录包含已修改和未跟踪的本地配置，本备份包含筛选规则内的实际工作区文件，并非仅导出该提交。
