# 验证记录

2026-11-05 完成备份及参考实现检查。

- 本次更正为 26 服务器 CPU MoE 核心备份：1621 个文件、22,337,873 字节；远端归档与本地归档 SHA256 一致，全部源文件逐一与远端 manifest 校验一致。
- 移除当前分支误传的通用 benchmark 目录；核心备份排除超过 5 MiB 文件、权重、二进制及运行产物，扫描 GitHub token 和私钥头模式。
- `python tests/test_trace_merge.py` 通过：T0/T2/Tunknown、CPU wall clock 故意错位但 RAW marker 正确映射、缺锚点拒绝、精确大时间戳裁剪、原事件保留、匹配 roundtrip 窗口并集与范围外常驻日志过滤、total 事件显式选择且不重复计时。
- Bash 包装器语法检查和 Python 源码语法检查通过；合成器命令行成功写出可解析的 trace、stats 和文本报告。

两个补丁通过隔离副本 git apply --check；total header 在 22 服务器以 g++ 标准库 fixture 验证默认 part/total 互斥与 T/E/C 字段。

未运行模型或重新编译完整 C++ 扩展，未启动或停止现有服务，没有新 NPU profiler 采集。校时接入补丁仍需隔离环境中的实际采集验证，并确认 Ascend exporter 的 host/device 时域映射。测试结果不等于精度正确或性能提升。

补齐 GGUF 工具：从 26 服务器补传5个转换/校验文件并逐一比对源 SHA256，核心备份现共1626个文件。此次仅核对源码和文档中的参数，未重新转换整套权重。用户确认此前只试过 DeepSeek V4 Flash 与0731版本，其他模型未测试。
