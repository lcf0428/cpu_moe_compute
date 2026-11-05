# 验证记录

2026-11-05 完成备份及参考实现检查。

- 远端压缩归档与本地归档 SHA256 一致；2637 个 benchmark 文件逐一与归档内容校验一致。
- 检查暂存 benchmark 数量为 2637；提交内容无超过 5 MiB 的文件；扫描未发现 GitHub token、私钥头和 AWS access key 模式。
- `python tests/test_trace_merge.py` 通过：T0/T2/Tunknown、CPU wall clock 故意错位但 RAW marker 正确映射、缺锚点拒绝、精确大时间戳裁剪、原事件保留、匹配 roundtrip 窗口并集与范围外常驻日志过滤、total 事件显式选择且不重复计时。
- Bash 包装器语法检查和 Python 源码语法检查通过；合成器命令行成功写出可解析的 trace、stats 和文本报告。

两个补丁通过隔离副本 git apply --check；total header 在 22 服务器以 g++ 标准库 fixture 验证默认 part/total 互斥与 T/E/C 字段。

未运行模型或重新编译完整 C++ 扩展，未启动或停止现有服务，没有新 NPU profiler 采集。校时接入补丁仍需隔离环境中的实际采集验证，并确认 Ascend exporter 的 host/device 时域映射。测试结果不等于精度正确或性能提升。
