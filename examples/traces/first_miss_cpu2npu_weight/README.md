# CPU 向 NPU 传输权重的 trace 参考

这份 `first_miss/cpu2npu_weight` 采集由用户提供，用于参考 CPU 计算、权重传输及 NPU 事件在合成时间线中的展示效果。文件按原始内容保存，未重新合成或修改时间戳。

[下载 trace_view_cpu_compute.json](trace_view_cpu_compute.json?raw=true)，然后在 Perfetto 或兼容 Chrome trace 的查看器中打开。文件约 84.8 MiB，GitHub 页面可能无法直接预览。SHA256 见同目录 `.sha256` 文件。

它是历史展示样例，不代表已通过本仓库新增校时流程验证，也不证明模型结果正确或存在性能收益。自己的实验仍应按合成指南采集同次日志并核对时钟。
