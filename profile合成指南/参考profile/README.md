# 参考profile：17:41 pipeline版本

**当前参考为17:41的逐专家提交pipeline版本**：CPU计算和预取采用共享内存flag+C++轮询，LRC使用C++；搬运按“准备一个专家、提交一个专家”的次序执行。

本样例来自 `dspark7_pipeline_on_20261108-174134`，用于参考CPU计算、专家搬运与预取在合成时间线中的展示方式。原始文件名为 `trace_view_cpu_compute.json`，内容保持原样，没有重新合成或修改时间戳。

[下载参考profile（gzip）](trace_view_cpu_compute.json.gz?raw=true)。原始JSON约312 MB，超过GitHub普通文件100 MB限制，因此以gzip保存。解压后用MindStudio打开 `trace_view_cpu_compute.json`。

```bash
gzip -dk trace_view_cpu_compute.json.gz
sha256sum -c trace_view_cpu_compute.json.sha256
```

解压后的校验值见 [SHA256](trace_view_cpu_compute.json.sha256)。本文件作为展示参考，合成自己的profile仍需同次CPU/NPU日志与可靠校时依据。
