# 校时与 T 标签实现说明

修改只在备份副本，未覆盖运行中的服务器源码。

## T 的真实语义

`transfer_count` 由 `device_transfer.py::prepare` 的 `active.sum()` 产生，经 `hybrid_executor.py` 把 `-1-T` 放入跳过的 CPU route sentinel；`cpu_compute_trace.h` 对重复 sentinel 取最大值，不求和。标签为 `L4_E1_C1_T2`。T 是该层 demand admission 所安排的专家权重集合数量，每个专家可能包含多个权重 tensor/多次底层 copy。T 不包含其他独立预取，且并不证明传输成功完成。旧日志没有字段时显示 `Tunknown`，不得臆造 T0。

## 原脚本不能直接用于时序比较

原合成脚本直接使用 CPU `wall_ns`，未做 CPU/NPU 时钟对齐。原 `clock_calibration.py` 只记录 NPU Event 调用周围的 CPU RAW 时间区间，没有把对应设备事件时间或唯一标记记录下来；只有这个 JSONL 不能建立可靠跨域映射。不能靠把 CPU 条形塞进 `NOTIFY_WAIT` 区间来校时。

## 下次采集的可操作接入

1. 将 `clock_anchor_helper.py` 放入采集 worker 可 import 的位置。
2. 在同一个处理 `cpu_moe.compute` 的 worker 进程中，torch profiler 真正开始记录后调用 `record_clock_anchor('start', PROF_RAW_DIR)`；在最后一批计算完成后、profiler 停止记录之前调用 `record_clock_anchor('stop', PROF_RAW_DIR)`。多 worker 每个 PID 单独做。原脚本 `delay_iterations=2`，因此不能简单在 profiler.start 刚调用时就打 start marker；应在延迟阶段结束后打。到达 `max_iterations` 自动停止前也要先打 stop。
3. 导出标准 Ascend `trace_view.json`。确认它保留了 `CPU_MOE_CLOCK_ANCHOR_<pid>_<uuid>` 两个 host `record_function` X 事件，以及相应 `cpu_clock_anchors_<pid>.jsonl`。文件中没有 marker、marker 重复、没有 marker 包围范围内的 CPU 记录时合成拒绝继续；服务常驻日志中窗口外记录会排除并计数。不要手填、猜测设备锚点。
4. 调用：

```bash
python3 merge_cpu_compute_trace.py "$TRACE" "$CPU_LOG" \
  --clock-alignment "$CAPTURE/cpu_clock_anchors_${PID}.jsonl" \
  --include-actual-compute --actual-compute-only \
  --out "$OUT/trace_view_cpu_compute.json" \
  --stats "$OUT/cpu_compute_stats.json" \
  --report "$OUT/cpu_compute_report.txt"
```

该实现把 CPU `mono_raw_ns` 用同 PID 的首末两个 host marker 做仿射映射，并保留原始 CLOCK_MONOTONIC duration，检测超过 1000 ppm 的不合理斜率；记录 marker 前后 CPU 包围区间的不确定性和拟合残差。它依赖 **Ascend exporter 已经将 device 时间映射到 host profiler 时域**。Host marker 自身不能修复一份 host/device 未对齐的原始 trace；若该前提不满足，还需要设备时钟与 host 的原始同步元数据。当前未取得同次真实校准采集，不能宣称完成了真机对齐验证。

标记的 frontend 调用有不确定性，用进入 record_function 前后 RAW 区间的中点估计，并在 stats 报告半区间宽度。先 warmup frontend 避免首次调用初始化明显放大误差。不应把误差大的采集用于微秒级结论。

仅需要最小计算日志：`KT_MOE_CPU_TRACE=<capture>/cpu_compute_%p.jsonl`，`CPU_MOE_PHASE_TRACE=1` 是参考启动脚本的条件开关；关闭 lifecycle/worker/phase/task/page 等内部诊断。现有 C++ Span 的 duration 从构造函数末尾开始到析构函数采样时结束，包含这段作用域内线程调度等待，但不等于 device roundtrip，也不是 CPU_THREAD_CPUTIME；实际调用作用域已核对，见下节。

## 验证范围

已通过 Python 标准库 fixture：T0/T2/Tunknown、故意不同 wall 时钟下的 RAW marker 映射、缺 marker 拒绝、1.8e15 微秒绝对时间戳下原始 0.125 微秒事件保持与边界精确裁剪。没有运行模型、没有启动/停止服务、没有取得新的端到端 NPU 采集。

## 当前源码的精确计算边界

调用点：`cpu_moe/dependencies/ktransformers-AK/kt-kernel/operators/llamafile/moe.hpp:1280` 的 `forward()`：

```cpp
kt::cpu_trace::Span trace(config_.layer_idx, tp_part_idx, qlen, k, expert_ids,
                         [this](int64_t id) { return config_.should_skip_expert(id); });
forward_impl(qlen, k, expert_ids, weights, input, output_in);
// Span 析构采样结束时间。
```

它涵盖该 NUMA/TP part 的 `forward_impl()` 到返回，本 part float 输出已经写入；不包含其后上层 `TP_MOE::merge_results()` 的跨 part 归约和输出类型转换，也不含任务 completion 通知、Python/上层返回和 H2D。即使 `threadpool_count=1`，后续输出 dtype 转换仍在范围外。因此现有最小开关适合本图的 **part compute**，不能将其标成完整 CPU 后端调用直到最终结果返回。如果研究目标必须覆盖后端完整返回，需要另加外层 span，不能直接重命名此字段。

E 为非 skip 专家 ID 去重数（固定容量数组；`expert_ids_truncated=true` 时可能不完整），C 为非 skip 路由条目总数 `expert_route_count`。L 为源码零起点 layer 加 1。

## 可审查的当前 profiler 接入补丁

随仓库提供 `clock_anchor_profiler.patch`，针对当前 `vllm_ascend/profiler/torch_npu_profiler.py`，以及原始快照和打补丁后的快照。只修改副本，未动运行服务器。

建议在隔离源码/容器里接入：

```bash
cd "$WS/vllm-ascend"
git apply --check "$BACKUP/scripts/clock_anchor_profiler.patch"
git apply "$BACKUP/scripts/clock_anchor_profiler.patch"
cp "$BACKUP/scripts/clock_anchor_helper.py" \
  vllm_ascend/expert_offload/cpu_backend/clock_anchor_helper.py
export CPU_MOE_CLOCK_ANCHORS=1
```

`$BACKUP/scripts` 应设为仓库内这些脚本的实际目录。原 `CPU_MOE_CLOCK_CALIBRATION=1` 可关闭，它的 Event 包围记录不是本工具输入。

补丁把 start marker 放在 `TorchNPUProfilerWrapper._start()` 的 `self.profiler.start()` 之后，把 stop marker 放在 `_stop()` 的 `torch.npu.synchronize()` 之后、`self.profiler.stop()` 之前。父类 `WorkerProfiler` 的 `delay_iterations=2` 到第二次 step 才真正调用 `_start()`，所以该接法能处理现有 delay=2；自动 max stop 也通过 `_stop()`，不会漏掉尾 marker。

为了第一轮复现容易审查，单独校准采集推荐 `delay_iterations=0,max_iterations=0`：当前父类仅 `_max_iters > 0` 时自动停止，0 表示手动停止。保持 CPU/NPU activities 两者开启。服务启动和 warmup 后：

```bash
curl -f -X POST http://127.0.0.1:8151/start_profile
# 运行一次有限 benchmark，等待全部响应完成。不要留下并发负载。
# 必须等待 benchmark 进程正常结束后再请求 stop。
curl -f -X POST http://127.0.0.1:8151/stop_profile
```

停止钩子的 NPU synchronize 会等待已排队 NPU 工作及其 CPU callback 结果回传，但不替代 benchmark 完成等待；尚未提交的 CPU 任务不能靠 synchronize 保证。随后等待 profiler 解析完成，按 PID 配对 CPU JSONL、marker JSONL 与 Ascend trace，按前述命令合成。已有旧采集没有这些 marker，补丁不能事后恢复缺失的时钟关系。

补丁尚未在新模型运行中验证；这里只做源码核对、Python 语法和标准库 fixture 验证。导出窗口使用本次已校时 CPU compute 与唯一匹配 roundtrip 的连续最早起点/最晚终点并集，保留窗口内所有原轨道原事件，窗口内部 ts/dur 原值不改，只精确裁剪跨界事件。

## 完整 CPU 后端计算结果的可选最小补丁

现有开关 `KT_DECODE_TIMING=1` / `KT_PREFILL_TIMING=1` 在 Python 路径记录 submit→sync 聚合打印，包含提交/排队等待，不是纯计算 span，也未输出本合成工具需要的逐任务 JSONL。`FORWARD_TIME_REPORT` 是 C++ 编译宏，包含详细打印而非最小开关。`KT_MOE_PHASE_TIMING` 是内部阶段耗时统计，本任务不需要。

`cpu_compute_total.patch` 提供一个小型、可审查的补充方案。应用根目录必须是：

```
/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/dependencies/ktransformers-AK/kt-kernel
```

两个目标文件真实存在：`cpu_backend/cpu_compute_trace.h`、`operators/moe-tp.hpp`。在隔离副本执行：

```bash
cd "$KT_KERNEL_COPY"
git apply --check "$BACKUP/scripts/cpu_compute_total.patch"
git apply "$BACKUP/scripts/cpu_compute_total.patch"
# 按当前容器和 kt_kernel 原有构建流程重编扩展，确认实际运行加载了新 .so。
export KT_MOE_CPU_TRACE="$CAPTURE/cpu_compute_%p.jsonl"
export KT_MOE_CPU_TRACE_KIND=total
```

默认不设 `KT_MOE_CPU_TRACE_KIND` 时保持旧 `cpu_moe.compute` 的 part 作用域；设为 `total` 时仅记录新增 `cpu_moe.compute_total`，禁用嵌套 part 事件。新增事件 `part=-1`，范围从外层 TP `forward` 调度 NUMA 计算之前，到 `merge_results` 完成并写出最终 CPU 输出之后，包含线程池调度/join、跨 NUMA 合并及 dtype 转换。不含任务队列排队、后续 completion 通知、Python返回以及 H2D；它代表“CPU计算完成、结果已写出”，不能当成CPU→NPU整体往返。保持旧事件语义，绝不能将 total 与内部 part 时长相加。

合成时显式选择：

```bash
python3 merge_cpu_compute_trace.py "$TRACE" "$CPU_LOG" \
  --compute-event cpu_moe.compute_total \
  --clock-alignment "$CAPTURE/cpu_clock_anchors_${PID}.jsonl" \
  --include-actual-compute --actual-compute-only \
  --out "$OUT/trace_view_cpu_compute.json" \
  --stats "$OUT/cpu_compute_stats.json" \
  --report "$OUT/cpu_compute_report.txt"
```

两个补丁均已在隔离原始副本通过 `git apply --check`。总计时 header 在 22 服务器任务日志目录通过独立 `g++ -std=c++20 -pthread` fixture：默认仅旧 compute、total 模式仅 compute_total，且 T/E/C 字段保持。合成 fixture 通过 total 显式选择且排除嵌套 part。**没有重编整个真实 kt_kernel 扩展，也没有运行模型验证该总计时接入**；补丁保留为待审方案，未应用到线上源码或服务。
