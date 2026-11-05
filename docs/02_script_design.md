# CPU MoE 脚本设计

参考源为 22 服务器 `/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/run_dpark_26.sh`，仓库保留原样快照 `reference/run_dpark_26.sh`。这是特定机器和代码版本的参考配置，移植时需要 agent 对齐硬件、容器可用 CPU、库文件、权重和当前配置实现；不能直接作为任意机器的通用启动脚本。

## 双份权重

当前说明和运行参考只支持双份权重方案：CPU 端通过 `weight_format: "gguf"` 与按层 `weight_path` 读取 GGUF；NPU 端从 `vllm serve` 指定的模型目录加载，并使用设备所需的预转换 NZ 缓冲。CPU GGUF 与 NPU 权重必须属于同一模型且层、专家编号和量化约定相容。不能只改路径就假定精度一致。

统一权重是后续方案，目前不成熟，本仓库不把源码中可能存在的 native 分支当作已经支持或验证的方案。

## 必须由后端绑定 CPU 核

`additional_config` 必须保留：

```json
{"enable_cpu_binding": false}
```

这里关闭的是 vLLM Ascend 的额外绑核，CPU MoE 后端仍然需要绑核。kt 后端已经管理计算线程亲和性，再让上层按另一套规则绑核可能缩窄可用 CPU 集合或覆盖已有安排，造成重复绑定、跨 NUMA 和线程争用。不要把 false 理解为允许计算线程任意迁移。

参考中的安排是 `cpuinfer_threads=32`、`threadpool_count=1`、`numa_nodes=[1]`、`KT_MOE_CORE_OFFSET=160`，预期工作核 160–191；poll 核 158，completion 核 159。脚本前面的旧注释仍提到 64–95 和 96/97，已经与实际 export 不一致；必须以运行时每个线程的 affinity 验证，不能照抄旧注释或启动 echo。

## NUMA 与 NPU 亲和性需要分别核对

首先确认计算线程及权重内存放在哪个 NUMA 节点，然后确认选中的物理 NPU 与该节点的 PCIe 或平台互联距离。只设置 `numa_nodes` 不足以证明 NPU 亲和性正确；只选择 NPU 最近节点也不足以证明线程数和 CPU 集合正确。

在目标宿主机及容器中核对：

```bash
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
cat /sys/devices/system/node/node1/cpulist
grep Cpus_allowed_list /proc/self/status
npu-smi info
npu-smi info -t topo
# 若当前 npu-smi 不支持 topo，使用该版本帮助及 PCIe BDF 对应关系查询
# 已确认 NPU 对应的 PCIe BDF 后：
cat /sys/bus/pci/devices/<BDF>/numa_node
cat /sys/bus/pci/devices/<BDF>/local_cpulist
```

`numa_node=-1` 代表内核未报告，不等于节点 0。`ASCEND_RT_VISIBLE_DEVICES=7` 是可见物理卡选择，进程内逻辑卡号可能变为 0，要结合实际映射核对。容器 cpuset 必须包含计算、poll 和 completion 所需核，32 个计算线程应对应允许的 32 个工作核，握手线程放在独立可用核。

启动后检查具体 worker PID 的 `ps -L -p <PID> -o pid,tid,psr,comm` 和 `/proc/<PID>/task/<TID>/status` 中 `Cpus_allowed_list`，并用 `numastat -p <PID>` 检查内存落点。`psr` 只是当时所在核，允许核集合才是绑定依据。需要进程级约束时可以按核对后的节点使用 `numactl --cpunodebind=<NODE> --membind=<NODE> bash <script>`，但外层集合必须容纳后端全部线程，不能再把整个进程固定到某一个计算核。

## 计算与传输调度

`transfer_policy=load_balance` 使用 CPU 计算和 H2D 权重传输的成本估计来选择本次迁入的专家；设备空间不足时由淘汰策略选择可用槽。当前参考实现按 `floor(M * t_cpu / (t_cpu + t_copy))` 计算迁入数量，再受可用槽和 `balance_max_transfers` 限制，M 为本次路由缺失专家数。70 和 450 微秒是配置估计，不能当成实测常数。

`transfer_scheduler=npu` 与自定义 copy 库必须和对应代码、CANN、图捕获流程配套。CPU 计算、权重 H2D、设备计算和结果完成通知是不同阶段；图中重叠关系必须经过时钟校准才能解释。

原脚本顶部写着 next-layer prefetch disabled，但 JSON 实际设置 `expert_prefetch_enabled=true`。解释和复现应以实际配置及代码分支为准，需要关闭预取时显式改为 false 并另行验证，不能沿用顶部注释。`T` 的按需迁入计数也不自动包含独立预取。

## 迁移步骤

先对齐代码和依赖，再确认两份权重与自定义动态库，随后检查 NPU 和 CPU 亲和性、实际绑核及内存落点。先验证 eager 基线和模型结果，再验证图模式，最后采集同次 CPU 与 NPU profiler。不要为了复现截图而移动时间区间，也不能根据截图证明模型正确或性能提升。

新任务的运行日志、临时采集及校时输出统一放 `/mnt/share/y50063808/codex_logs/<任务名>/`。原参考脚本的 `/mnt/sdc/...` 和 latest 链接属于历史环境，执行前应改成当前任务目录；不要直接执行原样快照。
