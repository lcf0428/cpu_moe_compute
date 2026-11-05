# CPU MoE 脚本设计

参考启动脚本相对于 `cpu_moe/` 的路径为 `run_dpark_26.sh`，`cpu_moe` 的绝对路径由实际环境确定。完整核心备份位于仓库的 `cpu_moe/`，另在 `reference/run_dpark_26.sh` 保留说明用快照。这是特定机器和代码版本的参考配置，移植时需要 agent 对齐硬件、容器可用 CPU、库文件、权重和当前配置实现；不能直接作为任意机器的通用启动脚本。

## 双份权重

当前说明和运行参考只支持双份权重方案：CPU 端通过 `weight_format: "gguf"` 与按层 `weight_path` 读取 GGUF；NPU 端从 `vllm serve` 指定的模型目录加载，并使用设备所需的预转换 NZ 缓冲。CPU GGUF 与 NPU 权重必须属于同一模型且层、专家编号和量化约定相容。不能只改路径就假定精度一致。

统一权重是后续方案，目前不成熟，本仓库不把源码中可能存在的 native 分支当作已经支持或验证的方案。

## 必须由后端绑定 CPU 核

**必须在 `additional_config` 中显式设置 `enable_cpu_binding=false`，不要省略后依赖默认值。**

```json
{"enable_cpu_binding": false}
```

这里关闭的是 vLLM Ascend 的额外绑核，CPU MoE 后端仍然需要绑核。kt 后端已经管理计算线程亲和性，再让上层按另一套规则绑核可能缩窄可用 CPU 集合或覆盖已有安排，造成重复绑定、跨 NUMA 和线程争用。不要把 false 理解为允许计算线程任意迁移。

参考中的安排是 `cpuinfer_threads=32`、`threadpool_count=1`、`numa_nodes=[1]`、`KT_MOE_CORE_OFFSET=160`，预期工作核 160–191；poll 核 158，completion 核 159。脚本前面的旧注释仍提到 64–95 和 96/97，已经与实际 export 不一致；必须以运行时每个线程的 affinity 验证，不能照抄旧注释或启动 echo。

## NUMA 与 NPU 亲和性需要分别核对

**每台服务器的 NUMA 拓扑、CPU 编号、NPU 连接关系和容器可用核都可能不同。必须根据本次实际使用的 NPU card 确定亲和性和绑核方案，不能直接照抄本参考的 NUMA 1、160–191、158/159；同一台服务器换一张 NPU card，也必须重新确认。**

大致流程如下：

1. **先确定实际使用的 NPU card。** 核对宿主机设备编号、芯片/设备映射及容器内可见设备。进程内的逻辑卡 0 可能对应宿主机另一张物理卡，不能拿逻辑编号直接套用亲和性结论。
2. **查询这张 NPU 对应的 CPU/NUMA 亲和性。** 根据该服务器的 NPU 拓扑工具、PCIe BDF 和系统报告，确定关联的 NUMA 节点及 CPU 集合；不能只凭卡号推断 NUMA 编号。多张卡要逐张核对，不能默认共用一套绑核。
3. **核对该节点实际可用且相对空闲的核。** 将节点 CPU 列表与容器 cpuset、进程允许集合对照，再持续观察候选核上的系统服务、其他用户进程、IRQ 和软中断负载，排除经常繁忙或被其他任务占用的核。不要只看一次 CPU 利用率就认定空闲；同时考虑权重和计算内存的 NUMA 落点。
4. **为计算与握手线程分别选核。** 据此设置 `numa_nodes`、`cpuinfer_threads`、`KT_MOE_CORE_OFFSET`，再为 poll 和 completion 选择独立可用核。参考后端若按 offset 分配连续工作核，就必须确认该整段 CPU 都属于目标节点并在允许集合内；不能跨到别的节点或超出 cpuset。
5. **关闭上层重复绑核。** 显式设置 `enable_cpu_binding=false`，由 kt 后端执行计算线程绑核。需要外层 NUMA 内存约束时，约束范围必须容纳全部计算及握手线程，不能与后端安排冲突。
6. **启动前确定并固定整套配置。** 将已确认的 NPU card、NUMA 节点、计算核、poll 核和 completion 核写入脚本，确认资源安排和容器允许集合一致后再启动。首次部署或更改配置时可以做一次实际绑核核验；日常运行不需要反复检查或动态换核，只有性能异常时再排查。

这里要同时解决两件事：**CPU 计算线程和内存如何落在合适的 NUMA 节点，以及该节点与实际 NPU card 是否亲近。** 只填 `numa_nodes` 或只挑空闲 CPU，都不足以完成这项配置。因此仓库提供流程与参考脚本，由使用者或 agent 按目标服务器重新对齐，不提供一套适用于所有服务器的固定核号。

在目标宿主机及容器中核对。下面的 `NODE` 和 `BDF` 必须使用上述查询得到的值：

```bash
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
cat "/sys/devices/system/node/node${NODE}/cpulist"
grep Cpus_allowed_list /proc/self/status
npu-smi info
npu-smi info -t topo
# 若当前 npu-smi 不支持 topo，使用该版本帮助及 PCIe BDF 对应关系查询
# 已确认 NPU 对应的 PCIe BDF 后：
cat "/sys/bus/pci/devices/${BDF}/numa_node"
cat "/sys/bus/pci/devices/${BDF}/local_cpulist"
```

`numa_node=-1` 代表内核未报告，不等于节点 0。`ASCEND_RT_VISIBLE_DEVICES=7` 是可见物理卡选择，进程内逻辑卡号可能变为 0，要结合实际映射核对。容器 cpuset 必须包含计算、poll 和 completion 所需核，32 个计算线程应对应允许的 32 个工作核，握手线程放在独立可用核。

首次部署或更改绑核配置时，可一次性检查具体 worker PID 的 `ps -L -p <PID> -o pid,tid,psr,comm` 和 `/proc/<PID>/task/<TID>/status` 中 `Cpus_allowed_list`，并用 `numastat -p <PID>` 检查内存落点。`psr` 只是当时所在核，允许核集合才是绑定依据。需要进程级约束时可以按核对后的节点使用 `numactl --cpunodebind=<NODE> --membind=<NODE> bash <script>`，但外层集合必须容纳后端全部线程，不能再把整个进程固定到某一个计算核。

### 选核前检查系统任务和中断

**亲和性合适不代表核空闲。选定 NUMA 节点后，还必须检查候选核是否被系统任务、中断或其他业务频繁占用。** 计算线程被抢占会增加延迟和抖动；poll、completion 核繁忙也可能延迟任务提交或完成通知。

可以在有代表性的系统负载下观察一段时间，而不是只取一次快照：

```bash
# 观察每个 CPU 的用户态、系统态、IRQ、softirq 和空闲比例。
mpstat -P ALL 1 30

# 观察线程所在 CPU 和运行状态；结合具体线程 affinity 判断实际可用范围。
ps -eLo pid,tid,psr,pcpu,comm --sort=-pcpu

# 前后采样比较各 CPU 计数增量，排查中断集中的核。
cat /proc/interrupts
cat /proc/softirqs
```

`mpstat` 需要目标环境提供 sysstat。`/proc/interrupts` 和 `/proc/softirqs` 是累计计数，必须看一段时间内的增量；`psr` 和一次 `%CPU` 快照不能证明某核长期闲置。不要为了选核而停止其他用户任务。

**这些检查应在启动前完成，用来提前确定哪些核适合使用。** 优先选择同一亲和 NUMA 节点内、允许使用且持续低负载的核，为计算、poll、completion 分别安排资源，并把选好的核固定到启动配置中。不要等服务启动后再边跑边选核，也不需要在正常运行期间不停检查或调整。资源占用不明确时，应先与管理员或相关使用者确认；启动后的争用和延迟分析只作为性能异常时的排障手段。

**绑核只限制线程可以在哪些 CPU 上运行，并不独占这些核，也不阻止系统任务和中断进入。** 如果仍有明显干扰，需要由管理员统一安排 CPU 资源预留、服务及 IRQ 亲和性，必要时评估 CPU 隔离。即使核已预留，也不能承诺完全没有调度或系统中断；共享服务器上不要直接搬移 IRQ、停服务或更改其他任务的亲和性。

## 权重固定与 CPU 后端批量

**当前权重传输方案必须显式设置 `KT_MOE_PIN_WEIGHTS=1`。** 按现有环境实测，未设为 1 时吞吐明显下降。这个设置服务于正常传输路径，不是仅采集时才开的诊断项。

`cpu_moe.max_num_tokens` **建议先固定为 32**。现有环境尝试调大没有明显收益，这是当前计算后端需要进一步处理的问题，不能简单把它当成提高吞吐的旋钮。若实际输入确实需要更大容量，应核对后端实现后再验证。

## 计算与传输调度

`transfer_policy=load_balance` 使用 CPU 计算和 H2D 权重传输的成本估计来选择本次迁入的专家；设备空间不足时由淘汰策略选择可用槽。当前参考实现按 `floor(M * t_cpu / (t_cpu + t_copy))` 计算迁入数量，再受可用槽和 `balance_max_transfers` 限制，M 为本次路由缺失专家数。70 和 450 微秒是配置估计，不能当成实测常数。

`transfer_scheduler=npu` 与自定义 copy 库必须和对应代码、CANN、图捕获流程配套。CPU 计算、权重 H2D、设备计算和结果完成通知是不同阶段；图中重叠关系必须经过时钟校准才能解释。

**建议首次复现和迁移时显式设置 `expert_prefetch_enabled=false`。** 先在无预取基线验证计算结果、图模式和时序，再单独试开预取。开启后会增加权重传输、缓存槽竞争和异步依赖，当前分支可能出现正确性、同步或性能问题，不应直接当作稳定优化启用。

原始脚本顶部写着 next-layer prefetch disabled，但 JSON 实际为 true；它是历史快照，迁移时应主动改成上述建议值。只有试开预取时才需要调整 `expert_prefetch_num`，建议从 1 开始。`hot_expert_preload` 是独立的热专家预加载开关，没有匹配的热专家统计文件时建议关闭。`T` 的按需迁入计数不自动包含独立预取。

## 动态库需要在目标环境准备

**原脚本中的 `.so` 路径不能直接照搬到另一台机器。** 本仓库不提供完整后端与这些二进制，使用者必须按目标 CPU 架构、Python、CANN、torch/torch_npu、编译器和对应源码版本准备或重新编译，再改成实际产物路径。相同 CPU 架构也不代表动态库 ABI 和运行时依赖一定兼容。

`KT_MOE_POLL_LIBRARY` 指向的 `libkt_ready_poll.so`、`transfer_copy_library` 指向的 `libnpu_expert_copy_dispatch.so`、kt_kernel 扩展和自定义 CANN 算子都要分别核对。使用对应 poll 或 NPU transfer 路径时，缺少匹配库不能靠保留原路径、创建同名文件或随意删掉配置解决。应依据各源码版本的构建说明编译，检查实际加载位置与依赖，并做小规模运行验证。

## 迁移步骤

先对齐代码和依赖，再确认两份权重与自定义动态库，随后检查 NPU 和 CPU 亲和性、实际绑核及内存落点。此前测试均使用图模式，建议沿用图模式并验证模型结果，最后采集同次 CPU 与 NPU profiler。不要为了复现截图而移动时间区间，也不能根据截图证明模型正确或性能提升。

运行日志、临时采集及校时输出按实际服务器的存储位置设置，使用独立任务目录。原参考脚本的 `/mnt/sdc/...` 和 latest 链接属于历史环境，执行前应改成当前任务目录；不要直接执行原样快照。
