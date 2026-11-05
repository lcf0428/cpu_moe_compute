# CPU MoE 参数设置建议

这份说明只保留影响 CPU MoE 运行、迁移和采集的设置。端口、监听地址等通用服务参数不再逐项解释。所有建议针对当前双份权重和 kt_kernel 方案；参考脚本中的数字是原机器的配置，不是通用默认值。

## 必须设置的项目

| 项目 | 设置要求 | 原因及不满足时的影响 |
|---|---|---|
| `additional_config.enable_cpu_binding` | **必须显式设置为 `false`，不要省略后依赖默认值。** | kt 后端已经对 CPU MoE 计算线程绑核。这里设为 false 是关闭 vLLM Ascend 上层的另一套绑核，避免再次修改线程亲和性或限制后端可用核，造成绑核冲突、争用和性能波动。CPU MoE 后端本身仍须完成绑核。 |
| `expert_offload_config.expert_offload`、`cpu_moe.enabled` | **使用本方案时均必须为 `true`。** | 分别打开专家卸载和 CPU 专家计算；不能只开其中一项就认为已启用完整 CPU MoE 路径。 |
| `cpu_moe.backend` | **本参考方案必须使用 `kt_kernel`。** | 当前脚本、绑核和采集实现围绕该后端组织；更换后端需重新对齐接口与行为。 |
| `cpu_moe.weight_format` | **当前方案必须设置为 `gguf`。** | 当前支持 CPU GGUF 与 NPU 模型双份权重。统一权重方案尚不成熟，不建议切换到源码中的实验分支。 |
| `cpu_moe.weight_path` 和 NPU 模型目录 | **必须改成目标机器实际路径，并保证两份权重对应。** | 模型版本、层号、专家编号及量化约定需要一致；只保证文件存在不足以保证结果正确。GGUF 权重未随仓库上传；后端源码现已备份在 `cpu_moe/`。 |
| `cpuinfer_threads`、`numa_nodes`、`KT_MOE_CORE_OFFSET` | **必须按目标机器重新设置并检查实际绑核。** | 参考的 32 线程、NUMA 1、160–191 核只适用于原环境。线程数、允许核集合、内存落点和 NPU 亲和性必须一起对齐，具体方法见脚本设计说明。 |
| `KT_MOE_POLL_CPU`、`KT_MOE_COMPLETION_CPU` | **使用当前 poll/completion 路径时必须选取实际可用核。** | 不能照抄 158/159。应避开计算工作核，并保证这些核包含在容器 cpuset 中。 |
| `KT_BUILD`、`PYTHONPATH`、自定义算子及动态库路径 | **必须按目标环境准备并修改。** | 仓库中的绝对路径属于原机器。路径存在也不代表代码版本或动态库 ABI 匹配，见下文动态库准备要求。 |

## 专家预取建议先关闭

**首次复现和迁移时，建议显式设置 `expert_prefetch_enabled=false`。** 先在无预取条件下验证结果、图模式和 CPU/NPU 时序，再单独测试开启预取的收益。预取会引入额外权重传输、槽位竞争和异步依赖；当前分支开启后可能遇到正确性、同步或性能问题，不能把它当成已经稳定的默认优化。

```json
{
  "enable_cpu_binding": false,
  "expert_offload_config": {
    "expert_prefetch_enabled": false
  }
}
```

这是需要合并进完整配置的关键字段片段，不能替代整个 `additional_config`。

| 参数 | 建议怎么设置 | 目的及影响 |
|---|---|---|
| `expert_prefetch_enabled` | **建议先设 `false`；基线通过后按需试开。** | 控制下一层专家预取。原始快照实际为 true，不代表新环境也建议开启。 |
| `expert_prefetch_num` | **预取关闭时无须调；开启后建议从 1 开始。** | 控制预取预算。增大可能占用更多带宽和缓存槽，不保证更快；它不是图中的 T 计数。 |
| `hot_expert_preload`、`hot_experts_file` | **按需启用；没有匹配的热专家文件时建议关闭预加载。** | 热专家预加载与下一层预取是两件事，不能以为关了预取就同时关了预加载。仓库已保留原环境的 `cpu_moe/expert_rank_gsm8k.json`；开启预加载时必须准备或确认对应模型、层号和专家编号的文件；更换数据分布后应重新评估。 |

## 动态库必须在目标环境准备

**不能直接复制原机器的 `.so` 路径就运行。** 仓库未提供这些二进制；迁移到另一台机器或容器时，需要按实际源码、CPU 架构、CANN、torch/torch_npu 和编译器环境准备或重新编译，再修改脚本路径。即使机器同为 aarch64，依赖版本变化也可能导致加载失败或运行异常。

| 路径或库 | 设置要求 |
|---|---|
| `KT_MOE_POLL_LIBRARY` / `libkt_ready_poll.so` | **使用 poll 提交路径时必须准备匹配的库。** 从对应 kt 运行时源码按其构建流程编译，不能只创建同名文件或保留原绝对路径。 |
| `transfer_copy_library` / `libnpu_expert_copy_dispatch.so` | **使用当前 `transfer_scheduler=npu` 路径时必须准备。** 按对应专家传输实现构建，并对齐调用接口和设备运行时。缺库时不能认为删掉配置就会等价回退。 |
| kt_kernel 后端扩展 | **必须安装或编译与当前 Python 及运行时匹配的扩展。** 应检查实际导入位置，避免 PYTHONPATH 指向新源码而仍加载旧 `.so`。 |
| `ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH` 指向的自定义算子 | **按目标环境安装或重新编译后设置。** 路径应指向实际构建产物，不能照抄原工作区目录。 |

先在目标环境查找对应源码的构建说明、CMake 或构建脚本，再按该版本流程编译。本仓库只是参考快照，不包含完整构建依赖，无法给出通用于所有机器的一条编译命令。编译后至少检查动态库依赖能否解析、后端实际加载位置及一次小规模运行；文件存在和 `ldd` 无缺项都不能替代运行验证。

## 建议保持的运行设置

| 参数 | 建议怎么设置 | 目的及影响 |
|---|---|---|
| `threadpool_count` | **建议先保持 `1`。** | 与当前单池参考路径一致；增加线程池会改变 NUMA 分配与并发资源，需要重新验证。 |
| `layers` | **复现当前方案建议保持 `all`。** | 仅在做指定层实验、且当前代码支持时按需缩小范围。 |
| `enable_multi_card` | **当前参考方案建议保持 `false`。** | 多卡卸载需单独对齐，不建议只改这个开关就扩展当前脚本。 |
| `prefill_delta_load`、`prefill_split_gmm` | **复现当前分支建议先保持 `true`。** | 分别选择 prefill 增量加载与拆分 GMM 路径；做消融时再逐项修改，避免同时改变多条执行路径。 |
| `cache_policy_enabled` | **复现当前缓存方案建议保持 `true`。** | 关闭会改变专家驻留和迁入行为，应作为单独实验。 |
| `KT_MOE_POLL_SUBMIT` | **当前异步参考路径建议保持 `1`。** | 依赖匹配的 poll 库和绑核配置；其他提交方式需按当前实现重新验证。 |
| `KT_MOE_MATCH_NPU_MXFP8` | **当前 MXFP8 路径建议保持 `1`。** | 用于匹配 CPU/NPU 激活计算约定；不能代替跨设备精度验证。 |
| `KT_MOE_PIN_WEIGHTS` | **当前权重传输方案建议保持 `1`，并检查主存余量。** | 固定主存权重以支持对应传输路径，会占用主存资源。 |
| `KT_MOE_COMPLETION_RT_PRIORITY` | **建议保持 `0`。** | 先避免实时线程带来的 RT throttling 问题。 |
| `KT_MOE_RT_PRIORITY` | **建议不设置。** | 不建议直接沿用持续 FIFO 实时优先级设置。 |
| `CPU_MOE_ASYNC_RESULT`、`CPU_MOE_ASYNC_VALIDATE_OUTPUTS` | **建议不设置。** | 当前参考未启用的实验性分支，不建议在首次复现时混入。 |
| `VLLM_USE_V1`、`VLLM_USE_V2_MODEL_RUNNER` | **复现当前分支先保持 `1`、`0`。** | 更换引擎或 runner 前需核对 CPU MoE 接入点，不能按其他版本默认值直接替换。 |

## 按实际负载调整的项目

| 参数 | 建议怎么设置 | 调整影响 |
|---|---|---|
| `num_device_experts` | **按 HBM 容量、命中率和传输量设置，不必固定为 `[84]`。** | 单值广播各 MoE 层；逐层列表需要对齐注册顺序。增加容量会占用更多 HBM，但可能减少缺失专家传输。 |
| `num_device_layers` | **建议先保持 `2`；需要调整时检查当前缓冲池实现。** | 它是卸载设备层缓冲参数，不能直接按模型层数填写。 |
| `cpu_moe.max_num_tokens` | **必须覆盖当前 CPU 路径实际可能收到的批量，参考值为 32。** | 与服务端的 token 调度预算不同；调大可能增加缓冲占用。 |
| `transfer_policy`、`transfer_scheduler` | **复现当前方案保持 `load_balance`、`npu`。** | 更换策略属于另一套调度实验，需要重新核对图模式和动态库。 |
| `balance_max_transfers` | **按可用槽与带宽设置，参考值 32。** | 限制单次迁入专家数，仍受可用槽位限制。 |
| `balance_cpu_expert_us`、`balance_copy_us` | **建议按目标环境实测后设置，不要把 70/450 微秒当通用常数。** | 当前比例公式为 `floor(M*t_cpu/(t_cpu+t_copy))`；提高 CPU 成本估计会增加迁入比例，提高 copy 成本估计会降低迁入比例。 |
| `KT_MOE_COMPLETION_SPIN_US` | **按延迟和 CPU 占用权衡，参考值 3000 微秒。** | 更长热轮询可减少休眠唤醒，但会持续占用 CPU。 |
| speculative config | **按模型与实验目的设置，首次基线可先不启用推测解码。** | dspark 和 7 个候选 token 是原实验设置，会改变额外计算量和批次形态。 |
| compilation config | **建议先验证 eager，再测试目标图模式。** | `FULL_DECODE_ONLY` 和 capture sizes 应适配实际批次；捕获更多尺寸会增加启动时间与内存。 |
| 并行规模、批次及 KV 配置 | **按模型、HBM 和后端支持设置。** | 不必照抄原服务配置；改变 TP、DP、PP 后要重新核对当前 CPU 后端与卸载路径支持。 |

## 当前路径无需调节的兼容项

`balance_cpu_extra_route_us`、`balance_npu_route_us`、`balance_min_gain_us` 在当前读取的比例策略公式中没有使用，**不需要为性能调优修改它们**。当前配置若要求字段存在，保留原值即可；只有确认解析器允许省略时才删除。它们不能笼统称为“设置不设置都一样”，因为其他版本或策略可能读取这些字段。

`VLLM_ASCEND_OFFLOAD_THRESHOLD` 的作用依分支和执行路径而定，**建议先保留参考值，不作为主要调优旋钮**；修改前确认实际读取点，manager 还存在由设备专家数与 topk 计算的阈值。

## 按需开启采集和诊断

| 参数 | 建议怎么设置 | 说明 |
|---|---|---|
| `CPU_MOE_PHASE_TRACE` / `KT_MOE_CPU_TRACE` | **只在采集合成图时开启；正常性能基线建议关闭。** | 前者是脚本条件开关，后者指定最小 CPU 日志路径，保留 `%p` 区分 PID。现有记录是 part compute，不包含上层结果归并。 |
| `CPU_MOE_CLOCK_ANCHORS`、`--clock-alignment` | **使用本仓库校时方案合成时必须启用并提供。** | 先应用 profiler 接入补丁，再采集同次配对锚点；缺锚点不能直接混合 CPU/NPU 时间。依赖 Ascend exporter 已完成原始 host/device 对齐。 |
| `CPU_MOE_CLOCK_CALIBRATION` | **本仓库新校时流程不依赖它，可关闭。** | 旧 Event 包围记录不能单独建立本合成器需要的时钟映射。 |
| `KT_MOE_CPU_TRACE_KIND` | **仅在需要完整 CPU 计算结果归并口径时设 `total`。** | 需要先应用可选 C++ 补丁并重编；默认 part 与 total 不应混加。完整扩展和模型仍待验证。 |
| `CPU_MOE_COMPUTE_EVENT` / `--compute-event` | **必须与实际日志事件一致。** | 默认 `cpu_moe.compute`；total 日志选择 `cpu_moe.compute_total`。 |
| `KT_MOE_LIFECYCLE_TRACE`、`KT_MOE_LIFECYCLE_WORKERS`、`KT_MOE_PHASE_TIMING`、`KT_MOE_TASK_TRACE`、`KT_MOE_PAGE_TRACE`、`KT_MOE_POLL_TRACE` | **最小采集建议全部不设置。** | 只在定位具体内部问题时按需开启，避免额外诊断开销。 |
| `VLLM_ASCEND_CPU_MOE_TRACE`、`VLLM_ASCEND_PROFILE_DIAG_LOG`、`moe_offload_debug` | **建议关闭；排障时按需开启。** | 不是最小合成所必需的统计。 |
| `decode_stats_enabled`、`benchmark_route_stats_enabled` | **最小合成可设为 `false`；研究这些指标时再开启。** | decode 汇总与 benchmark 路由统计不参与最小 CPU compute 合成。 |
| `torch_profiler_with_stack`、`ignore_frontend` | **最小采集建议分别为 `false`、`true`。** | 减少调用栈及前端采集开销，仍须保留 worker 所需 CPU/NPU activities。 |
| `delay_iterations`、`max_iterations` | **首次校时建议 `0`、`0`，手动 start/stop；随后按需限制窗口。** | 此建议针对当前父类中 0 表示不自动按最大迭代数停止的实现。有限 benchmark 全部响应结束后再 stop，保证锚点与计算在同次窗口。 |
| profiler 及日志目录 | **必须使用本次任务独立目录。** | 避免 latest 更新或多 PID、多次运行混用；新远端任务放 `codex_logs/<任务名>/`。 |

合成命令与完整 CPU 计时边界见 [profile 合成指南](01_profile_merge_guide.md)；硬件亲和性及移植步骤见 [脚本设计](02_script_design.md)。
