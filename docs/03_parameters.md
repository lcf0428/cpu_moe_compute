# CPU MoE 参考脚本参数说明

对应 `reference/run_dpark_26.sh` 的配置快照。下面的值用于理解实验设置，修改前应对齐当前 vLLM Ascend 和 kt 后端实现。标记为估计或兼容项的参数不能当成已经测出的性能结论。

## 服务与调度

| 参数 | 参考值 | 目的与影响 |
|---|---|---|
| CARDS / ASCEND_RT_VISIBLE_DEVICES | 7 | 选择可见物理 NPU；不是 7 张卡。进程内逻辑编号需另行确认。 |
| PORT / --port | 8151 | HTTP 端口；多实例需避免冲突。 |
| --host | 0.0.0.0 | 监听所有接口；按目标机器网络需求设置。 |
| SERVED_NAME / --served-model-name | deepseek_v4 | API 请求使用的模型名，不是权重路径。 |
| vllm serve 模型路径 | /mnt/share/weight/DeepSeek-V4-Flash-0731 | NPU 侧模型来源，与 CPU GGUF 必须一致。 |
| MAX_SEQS / --max-num-seqs | 8 | 同时调度序列数上限；增大影响批量、延迟、KV 占用。 |
| GPU_MEM / --gpu-memory-utilization | 0.95 | NPU 显存预算比例；余量小会增加 OOM 风险。 |
| --max-model-len | 267000 | 最大上下文长度，影响容量检查与 KV 需求。 |
| --max-num-batched-tokens | 32000 | 每次调度的 token 预算，影响 prefill 吞吐和峰值内存。 |
| --data-parallel-size | 1 | DP 副本数。 |
| --tensor-parallel-size | 1 | TP 切分数；改变后要重新核对 CPU 后端支持。 |
| --pipeline-parallel-size | 1 | PP stage 数。 |
| --enable-expert-parallel | 开启 | 专家并行路径开关；单卡配置并不因此产生多卡并行。 |
| --seed | 1024 | 随机种子；不保证所有异步实现位级确定性。 |
| --trust-remote-code | 开启 | 允许模型自定义代码。 |
| --tokenizer-mode / --tool-call-parser / --reasoning-parser | deepseek_v4 | 对应模型的分词、工具调用和推理输出解析。 |
| --enable-auto-tool-choice | 开启 | 允许自动工具选择的输出解析流程。 |
| --aggregate-engine-logging | 开启 | 聚合引擎日志。 |
| --api-server-count | 1 | API 前端进程数。 |
| --safetensors-load-strategy | prefetch | 模型文件加载策略；影响加载时的 IO 和内存行为。 |
| --kv-cache-dtype | fp8 | 降低 KV 内存占用，同时需要验证精度影响。 |
| --shutdown-timeout | 60 | 关闭超时，单位秒。 |
| --additional-config | JSON | Ascend 和 CPU MoE 扩展参数入口。 |

## 推测解码与图模式

| 参数 | 值 | 目的与影响 |
|---|---|---|
| speculative_config.method | dspark | 选择对应推测解码实现，需要匹配当前分支。 |
| speculative_config.num_speculative_tokens | 7 | 每轮候选 token 数；影响接受率、额外计算和批次形态。 |
| compilation_config.cudagraph_mode | FULL_DECODE_ONLY | decode 使用完整图模式；eager 正确不代表图捕获与 replay 正确。 |
| compilation_config.cudagraph_capture_sizes | 1,2,3,4,8,16,24,32,48 | 预捕获批次尺寸；增加尺寸会增加启动时间和图内存。 |
| VLLM_USE_V1 | 1 | 使用 V1 引擎。 |
| VLLM_USE_V2_MODEL_RUNNER | 0 | 保持当前旧 runner 路径，需与分支兼容。 |

## 卸载与缓存

| 参数 | 值 | 目的与影响 |
|---|---|---|
| enable_cpu_binding | false | **必须关闭上层额外绑核**，由 CPU MoE 后端绑定工作线程。详见设计说明。 |
| expert_offload | true | 开启专家卸载。 |
| prefill_delta_load | true | 选择 prefill 的增量权重加载路径，实际驻留与复用依实现。 |
| prefill_split_gmm | true | 选择拆分的 prefill GMM 路径。 |
| enable_multi_card | false | 关闭该卸载模块的多卡方案，与 TP/DP 参数是不同层次。 |
| NDE_ARRAY / NDE_JSON / num_device_experts | 84 / [84] | 设备专家容量配置；单值广播各 MoE 层，逐层列表须覆盖注册顺序。影响命中率、传输和 HBM。 |
| num_device_layers | 2 | 卸载模块的设备层缓冲容量参数，不是模型只有两层。修改前检查当前 prefill 池和缓存实现。 |
| cache_policy_enabled | true | 开启专家缓存策略。 |
| expert_prefetch_enabled | true | 实际开启下一层专家预取；原脚本顶部注释与此冲突，不能按注释当成关闭。 |
| expert_prefetch_num | 1 | 预取专家预算，影响带宽、槽位与命中率；不是 T 标签。 |
| hot_expert_preload | true | 使用热专家排序进行预加载。 |
| hot_experts_file | expert_rank_gsm8k.json | 热专家统计文件；与模型编号和数据分布绑定，更换任务需重新评估。此文件未在 benchmark 内，需额外提供。 |
| MOE_DEBUG / moe_offload_debug | false | 关闭详细调试，避免额外日志影响性能。 |
| VLLM_ASCEND_OFFLOAD_THRESHOLD | 32 | 分支相关卸载阈值入口；manager 还存在由设备专家数/topk计算的阈值，不能认定这个环境变量在每条路径都有效，移植时追踪读取点。 |

## CPU 后端和权重

| 参数 | 值 | 目的与影响 |
|---|---|---|
| cpu_moe.enabled | true | 启用 CPU 路由专家计算。 |
| backend | kt_kernel | 使用当前 kt 后端，依赖匹配的 C++ 库。 |
| weight_format | gguf | **当前支持方案为 CPU GGUF 与 NPU 双份权重**。统一权重方案不属于成熟支持路径。 |
| weight_path | dsv4_layer{layer_idx}_mxfp4.gguf | 按 layer_idx 展开文件名。权重未上传，应单独准备。 |
| layers | all | 对全部适用层启用 CPU 后端。 |
| max_num_tokens | 32 | CPU MoE 单批容量配置；不能与服务的 32000 token 调度预算混为一谈。 |
| cpuinfer_threads | 32 | CPU 计算线程数，需与实际允许核和 NUMA 节点容量匹配。 |
| threadpool_count | 1 | 后端线程池数量，改变会影响 NUMA 分布和并发资源。 |
| numa_nodes | [1] | 后端使用的 NUMA 节点，必须结合 NPU 亲和性重新确认。 |
| KT_BUILD / PYTHONPATH | kt_async_runtime 加两源码目录 | 选择实际导入的 Python 和后端实现；路径顺序可改变加载版本。 |
| KT_MOE_CORE_OFFSET | 160 | 后端工作核偏移入口；要检查线程实际 affinity，不要只看 echo。 |
| KT_MOE_MATCH_NPU_MXFP8 | 1 | CPU 激活计算匹配 NPU MXFP8 约定，跨设备迁移仍需精度验证。 |
| KT_MOE_PIN_WEIGHTS | 1 | 尝试固定权重主存以支持传输；影响主存资源占用。 |
| KT_MOE_POLL_SUBMIT | 1 | 通过轮询提交 CPU 任务。 |
| KT_MOE_POLL_CPU | 158 | poll 线程目标核，需避开工作核且位于允许集合。 |
| KT_MOE_COMPLETION_CPU | 159 | 完成通知线程目标核，同样需要独立可用核。 |
| KT_MOE_COMPLETION_RT_PRIORITY | 0 | 不启用实时优先级，避免持续 FIFO 导致 RT throttling。 |
| KT_MOE_COMPLETION_SPIN_US | 3000 | 完成线程热轮询窗口，单位微秒；减少休眠唤醒延迟但消耗 CPU。 |
| KT_MOE_POLL_LIBRARY | libkt_ready_poll.so | 轮询动态库绝对路径，需与调用接口兼容。 |
| KT_MOE_RT_PRIORITY / KT_MOE_POLL_TRACE | unset | 关闭旧实时优先级入口和轮询诊断。 |
| CPU_MOE_ASYNC_RESULT / CPU_MOE_ASYNC_VALIDATE_OUTPUTS | unset | 不启用这两项实验性异步结果及验证分支。 |

## 动态迁入负载均衡

| 参数 | 值 | 目的与影响 |
|---|---|---|
| transfer_policy | load_balance | 使用按 CPU/copy 成本比例分配缺失专家的策略。 |
| balance_max_transfers | 32 | 每次最多迁入专家数量，还受可用槽位限制。 |
| balance_cpu_expert_us | 70.0 | 每专家 CPU 成本估计，增大会提高迁入比例。 |
| balance_copy_us | 450.0 | 每专家权重复制成本估计，增大会降低迁入比例。 |
| balance_cpu_extra_route_us | 10.0 | 配置兼容项；本次读取的比例策略构造和公式未使用，不能宣称改它会改变分配。 |
| balance_npu_route_us | 2.0 | 同上，当前比例公式未纳入 NPU 计算估计。 |
| balance_min_gain_us | 20.0 | 同上，当前比例公式不使用该门槛。 |
| transfer_scheduler | npu | 选择设备侧传输调度路径，需验证图捕获兼容。 |
| transfer_copy_library | libnpu_expert_copy_dispatch.so | 自定义专家复制库路径；二进制不在备份中。 |

当前比例策略见 `reference/load_balance_transfer_policy.py`：`k=floor(M*70/(70+450))`，再限制到可用槽与最大传输数；专家准入排序和 LRC 淘汰分开处理。`T` 是实际采集的迁入专家数，不是按该公式事后估算的数值。

## 最小采集与统计

| 参数 | 值 | 目的与影响 |
|---|---|---|
| PROFILE_ROOT / RUN_STAMP / PROF_RAW_DIR | 根目录加时间戳 | 为每次采集隔离输出；改到本次 codex_logs 任务目录，避免混用 latest。 |
| LOG_DIR | 历史 logs 目录 | decode_stats_path 的基础目录，移植时改到任务目录。 |
| CPU_MOE_PHASE_TRACE | 默认 1 | shell 中控制是否设置 KT_MOE_CPU_TRACE；设为 0 则取消该变量。 |
| KT_MOE_CPU_TRACE | cpu_compute_%p.jsonl | 最小 CPU 计算区间采集，%p 区分进程；具体边界见合成指南。 |
| CPU_MOE_CLOCK_CALIBRATION | 1 | 请求 start/stop 校准采集；**旧开关本身不能保证合成已校时**，还需合成器可识别的配对锚点。 |
| KT_MOE_LIFECYCLE_TRACE / KT_MOE_LIFECYCLE_WORKERS / KT_MOE_PHASE_TIMING | unset | 关闭内部生命周期、worker 和细分阶段统计。 |
| KT_MOE_TASK_TRACE / KT_MOE_PAGE_TRACE | unset | 关闭任务及页面诊断。 |
| VLLM_ASCEND_CPU_MOE_TRACE / VLLM_ASCEND_PROFILE_DIAG_LOG | unset | 关闭额外 Python/运行时诊断。 |
| VLLM_TORCH_PROFILER_DIR | PROF_RAW_DIR | profiler 输出目录。 |
| VLLM_SERVER_DEV_MODE | 1 | 启用采集控制所需开发接口，按运行分支核对。 |
| profiler_config.profiler | torch | 选择 torch profiler。 |
| torch_profiler_dir | PROF_RAW_DIR | profiler JSON 中的输出目录。 |
| torch_profiler_with_stack | false | 不采集调用栈，减少额外开销与体积。 |
| ignore_frontend | true | 排除前端采集；仍须核对正确 worker 的输出。 |
| delay_iterations | 2 | 延后迭代采集，需核对与 start/stop 窗口的关系。 |
| max_iterations | 200 | 最大采集迭代数，限制输出体积。 |
| decode_stats_enabled | true | 独立的 decode 汇总统计；最小合成并不需要，可以在不依赖它时关闭。 |
| benchmark_route_stats_enabled | false | 不开额外 benchmark 路由统计。 |
| decode_stats_path | cpu_moe_<时间>_stats | decode 统计输出前缀，不是 CPU 计算日志路径。 |

## 环境与运行控制

| 参数 | 值 | 目的与影响 |
|---|---|---|
| set -o pipefail | 开启 | 管道中 vllm 失败不会被 tee 的成功退出掩盖；脚本当前实际没有自动 tee。 |
| VLLM_LOGGING_LEVEL | INFO | 日志级别。 |
| PYTHONUNBUFFERED | 1 | Python 即时输出，便于诊断。 |
| TMPDIR | 已设值或 /tmp | 临时文件目录，需保证容量与权限。 |
| CANN set_env.sh | import acl 失败时 source | 初始化运行环境；import acl 成功也不能证明所有动态库版本正确。 |
| PYTORCH_NPU_ALLOC_CONF | expandable_segments:True | 内存分配行为配置，影响碎片和兼容性。 |
| ASCEND_CUSTOM_OPP_PATH | custom_transformer 路径 | 自定义算子定位。 |
| LD_LIBRARY_PATH | op_api/lib 前置 | 自定义算子动态库搜索路径。 |
| ASCEND_CONNECT_TIMEOUT | 10000 | CANN 连接超时配置，单位以当前运行时实现为准。 |
| ASCEND_TRANSFER_TIMEOUT | 10000 | CANN 传输超时配置，不能当成单次专家迁移成本。 |
| VLLM_ENGINE_READY_TIMEOUT_S | 10000 | 引擎就绪等待秒数。 |
| VLLM_RPC_TIMEOUT | 3600000 | RPC 超时配置，具体单位与读取点按当前 vLLM 分支确认。 |
| VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS | 30000 | 模型执行等待秒数；很大超时会延后暴露死锁。 |

参考脚本还会更新 PROFILE_ROOT/latest 符号链接。复现时应显式记录本次路径、PID、代码版本、权重来源、CPU affinity、NPU 映射和采集窗口，避免另一次运行更新 latest 后误合成。

## 备份参考实现新增参数

| 参数 | 默认值 | 目的与影响 |
|---|---|---|
| CPU_MOE_CLOCK_ANCHORS | 未设置 | 应用 profiler 接入补丁后设为 1，在真实开始和停止采集时记录配对锚点；不同于原 CPU_MOE_CLOCK_CALIBRATION。 |
| --clock-alignment | 必填 | 合成器接收本次 CPU RAW 与 profiler host 的配对 JSONL，缺失拒绝合成。 |
| KT_MOE_CPU_TRACE_KIND | part 模式 | 可选 C++ 补丁新增；total 记录完整外层 CPU compute_total，并停用 part 记录。尚需编译和真机验证。 |
| CPU_MOE_COMPUTE_EVENT | cpu_moe.compute | 本仓库包装器选择的事件；改为 cpu_moe.compute_total 以处理可选补丁生成的完整计算区间。 |
| --compute-event | cpu_moe.compute | 核心合成器的显式事件选择，禁止将两种边界混加。 |
| --out / --stats / --report | 包装器在 OUT_DIR 命名 | 分别生成合成 trace、机器可读统计和文本报告，不得覆盖输入。 |
| --include-actual-compute / --actual-compute-only | 包装器启用 | 展示实际计算轨道并保留匹配边界参数。 |

包装器位置参数为 TRACE、CPU_LOG、CLOCK_ANCHORS 和可选 OUT_DIR。它不再通过旧 latest 自动猜测输入。CPU 日志匹配辅助脚本保留供参考，不能替代同次 PID 与校时范围核对。
