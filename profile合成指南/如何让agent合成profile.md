# 如何让agent合成profile（离线版）

填写下面的输入信息，将**整个文本框**发给有本地文件/命令执行能力的AI。规则、字段和算法都包含在文本中，不需要访问GitHub或读取其他说明；没有现成合成脚本时，AI直接用Python标准库实现。AI仍需要实际日志和可用的本地/SSH访问权限。

## 整段复制给AI

```text
请离线合成本次CPU MoE与Ascend/NPU的profile，并交付一个可用MindStudio打开的Chrome Trace JSON。
不要访问GitHub、网页、在线文档或下载依赖。下面已给出完整任务规则与实现方法；不要求先找到某个仓库或既有脚本。没有现成匹配工具时，请用Python标准库（json、decimal、pathlib、collections、hashlib等）编写合成程序并实际执行，不要只给方案。可以使用已授权SSH连接读取服务器上的数据。

【我提供的输入】
数据位置：<本地目录，或服务器/SSH别名及本次采集目录>
本次原始NPU trace：<已知则填写，不知道则在本次目录内查找>
同次CPU计算日志JSONL：<已知则填写，不知道则查找>
同次时钟锚点JSONL：<已知则填写，不知道则查找>
同次预取/DMA/原生生命周期日志：<有则填写，不知道则查找，没有则不伪造>
本地输出目录：<可留空；默认当前用户桌面下新建 profile_<本次采集时间> 子目录>

【1. 找齐并配对输入】
- 只检查本次采集目录及必要的关联文件，避免全机扫描。原始NPU文件常为 <采集目录>/*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json。
- CPU文件常为 cpu_compute_<PID>.jsonl 或 cpu_compute_<时间>_<PID>.jsonl；时钟文件常为 cpu_clock_anchors_<PID>.jsonl。以内容、PID、采集时间窗核对，不只看文件名，不盲选latest。
- 确认profiler解析已结束，输入文件完整且不再变化。trace_view_cpu_compute*.json 通常已合成，不可直接当原始trace重复追加。
- NPU trace可为事件数组，或含traceEvents数组的对象；保留原有顶层字段。CPU/锚点JSONL逐行读JSON。若有多次采集、多PID或多rank，分别配对和校时，不混合计算。
- 输入有歧义时只询问缺少的路径/采集信息；缺少CPU记录、可靠校时依据或访问权限时明确报缺项，不能假装成功。不得重启服务来补采集，除非我另行授权。

【2. 识别CPU实际计算记录】
常见计算记录字段如下（数值仅示意，不是本次数据）：
{"event":"cpu_moe.compute","pid":123,"tid":456,"layer":0,"part":0,"task_id":7,"mono_raw_ns":1000000000,"duration_ns":90000,"expert_count":1,"expert_route_count":1,"transfer_count":0}
- event=cpu_moe.compute 表示实际CPU专家forward区间，开始时钟为CLOCK_MONOTONIC_RAW纳秒字段mono_raw_ns，duration_ns为记录的计算持续纳秒；不要拿Python submit到sync的等待时间冒充实际计算。
- 也可能有 cpu_moe.compute_total，表示更外层的计算/归并边界。先确认字段和含义，只选一种主要计算边界，不能把total和part相加。多个part可分轨显示，不能把并行part时长总和称为层延迟。
- layer通常从0开始，显示层号可加1；先核对实际日志约定。E=expert_count（唯一CPU专家数），C=expert_route_count（expert-token路由工作量），T=transfer_count（本层实际按需迁入专家数）。标签使用 L<层号>_E<E>_C<C>_T<T>。
- 缺C或T时写unknown，不能默认0。一个专家处理两个token是两份C，仍可能只搬一次权重；一个专家的多次权重/scale memcpy也不能计成多个专家。预取数量与按需T计数分开。
- 只按实际存在、字段含义明确的记录生成区间，不因专家数为0就推测没有其他调度开销。

【3. 精确校时，不能凭图形猜偏移】
NPU导出的Chrome trace通常以微秒为单位，并已由Ascend导出器映射到主机时间线；CPU原始纳秒不能直接与其相减。先核对导出的时间单位及设备/主机时间域关系。常用校时输入：
锚点JSONL：
{"pid":123,"tag":"start","marker_name":"CPU_MOE_CLOCK_ANCHOR_123_abc","before_raw_ns":1000000000,"after_enter_raw_ns":1000002000}
原始trace中对应事件：
{"ph":"X","name":"CPU_MOE_CLOCK_ANCHOR_123_abc","pid":123,"tid":456,"ts":"1800000000000000.0","dur":1}
- 每个marker_name必须在trace中唯一匹配；before/after来自CLOCK_MONOTONIC_RAW，after不得小于before。相同PID至少需要两个不同时间的有效锚点，且应包围要导出的CPU区间。
- 读JSON时使用parse_float=Decimal，避免小数先被float舍入；全程用Decimal(str(value))处理时钟。每个锚点原始时间 x=(before_raw_ns+after_enter_raw_ns)/2，目标时间 y=该marker的ts（微秒）。以最早/最晚锚点建立线性映射：
  a=(y_last-y_first)/(x_last-x_first)
  map_raw_ns(x)=y_first+(x-x_first)*a
  a应接近0.001微秒/纳秒；偏离超过1000ppm则停止并排查。此阈值只是异常检查，不是精度保证。
- 有额外锚点时计算拟合残差；记录锚点夹逼误差约为(after-before)/2000微秒，并结合漂移/残差说明对齐不确定性，不声称纳秒精度。
- CPU区间起点用map_raw_ns(mono_raw_ns)，持续时间用duration_ns/1000微秒。若日志提供同一原始时钟的结束字段，优先映射真实起止点。记录持续时间所用时钟与RAW可能略有速率差，该差异需计入不确定性；不为制造重叠而缩放时长。
- 不对锚点范围外的记录无依据外推：排除并统计，或找到覆盖它们的可靠锚点后再处理。有效CPU区间必须非空。
- CPU生命周期常用mono_ns（CLOCK_MONOTONIC），不能代入RAW映射。若同条记录同时带mono_ns与mono_raw_ns，可建立同次、同PID的桥接映射；只有wall_ns不能直接宣称已对齐MONOTONIC。没有桥接证据时，只计算生命周期之间同一mono_ns时钟的差值，并把统计附在对应task的args中，不把它伪造为已对齐的轨道。
- 若日志不用上述字段，先检查本地已有记录或源码明确时间单位/时钟；不能仅凭字段名称猜测。没有可靠映射就停止并报告具体缺项。

【4. 原生启动、完成、预取和搬运】
- 原生记录常为 event=cpu_moe.lifecycle，包含pid、task_id、layer、phase、mono_ns等。按(pid,task_id)及part/线程等实际上下文配对，不能跨任务拼接；同phase重复时先检查是否对应不同part，不任意取最后一条。
- 常见phase：callback_enter、admission_end、queue_publish、queue_take、compute_begin、compute_end、completion_wait_end、output_ready_recorded、dma_ready_recorded。
- 在当前原生路径中，历史名callback_enter表示CPU监测到输入就绪信号，不等于Python host callback。
- 对存在的合法配对计算：启动=compute_begin-callback_enter；计算=compute_end-compute_begin；计算后返回提交=output_ready_recorded-compute_end。以上用同一mono_ns相减再除1000得到微秒。output_ready_recorded表示就绪操作已提交，不代表设备已经消费结果。
- 缺阶段就标缺失，不补零；出现负时长或配对冲突需排查。CPU实际forward的duration仍按计算记录，不强迫它等于包含其他调度边界的生命周期差值。
- 预取、DMA构造、拷贝API提交、设备MEMCPY执行是不同阶段。仅在日志有可靠起止、身份与可用时钟映射时追加对应轨道；实际设备拷贝以原始NPU事件为准，不能把host API耗时画成设备拷贝。
- 没有预取记录时仍可交付有效CPU合成，但必须说明未合入预取，不能仅因服务开了预取或文件名含prefetch就宣称画出了预取。也不能把所有MEMCPY都算成专家权重搬运。
- 原生路径不得套用“固定三个输入拷贝+HOSTFUNC_CALLBACK”的旧往返匹配规则；只有日志明确支持的任务/事件关联才可用来配对。旧路径有可靠往返证据时可以保留对应区间，否则说明往返尚未识别。

【5. 生成可阅读的Chrome Trace】
- 保留所有本次窗口内原始轨道：NPU设备流、CANN、Python、draft、Overlap Analysis及元数据。不要另做只剩CPU和Computing的简化两轨图，不过滤活跃流。
- 保持原始事件pid/tid、名称、算子、参数、时间戳与持续时间。CPU追加事件用ph="X"、cat="cpu_moe_actual_compute"、ts和dur（微秒），args保留原始PID/TID、task_id、layer/part、E/C/T及校时/阶段统计。
- 不覆盖原有轨道：从未使用的tid中为新增CPU/预取/DMA轨道分配ID。可使用设备进程pid，在args另存原CPU PID/TID；或选择查看器支持的独立进程轨道。保留原Stream名称和数字顺序，通过thread_name/thread_sort_index等元数据把CPU实际计算放在设备流上方，不改原设备事件的pid/tid。
- 裁剪窗口取本次所有已校时CPU计算区间，以及有可靠身份配对的输入/输出往返和相关预取/DMA阶段的连续最早起点到最晚终点。只收录本次任务关联事件来确定边界；不能用无关常驻CPU日志或全局最早事件无限扩大窗口。
- 对ph="X"区间按Decimal判断是否与窗口相交：完全在窗口内则保持原ts/dur的值和类型；只对跨界区间裁剪，dur不得为负；窗口外移除。元数据保留；点事件按所在时间判断。B/E、异步和flow事件应成对保留/处理，避免单边悬空；不明确时保持完整关联并报告边界处理策略。
- 不把1.8e15量级的绝对微秒时间戳转成float再做减法，这会制造约0.25微秒的量化误差和假重叠。写JSON时也不能将Decimal偷偷转成float：保持原数值token（或使用支持的精确序列化），新增/裁剪时间用查看器支持的精确小数字符串或精确数值输出。
- 不重新计算或伪造原来的Overlap Analysis；它保留为原始派生轨道。新增事件不是原生硬件采样，需要在cat/args中清楚区分。

【6. 验证后才交付】
- 输出JSON可以正常解析，有traceEvents/事件数组，CPU实际计算区间数大于0、dur非负、校时区间有效；不要交付空合成。
- 检查每个PID的锚点数量、残差/不确定性、被排除的记录和未匹配任务。检查cpu_moe.trace_flush等writer记录里的dropped；没有writer记录时写未知，不能声称无丢失。缺预取、缺往返匹配等情况如实说明。
- 对每条原始设备Stream，将原始事件按同一窗口精确裁剪后，与导出中的对应事件比较：名称、args、pid/tid、区间均一致；内部事件原ts/dur不变。不因合成增加同流假重叠。原始轨道本来存在的重叠不能“修平”。
- 本地JSON写到新文件，不覆盖原始输入。必要的中间脚本、统计和报告可留在服务器或临时工作目录供验证，不作为下载交付物。

【7. 本地只交付一个最终trace JSON】
- 默认目标为本地当前用户的桌面，新建 profile_<本次采集时间> 子目录；我指定了绝对路径则优先使用指定路径。已有目录重名时创建不冲突的新目录，不覆盖已有文件。
- 只有实际合入预取事件才命名 trace_view_cpu_compute_prefetch.json，否则命名 trace_view_cpu_compute.json。不要改名冒充预取已合入。
- 只交付这一个最终trace JSON。不要下载原始trace、CPU JSONL、其他统计JSON、TXT、数据库、锚点、图片、压缩包或整个profiler目录。
- 远端大文件可临时压缩传输，但本地解压并校验后不保留本次临时压缩包；最终目录只保留trace JSON。不要删除用户已有文件。
- 远端合成时比较远端最终JSON和本地JSON的SHA256及大小，一致后才报告成功。本地合成时校验最终文件可解析，并记录SHA256。
- 最终简短告诉我：本地文件路径、实际CPU区间数量、预取是否合入、对齐是否有效，以及必须说明的缺失/误差。没有完成验证就明确说尚未完成。
```

这份文本不依赖外部链接或仓库脚本。仅有NPU trace、没有CPU日志或校时证据时，AI应指出缺项，而不是自行猜测出“合成结果”。
