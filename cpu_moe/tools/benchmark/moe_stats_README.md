# 投机采信率与专家统计

文件：`moe_stats.py`，仅依赖 Python 标准库。只读统计，不开启 profiler，不改变服务或日志开关。
在 cpu_dspark 容器内使用，工作目录为本文件所在的 cpu_moe。

## 统计一次评测区间

```bash
cd /mnt/share/y50063808/vllm-ascend-workspace/cpu_moe
STATS_DIR=$(mktemp -d ../.vaws-local/moe-stats-XXXXXXXX)
python3 moe_stats.py snapshot --out "$STATS_DIR/before.json" >/dev/null
# 此处运行自己的评测，等待全部完成，等待服务周期性 metrics 更新完成。
python3 moe_stats.py snapshot --out "$STATS_DIR/after.json" >/dev/null
python3 moe_stats.py report --before "$STATS_DIR/before.json" --after "$STATS_DIR/after.json" --out "$STATS_DIR/result.json"
```

服务地址默认 http://127.0.0.1:8150/metrics，可通过 snapshot --url 修改，抓取时自动绕过代理。
采信率 acceptance_rate = 接受 draft token 数 / 提交 draft token 数。
平均接受长度 mean_accepted_length_including_bonus = 1 + 接受 draft token 数 / draft 轮数。
比例字段范围 0~1，乘100即百分数；空值表示无法计算，不是0。
计数差按同一 model/engine 的同一组指标标签逐项核对。计数下降时拒绝输出；请勿跨服务重启取差，重启后计数又超过起始值无法仅凭指标检测。
所有并发请求、预热都可能计入该区间。只统计正式样本时，应在预热结束且指标更新后保存 before，评测完成且指标更新后保存 after；不要在区间内运行其他请求。
默认 model=deepseek_v4，engine=0，可使用 report --model/--engine 修改。

## 离线读取已有指标

```bash
python3 moe_stats.py report --before ../.vaws-local/automation-3-8150-tpot/metrics_before.txt --after ../.vaws-local/automation-3-8150-tpot/metrics_after.txt
```

## 专家缓存命中率

```bash
python3 moe_stats.py report --hit-stats hit_test_20261014/hit_stats.json
```

兼容原来逐层 hit_stats.json：hits/selections 为按路由次数加权的命中率；unique_hits/unique_selections 为每次调用去重后的命中率。统计范围严格取决于该文件的采集范围，脚本不会自动把旧 hit_stats 与新的评测区间对齐。
当前 NPU 侧维护映射的方案，仅 KT_MOE_CPU_TRACE 并不记录完整的初始路由总数和命中数，无法还原精确的初始缓存命中率。本脚本不会为此安装旧 Python hook 或增加同步拷贝；没有完整计数时该项显示 null。原来的旧 hook 对当前 NPU 设备端路径不适用。

## 当前 CPU 日志统计

```bash
python3 moe_stats.py report --cpu-log /absolute/path/cpu_compute_PID.jsonl --tokens 8 --out /tmp/cpu_stats.json
```

--tokens 8 仅用于当前 spec7、batch1 的8-token验证调用；这是形状筛选，不是通用 decode 检测。不要直接用于其他投机长度或多请求混跑，也不能保证排除恰为8-token的 prefill。
可加 --since-ns START --until-ns END，按 wall_ns 过滤 [START,END)。不指定则扫描整个文件。
输出 CPU 唯一专家数的均值/中位数/p90/max、CPU 路由次数、选中搬运专家总数和每记录均值、各层记录数、日志丢失提示。
T 是决定搬运的专家数，不是已完成 DMA 的确认计数。CPU 专家数是转移选择后仍由 CPU 计算的专家数，不能直接当初始 miss，也不能用于推导缓存命中率。
每条记录对应一个 layer/partition 调用，不擅自拼成全模型一步。日志丢失量是生命周期总量，非筛选区间增量。
推荐读取停止写入或已复制完整的日志；不要同时传入重叠/重复文件。CPU 日志仅需 KT_MOE_CPU_TRACE，不需内部详细耗时或 NPU profiler。
上述 report 参数可组合使用，JSON 中分别列出 speculative、expert_cache、cpu，各自记录统计范围。
