# vLLM 0.30：当前单卡 CPU MoE 实验

本次实验固定 **vLLM 0.30.0**，153 容器实际版本字符串是 `0.30.0+empty`。`+empty` 是当前构建的本地版本后缀。本次使用包含 DeepSeek V4/DSpark 支持的实验构建，保留这套已验证环境，不在复现时升级 vLLM，也不将同名上游安装包视为完全相同的构建。

vLLM Ascend 使用 [yxk-fdu/vllm-ascend 的 CPU 分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928)，本次提交固定为 [61afe18c](https://github.com/yxk-fdu/vllm-ascend/commit/61afe18c467110387fa7038725d96067e1be6e4c)。运行环境为 torch `2.10.0+cpu`、torch-npu `2.10.0.post4`、CANN 9.2.0。torch 的版本不改变 vLLM 0.30 这个实验基线。

## 这次主要改动

1. **通信与任务唤醒**：NPU 发布共享内存序列信号，C++ 监测线程直接规划和提交 CPUInfer 计算，独立完成线程发布 CPU 结果就绪。下一层预取也接入同一个原生监测器，不用 ACL→Python host callback 唤醒计算或预取。实际权重拷贝仍由独立 Python/Torch DMA 线程提交。
2. **搬运次序与流水线**：本层按需权重就绪后才触发下一层预取，避免预取抢在本层按需搬运前面。构造一个专家就提交该专家的拷贝，再构造下一个。权重、量化属性和映射完成后才发布就绪，下一层 MoE 使用前仍要等待。
3. **专家调度**：CPU 成本按 expert-token 路由次数计数，一个专家处理两个 token 就计两份计算量，搬运权重仍只算一次。优先考虑被选中次数多的缺失专家，用简单的 CPU/搬运/NPU 计算时间估计选择搬运数量。共享 C++ LRC 决定淘汰谁，预取只读取下一层历史，不把预测当成实际需求再更新一次 LRC。

当前估计为 `max(剩余CPU路由数×93.16μs, 搬运专家数×275.828μs + 200μs)`，200μs 是每层一次的 NPU 计算估计，不是每专家加一次。参数默认值与当前实验显式取值不同，复现以启动脚本为准。

## 准备已有环境

先进入已验证的 `cpu_moe` 容器，保留其中的 vLLM 0.30 构建。需要模型原始 NPU 权重和现有逐层 GGUF 两份数据。路径通过环境变量传入，计算和调度默认值保留本次取值。启动前设置 `WORKSPACE`、`ASCEND_REPO`、`KT_BUILD`、`MODEL_PATH` 和 `GGUF_PATH`。

vLLM Ascend 安装本次源码，并构建两个小扩展：

```bash
export ASCEND_REPO=/workspace/vllm-ascend
cd "$ASCEND_REPO"
# 不升级已有 vLLM / Torch / Torch NPU。
python -m pip install -e . --no-build-isolation --no-deps
python -m pip install -e ./csrc/lrc_policy --no-build-isolation --no-deps
(cd csrc/lrc_policy && python setup_npu_wait.py build_ext --inplace)
```

kt-kernel 必须导出 `native_dispatch_api()` 和 `native_cpu_handle()`，并保留 `libkt_ready_poll.so`。153 当前运行包已经具备这些接口。若使用本仓库原始 kt 源码快照，先应用 [原生接口补丁](../patches/kt_native_dispatch_api.patch)，再沿用原有 Ascend 构建配置重编译 `kt_kernel_ext`，不要改变已有数学内核或量化配置：

```bash
# 在 cpu_moe_compute 仓库根目录执行，仅对尚未打过补丁的源码使用。
git apply --check --directory=cpu_moe/dependencies/ktransformers-AK patches/kt_native_dispatch_api.patch
git apply --directory=cpu_moe/dependencies/ktransformers-AK patches/kt_native_dispatch_api.patch
# KT_BUILD_DIR 是该份源码已配置好的 CMake 构建目录。
cmake --build "$KT_BUILD_DIR" --target kt_kernel_ext -j2
# 将新扩展安装到 KT_BUILD 指向的运行包；动态库依赖须与当前环境匹配。
```

## 启动

使用 [run_cpu_moe_vllm030.sh](../scripts/run_cpu_moe_vllm030.sh)。这是按本次实际运行参数整理的单卡入口，脚本不安装或升级依赖，启动前会检查 vLLM 版本和原生接口。默认 BS1、DSpark7、CPU 计算开启、预取关闭。

```bash
# 路径是示例，替换为当前容器已经可用的目录；不移动或复制权重。
export WORKSPACE=/workspace
export ASCEND_REPO="$WORKSPACE/vllm-ascend"
export KT_BUILD=/workspace/cpu_moe/kt-native-profile-package
export MODEL_PATH=/models/DeepSeek-V4-Flash-0731
export GGUF_PATH='/models/gguf/dsv4_layer{layer_idx}_mxfp4.gguf'
export LOG_ROOT=/workspace/logs/cpu_moe
# 以下命令在容器内执行；一次只保留一个配置的服务。
BS=1 DSPARK=7 PREFETCH=0 bash scripts/run_cpu_moe_vllm030.sh
BS=1 DSPARK=7 PREFETCH=1 bash scripts/run_cpu_moe_vllm030.sh
BS=1 DSPARK=0 PREFETCH=0 bash scripts/run_cpu_moe_vllm030.sh
BS=1 DSPARK=0 PREFETCH=1 bash scripts/run_cpu_moe_vllm030.sh
```

切换配置前先正常停止上一进程，确认 API 和 EngineCore 均退出。不要用宽泛 `pkill` 或停止别人的容器。脚本前台运行，Ctrl+C 正常退出；需要后台运行时自行保留对应 PID 与日志。日志和采集目录由 `LOG_ROOT` 指定，容器保留。

## 重要参数

| 参数 | 本次取值 | 含义 |
| --- | --- | --- |
| `DSPARK` | `0` / `7` | 0 不传 speculative-config；7 使用 DSpark、每次草拟 7 个 token。 |
| `BS` | `1` | 服务 `max_num_seqs`，AISBench 并发也必须相同。BS4 的旧结果另列，不混入本次 BS1。 |
| `PREFETCH` / `expert_prefetch_num` | 0 或 1 / `1` | 开关下一层预取；开启时最多预取一个专家。 |
| `native_dispatch` | `true` | 使用共享信号与 C++ 原生监测、CPU 提交和完成路径。 |
| `transfer_policy` | `load_balance` | 按实际路由计算量与搬运/NPU 时间估计分配缺失专家。 |
| CPU / 拷贝 / NPU 固定成本 | 93.16 / 275.828 / 200 μs | CPU 按 expert-token，拷贝按专家，NPU 固定成本按层。仅是当前硬件与内核的校准值。 |
| `NUMA_NODE` / `CPU_THREADS` | `1` / `30` | GGUF 和 CPU 计算线程同 NUMA；153 的计算核为 32–61。 |
| `POLL_CPU` / `COMPLETION_CPU` / `DMA_CPU` | 62 / 63 / 127 | 监测、完成、DMA 线程不占计算核；153 的 63 与 127 是 SMT 兄弟核。移植时先查拓扑。 |
| `HOST_WEIGHT_NUMA_NODE` | `0` | NPU 搬运源的锁页权重缓冲位置，不是 CPU GGUF 的位置。 |
| `num_device_experts` / `num_device_layers` | `[84]` / `2` | NPU 专家驻留配置，直接影响显存与需要搬运的专家数。 |
| `max_num_tokens` | `48` | CPU 后端 token 容量。并发与 DSpark 组合不能超过该容量。 |
| `KT_MOE_MXFP4_TOKEN_REUSE` | `0` | 这次关闭多激活权重复用，是线性 CPU 成本校准的前提之一。 |
| `KT_MOE_MATCH_NPU_MXFP8` | `1` | 对齐 CPU/NPU 激活计算路径，专家在两侧迁移时保持同一实验设置。 |
| `enable_cpu_binding` | `false` | 不让 vLLM 自动绑核覆盖 kt 后端的 NUMA/线程设置。 |

其余模型长度、KV 格式、超时等参数按已验证启动配置保留，不作为这次对比变量。预取就绪使用独立设备标志，不能与 CPU 结果就绪混用；不要把同步等待删除来换吞吐。

## CPU 耗时采集

需要单独采集CPU侧耗时时，用 `CPU_MOE_PROFILE=1` 启动，再按现有 [日志合成说明](01_profile_merge_guide.md) 操作。当前profiler配置delay=2、max_iterations=200；DSpark7短采集请求使用max_tokens=20并自然结束。服务启动不自动开始采集。

本仓库只保存CPU计算的代码依赖、启动入口和使用说明，性能统计结果保存在本地结果目录。
