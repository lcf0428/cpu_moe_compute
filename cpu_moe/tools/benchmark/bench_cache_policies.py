#!/usr/bin/env python3
"""Standalone NPU bench for the fused expert-cache eviction policies (LRC vs LRU).

No vllm service needed: builds LoadBalanceTransferPolicy directly and drives
the fused Triton path (ModelNew, what the hybrid stack runs under graph replay)
against the trusted pure-torch reference, on one NPU card.

  1. Correctness: fused vs reference step-by-step over a skewed trace
     (mapping / candidates / victims / active must match exactly).
  2. Latency: event-timed per-step microseconds for fused LRC, fused LRU,
     and both unfused torch references, at each token-row count.

Run inside cpu_dspark:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=7 python3 bench_cache_policies.py
"""
import argparse
import statistics
import sys
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401  registers the npu backend

sys.path.insert(0, "/mnt/share/y50063808/vllm-ascend-workspace/vllm-ascend")

from vllm_ascend.expert_offload.device_lrc_policy import DeviceLRCHotness
from vllm_ascend.expert_offload.device_lru_policy import DeviceLRUHotness
from vllm_ascend.expert_offload.load_balance_transfer_policy import LoadBalanceTransferPolicy


def make_config():
    # Field values mirror the production run script (run_cpu_moe.sh).
    return SimpleNamespace(
        cache_recent_window=32,
        cache_ema_beta=0.9,
        cache_recent_weight=1.0,
        cache_ema_weight=0.5,
        cache_router_weight=0.3,
        cache_age_weight=0.01,
        balance_max_transfers=32,
        balance_cpu_expert_us=70.0,
        balance_copy_us=450.0,
    )


def make_seed_state(experts, window, hot_count):
    """Mimic LRCLayerState right after hot-expert preload seeding."""
    freq = [0] * experts
    router = [0.0] * experts
    for rank in range(hot_count):
        norm = (hot_count - rank) / hot_count
        freq[rank] = max(1, round(norm * window))
        router[rank] = norm
    return SimpleNamespace(
        freq=freq,
        ema=[0.0] * experts,
        router_score=router,
        last_used=[-1] * experts,
        step=0,
    )


def build_policy(mode, experts, slots, device, cfg, seed_state):
    policy = LoadBalanceTransferPolicy(experts, slots, device, cfg)
    if mode == "lrc":
        policy.lrc = DeviceLRCHotness(experts, device, cfg, seed_state)
    elif mode == "lru":
        policy.lrc = DeviceLRUHotness(experts, device, cfg, seed_state)
    else:
        raise ValueError(mode)
    return policy


def initial_mapping(experts, slots, device):
    mapping = torch.full((experts,), -1, dtype=torch.int32)
    mapping[:slots] = torch.arange(slots, dtype=torch.int32)
    return mapping.to(device)


def make_trace(steps, rows, topk, experts, device, seed, skew):
    """Zipf-skewed route ids + softmax-ish weights; deterministic per seed."""
    gen = torch.Generator().manual_seed(seed)
    probs = 1.0 / torch.arange(1, experts + 1, dtype=torch.float64) ** skew
    probs /= probs.sum()
    trace = []
    for _ in range(steps):
        idx = torch.multinomial(probs, rows * topk, replacement=True, generator=gen)
        ids = idx.reshape(rows, topk).to(device=device, dtype=torch.int32)
        w = torch.rand(rows, topk, generator=gen).to(device=device)
        trace.append((ids, w))
    return trace


def check_correctness(mode, experts, slots, rows, topk, steps, device, cfg, seed_state, seed, skew):
    fused = build_policy(mode, experts, slots, device, cfg, seed_state)
    ref = build_policy(mode, experts, slots, device, cfg, seed_state)
    mapping_f = initial_mapping(experts, slots, device)
    mapping_r = initial_mapping(experts, slots, device)
    trace = make_trace(steps, rows, topk, experts, device, seed, skew)
    for step, (ids, w) in enumerate(trace):
        cf, vf, af = fused.select(ids, mapping_f, w)
        cr, vr, ar = ref.reference(ids, mapping_r, w)
        if not torch.equal(mapping_f, mapping_r):
            diff = (mapping_f != mapping_r).nonzero().flatten().tolist()
            print(f"[{mode}] FAIL step={step}: mapping differs at experts {diff[:10]}")
            return False
        if not (torch.equal(cf, cr) and torch.equal(vf, vr) and torch.equal(af, ar)):
            print(f"[{mode}] FAIL step={step}: "
                  f"cand_eq={torch.equal(cf, cr)} vict_eq={torch.equal(vf, vr)} act_eq={torch.equal(af, ar)}")
            return False
    resident = int((mapping_f >= 0).sum())
    print(f"[{mode}] correctness PASS: {steps} steps, mapping/candidates/victims/active all equal; "
          f"final resident={resident}/{slots}")
    return True


def time_path(call, trace, warmup, rounds, offset):
    for i in range(warmup):
        call(trace[(offset + i) % len(trace)])
    torch.npu.synchronize()
    per_step = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for i in range(len(trace)):
            call(trace[i])
        end.record()
        torch.npu.synchronize()
        per_step.append(start.elapsed_time(end) * 1000.0 / len(trace))
    return min(per_step), statistics.median(per_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--slots", type=int, default=84)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--rows", default="1,8")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--skew", type=float, default=1.1)
    args = parser.parse_args()

    device = torch.device("npu")
    cfg = make_config()
    seed_state = make_seed_state(args.experts, cfg.cache_recent_window, args.slots)
    ok = True
    for rows in (int(x) for x in args.rows.split(",")):
        for mode in ("lrc", "lru"):
            ok &= check_correctness(mode, args.experts, args.slots, rows, args.topk,
                                    args.steps, device, cfg, seed_state,
                                    args.seed + rows, args.skew)
    if not ok:
        sys.exit(1)

    print()
    header = f"latency: E={args.experts} S={args.slots} K={args.topk} " \
             f"steps={args.steps} rounds={args.rounds} (µs per select(), lower is better)"
    print(header)
    print(f"{'rows':>4}  {'mode':>4}  {'path':>6}  {'min':>10}  {'median':>10}")
    for rows in (int(x) for x in args.rows.split(",")):
        trace = make_trace(args.steps, rows, args.topk, args.experts, device,
                           args.seed + rows, args.skew)
        for mode in ("lrc", "lru"):
            fused = build_policy(mode, args.experts, args.slots, device, cfg, seed_state)
            mapping = initial_mapping(args.experts, args.slots, device)
            lo, med = time_path(lambda t: fused.select(t[0], mapping, t[1]),
                                trace, args.warmup, args.rounds, rows)
            print(f"{rows:>4}  {mode:>4}  {'fused':>6}  {lo:>10.2f}  {med:>10.2f}")
            ref = build_policy(mode, args.experts, args.slots, device, cfg, seed_state)
            mapping_r = initial_mapping(args.experts, args.slots, device)
            lo, med = time_path(lambda t: ref.reference(t[0], mapping_r, t[1]),
                                trace, args.warmup, args.rounds, rows)
            print(f"{rows:>4}  {mode:>4}  {'torch':>6}  {lo:>10.2f}  {med:>10.2f}")


if __name__ == "__main__":
    main()
