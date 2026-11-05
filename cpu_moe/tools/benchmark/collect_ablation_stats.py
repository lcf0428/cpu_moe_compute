#!/usr/bin/env python3
"""Collect TPOT/throughput and expert hit rates for one ablation run dir.

Reads the newest ais_bench result CSV (TPOT/E2EL/TTFT/throughput) and the
decode-stats benchmark counters (expert hit rates, NPU/CPU route split).
Standard library only.
"""
import glob
import json
import os
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: collect_ablation_stats.py <run_dir>")
        return 2
    run_dir = sys.argv[1]
    case = os.path.basename(os.path.normpath(run_dir))

    perf = {}
    csvs = sorted(glob.glob(os.path.join(run_dir, "outputs", "*", "performances", "*", "gsm8k.csv")))
    if not csvs:
        print(f"[{case}] ERROR: no ais_bench result csv under {run_dir}/outputs")
        return 1
    with open(csvs[-1]) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                perf[parts[0]] = {"avg": parts[2], "median": parts[5], "min": parts[3], "max": parts[4]}

    req_rate = None
    js = sorted(glob.glob(os.path.join(run_dir, "outputs", "*", "performances", "*", "gsm8k.json")))
    if js:
        try:
            with open(js[-1]) as f:
                data = json.load(f)
            req_rate = data.get("Request Throughput", {}).get("total")
        except Exception:
            pass

    print("=" * 78)
    print(f"[{case}]  ais_bench: {os.path.relpath(csvs[-1], run_dir)}")
    print("=" * 78)
    for key in ("E2EL", "TTFT", "TPOT", "ITL", "OutputTokens", "OutputTokenThroughput"):
        if key in perf:
            p = perf[key]
            print(f"  {key:<22} avg={p['avg']:>12}  median={p['median']:>12}  min={p['min']:>12}  max={p['max']:>12}")
    if req_rate:
        print(f"  {'RequestThroughput':<22} {req_rate:>12}")

    counters = sorted(glob.glob(os.path.join(run_dir, "stats", "benchmark_counters_rank0.json")))
    if not counters:
        print(f"\n[{case}] WARNING: no benchmark_counters_rank0.json — expert hit rates unavailable")
        return 0
    with open(counters[-1]) as f:
        d = json.load(f)
    totals = d.get("totals") or {}
    cols = d.get("columns") or []
    per_layer = d.get("per_layer")

    def rate(num, den):
        return totals[num] / totals[den] if totals.get(den) else float("nan")

    print("-" * 78)
    print(f"[{case}]  expert cache counters:")
    print(f"  avg expert hit rate (post): {rate('post_hit_experts', 'post_experts'):.4f}")
    print(f"  avg expert hit rate (pre):  {rate('pre_hit_experts', 'pre_experts'):.4f}")
    print(f"  avg route hit rate (post):  {rate('post_hit_routes', 'post_routes'):.4f}")
    cpu = totals.get("cpu_routes") or 0
    npu = totals.get("npu_routes") or 0
    if cpu + npu:
        print(f"  route split: npu={npu} ({npu / (cpu + npu) * 100:.1f}%)  cpu={cpu} ({cpu / (cpu + npu) * 100:.1f}%)"
              f"  admitted_experts={totals.get('admitted_experts')}")

    def layer_value(v, col):
        if isinstance(v, dict):
            return v.get(col, 0)
        if isinstance(v, list):
            return v[cols.index(col)] if col in cols else 0
        return 0

    if per_layer:
        hash_layers = {int(x) for x in (d.get("excluded_hash_layers") or [])}
        items = per_layer.items() if isinstance(per_layer, dict) else enumerate(per_layer)
        rates = []
        for key, v in items:
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            if key in hash_layers:
                continue
            pe = layer_value(v, "post_experts")
            ph = layer_value(v, "post_hit_experts")
            if pe:
                rates.append(ph / pe)
        if rates:
            print(f"  per-layer hit rates (hash layers excluded): n={len(rates)} mean={sum(rates) / len(rates):.4f} "
                  f"min={min(rates):.4f} max={max(rates):.4f}")
            print("  rates:", [round(r, 4) for r in rates])

    sums = sorted(glob.glob(os.path.join(run_dir, "stats", "decode_stats_summary_*.txt")))
    if sums:
        print("-" * 78)
        with open(sums[-1]) as f:
            for line in f:
                if line.startswith(("run ", "offload ", "decode ", "measured ")):
                    print("  | " + line.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
