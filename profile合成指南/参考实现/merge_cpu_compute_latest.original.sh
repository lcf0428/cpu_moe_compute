#!/bin/bash
set -euo pipefail
WS=/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe
CPU_DIR="${CPU_COMPUTE_TRACE_DIR:-/tmp/cpu_moe_compute}"
TRACE="${1:-}"
CPU_LOG="${2:-}"
# Archived CPU logs in the capture directory also work outside the container.
mapfile -t INPUTS < <(python3 - "$WS" "$CPU_DIR" "$TRACE" "$CPU_LOG" <<'PY'
import pathlib, sys
ws, cpu_dir, trace, cpu_log = sys.argv[1:]
def latest(paths):
    paths = [p for p in paths if p.is_file()]
    return str(max(paths, key=lambda p: p.stat().st_mtime)) if paths else ''
if not trace:
    profile = pathlib.Path(ws) / 'cpu_profile' / 'latest'
    trace = latest(profile.resolve().rglob('trace_view.json')) if profile.exists() else ''
print(trace)
print(cpu_log)
PY
)
TRACE="${INPUTS[0]:-}"
CPU_LOG="${INPUTS[1]:-}"
if [[ -z "$CPU_LOG" && -f "$TRACE" ]]; then
    CPU_LOG=$(python3 "$WS/select_cpu_compute_log.py" "$TRACE" "$CPU_DIR")
fi
[[ -f "$TRACE" && -f "$CPU_LOG" ]] || { echo "Need Ascend trace and /tmp CPU log from the same run; run this script inside cpu_moe or pass both paths." >&2; exit 1; }
OUT_DIR="${TRACE%/*}"
printf 'Ascend trace: %s\nCPU log: %s\n' "$TRACE" "$CPU_LOG"
python3 "$WS/merge_cpu_compute_trace.py" "$TRACE" "$CPU_LOG" \
    --include-actual-compute --actual-compute-only \
    --out "$OUT_DIR/trace_view_cpu_compute.json" \
    --stats "$OUT_DIR/cpu_compute_stats.json" \
    --report "$OUT_DIR/cpu_compute_report.txt"
cat "$OUT_DIR/cpu_compute_report.txt"
