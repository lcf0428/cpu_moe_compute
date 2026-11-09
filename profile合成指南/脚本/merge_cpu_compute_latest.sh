#!/usr/bin/env bash
# Portable reference wrapper: always identify one capture explicitly.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if (( $# < 3 || $# > 4 )); then
  echo 'Usage: merge_cpu_compute_latest.sh TRACE CPU_LOG CLOCK_ANCHORS [OUT_DIR]' >&2
  exit 2
fi
TRACE=$1
CPU_LOG=$2
CLOCK_ANCHORS=$3
OUT_DIR=${4:-$(dirname -- "$TRACE")}
for path in "$TRACE" "$CPU_LOG" "$CLOCK_ANCHORS"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p -- "$OUT_DIR"
python3 "$HERE/merge_cpu_compute_trace.py" "$TRACE" "$CPU_LOG" \
  --clock-alignment "$CLOCK_ANCHORS" \
  --compute-event "${CPU_MOE_COMPUTE_EVENT:-cpu_moe.compute}" \
  --include-actual-compute --actual-compute-only \
  --out "$OUT_DIR/trace_view_cpu_compute.json" \
  --stats "$OUT_DIR/cpu_compute_stats.json" \
  --report "$OUT_DIR/cpu_compute_report.txt"
cat -- "$OUT_DIR/cpu_compute_report.txt"
