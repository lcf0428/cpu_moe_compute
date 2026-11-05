#!/usr/bin/env bash
# Run inside cpu_dspark. Keep all existing ais_bench arguments after this script.
set -euo pipefail
BASE=/mnt/share/y50063808/vllm-ascend-workspace
CODE=/mnt/share/y50063808/vllm-ascend-workspace/vllm-ascend
PROFILE_RUN=$(basename "$(readlink -f "$BASE/cpu_moe/cpu_profile/latest")")
COUNTERS="$BASE/logs/cpu_moe_${PROFILE_RUN}_stats/benchmark_counters_rank0.json"
OUT="$BASE/outputs/aisbench_stats_$(date +%Y%m%d_%H%M%S_%N)"
if [ "$#" -eq 0 ]; then
    echo "Usage: bash $0 ais_bench <your usual arguments>" >&2
    exit 2
fi
exec python3 "$CODE/tools/aisbench_with_stats.py" \
    --counter-file "$COUNTERS" --out "$OUT" -- "$@"
