#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PERF_PROFILE_PATH:-}" ]]; then
  echo "[perf] PERF_PROFILE_PATH not set; skipping render benchmark." >&2
  exit 0
fi

echo "[perf] Running profile_draw on $PERF_PROFILE_PATH"
"$(dirname "$0")/profile_draw.py" "$PERF_PROFILE_PATH"

echo "[perf] Running renderer benchmark"
bench_cmd=("$(dirname "$0")/benchmark_renderers.py" --iterations 4)
if [[ -z "${PERF_ENABLE_GPU_BENCH:-}" ]]; then
  bench_cmd+=(--skip-gpu)
fi
"${bench_cmd[@]}"
