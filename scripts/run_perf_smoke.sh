#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PERF_PROFILE_PATH:-}" ]]; then
  echo "[perf] PERF_PROFILE_PATH not set; skipping render benchmark." >&2
  exit 0
fi

echo "[perf] Running profile_draw on $PERF_PROFILE_PATH"
"$(dirname "$0")/profile_draw.py" "$PERF_PROFILE_PATH"
