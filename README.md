# EDFView (v0.2)

A lightweight, multi-channel EDF viewer built for fast skimming through long sleep studies.  
Version 0.2 introduces an overscan renderer that pre-renders ±2 viewport windows so pans feel instant while the data source (EDF or Zarr) loads quietly in the background.

## Features

- Multi-channel stacked viewer with shared time axis (seconds-first UI)
- Automatic EDF → Zarr background ingest with parity checks
- Prefetch ring buffer and memory budgets exposed in the UI
- Overscan tile renderer (v0.2) keeps curve data hot within ±2 windows for stutter-free panning
- Auto-detects companion CSV/Staging files (APPLES-style) and overlays events/stages on the timeline
- Dedicated hypnogram lane with stage curves and an event navigator (list + next/prev) plus full-screen shaded event regions with hover tooltips
- Configurable via `config.ini` or CLI overrides; persists user knob values with `QSettings`
- Tested helper utilities for timebase conversions, decimation, and overscan slicing

## Quick start

```bash
uv pip install -r requirements.txt
uv run python app.py raw/<study>.edf
```

1. Drop EDFs into `raw/` (ignored by git). Derived artifacts land in `processed/`.
2. Launch the viewer with the EDF path. If no Zarr cache exists the app builds one in the background and swaps automatically.
3. Pan within ±2 windows to stay inside the overscan buffer. Crossing the boundary triggers a new render—watch the log for `Overscan render failed`.

## Project layout

```
core/          loaders, timebase helpers, overscan utilities
ui/            Qt widgets + main window
scripts/       packaging, profiling, validation helpers
tests/         pytest suite (timebase, loader, overscan, view window)
```

See `agents.md` for the full working playbook (milestones, runbooks, decision log).

## Testing

```
uv run python -m pytest
```

The suite covers loader slices, timebase math, overscan slicing, and view-window helpers. Zarr cache parity tests are planned; perf smoke scripts live under `scripts/`.

## Performance tuning

Short EDFs can be promoted into the int16 RAM cache for instant pans. Toggle the feature in the `[cache]` section of `config.ini` (see `docs/perf_comparison.md`) to cap the size and choose whether to keep the cache purely in memory or spill to a memmap backing file.

## Roadmap

- Phase 4: CSV annotations import & overlay (detailed plan in `agents.md`)
- Channel toggles + per-channel scaling (Phase 3)
- Perf smoke automation in CI
- Longer-term: multiresolution pyramids, Parquet events, GPU-friendly envelopes

---

Questions, findings, or next steps? Capture them in `agents.md` before diving in.
