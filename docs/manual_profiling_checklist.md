# Manual profiling checklist

Use this runbook to capture regressions whenever overscan behaviour or draw
performance changes. The steps assume the synthetic fixtures already ship with
`test.edf` in the `raw/` directory.

## Quick smoke

1. Export the offscreen platform so Qt will run headless: `export QT_QPA_PLATFORM=offscreen`.
2. Launch the viewer against the bundled fixture: `python app.py raw/test.edf`.
3. Immediately zoom out to ≥120 s and confirm a coarse preview appears within one frame,
   then refines within ~200 ms.
4. Pan left/right in ±2 window increments and confirm previews keep pace without
   clearing the curves.

## Record timings

1. Run `python scripts/profile_draw.py raw/test.edf --windows 10 60 120`.
2. Capture the reported preview/final timings and compare against the existing
   baseline in the task tracker.
3. If any window exceeds 1.0 s for the refined draw, capture a profiler trace
   (e.g. `python -m cProfile -o overscan.prof scripts/profile_draw.py ...`).

## GPU vs CPU parity (overscan overlays)

1. Ensure the overscan cache matches production defaults: `overscan.window_preload=2`
   and `overscan.preview_stride=2` in `config.ini` (no overrides in your
   environment variables).
2. Launch the renderer benchmark with representative channel counts:
   ``python scripts/benchmark_renderers.py --channels 6 --channels 12 --channels 24 --iterations 6``.
3. For GPU-equipped workstations, re-run without `--skip-gpu`; on CPU-only CI,
   keep the flag so the harness still exercises the CPU overlays.
4. Record both the average frame time and reported vertex totals for each batch
   (values are printed to stdout).  Investigate any GPU frame time that trails
   the CPU by more than 15 % or increases the peak Python memory footprint by
   ≥25 %.
5. When investigating a regression, repeat the run with `QT_QPA_PLATFORM=offscreen`
   to eliminate desktop compositor differences.

## Screenshot regression check

1. With the viewer still pointed at `raw/test.edf`, grab a headless screenshot
   (``pytest tests/test_headless_ui.py::test_main_window_headless_smoke``
   already produces `headless_smoke.png`).
2. Visually confirm the preview/final swap renders the expected tile without
   leaving empty channels.

## Automated guardrails

* Run `pytest tests/test_headless_ui.py::test_main_window_overscan_preview_then_final`
  to assert coarse tiles arrive before refinement.
* Run `pytest tests/test_prefetch.py::test_prefetch_preview_runs_before_final`
  to ensure the prefetch queue prioritises preview tiles.
