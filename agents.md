# edfview • agents.md
A lightweight playbook for Future-You (and any local “agent”) to work consistently on the EDF viewer stack.

---

## 0) Purpose
Keep tasks small, interfaces clean, and decisions visible. This doc is the “how we work” guide for building a fast, reliable EDF/annotation viewer—step by step.

> **Current status:** v0.2 (2025-09-25) — overscan tile renderer delivers instant ±2-window pans; CSV annotations are next on deck.

---

## 1) Repo map (living)
edfview/
app.py
requirements.txt
agents.md            <- you are here
.gitignore
raw/            # input EDF/CSV (ignored)
processed/      # cache/HDF/Zarr/parquet (ignored)
core/
edf_loader.py
timebase.py
annotations.py
overscan.py
ui/
main_window.py
time_axis.py
tests/
test_timebase.py
test_loader.py
test_overscan.py
test_annotations_csv.py

---

## 2) Data paths
- **RAW_DIR**: `raw_data/`
- **PROC_DIR**: `processed_data/`
- Never write to `raw_data/`. All derived data goes to `processed_data/`.
- Tiny synthetic fixtures may live in `tests/data/` (tracked).

---

## 3) Coding conventions (short & sharp)
- **Seconds are king** in the UI. Use `Timebase` for all conversions.
- Never render > ~2× #pixels per channel; decimate or envelope.
- Public functions return **numpy arrays, not lists**.
- All loaders return immutable metadata (dataclasses).
- One module = one responsibility. Resist “just this once.”

---

## 4) Runbooks

### 4.1 Open an EDF & view 30 s
1. Put file in `raw_data/`.
2. `python app.py raw_data/<file>.edf`
3. In UI: pick channel → set Start + Window → verify axis shows HH:MM:SS.

### 4.2 Add external CSV annotations
1. Create a mapping YAML (see snippet below).
2. Wire `annotations.from_csv()` in `main_window.py`.
3. Toggle “Show events” and confirm spans/markers align.

### 4.3 Quick performance sanity
- Keep window ≤ 120 s.
- If total samples to draw > 4× plot width → decimate.
- Use a 60 ms debounce on sliders (no constant re-draw).

### 4.4 Build Zarr cache (Phase 5)
1. Opening an EDF automatically kicks off `EdfToZarr` in the background (progress shown in the control panel).
2. Cached stores land in `processed/<stem>.zarr/` (created if missing).
3. Viewer still reads from EDF; parity tests ensure Zarr mirrors the source for later swap-over.

### 4.5 Overscan sanity (v0.2)
1. With an EDF open, pan left/right in ≤2-window increments.
2. Confirm curves stay static (no redraw shimmer) and axis updates immediately.
3. Watch logs for `Overscan render failed`—treat as regression if emitted.

### 4.6 Annotations auto-import
1. Place companion CSVs (`<stem>.csv`, `<stem>STAGE.csv`) beside the EDF.
2. Open the EDF and ensure the control panel toggles "Show annotations" automatically.
3. Verify event markers render as vertical lines and the stage label updates while panning.

---

## 5) Phase milestones (don’t skip)

**Phase 1 – Loader (no cache)**
- [x] `EdfLoader.read(i, t0, t1) -> (t_s, x)`
- [x] `EdfLoader.read_annotations()` (EDF+)

**Phase 2 – Single-channel viewer**
- [x] Time axis HH:MM:SS
- [x] Channel dropdown, start + window controls *(superseded by multi-channel global controls)*

**Phase 3 – Multi-channel stack**
- [ ] Checkbox list to toggle channels
- [ ] Per-channel scaling (z-score from first view)

**Phase 4 – Annotations overlays**
- [ ] EDF+ events (onset, spans)
- [ ] CSV adapter with mapping

**Phase 5 – Zarr cache (single-study)**
- [x] `EdfToZarr.build()` writes chunked per-channel arrays to `processed/<study>.zarr`
- [x] Store recording metadata (start_dt, duration, fs, units, channel map)
- [x] Smoke-test on synthetic EDF (MemoryStore) + round-trip via loader shim
- [x] Viewer auto-builds cache in background, swaps to `ZarrLoader` post-ingest

**Phase 6 – Tolerable skimming**
- [x] `decimate.minmax(t, x, px)` utility with pytest coverage
- [x] Integrate decimation + pan/zoom UI with debounce
- [x] Max window cap enforced at loader/Zarr loader
- [ ] Perf smoke tests (profiling script ± synthetic 500 Hz data)
- [x] Prefetch ring-buffer prototype for smoother scrubbing
- [x] Overscan renderer precomputes ±2 windows and reuses cached curves (v0.2)

**Phase 7 – Prefetch ring buffer**
- [x] Async read-ahead for prev/next window
- [x] LRU in-RAM tiles (PrefetchService with tile/time budget)
- [x] Prefetch knobs surfaced via CLI/INI/UI

> **Later (Phase 9+)**: HDF5/Zarr multiresolution pyramids, Parquet events, GPU-friendly envelopes.

---

## 6) Local agents (roles & checklists)

### Loader Agent
**Goal:** deterministic slices from EDF without surprises.  
**Checklist:**
- [x] `sec_to_idx` & `idx_to_time` via `Timebase` only
- [x] Clamp `[t0, t1]` to file duration
- [x] Handle different `fs` per channel
- [x] Unit tests: start-of-file, end-of-file, sub-second windows

### Viewer Agent
**Goal:** draw only what eyes can see.  
**Checklist:**
- [x] Choose window `[t0, t1]` → compute `sec_per_px`
- [x] If `n_samples > 4*px`: decimate (min/max per bin)
- [x] One `PlotCurveItem` per visible channel
- [x] Shared X axis; per-lane vertical offsets
- [x] Prefetch knobs exposed in UI (tile_s, max MB)
- [x] Overscan tile cache keeps curves hot within ±2 windows for instant pans
- [x] Hypnogram lane + shaded event regions with hover tooltips and navigator list


### Annotation Agent
**Goal:** fast interval queries.  
**Checklist:**
- [ ] Normalize to schema `(start_s, end_s, label, chan)`
- [ ] Sort by `start_s`
- [ ] `between(t0, t1)` uses binary search + overlap mask
- [ ] Label relabeling mapping (e.g., “Obstructive apnea” → “OA”)

### Prefetch Agent (Phase 7)
**Goal:** hide I/O latency.  
**Checklist:**
- [x] Tile size ≈ 5–10 s per channel
- [x] Background thread prefetches prev/next tiles
- [x] LRU cache with memory budget (e.g., 128 MB)

### Cache Architect (Phase 9+)
**Goal:** O(pixels) rendering, any window size.  
**Checklist:**
- [ ] Build L0 raw, L1/L2… envelopes (min/max)
- [ ] Store in chunked HDF5/Zarr with blosc/zstd
- [ ] Parquet events with `start_s` sort key
- [ ] Integrity: store original EDF hash

---

## 7) CLI & scripts (living)
- `scripts/pack_study.py <edf> [--csv …]` → writes `processed/<study>/…`
- `scripts/validate_alignment.py <edf> <csv>` → drift & offset report
- `scripts/make_fixture.py` → generate 60 s synthetic EDF for tests
- `scripts/edf_to_zarr.py <edf> [--out dir]` → one-shot ingest using `EdfToZarr`
- `scripts/profile_draw.py <path>` → render timings (10/60/120 s)
- `scripts/run_perf_smoke.sh` → CI helper (requires PERF_PROFILE_PATH env)
- TODO: Perf smoke in CI runs this script on a small fixture (capture baseline).

---

## 8) YAML snippet for CSV annotations
```yaml
# apples_events.yml
start: "StartSec"
duration: "DurSec"     # or use 'end': "EndSec"
unit: "s"              # 's' or 'ms'
label: "Event"
chan: "Signal"         # optional
offset_s: 0.0
```
---

## 9) Testing strategy (minimal but real)
- test_timebase.py *(added)*
- sec↔idx roundtrips (various fs)
- clamp_window edges
- test_loader.py *(using fake pyEDFlib reader added)*
- short EDF fixture: exact sample counts at edges *(covered via synthetic arrays for now)*
- annotations: NaN durations → 0 span *(pending)*
- Golden images (optional): save PNG of 10 s window and diff on CI.
- test_overscan.py *(new)* verifies slice+decimate helper for overscan tiles
- test_zarr_cache.py *(planned)*
  - MemoryStore ingest → asserts attrs, chunk sizes, dtype
  - Loader shim reads from Zarr with identical API as `EdfLoader`

> **Test runner:** `uv pip install pytest` then `uv run python -m pytest` (CI todo).

---

## 10) Troubleshooting quickies
- Axis shows years → you plotted datetimes instead of seconds. Use TimeAxis / Timebase.
- Legend explosion → don’t autogenerate legends; one curve per lane, no legend by default.
- Jank on pan → add 60 ms debounce; cap window; decimate-to-pixels.
- Misaligned CSV → check offset_s; verify EDF start datetime vs CSV epoch.

---

## 11) Decision log (append entries)
- 2025-09-23: UI uses seconds-from-start, absolute datetimes only for tooltips.
- 2025-09-23: Max default window = 60 s; hard cap = 120 s.
- 2025-09-24: Multi-channel stacked viewer replaces dropdown; labels pinned left, shared absolute axis.
- 2025-09-24: Dependency installs handled via `uv`; pytest suite (loader, timebase) is baseline gate.
- 2025-09-24: Zarr cache ingestion planned; target `EdfToZarr` writer, loader shim, MemoryStore unit tests.
- 2025-09-24: Viewer swaps to Zarr cache post-build; source badge shows active backend.
- 2025-09-24: Skimming UI (pan/zoom, decimation) live; PrefetchService warms ±1 window; full-night zoom enabled via Zarr; prefetch knobs adjustable in UI/INI.
- 2025-09-24: App auto-ingests EDF → Zarr with progress UI; Zarr parity verified post-write.
- 2025-09-25: v0.2 adds overscan tile renderer (±2 windows) with reusable curves + `core.overscan` helper.
- 2025-09-25: Phase 4 (CSV annotations) is next; formal plan captured below.
- 2025-09-26: Auto-detects APPLES CSV/stage files; events plot as markers, stage label updates during pan.

---

## 12) Prompts / task tickets (copy-ready)

### Ticket: Implement Phase 1 loader

Build EdfLoader with read(i, t0, t1) returning (t_s, x), and read_annotations() for EDF+. Use Timebase.sec_to_idx & time_vector. Add unit tests for edges.

### Ticket: CSV annotations (Phase 4)

1. **Auto-detect siblings** – when opening `<stem>.edf`, look for `<stem>.csv` (event log) and `<stem>STAGE.csv` (sleep stages). Allow manual import via menu if files live elsewhere; remember last-used directory.
2. **Event parser** – implement `core.annotations.from_csv(path, mapping, *, default_chan=None, tz_hint=None)` returning immutable records `(start_s, end_s, label, chan, attrs)` and metadata (source path, tz, validation flag). Handle string durations with units (`"0.0 cmH2O"`, `"5 (43)"`), blank `Duration` values, and optional channel/body-position columns. Normalize times against EDF start time using mapping fields (`start`, `duration` *or* `end`).
3. **Stage parser** – detect one-column numerical staging files (values 11/12/13/14). Convert to `(start_s, end_s, label)` spans assuming 30 s epochs (confirm APPLES docs). Provide configurable mapping (e.g., 11=W, 12=N1, 13=N2, 14=REM/N3) so sites can override.
4. **Index** – add `AnnotationIndex` with `.between(t0, t1)` using bisect + vectorized overlap mask; support channel filtering, stage+event separation, and metadata-driven styling. Cover with pytest fixtures for overlapping events, multichannel labels, and stage files.
5. **UI integration** –
   - Show auto-detected CSVs in a modal before rendering; allow opt-in/opt-out per file.
   - Add "Import CSV annotations…" action to re-run parser manually; surface load status + error messages in control panel.
   - Render markers/spans atop overscan tiles without breaking pan performance (reuse existing overscan cache; overlay drawn from index results).
   - Add channel/stage filters, legend snippets, and badge with total counts.
6. **Validation & tooling** – update `scripts/validate_alignment.py` to reuse parser; add drift report and stage histogram. Log warnings when validation column isn't `*`. Provide docs snippet (`agents.md` §8) describing required columns.

### Ticket: Decimation util

Given (t, x) and plot_width_px, return min/max per pixel bin. Target ≤ 2 samples/px. Benchmark on 120 s with 500 Hz.

### Ticket: Prefetch ring buffer

Async worker preloads +/- one window of tiles into an LRU cache; UI reads from cache if available.

---

## 13) Glossary (micro)
	•	Window: [t0, t1] seconds shown in the viewport.
	•	LOD: level of detail; choose coarser envelopes when zoomed out.
	•	Tile: contiguous chunk of per-channel samples (e.g., 5 s).

---

## 14) Style

Direct, deterministic, boring-on-purpose. Fancy comes later.
