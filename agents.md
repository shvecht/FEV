# edfview • agents.md
A lightweight playbook for Future-You (and any local “agent”) to work consistently on the EDF viewer stack.

---

## 0) Purpose
Keep tasks small, interfaces clean, and decisions visible. This doc is the “how we work” guide for building a fast, reliable EDF/annotation viewer—step by step.

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
ui/
main_window.py
time_axis.py
tests/
test_timebase.py
test_loader.py

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

**Phase 6 – Tolerable skimming**
- [ ] On-the-fly decimation
- [ ] Max window cap
- [ ] Debounced redraw

**Phase 7 – Prefetch ring buffer**
- [ ] Async read-ahead for prev/next window
- [ ] LRU in-RAM tiles

> **Later (Phase 9+)**: HDF5/Zarr multiresolution pyramids, Parquet events, GPU-friendly envelopes.

---

## 6) Local agents (roles & checklists)

### Loader Agent
**Goal:** deterministic slices from EDF without surprises.  
**Checklist:**
- [ ] `sec_to_idx` & `idx_to_time` via `Timebase` only
- [ ] Clamp `[t0, t1]` to file duration
- [ ] Handle different `fs` per channel
- [ ] Unit tests: start-of-file, end-of-file, sub-second windows

### Viewer Agent
**Goal:** draw only what eyes can see.  
**Checklist:**
- [ ] Choose window `[t0, t1]` → compute `sec_per_px`
- [ ] If `n_samples > 4*px`: decimate (min/max per bin)
- [ ] One `PlotCurveItem` per visible channel
- [ ] Shared X axis; per-lane vertical offsets

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
- [ ] Tile size ≈ 5–10 s per channel
- [ ] Background thread prefetches prev/next tiles
- [ ] LRU cache with memory budget (e.g., 128 MB)

### Cache Architect (Phase 9+)
**Goal:** O(pixels) rendering, any window size.  
**Checklist:**
- [ ] Build L0 raw, L1/L2… envelopes (min/max)
- [ ] Store in chunked HDF5/Zarr with blosc/zstd
- [ ] Parquet events with `start_s` sort key
- [ ] Integrity: store original EDF hash

---

## 7) CLI & scripts (future slots)
- `scripts/pack_study.py <edf> [--csv …]` → writes `processed_data/<study>/…`
- `scripts/validate_alignment.py <edf> <csv>` → drift & offset report
- `scripts/make_fixture.py` → generate 60 s synthetic EDF for tests

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

---

## 12) Prompts / task tickets (copy-ready)

### Ticket: Implement Phase 1 loader

Build EdfLoader with read(i, t0, t1) returning (t_s, x), and read_annotations() for EDF+. Use Timebase.sec_to_idx & time_vector. Add unit tests for edges.

### Ticket: Add CSV annotations

Implement annotations.from_csv(path, mapping) → normalized schema. Add between(t0, t1). Provide YAML mapping for APPLES.

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
