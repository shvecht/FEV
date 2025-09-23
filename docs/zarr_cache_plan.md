# EDF → Zarr Ingestion Plan

Goal: create a repeatable pipeline that converts an EDF recording into a chunked Zarr store for fast random access and skimming, while keeping the public loader API unchanged for the viewer.

## 1. Target layout

```
processed_data/<stem>.zarr/
  attrs:
    edf_path: original path (absolute)
    created_at: ISO timestamp
    start_dt: ISO timestamp (if available)
    duration_s: float
    timezone: optional tz info
  channels/
    <index>/
      attrs:
        name: channel label from EDF
        unit: physical dimension
        fs: sample rate (Hz)
        n_samples: int
      data.zarr: float32 samples chunked in time
  timebase/
    idx_to_seconds: chunked float64 array (optional fast map)
```

- Chunking: 5-second windows per channel (e.g. `chunks=(fs*5,)` capped at ~4096 samples) to balance sequential reads vs random hops.
- Compression: default Zstd level via numcodecs (later phases can tweak); start with no compression for simplicity.
- Metadata: duplicate EDF header info so downstream tools stay self-contained.

## 2. Core components

### `core/zarr_cache.py`

- `EdfToZarr` class
  - `__init__(edf_path, out_path=None, *, chunk_duration=5.0, store_factory=None)`
  - `build()` orchestrates streaming read → write
  - Uses `EdfLoader` for slices to avoid duplicating EDF parsing logic
  - Writes through `zarr.open_group` (auto-creates directory store)
  - Accepts optional `store_factory` (defaults to filesystem) to ease testing.
- Helper: `_write_channel(group, channel_info, data_iter)` where `data_iter` yields `(t, x)` windows.
- Expose `open_zarr(path)` returning a lightweight `ZarrLoader` with `read(i, t0, t1)` API parity.

### Loader shim (`core/zarr_loader.py`)

- Mirrors `EdfLoader` surface: `channels`, `fs(i)`, `read`, `duration_s`, etc.
- Implements `read` by slicing Zarr arrays (leveraging chunk boundaries).
- Accepts optional `Timebase` reuse (constructed from attrs).

## 3. Ingestion flow

1. CLI / viewer detects request → call `EdfToZarr.build()` into `processed_data/<stem>.zarr` when missing or stale (compare mtime/hash later).
2. Viewer chooses between `EdfLoader` and `ZarrLoader` based on availability; for Phase 5 we prefer Zarr if present else fallback to EDF.
3. Maintain backwards compatibility: `app.py` can call helper `load_study(edf_path)` returning loader + metadata.

## 4. Testing strategy

- `tests/test_zarr_cache.py`
  - Use `MemoryStore` to capture writes without touching disk.
  - Patch `EdfLoader` to return deterministic data (reuse FakeEdfReader) and assert arrays/attrs.
  - Validate chunk sizes (`.chunks`) and dtype (float32).
  - Round-trip: instantiate `ZarrLoader` from store and compare `read` outputs to original loader.
- Add fixture for temporary directory store to smoke-test real filesystem creation.

## 5. Docs & tooling

- Update `agents.md` (done) with new phase tasks and CLI entry.
- Add README snippet (future) describing processed-data expectations.
- Extend `scripts` section once CLI wiring lands.

## 6. Next steps (implementation order)

1. Implement `core/zarr_cache.py` scaffolding + writer using `MemoryStore`, with unit tests.
2. Implement `core/zarr_loader.py` and tests verifying parity with EDF loader on synthetic data.
3. Add façade in `core/study.py` (or similar) to pick loader (EDF vs Zarr) and integrate with `app.py`.
4. Optional CLI wrapper under `scripts/edf_to_zarr.py` calling `EdfToZarr`.
5. Document CLI usage + update `agents.md` when features land.

## 7. Open questions

- Hashing/invalidating Zarr cache (Phase 7+).
- Compression choice (introduce numcodecs later).
- Multi-study directory layout if future CLI bundles annotations.

```
Owner: Loader Agent → Zarr Cache Agent handoff.
Reviewers: Viewer Agent (for loader swap), Annotation Agent (future overlay alignment).
```
