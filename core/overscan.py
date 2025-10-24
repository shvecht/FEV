"""Helpers for overscan window management."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import zarr

from core.decimate import min_max_bins

__all__ = [
    "SignalChunk",
    "chunk_from_arrays",
    "chunk_from_envelope",
    "slice_and_decimate",
    "select_lod_duration",
    "envelope_to_series",
    "choose_lod_duration",
    "build_envelopes",
    "OverscanRenderer",
]

DEFAULT_PROCESSED_DIR = Path("processed")


@dataclass(frozen=True)
class SignalChunk:
    """Lightweight container for time/value pairs used by overscan pipelines."""

    t: np.ndarray
    x: np.ndarray
    lod_duration_s: float | None = None
    source_start: float | None = None
    source_end: float | None = None

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if self.t.shape != self.x.shape:
            raise ValueError("t and x must have matching shapes")

    def __iter__(self):
        yield self.t
        yield self.x

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.t, self.x

    def slice(self, start: float, end: float) -> "SignalChunk":
        if self.t.size == 0:
            return self
        idx_start = int(np.searchsorted(self.t, start, side="left"))
        idx_end = int(np.searchsorted(self.t, end, side="right"))
        if idx_end <= idx_start:
            empty = self.t[:0]
            return SignalChunk(
                empty,
                self.x[:0],
                lod_duration_s=self.lod_duration_s,
                source_start=self.source_start,
                source_end=self.source_end,
            )
        t_slice = self.t[idx_start:idx_end]
        x_slice = self.x[idx_start:idx_end]
        new_start = float(t_slice[0]) if t_slice.size else start
        new_end = float(t_slice[-1]) if t_slice.size else end
        return SignalChunk(
            t_slice,
            x_slice,
            lod_duration_s=self.lod_duration_s,
            source_start=new_start,
            source_end=new_end,
        )


def chunk_from_arrays(
    t: np.ndarray,
    x: np.ndarray,
    *,
    lod_duration_s: float | None = None,
    source_start: float | None = None,
    source_end: float | None = None,
) -> SignalChunk:
    """Create a :class:`SignalChunk` from raw arrays."""

    return SignalChunk(
        np.asarray(t),
        np.asarray(x),
        lod_duration_s=lod_duration_s,
        source_start=source_start,
        source_end=source_end,
    )


def chunk_from_envelope(
    bin_start: float,
    bin_duration: float,
    mins: Sequence[float],
    maxs: Sequence[float],
) -> SignalChunk:
    """Materialise a min/max envelope into time/value series for drawing."""

    mins_arr = np.asarray(mins, dtype=np.float32)
    maxs_arr = np.asarray(maxs, dtype=np.float32)
    if mins_arr.size != maxs_arr.size:
        raise ValueError("mins and maxs must be the same length")
    if mins_arr.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return SignalChunk(empty, empty.astype(np.float32), lod_duration_s=float(bin_duration))

    valid = ~(np.isnan(mins_arr) & np.isnan(maxs_arr))
    if not np.all(valid):
        mins_arr = mins_arr[valid]
        maxs_arr = maxs_arr[valid]

    n_bins = mins_arr.size
    if n_bins == 0:
        empty = np.zeros(0, dtype=np.float64)
        return SignalChunk(empty, empty.astype(np.float32), lod_duration_s=float(bin_duration))

    edges = bin_start + np.arange(n_bins + 1, dtype=np.float64) * float(bin_duration)
    t = np.empty(n_bins * 2, dtype=np.float64)
    x = np.empty(n_bins * 2, dtype=np.float32)
    t[0::2] = edges[:-1]
    t[1::2] = edges[1:]
    x[0::2] = mins_arr
    x[1::2] = maxs_arr
    return SignalChunk(
        t,
        x,
        lod_duration_s=float(bin_duration),
        source_start=float(edges[0]),
        source_end=float(edges[-1]),
    )


def slice_and_decimate(
    t: np.ndarray | SignalChunk,
    x: np.ndarray | None,
    start: float,
    end: float,
    pixels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a view of ``t``/``x`` within ``[start, end]`` decimated to fit pixels."""

    if isinstance(t, SignalChunk):
        chunk = t.slice(start, end)
        t_arr, x_arr = chunk.as_tuple()
        if pixels and pixels > 0 and x_arr.size > pixels * 2:
            return min_max_bins(t_arr, x_arr, pixels)
        return t_arr, x_arr

    if t.size == 0:
        return t[:0], (x[:0] if x is not None else np.zeros(0, dtype=np.float32))
    if x is None:
        raise ValueError("x must be provided when passing raw arrays")
    idx_start = int(np.searchsorted(t, start, side="left"))
    idx_end = int(np.searchsorted(t, end, side="right"))
    if idx_end <= idx_start:
        return t[:0], x[:0]
    t_slice = t[idx_start:idx_end]
    x_slice = x[idx_start:idx_end]
    if pixels and pixels > 0 and x_slice.size > pixels * 2:
        t_slice, x_slice = min_max_bins(t_slice, x_slice, pixels)
    return t_slice, x_slice


def select_lod_duration(
    view_duration: float,
    durations: Sequence[float] | Iterable[float],
    min_bin_multiple: float,
    *,
    min_view_duration: float | None = None,
) -> float | None:
    """Pick the coarsest LOD duration suitable for ``view_duration``.

    Parameters
    ----------
    view_duration:
        Length of the visible window in seconds.
    durations:
        Available LOD bin durations in seconds.
    min_bin_multiple:
        Require at least this many bins inside the window before selecting a
        particular duration. A value of 2.0 means the window must span at least
        two bins before that LOD level is considered.
    min_view_duration:
        Optional floor (in seconds) below which no LOD duration will be
        selected regardless of ``min_bin_multiple``.
    """

    if view_duration <= 0:
        return None
    if min_view_duration is not None:
        try:
            min_view = float(min_view_duration)
        except (TypeError, ValueError):
            min_view = 0.0
        if min_view > 0.0 and view_duration < min_view:
            return None

    try:
        iterator = iter(durations)
    except TypeError:
        return None

    min_multiple = max(1.0, float(min_bin_multiple))
    candidates: list[float] = []
    for value in iterator:
        duration = float(value)
        if duration <= 0:
            continue
        if view_duration >= min_multiple * duration:
            candidates.append(duration)
    if not candidates:
        return None
    return max(candidates)


def envelope_to_series(
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    bin_duration: float,
    start_bin: int,
    window_start: float,
    window_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert min/max bins into a draw-ready time/value series."""

    if mins.size == 0 or maxs.size == 0:
        return mins[:0], maxs[:0]
    if mins.size != maxs.size:
        raise ValueError("mins and maxs must have identical length")
    if bin_duration <= 0:
        raise ValueError("bin_duration must be positive")
    if window_end <= window_start:
        return mins[:0], maxs[:0]

    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)
    valid_mask = np.isfinite(mins) & np.isfinite(maxs)
    if not np.any(valid_mask):
        return mins[:0], maxs[:0]

    indices = np.nonzero(valid_mask)[0]
    if indices.size == 0:
        return mins[:0], maxs[:0]

    mins = mins[indices]
    maxs = maxs[indices]
    bin_indices = start_bin + indices.astype(np.int64, copy=False)

    bin_starts = bin_indices.astype(np.float64) * float(bin_duration)
    bin_ends = bin_starts + float(bin_duration)

    np.maximum(bin_starts, window_start, out=bin_starts)
    np.minimum(bin_ends, window_end, out=bin_ends)

    span_mask = bin_ends > bin_starts
    if not np.any(span_mask):
        return mins[:0], maxs[:0]

    bin_starts = bin_starts[span_mask]
    bin_ends = bin_ends[span_mask]
    mins = mins[span_mask]
    maxs = maxs[span_mask]

    count = bin_starts.size
    t_out = np.empty(count * 2, dtype=np.float64)
    x_out = np.empty(count * 2, dtype=np.float32)

    for idx, (t_start, t_end, mn, mx) in enumerate(zip(bin_starts, bin_ends, mins, maxs)):
        base = idx * 2
        t_out[base] = t_start
        t_out[base + 1] = t_end
        if mn <= mx:
            x_out[base] = mn
            x_out[base + 1] = mx
        else:
            # Handle inverted calibration slopes defensively.
            x_out[base] = mx
            x_out[base + 1] = mn

    return t_out, x_out


def choose_lod_duration(
    view_duration: float,
    available: Iterable[float],
    *,
    ratio: float,
) -> float | None:
    """Pick the coarsest LOD duration allowed by ``ratio`` and ``view_duration``.

    ``ratio`` represents the minimum number of bins the visible window should
    cover before we drop to an envelope. A ratio <= 0 disables LOD selection.
    """

    if ratio <= 0 or not np.isfinite(ratio):
        return None

    best: float | None = None
    for duration in sorted({float(d) for d in available}):
        if duration <= 0:
            continue
        if view_duration >= ratio * duration:
            if best is None or duration > best:
                best = duration
    return best


def build_envelopes(
    loader,
    levels: Sequence[float] | Iterable[float],
    *,
    processed_dir: str | Path | None = None,
) -> Path:
    """Pre-compute per-channel envelopes and persist them to disk.

    Parameters
    ----------
    loader:
        Loader exposing EDF/Zarr-style methods ``fs(i)``, ``read(i, t0, t1)`` and
        channel metadata via ``info``.
    levels:
        Iterable of target bin durations (seconds).
    processed_dir:
        Optional override for the processed directory root. Defaults to
        ``processed`` beneath the current working directory.

    Returns
    -------
    Path
        Path to the backing Zarr store where envelopes were written.
    """

    durations = _normalise_lod_levels(levels)
    processed_root = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
    processed_root.mkdir(parents=True, exist_ok=True)

    study = _resolve_study_name(loader)
    store_path = processed_root / f"{study}.zarr"
    root = zarr.open_group(str(store_path), mode="a")

    if not durations:
        return store_path

    info_seq = getattr(loader, "info", None)
    if info_seq is None:
        raise AttributeError("loader must expose channel metadata via `info`")

    max_window = float(getattr(loader, "max_window_s", 60.0))

    for idx, meta in enumerate(info_seq):
        fs = float(getattr(meta, "fs", getattr(loader, "fs", lambda _i: 0.0)(idx)))
        if not math.isfinite(fs) or fs <= 0:
            continue
        total_samples = int(getattr(meta, "n_samples", round(fs * float(getattr(loader, "duration_s", 0.0)))))
        if total_samples <= 0:
            continue

        channel_group = root.require_group(f"LOD{idx}")
        for duration in durations:
            bin_size = max(1, int(round(duration * fs)))
            actual_duration = bin_size / fs
            bins = int(math.ceil(total_samples / bin_size))
            if bins <= 0:
                continue

            mins = np.full(bins, np.nan, dtype=np.float32)
            maxs = np.full(bins, np.nan, dtype=np.float32)

            loader_duration = float(getattr(loader, "duration_s", total_samples / fs))
            if not math.isfinite(loader_duration) or loader_duration <= 0:
                loader_duration = total_samples / fs
            window_hint = max_window if math.isfinite(max_window) and max_window > 0 else loader_duration
            chunk_window = min(loader_duration, window_hint)
            if chunk_window <= 0:
                chunk_window = loader_duration
            chunk_samples = max(bin_size, int(round(chunk_window * fs)))

            for start_idx in range(0, total_samples, chunk_samples):
                end_idx = min(total_samples, start_idx + chunk_samples)
                if end_idx <= start_idx:
                    continue
                start_t = start_idx / fs
                end_t = end_idx / fs
                chunk = loader.read(idx, start_t, end_t)
                t_arr, x_arr = chunk.as_tuple()
                if x_arr.size == 0:
                    continue
                start_time = float(t_arr[0]) if t_arr.size else start_t
                sample_offset = int(round(start_time * fs))
                _accumulate_envelope(mins, maxs, x_arr, sample_offset, bin_size)

            if not (np.isfinite(mins).any() or np.isfinite(maxs).any()):
                continue

            dataset_name = _lod_dataset_name(actual_duration)
            if dataset_name in channel_group:
                del channel_group[dataset_name]
            chunk_len = max(1, min(bins, 1024))
            dataset = channel_group.create_dataset(
                dataset_name,
                shape=(bins, 2),
                chunks=(chunk_len, 2),
                dtype="float32",
                overwrite=True,
            )
            dataset[:, 0] = mins.astype(np.float32, copy=False)
            dataset[:, 1] = maxs.astype(np.float32, copy=False)
            dataset.attrs["bin_duration_s"] = float(actual_duration)
            dataset.attrs["bin_size"] = int(bin_size)
            dataset.attrs["columns"] = ["min", "max"]
            dataset.attrs["source_fs"] = float(fs)

    return store_path


class OverscanRenderer:
    """Render helper selecting between raw reads and pre-built envelopes."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def choose_envelope_duration(
        self,
        channel: int,
        window_duration: float,
        plot_width_px: int,
    ) -> float | None:
        """Return the most suitable envelope duration for the current viewport."""

        if window_duration <= 0 or plot_width_px <= 0:
            return None
        fs_fn = getattr(self._loader, "fs", None)
        if fs_fn is None:
            raise AttributeError("loader must expose an fs(idx) method")
        fs = float(fs_fn(channel))
        if not math.isfinite(fs) or fs <= 0:
            return None

        samples_per_pixel = window_duration * fs / float(plot_width_px)
        if not math.isfinite(samples_per_pixel) or samples_per_pixel <= 0:
            return None

        lod_durations = self._lod_durations_for(channel)
        best_duration: float | None = None
        best_stride = -1.0
        for duration in lod_durations:
            bin_size = max(1, int(round(duration * fs)))
            if bin_size <= samples_per_pixel + 1e-6:
                stride = bin_size / fs
                if best_duration is None or stride > best_stride:
                    best_duration = float(duration)
                    best_stride = stride
        return best_duration

    def render(
        self,
        channel: int,
        start: float,
        end: float,
        plot_width_px: int,
    ) -> SignalChunk:
        """Return data for the requested window using the best available LOD."""

        window_duration = max(0.0, float(end) - float(start))
        duration = None
        try:
            duration = self.choose_envelope_duration(channel, window_duration, plot_width_px)
        except AttributeError:
            duration = None

        if duration is not None:
            read_lod = getattr(self._loader, "read_lod_window", None)
            if callable(read_lod):
                try:
                    return read_lod(channel, float(start), float(end), duration)
                except KeyError:
                    pass

        read_fn = getattr(self._loader, "read")
        return read_fn(channel, float(start), float(end))

    def _lod_durations_for(self, channel: int) -> tuple[float, ...]:
        lod_durations_fn = getattr(self._loader, "lod_durations", None)
        if callable(lod_durations_fn):
            durations = lod_durations_fn(channel)
            return tuple(float(v) for v in durations)
        lod_levels_fn = getattr(self._loader, "lod_levels", None)
        if callable(lod_levels_fn):
            return tuple(float(v) for v in lod_levels_fn(channel))
        return ()


def _normalise_lod_levels(levels: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    values: list[float] = []
    for value in levels:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration <= 0 or not math.isfinite(duration):
            continue
        values.append(duration)
    deduped = sorted({round(v, 9) for v in values})
    return tuple(deduped)


def _resolve_study_name(loader) -> str:
    path = getattr(loader, "path", None)
    if path:
        stem = Path(path).stem
        if stem:
            return _sanitise_name(stem)
    candidate = getattr(loader, "study_id", None)
    if candidate:
        return _sanitise_name(str(candidate))
    return _sanitise_name(loader.__class__.__name__)


def _sanitise_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", text.strip())
    return cleaned or "study"


def _lod_dataset_name(duration: float) -> str:
    formatted = format(duration, "g").replace("-", "neg").replace(".", "p")
    return f"sec_{formatted}"


def _accumulate_envelope(
    mins: np.ndarray,
    maxs: np.ndarray,
    data: np.ndarray,
    start_sample: int,
    bin_size: int,
) -> None:
    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        return
    start_sample = max(0, int(start_sample))
    indices = start_sample + np.arange(data.size, dtype=np.int64)
    bin_indices = indices // bin_size
    bin_indices = np.minimum(bin_indices, mins.size - 1)
    unique_bins, starts = np.unique(bin_indices, return_index=True)
    stops = np.append(starts[1:], data.size)
    for bin_idx, seg_start, seg_stop in zip(unique_bins, starts, stops):
        segment = data[seg_start:seg_stop]
        if segment.size == 0:
            continue
        local_min = float(np.min(segment))
        local_max = float(np.max(segment))
        current_min = mins[bin_idx]
        if np.isnan(current_min):
            mins[bin_idx] = local_min
        else:
            mins[bin_idx] = min(current_min, local_min)
        current_max = maxs[bin_idx]
        if np.isnan(current_max):
            maxs[bin_idx] = local_max
        else:
            maxs[bin_idx] = max(current_max, local_max)
