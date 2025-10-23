"""Helpers for overscan window management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from core.decimate import min_max_bins

__all__ = [
    "SignalChunk",
    "slice_and_decimate",
    "chunk_from_arrays",
    "chunk_from_envelope",
    "select_lod_duration",
]


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
