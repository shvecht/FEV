"""Helpers for overscan window management."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from core.decimate import min_max_bins

__all__ = [
    "slice_and_decimate",
    "select_lod_duration",
    "envelope_to_series",
]


def slice_and_decimate(
    t: np.ndarray,
    x: np.ndarray,
    start: float,
    end: float,
    pixels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a view of ``t``/``x`` within ``[start, end]`` decimated to fit pixels."""
    if t.size == 0:
        return t[:0], x[:0]
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
