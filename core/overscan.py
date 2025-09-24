"""Helpers for overscan window management."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from core.decimate import min_max_bins

__all__ = ["slice_and_decimate"]


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
