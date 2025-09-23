"""Utilities for level-of-detail decimation of dense time-series."""
from __future__ import annotations

import numpy as np

__all__ = ["min_max_bins"]


def min_max_bins(t: np.ndarray, x: np.ndarray, pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return min/max envelope sampled to ~2 points per pixel.

    Parameters
    ----------
    t : np.ndarray
        Monotonic time vector (float64) for each sample.
    x : np.ndarray
        Signal values (float32/float64).
    pixels : int
        Available horizontal pixels for drawing.

    Returns
    -------
    t_out, x_out : np.ndarray
        Envelope series. Length <= 2 * pixels. If the input already meets the
        budget, the original arrays are returned (views).
    """
    if pixels <= 0:
        raise ValueError("pixels must be positive")
    if t.size != x.size:
        raise ValueError("t and x must have the same length")
    n = t.size
    if n == 0 or n <= pixels * 2:
        return t, x

    # Ensure numpy arrays for fancy indexing
    t = np.asarray(t)
    x = np.asarray(x)

    # Determine bin edges in sample space
    edges = np.linspace(0, n, num=pixels + 1, dtype=np.int64)
    # Ensure edges are strictly increasing (monotonic), adjusting duplicates
    edges = np.minimum(edges, n)
    edges[0] = 0
    edges[-1] = n
    # Fix any collapse by forcing strictly increasing indices
    np.maximum.accumulate(edges, out=edges)
    edges = np.unique(edges)
    if edges.size <= 1:
        return t[:1], x[:1]

    out_t = []
    out_x = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = x[start:end]
        if segment.size == 0:
            continue
        # Get indices of min and max within the segment
        local_min_idx = np.argmin(segment)
        local_max_idx = np.argmax(segment)
        idxs = np.array([local_min_idx, local_max_idx], dtype=np.int64)
        # Sort so time ordering preserved
        idxs = np.unique(np.sort(idxs))
        # Map back to global indices
        global_idxs = start + idxs
        out_t.append(t[global_idxs])
        out_x.append(x[global_idxs])

    if not out_t:
        return t[:1], x[:1]

    t_out = np.concatenate(out_t)
    x_out = np.concatenate(out_x)

    # Guarantee monotonic time ordering (min/max order may produce duplicates)
    order = np.argsort(t_out, kind="mergesort")
    t_out = t_out[order]
    x_out = x_out[order]

    # Enforce budget limit by trimming if necessary
    if t_out.size > pixels * 2:
        step = max(1, t_out.size // (pixels * 2))
        t_out = t_out[::step]
        x_out = x_out[::step]

    return t_out, x_out
