"""Helpers for hover overlays that avoid GUI imports for testing."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _sample_at_time(curve: Any, t: float) -> Tuple[float, float] | None:
    """Return the nearest (time, value) pair from a curve for the given time."""

    x_data = getattr(curve, "xData", None)
    y_data = getattr(curve, "yData", None)
    if x_data is None or y_data is None:
        return None
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    if x.size == 0 or y.size == 0:
        return None
    if x.shape[0] != y.shape[0]:
        size = min(x.shape[0], y.shape[0])
        x = x[:size]
        y = y[:size]
    idx = int(np.searchsorted(x, t, side="left"))
    if idx <= 0:
        best = 0
    elif idx >= x.shape[0]:
        best = x.shape[0] - 1
    else:
        left = idx - 1
        right = idx
        if abs(t - x[left]) <= abs(x[right] - t):
            best = left
        else:
            best = right
    return float(x[best]), float(y[best])

