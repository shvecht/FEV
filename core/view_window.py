"""Helpers for managing view windows (start/duration) with clamping."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowLimits:
    duration_min: float = 0.1
    duration_max: float = 120.0


def clamp_window(start: float, duration: float, *, total: float, limits: WindowLimits) -> tuple[float, float]:
    duration_clamped = max(limits.duration_min, min(limits.duration_max, duration))
    start_clamped = max(0.0, min(start, max(0.0, total - duration_clamped)))
    # adjust duration if recording shorter than min window
    if total < limits.duration_min:
        duration_clamped = total
        start_clamped = 0.0
    elif start_clamped + duration_clamped > total:
        start_clamped = max(0.0, total - duration_clamped)
    return start_clamped, duration_clamped


def pan_window(start: float, duration: float, delta: float, *, total: float, limits: WindowLimits) -> tuple[float, float]:
    start_new = start + delta
    return clamp_window(start_new, duration, total=total, limits=limits)


def zoom_window(start: float, duration: float, factor: float, *, anchor: float, total: float, limits: WindowLimits) -> tuple[float, float]:
    if factor <= 0:
        raise ValueError("factor must be positive")
    duration_new = duration * factor
    duration_new = max(limits.duration_min, min(limits.duration_max, duration_new))

    # keep anchor position (relative 0..1) within window
    rel = 0.0
    if duration > 0:
        rel = (anchor - start) / duration
    rel = min(1.0, max(0.0, rel))

    start_new = anchor - rel * duration_new
    return clamp_window(start_new, duration_new, total=total, limits=limits)
