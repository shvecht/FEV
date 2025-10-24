"""Shared tick formatter for time-based axes."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Iterable, Protocol


class _SupportsToDatetime(Protocol):
    def to_datetime(self, value: float) -> datetime:
        ...


class TimeTickFormatter:
    """Format seconds into clock-style labels with optional absolute mode."""

    __slots__ = ("_timebase", "_mode")

    def __init__(self, *, timebase: _SupportsToDatetime | None = None, mode: str = "relative") -> None:
        self._timebase = timebase
        self._mode = "absolute" if str(mode).lower() == "absolute" else "relative"

    @property
    def mode(self) -> str:
        return self._mode

    def set_timebase(self, timebase: _SupportsToDatetime | None) -> None:
        self._timebase = timebase

    def set_mode(self, mode: str) -> None:
        self._mode = "absolute" if str(mode).lower() == "absolute" else "relative"

    def format_ticks(self, values: Iterable[float]) -> list[str]:
        return [self._format_single(v) for v in values]

    def _format_single(self, value: float) -> str:
        if value is None:
            return ""
        try:
            numeric = float(value)
        except Exception:
            return ""
        if not isfinite(numeric):
            return ""

        if self._mode == "absolute" and self._timebase is not None:
            try:
                dt = self._timebase.to_datetime(numeric)
            except Exception:
                dt = None
            if isinstance(dt, datetime):
                return dt.strftime("%H:%M:%S")
            if dt is not None:
                try:
                    return datetime.fromisoformat(str(dt)).strftime("%H:%M:%S")
                except Exception:
                    pass

        seconds = int(round(numeric))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def __call__(self, values):  # pragma: no cover - exercised via AxisWidget
        return self.format_ticks(values)

