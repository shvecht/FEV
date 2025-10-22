# core/timebase.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple
import numpy as np

@dataclass(frozen=True)
class Timebase:
    """
    Canonical time keeper for one EDF recording.
    - t=0.0 is the file's recording start.
    - Supports sec <-> sample index for any channel fs.
    - Supports absolute datetime mapping (optional tz-aware).
    """
    start_dt: datetime     # naive or tz-aware; comes from EDF header
    duration_s: float

    def to_datetime(self, t_s: float | np.ndarray) -> datetime | np.ndarray:
        """
        Convert seconds-from-start to absolute datetime(s).
        """
        if isinstance(t_s, np.ndarray):
            # Return ndarray[datetime64[ns]] for vector ops
            base_ns = np.datetime64(self.start_dt.isoformat())
            dt = base_ns + (t_s * 1e9).astype('timedelta64[ns]')
            return dt
        return self.start_dt + timedelta(seconds=float(t_s))

    def to_seconds(self, when: datetime) -> float:
        """
        Convert absolute datetime -> seconds-from-start.
        """
        delta = when - self.start_dt
        return delta.total_seconds()

    # ----- sample index conversions (per-channel fs) -----

    @staticmethod
    def sec_to_idx(t0_s: float, t1_s: float, fs: float) -> Tuple[int, int]:
        """
        Convert [t0, t1) seconds to integer [start_idx, n_samples] at sampling rate fs.
        Clamps negative t0 to 0; returns n=0 if window is empty.
        """
        if fs <= 0:
            raise ValueError("fs must be positive")

        start = max(0.0, t0_s)
        end = max(start, t1_s)

        s0 = int(np.floor(start * fs))
        n = int(max(0.0, np.ceil((end - start) * fs)))
        return s0, n

    @staticmethod
    def idx_to_time(idx: int | np.ndarray, fs: float) -> float | np.ndarray:
        """
        Convert sample index/indices to seconds-from-start.
        """
        if fs <= 0:
            raise ValueError("fs must be positive")
        return np.asarray(idx, dtype=float) / fs if isinstance(idx, np.ndarray) else float(idx) / fs

    @staticmethod
    def time_vector(s0: int, n: int, fs: float) -> np.ndarray:
        """
        Build a time vector in seconds for a window starting at sample s0 of length n at fs.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)
        return (s0 + np.arange(n, dtype=np.int64)) / float(fs)

    # ----- helpers -----

    def clamp_window(self, t0_s: float, t1_s: float) -> Tuple[float, float]:
        """
        Clamp [t0, t1] to recording bounds [0, duration_s].
        Ensures t1 >= t0 after clamping.
        """
        t0 = max(0.0, min(t0_s, self.duration_s))
        t1 = max(t0, min(t1_s, self.duration_s))
        return t0, t1

    def make_axis_labels(self, ticks_s: Iterable[float]) -> list[str]:
        """
        Format tick labels as HH:MM:SS (relative).
        """
        out = []
        for v in ticks_s:
            sec = int(round(v))
            h, r = divmod(sec, 3600); m, s = divmod(r, 60)
            out.append(f"{h:02d}:{m:02d}:{s:02d}")
        return out