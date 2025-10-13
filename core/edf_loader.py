# core/edf_loader.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
import threading
import numpy as np
import pyedflib

from core.int16_cache import Int16Cache, build_int16_cache as _build_int16_cache
from core.timebase import Timebase

@dataclass
class ChannelInfo:
    name: str
    fs: float
    n_samples: int
    unit: str

class EdfLoader:
    def __init__(self, path: str, *, max_window_s: float = 120.0):
        if max_window_s <= 0:
            raise ValueError("max_window_s must be positive")
        self.path = path
        self._r = pyedflib.EdfReader(path)
        self._lock = threading.RLock()
        self.max_window_s = float(max_window_s)
        self.duration_s = float(self._r.file_duration)
        self.start_dt: datetime = self._r.getStartdatetime()
        self.n_channels = self._r.signals_in_file
        self.channels = self._r.getSignalLabels()
        self._fs = [self._r.getSampleFrequency(i) for i in range(self.n_channels)]
        ns = self._r.getNSamples()
        self._ns = [ns[i] for i in range(self.n_channels)]
        self.info = [ChannelInfo(self.channels[i], self._fs[i], self._ns[i],
                                 self._r.getPhysicalDimension(i)) for i in range(self.n_channels)]
        self.timebase = Timebase(self.start_dt, self.duration_s)
        self._cache: Int16Cache | None = None
        self._scratch_float32: Dict[int, np.ndarray] = {}

    def fs(self, i: int) -> float:
        return self._fs[i]

    def has_cache(self) -> bool:
        return self._cache is not None

    def cache_bytes(self) -> int:
        cache = self._cache
        return cache.total_bytes if cache is not None else 0

    def build_int16_cache(self, max_mb: float, prefer_memmap: bool = True) -> bool:
        max_bytes = int(max(0.0, float(max_mb)) * 1024 * 1024)
        with self._lock:
            cache = _build_int16_cache(
                self._r,
                max_bytes=max_bytes,
                prefer_memmap=prefer_memmap,
                memmap_dir=None,
            )
            if cache is None:
                return False
            if cache.total_bytes > max_bytes and max_bytes > 0:
                return False
            self._cache = cache
            self._scratch_float32.clear()
            return True

    def read(self, i: int, t0: float, t1: float):
        fs = self._fs[i]
        with self._lock:
            t0_c, t1_c = self.timebase.clamp_window(t0, t1)
            t1_c = min(t0_c + self.max_window_s, t1_c)
            s0, n = Timebase.sec_to_idx(t0_c, t1_c, fs)
            total = self._ns[i]
            if s0 >= total:
                return np.zeros(0, dtype=float), np.zeros(0, dtype=np.float32)
            n = min(n, max(0, total - s0))
            if n <= 0:
                return np.zeros(0, dtype=float), np.zeros(0, dtype=np.float32)
            cache = self._cache
            if cache is not None:
                scratch = self._scratch_float32.get(i)
                if scratch is None or scratch.shape[0] != n:
                    scratch = np.empty(n, dtype=np.float32)
                    self._scratch_float32[i] = scratch
                cache.fill_float32(i, s0, s0 + n, out=scratch)
                x = scratch.copy()
            else:
                x = self._r.readSignal(i, start=s0, n=n).astype(np.float32)
        t = Timebase.time_vector(s0, x.size, fs)
        return t, x

    def read_annotations(self):
        # Returns lists: onset(s), duration(s), text
        with self._lock:
            try:
                return self._r.readAnnotations()
            except Exception:
                return ([], [], [])

    def close(self):
        with self._lock:
            self._cache = None
            self._scratch_float32.clear()
            self._r.close()
