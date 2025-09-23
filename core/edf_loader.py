# core/edf_loader.py
from dataclasses import dataclass
from datetime import datetime
import threading
import numpy as np
import pyedflib

from core.timebase import Timebase

@dataclass
class ChannelInfo:
    name: str
    fs: float
    n_samples: int
    unit: str

class EdfLoader:
    def __init__(self, path: str):
        self.path = path
        self._r = pyedflib.EdfReader(path)
        self._lock = threading.RLock()
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

    def fs(self, i: int) -> float:
        return self._fs[i]

    def read(self, i: int, t0: float, t1: float):
        fs = self._fs[i]
        with self._lock:
            t0_c, t1_c = self.timebase.clamp_window(t0, t1)
            s0, n = Timebase.sec_to_idx(t0_c, t1_c, fs)
            total = self._ns[i]
            if s0 >= total:
                return np.zeros(0, dtype=float), np.zeros(0, dtype=np.float32)
            n = min(n, max(0, total - s0))
            if n <= 0:
                return np.zeros(0, dtype=float), np.zeros(0, dtype=np.float32)
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
        self._r.close()
