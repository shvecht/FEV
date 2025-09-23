# core/edf_loader.py
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pyedflib

@dataclass
class ChannelInfo:
    name: str
    fs: float
    n_samples: int
    unit: str

class EdfLoader:
    def __init__(self, path: str):
        self._r = pyedflib.EdfReader(path)
        self.duration_s = float(self._r.file_duration)
        self.start_dt: datetime = self._r.getStartdatetime()
        self.n_channels = self._r.signals_in_file
        self.channels = self._r.getSignalLabels()
        self._fs = [self._r.getSampleFrequency(i) for i in range(self.n_channels)]
        self._ns = [self._r.getNSamples()[i] for i in range(self.n_channels)]
        self.info = [ChannelInfo(self.channels[i], self._fs[i], self._ns[i],
                                 self._r.getPhysicalDimension(i)) for i in range(self.n_channels)]

    def fs(self, i: int) -> float:
        return self._fs[i]

    def read(self, i: int, t0: float, t1: float):
        fs = self._fs[i]
        s0 = max(0, int(np.floor(t0 * fs)))
        n  = max(0, int(np.ceil((t1 - t0) * fs)))
        x = self._r.readSignal(i, start=s0, n=n).astype(np.float32)
        t = (s0 + np.arange(x.size)) / fs
        return t, x

    def read_annotations(self):
        # Returns lists: onset(s), duration(s), text
        try:
            return self._r.readAnnotations()
        except Exception:
            return ([], [], [])

    def close(self):
        self._r.close()