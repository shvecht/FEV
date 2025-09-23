from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import zarr

from core.timebase import Timebase


@dataclass(frozen=True)
class ChannelMeta:
    name: str
    fs: float
    n_samples: int
    unit: str


class ZarrLoader:
    def __init__(self, store_path: str | Path, *, max_window_s: float | None = None):
        path = Path(store_path)
        self.path = str(path)
        self._root = zarr.open_group(str(path), mode="r")
        self.channels = list(self._root.attrs.get("channels", []))
        self.duration_s = float(self._root.attrs.get("duration_s", 0.0))
        start = self._root.attrs.get("start_dt")
        self.start_dt: datetime | None = datetime.fromisoformat(start) if start else None
        default_cap = float(self._root.attrs.get("max_window_s", 120.0))
        self.max_window_s = float(default_cap if max_window_s is None else max_window_s)

        self._channel_arrays: List[zarr.Array] = []
        self._channel_meta: List[ChannelMeta] = []

        ch_group = self._root["channels"]
        idx = 0
        while str(idx) in ch_group:
            arr = ch_group[str(idx)]
            name = arr.attrs.get("name", f"ch{idx}")
            unit = arr.attrs.get("unit", "")
            fs = float(arr.attrs.get("fs", 1.0))
            n_samples = int(arr.attrs.get("n_samples", arr.size))
            self._channel_arrays.append(arr)
            self._channel_meta.append(ChannelMeta(name, fs, n_samples, unit))
            idx += 1

        self.n_channels = len(self._channel_arrays)
        if not self.channels:
            self.channels = [meta.name for meta in self._channel_meta]

        self.info = self._channel_meta
        self.timebase = Timebase(
            start_dt=self.start_dt or datetime.fromtimestamp(0),
            duration_s=self.duration_s,
        )

    # ------------------------------------------------------------------

    def fs(self, idx: int) -> float:
        return self._channel_meta[idx].fs

    def read(self, idx: int, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
        meta = self._channel_meta[idx]
        arr = self._channel_arrays[idx]
        t0_c, t1_c = self.timebase.clamp_window(t0, t1)
        t1_c = min(t0_c + self.max_window_s, t1_c)
        s0, n = Timebase.sec_to_idx(t0_c, t1_c, meta.fs)
        total = meta.n_samples
        if s0 >= total or n <= 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=np.float32)
        n = min(n, total - s0)
        data = arr[s0 : s0 + n]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        t = Timebase.time_vector(s0, data.size, meta.fs)
        return t, data

    def close(self):
        pass


__all__ = ["ZarrLoader", "ChannelMeta"]
