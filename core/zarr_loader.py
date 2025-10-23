from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import math

import numpy as np
import zarr

from core.timebase import Timebase
from core import annotations as annotations_core


@dataclass(frozen=True)
class ChannelMeta:
    name: str
    fs: float
    n_samples: int
    unit: str


class ZarrLoader:
    def __init__(
        self,
        store_path: str | Path,
        *,
        max_window_s: float | None = None,
        annotations: annotations_core.Annotations | None = None,
    ):
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
        self._lod_levels: Dict[int, Dict[float, zarr.Array]] = {}

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

        if "lod" in self._root:
            self._load_lod_groups(self._root["lod"])

        self.info = self._channel_meta
        self.timebase = Timebase(
            start_dt=self.start_dt or datetime.fromtimestamp(0),
            duration_s=self.duration_s,
        )
        self._annotations = annotations

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

    def annotations(self) -> annotations_core.Annotations:
        if self._annotations is None:
            return annotations_core.Annotations.empty()
        return self._annotations

    def set_annotations(self, annotations: annotations_core.Annotations | None):
        self._annotations = annotations

    # ------------------------------------------------------------------

    def lod_levels(self, idx: int) -> List[float]:
        levels = self._lod_levels.get(idx)
        if not levels:
            return []
        return sorted(levels.keys())

    def read_lod(self, idx: int, duration_s: float) -> Tuple[np.ndarray, np.ndarray, float]:
        levels = self._lod_levels.get(idx)
        if not levels:
            raise KeyError(f"no LOD levels for channel {idx}")
        key = self._lod_key(duration_s)
        if key not in levels:
            available = ", ".join(str(v) for v in sorted(levels.keys()))
            raise KeyError(f"LOD duration {duration_s} not found for channel {idx}; available: {available}")
        dataset = levels[key]
        data = dataset[:]
        mins = data[:, 0]
        maxs = data[:, 1]
        duration = float(dataset.attrs.get("bin_duration_s", duration_s))
        return mins, maxs, duration

    def read_lod_window(
        self,
        idx: int,
        t0: float,
        t1: float,
        duration_s: float,
    ) -> Tuple[np.ndarray, float, int]:
        levels = self._lod_levels.get(idx)
        if not levels:
            raise KeyError(f"no LOD levels for channel {idx}")
        key = self._lod_key(duration_s)
        if key not in levels:
            available = ", ".join(str(v) for v in sorted(levels.keys()))
            raise KeyError(f"LOD duration {duration_s} not found for channel {idx}; available: {available}")
        dataset = levels[key]
        bin_duration = float(dataset.attrs.get("bin_duration_s", duration_s))
        if bin_duration <= 0:
            raise ValueError("LOD bin duration must be positive")
        total_bins = dataset.shape[0]
        start_bin = max(0, int(math.floor(t0 / bin_duration)))
        end_bin = int(math.ceil(t1 / bin_duration))
        if end_bin <= start_bin:
            return dataset[:0], bin_duration, start_bin
        start_bin = min(start_bin, total_bins)
        end_bin = min(max(start_bin, end_bin), total_bins)
        return dataset[start_bin:end_bin], bin_duration, start_bin

    def close(self):
        pass

    # ------------------------------------------------------------------

    def _load_lod_groups(self, lod_root: zarr.Group) -> None:
        for name in lod_root.group_keys():
            try:
                idx = int(name)
            except ValueError:
                continue
            ch_group = lod_root[name]
            durations: Dict[float, zarr.Array] = {}
            for level_name in ch_group.array_keys():
                arr = ch_group[level_name]
                duration = float(arr.attrs.get("bin_duration_s", 0.0))
                key = self._lod_key(duration)
                durations[key] = arr
            if durations:
                self._lod_levels[idx] = durations

    @staticmethod
    def _lod_key(duration: float) -> float:
        return round(float(duration), 9)


__all__ = ["ZarrLoader", "ChannelMeta"]
