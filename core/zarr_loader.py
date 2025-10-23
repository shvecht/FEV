from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import zarr

from core.timebase import Timebase
from core import annotations as annotations_core
from core.overscan import SignalChunk, chunk_from_arrays, chunk_from_envelope


@dataclass(frozen=True)
class ChannelMeta:
    name: str
    fs: float
    n_samples: int
    unit: str


@dataclass(frozen=True)
class LodLevel:
    duration_s: float
    dataset: zarr.Array
    bin_size: int


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
        self._lod_levels: Dict[int, Dict[float, LodLevel]] = {}

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

    def read(self, idx: int, t0: float, t1: float) -> SignalChunk:
        meta = self._channel_meta[idx]
        arr = self._channel_arrays[idx]
        t0_c, t1_c = self.timebase.clamp_window(t0, t1)
        t1_c = min(t0_c + self.max_window_s, t1_c)
        s0, n = Timebase.sec_to_idx(t0_c, t1_c, meta.fs)
        total = meta.n_samples
        if s0 >= total or n <= 0:
            empty_t = np.zeros(0, dtype=float)
            empty_x = np.zeros(0, dtype=np.float32)
            return chunk_from_arrays(empty_t, empty_x)
        n = min(n, total - s0)
        data = arr[s0 : s0 + n]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        t = Timebase.time_vector(s0, data.size, meta.fs)
        source_end = float(t[-1]) if t.size else t0_c
        return chunk_from_arrays(
            t,
            data,
            source_start=float(t[0]) if t.size else t0_c,
            source_end=source_end,
        )

    def annotations(self) -> annotations_core.Annotations:
        if self._annotations is None:
            return annotations_core.Annotations.empty()
        return self._annotations

    def set_annotations(self, annotations: annotations_core.Annotations | None):
        self._annotations = annotations

    # ------------------------------------------------------------------

    def lod_levels(self, idx: int) -> List[float]:
        return list(self.lod_durations(idx))

    def lod_durations(self, idx: int) -> Tuple[float, ...]:
        levels = self._lod_levels.get(idx)
        if not levels:
            return ()
        return tuple(sorted(levels.keys()))

    def read_lod(self, idx: int, duration_s: float) -> Tuple[np.ndarray, np.ndarray, float]:
        level = self._get_lod(idx, duration_s)
        data = level.dataset[:]
        mins = data[:, 0]
        maxs = data[:, 1]
        return mins, maxs, level.duration_s

    def read_lod_window(
        self,
        idx: int,
        t0: float,
        t1: float,
        duration_s: float,
    ) -> SignalChunk:
        level = self._get_lod(idx, duration_s)
        meta = self._channel_meta[idx]
        t0_c, t1_c = self.timebase.clamp_window(t0, t1)
        t1_c = min(t0_c + self.max_window_s, t1_c)
        s0, n = Timebase.sec_to_idx(t0_c, t1_c, meta.fs)
        total = meta.n_samples
        if s0 >= total or n <= 0:
            empty = np.zeros(0, dtype=np.float32)
            return chunk_from_arrays(np.zeros(0, dtype=np.float64), empty)
        s1 = min(total, s0 + n)
        bin_size = level.bin_size
        bin_start = max(0, s0 // bin_size)
        bin_stop = min(level.dataset.shape[0], int(np.ceil(s1 / bin_size)))
        if bin_stop <= bin_start:
            empty = np.zeros(0, dtype=np.float32)
            return chunk_from_arrays(np.zeros(0, dtype=np.float64), empty, lod_duration_s=level.duration_s)
        data = level.dataset[bin_start:bin_stop]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        mins = data[:, 0]
        maxs = data[:, 1]
        start_time = bin_start * level.duration_s
        return chunk_from_envelope(start_time, level.duration_s, mins, maxs)


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
            durations: Dict[float, LodLevel] = {}
            for level_name in ch_group.array_keys():
                arr = ch_group[level_name]
                duration = float(arr.attrs.get("bin_duration_s", 0.0))
                key = self._lod_key(duration)
                bin_size = int(arr.attrs.get("bin_size", 0))
                if bin_size <= 0:
                    fs = float(self._channel_meta[idx].fs)
                    bin_size = max(1, int(round(fs * duration)))
                durations[key] = LodLevel(
                    duration_s=float(duration),
                    dataset=arr,
                    bin_size=bin_size,
                )
            if durations:
                self._lod_levels[idx] = durations

    @staticmethod
    def _lod_key(duration: float) -> float:
        return round(float(duration), 9)

    def _get_lod(self, idx: int, duration_s: float) -> LodLevel:
        levels = self._lod_levels.get(idx)
        if not levels:
            raise KeyError(f"no LOD levels for channel {idx}")
        key = self._lod_key(duration_s)
        if key not in levels:
            available = ", ".join(str(v) for v in sorted(levels.keys()))
            raise KeyError(
                f"LOD duration {duration_s} not found for channel {idx}; available: {available}"
            )
        return levels[key]


__all__ = ["ZarrLoader", "ChannelMeta"]
