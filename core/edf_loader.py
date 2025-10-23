# core/edf_loader.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
import threading
import numpy as np
import pyedflib

from core.int16_cache import Int16Cache, build_int16_cache as _build_int16_cache
from core.timebase import Timebase
from core import annotations as annotations_core
from core.overscan import SignalChunk, chunk_from_arrays, chunk_from_envelope

@dataclass
class ChannelInfo:
    name: str
    fs: float
    n_samples: int
    unit: str
    raw_index: int

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
        (
            self.channels,
            self._fs,
            self._ns,
            self.info,
            self._channel_map,
        ) = self._build_channel_metadata()
        self.n_channels = len(self.channels)
        self.timebase = Timebase(self.start_dt, self.duration_s)
        self._cache: Int16Cache | None = None
        self._scratch_float32: Dict[int, np.ndarray] = {}
        self._annotations: annotations_core.Annotations | None = None

    def _build_channel_metadata(
        self,
    ) -> Tuple[List[str], List[float], List[int], List[ChannelInfo], Tuple[int, ...]]:
        labels = self._r.getSignalLabels()
        ns = self._r.getNSamples()
        units = [self._r.getPhysicalDimension(i) for i in range(self._r.signals_in_file)]
        selected_labels: List[str] = []
        fs_list: List[float] = []
        ns_list: List[int] = []
        info_list: List[ChannelInfo] = []
        channel_map: List[int] = []

        for raw_idx in range(self._r.signals_in_file):
            fs = float(self._r.getSampleFrequency(raw_idx))
            n_samples = int(ns[raw_idx])
            if fs <= 0 or n_samples <= 0:
                continue
            channel_map.append(raw_idx)
            selected_labels.append(labels[raw_idx])
            fs_list.append(fs)
            ns_list.append(n_samples)
            info_list.append(
                ChannelInfo(
                    labels[raw_idx],
                    fs,
                    n_samples,
                    units[raw_idx],
                    raw_idx,
                )
            )

        return selected_labels, fs_list, ns_list, info_list, tuple(channel_map)

    def fs(self, i: int) -> float:
        return self._fs[i]

    def has_cache(self) -> bool:
        return self._cache is not None

    def cache_bytes(self) -> int:
        cache = self._cache
        return cache.total_bytes if cache is not None else 0

    def lod_levels(self, i: int) -> list[float]:
        cache = self._cache
        if cache is None:
            return []
        raw_idx = self._channel_map[i]
        try:
            return cache.lod_levels(raw_idx)
        except IndexError:
            return []

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

    def read(self, i: int, t0: float, t1: float) -> SignalChunk:
        raw_idx = self._channel_map[i]
        fs = self._fs[i]
        with self._lock:
            t0_c, t1_c = self.timebase.clamp_window(t0, t1)
            t1_c = min(t0_c + self.max_window_s, t1_c)
            s0, n = Timebase.sec_to_idx(t0_c, t1_c, fs)
            total = self._ns[i]
            if s0 >= total:
                empty_t = np.zeros(0, dtype=float)
                empty_x = np.zeros(0, dtype=np.float32)
                return chunk_from_arrays(empty_t, empty_x)
            n = min(n, max(0, total - s0))
            if n <= 0:
                empty_t = np.zeros(0, dtype=float)
                empty_x = np.zeros(0, dtype=np.float32)
                return chunk_from_arrays(empty_t, empty_x)
            cache = self._cache
            if cache is not None:
                scratch = self._scratch_float32.get(i)
                if scratch is None or scratch.shape[0] != n:
                    scratch = np.empty(n, dtype=np.float32)
                    self._scratch_float32[i] = scratch
                cache.fill_float32(raw_idx, s0, s0 + n, out=scratch)
                x = scratch.copy()
            else:
                x = self._r.readSignal(raw_idx, start=s0, n=n).astype(np.float32)
        t = Timebase.time_vector(s0, x.size, fs)
        source_end = float(t[-1]) if t.size else t0_c
        return chunk_from_arrays(
            t,
            x,
            source_start=float(t[0]) if t.size else t0_c,
            source_end=source_end,
        )

    # ------------------------------------------------------------------

    def lod_durations(self, i: int) -> tuple[float, ...]:
        cache = self._cache
        if cache is None:
            return ()
        raw_idx = self._channel_map[i]
        return cache.lod_durations(raw_idx)

    def read_lod_window(self, i: int, t0: float, t1: float, duration_s: float) -> SignalChunk:
        cache = self._cache
        if cache is None:
            raise KeyError("LOD data unavailable; build_int16_cache first")
        raw_idx = self._channel_map[i]
        fs = self._fs[i]
        with self._lock:
            t0_c, t1_c = self.timebase.clamp_window(t0, t1)
            t1_c = min(t0_c + self.max_window_s, t1_c)
            s0, n = Timebase.sec_to_idx(t0_c, t1_c, fs)
            total = self._ns[i]
            if s0 >= total or n <= 0:
                empty = np.zeros(0, dtype=np.float32)
                return chunk_from_arrays(np.zeros(0, dtype=np.float64), empty)
            s1 = min(total, s0 + n)
            level = cache.lod_level(raw_idx, duration_s)
            bin_size = level.bin_size
            bin_start = max(0, s0 // bin_size)
            bin_stop = min(level.data.shape[0], int(np.ceil(s1 / bin_size)))
            if bin_stop <= bin_start:
                empty = np.zeros(0, dtype=np.float32)
                return chunk_from_arrays(np.zeros(0, dtype=np.float64), empty, lod_duration_s=level.duration_s)
            mins = level.data[bin_start:bin_stop, 0]
            maxs = level.data[bin_start:bin_stop, 1]
            start_time = bin_start * level.duration_s
            return chunk_from_envelope(start_time, level.duration_s, mins, maxs)

    def annotations(self) -> annotations_core.Annotations:
        with self._lock:
            if self._annotations is None:
                self._annotations = annotations_core.from_edfplus(self._r)
            return self._annotations

    def read_annotations(self):
        ann = self.annotations()
        if ann.size == 0:
            return ([], [], [])
        starts = ann.data["start_s"]
        durations = ann.data["end_s"] - ann.data["start_s"]
        labels = ann.data["label"]
        return starts.tolist(), durations.tolist(), labels.tolist()

    def close(self):
        with self._lock:
            self._cache = None
            self._scratch_float32.clear()
            self._r.close()
