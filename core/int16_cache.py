# core/int16_cache.py
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import pyedflib


Int16Array = NDArray[np.int16]


@dataclass(frozen=True)
class EnvelopeLevel:
    duration_s: float
    bin_size: int
    data: NDArray[np.float32]

    def slice(self, start: int, stop: int) -> NDArray[np.float32]:
        start = max(0, int(start))
        stop = max(start, int(stop))
        return self.data[start:stop]


@dataclass(frozen=True)
class ChannelCalibration:
    digital_min: int
    digital_max: int
    physical_min: float
    physical_max: float
    sample_frequency: float
    n_samples: int
    slope: float
    offset: float


@dataclass
class Int16Cache:
    channels: Sequence[Int16Array]
    calibration: Sequence[ChannelCalibration]
    channel_names: Optional[Sequence[str]] = None
    lod_envelopes: Optional[Sequence[Dict[float, EnvelopeLevel]]] = None
    _locks: List[threading.RLock] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        channel_count = len(self.channels)
        if channel_count != len(self.calibration):
            raise ValueError("channels and calibration must have the same length")
        for arr in self.channels:
            if arr.dtype != np.int16:
                raise TypeError("channel arrays must have dtype int16")
        if self.channel_names is None:
            object.__setattr__(self, "channel_names", tuple(str(i) for i in range(channel_count)))
        elif len(self.channel_names) != channel_count:
            raise ValueError("channel_names length must match number of channels")
        object.__setattr__(self, "channels", tuple(arr for arr in self.channels))
        object.__setattr__(self, "calibration", tuple(self.calibration))
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        if self.lod_envelopes is None:
            lod = tuple({} for _ in range(channel_count))
        else:
            if len(self.lod_envelopes) != channel_count:
                raise ValueError("lod_envelopes length must match number of channels")
            lod = tuple(dict(levels) for levels in self.lod_envelopes)
        object.__setattr__(self, "lod_envelopes", lod)
        locks = [threading.RLock() for _ in range(channel_count)]
        object.__setattr__(self, "_locks", locks)

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def total_bytes(self) -> int:
        return int(sum(arr.nbytes for arr in self.channels))

    def channel_length(self, channel: int) -> int:
        return self.calibration[channel].n_samples

    def _validate_indices(self, channel: int, start: int, stop: int) -> None:
        if not 0 <= channel < self.channel_count:
            raise IndexError("channel index out of range")
        if start < 0 or stop < start:
            raise ValueError("invalid slice range")
        if stop > self.calibration[channel].n_samples:
            raise ValueError("slice exceeds cached samples")

    def raw_slice(self, channel: int, start: int, stop: int) -> Int16Array:
        self._validate_indices(channel, start, stop)
        with self._locks[channel]:
            return self.channels[channel][start:stop]

    def fill_float32(
        self,
        channel: int,
        start: int,
        stop: int,
        out: Optional[NDArray[np.float32]] = None,
    ) -> NDArray[np.float32]:
        self._validate_indices(channel, start, stop)
        meta = self.calibration[channel]
        length = stop - start
        with self._locks[channel]:
            segment = self.channels[channel][start:stop]
            if out is None:
                result = segment.astype(np.float32)
                if meta.slope != 1.0:
                    result *= meta.slope
                if meta.offset != 0.0:
                    result += meta.offset
                return result
            if out.dtype != np.float32:
                raise TypeError("out buffer must have dtype float32")
            if out.shape != (length,):
                raise ValueError("out buffer has incorrect shape")
            np.multiply(segment, meta.slope, out=out, dtype=np.float32)
            if meta.offset != 0.0:
                out += meta.offset
            return out

    def calibration_for(self, channel: int) -> ChannelCalibration:
        if not 0 <= channel < self.channel_count:
            raise IndexError("channel index out of range")
        return self.calibration[channel]

    def lod_levels(self, channel: int) -> list[float]:
        if not 0 <= channel < self.channel_count:
            raise IndexError("channel index out of range")
        levels = self.lod_envelopes[channel]
        if not levels:
            return []
        return sorted(levels.keys())

    def lod_level(self, channel: int, duration: float) -> EnvelopeLevel:
        if not 0 <= channel < self.channel_count:
            raise IndexError("channel index out of range")
        levels = self.lod_envelopes[channel]
        if not levels:
            raise KeyError(f"no LOD levels for channel {channel}")
        key = _lod_key(duration)
        if key not in levels:
            available = ", ".join(str(value) for value in sorted(levels.keys()))
            raise KeyError(
                f"LOD duration {duration} not found for channel {channel}; available: {available}"
            )
        return levels[key]


def _digital_read_fn(reader: pyedflib.EdfReader) -> Callable[[int], NDArray[np.int64]]:
    if hasattr(reader, "read_digital_signal"):
        return reader.read_digital_signal
    if hasattr(reader, "readSamples"):
        return reader.readSamples
    raise AttributeError("reader does not expose a digital read method")


def build_int16_cache(
    reader: pyedflib.EdfReader,
    *,
    max_bytes: int,
    prefer_memmap: bool,
    memmap_dir: Optional[Path],
    lod_durations: Sequence[float] | None = (1.0, 5.0, 30.0),
) -> Optional[Int16Cache]:
    if max_bytes <= 0:
        return None

    channel_count = reader.signals_in_file
    labels = tuple(reader.getSignalLabels())
    ns = tuple(int(n) for n in reader.getNSamples())
    read_fn = _digital_read_fn(reader)
    prefer_memmap = prefer_memmap and memmap_dir is not None
    if prefer_memmap and memmap_dir is not None:
        memmap_dir.mkdir(parents=True, exist_ok=True)

    channels: List[Int16Array] = []
    calibration: List[ChannelCalibration] = []
    lod_levels: List[Dict[float, EnvelopeLevel]] = []
    running_bytes = 0
    memmap_paths: List[Path] = []

    for ch in range(channel_count):
        expected_bytes = ns[ch] * np.dtype(np.int16).itemsize
        if running_bytes + expected_bytes > max_bytes:
            for path in memmap_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            return None

        raw = read_fn(ch)
        data = np.asarray(raw, dtype=np.int16)
        if data.shape != (ns[ch],):
            data = data.reshape((ns[ch],))

        if prefer_memmap and memmap_dir is not None:
            path = memmap_dir / f"channel_{ch}.int16"
            mm = np.memmap(path, dtype=np.int16, mode="w+", shape=(ns[ch],))
            mm[:] = data
            channels.append(mm)
            memmap_paths.append(path)
        else:
            channels.append(np.ascontiguousarray(data))

        digital_min = int(reader.getDigitalMinimum(ch))
        digital_max = int(reader.getDigitalMaximum(ch))
        physical_min = float(reader.getPhysicalMinimum(ch))
        physical_max = float(reader.getPhysicalMaximum(ch))
        sample_frequency = float(reader.getSampleFrequency(ch))

        denom = digital_max - digital_min
        if denom == 0:
            slope = 0.0
            offset = physical_min
        else:
            slope = (physical_max - physical_min) / denom
            offset = physical_min - slope * digital_min

        calibration.append(
            ChannelCalibration(
                digital_min=digital_min,
                digital_max=digital_max,
                physical_min=physical_min,
                physical_max=physical_max,
                sample_frequency=sample_frequency,
                n_samples=ns[ch],
                slope=slope,
                offset=offset,
            )
        )

        lod_levels.append(
            _compute_envelopes(
                data,
                sample_frequency,
                slope,
                offset,
                lod_durations,
            )
        )

        running_bytes += expected_bytes

    return Int16Cache(
        channels=channels,
        calibration=calibration,
        channel_names=labels,
        lod_envelopes=lod_levels,
    )


def _compute_envelopes(
    data: np.ndarray,
    fs: float,
    slope: float,
    offset: float,
    lod_durations: Sequence[float] | None,
) -> Dict[float, EnvelopeLevel]:
    if lod_durations is None:
        return {}
    if fs <= 0 or data.size == 0:
        return {}
    envelopes: Dict[float, EnvelopeLevel] = {}
    total_samples = int(data.size)
    for value in lod_durations:
        duration = float(value)
        if not math.isfinite(duration) or duration <= 0:
            continue
        bin_size = max(1, int(round(fs * duration)))
        if bin_size <= 0:
            continue
        # Recompute duration from integer bin size to keep loader metadata consistent.
        bin_duration = bin_size / fs
        bins = int(math.ceil(total_samples / bin_size))
        if bins <= 0:
            continue
        out = np.full((bins, 2), np.nan, dtype=np.float32)
        for idx in range(bins):
            start = idx * bin_size
            stop = min(total_samples, start + bin_size)
            segment = data[start:stop]
            if segment.size == 0:
                continue
            local_min = float(segment.min())
            local_max = float(segment.max())
            if slope != 1.0 or offset != 0.0:
                min_val = float(local_min * slope + offset)
                max_val = float(local_max * slope + offset)
            else:
                min_val = local_min
                max_val = local_max
            if min_val <= max_val:
                out[idx, 0] = min_val
                out[idx, 1] = max_val
            else:
                out[idx, 0] = max_val
                out[idx, 1] = min_val
        envelopes[_lod_key(bin_duration)] = EnvelopeLevel(
            duration_s=bin_duration,
            bin_size=bin_size,
            data=out,
        )
    return envelopes


def _lod_key(duration: float) -> float:
    return round(float(duration), 9)
