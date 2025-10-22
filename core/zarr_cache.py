from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, cast

import numpy as np
import zarr

try:
    from zarr.storage import BaseStore
except ImportError:  # pragma: no cover - zarr >= 3 compatibility
    class BaseStore(Protocol):  # type: ignore[override]
        def __getitem__(self, key: str) -> bytes: ...

        def __setitem__(self, key: str, value: bytes) -> None: ...

try:
    from zarr.storage import DirectoryStore as _DirectoryStore
except ImportError:  # pragma: no cover - zarr >= 3
    _DirectoryStore = None
    try:
        from zarr.storage import LocalStore as _LocalStore
    except ImportError:  # pragma: no cover - legacy fallback
        _LocalStore = None
else:
    _LocalStore = None

from core.edf_loader import EdfLoader

DEFAULT_PROCESSED_DIR = Path("processed")


def resolve_output_path(edf_path: str | Path, out_path: Optional[str | Path] = None) -> Path:
    base = Path(edf_path)
    if out_path is not None:
        return Path(out_path)
    stem = base.stem
    return DEFAULT_PROCESSED_DIR / f"{stem}.zarr"


@dataclass
class EdfToZarr:
    edf_path: str
    out_path: Optional[str] = None
    chunk_duration: float = 5.0
    max_chunk_samples: int = 4096
    lod_durations: Sequence[float] = (1.0, 5.0, 30.0)
    store_factory: Optional[Callable[[Path], BaseStore]] = None
    loader_factory: Optional[Callable[[str], EdfLoader]] = None
    progress_callback: Optional[Callable[[int, int], None]] = None
    owns_loader: bool = True

    def build(self) -> zarr.Group:
        edf_path = Path(self.edf_path)
        output_path = resolve_output_path(edf_path, self.out_path)

        if self.store_factory is not None:
            store = self.store_factory(output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            store = self._create_default_store(output_path)

        loader = self._create_loader()
        total_samples = sum(int(info.n_samples) for info in loader.info)
        self._progress_total = total_samples
        self._progress_done = 0
        self._report_progress(0)
        try:
            group = zarr.open_group(store=store, mode="w")
            self._write_root_attrs(group, loader, output_path)
            self._write_channels(group, loader)
        finally:
            if self.owns_loader:
                loader.close()

        return group

    # ------------------------------------------------------------------

    def _create_loader(self) -> EdfLoader:
        if self.loader_factory is not None:
            return self.loader_factory(self.edf_path)
        return EdfLoader(self.edf_path)

    def _create_default_store(self, output_path: Path) -> BaseStore:
        if _DirectoryStore is not None:
            return cast(BaseStore, _DirectoryStore(str(output_path)))
        if _LocalStore is not None:
            return cast(BaseStore, _LocalStore(str(output_path)))
        raise RuntimeError("No compatible Zarr store backend found")

    def _write_root_attrs(self, group: zarr.Group, loader: EdfLoader, output_path: Path) -> None:
        attrs = group.attrs
        attrs["schema_version"] = "1.0"
        attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        attrs["edf_path"] = str(Path(self.edf_path).resolve())
        attrs["output_path"] = str(output_path)
        attrs["duration_s"] = float(loader.duration_s)
        if loader.start_dt is not None:
            attrs["start_dt"] = loader.start_dt.isoformat()
        attrs["channels"] = list(loader.channels)
        attrs["chunk_duration_s"] = float(self.chunk_duration)
        attrs["max_chunk_samples"] = int(self.max_chunk_samples)
        attrs["max_window_s"] = float(getattr(loader, "max_window_s", 120.0))
        lod_levels = self._normalized_lod_durations()
        if lod_levels:
            attrs["lod_durations_s"] = list(lod_levels)

    def _write_channels(self, group: zarr.Group, loader: EdfLoader) -> None:
        channels_group = group.create_group("channels")
        lod_levels = self._normalized_lod_durations()
        lod_root: zarr.Group | None = None
        if lod_levels:
            lod_root = group.create_group("lod")

        for idx, info in enumerate(loader.info):
            chunk_len = self._compute_chunk_len(info.fs)
            array = channels_group.create_dataset(
                str(idx),
                shape=(info.n_samples,),
                chunks=(chunk_len,),
                dtype="float32",
                overwrite=True,
            )
            array.attrs["name"] = info.name
            array.attrs["unit"] = info.unit
            array.attrs["fs"] = float(info.fs)
            array.attrs["n_samples"] = int(info.n_samples)

            envelope_writers: list[_EnvelopeWriter] = []
            if lod_root is not None:
                envelope_writers = self._create_lod_writers(lod_root, idx, info, lod_levels)

            self._stream_channel(loader, idx, info, array, chunk_len, envelope_writers)

        # ensure callback sees completion state
        self._progress_done = self._progress_total
        self._report_progress(0)

    def _compute_chunk_len(self, fs: float) -> int:
        if fs <= 0:
            return 1
        samples = int(round(fs * self.chunk_duration))
        samples = max(1, samples)
        return min(self.max_chunk_samples, samples)

    def _stream_channel(
        self,
        loader: EdfLoader,
        idx: int,
        info,
        array: zarr.Array,
        chunk_len: int,
        envelope_writers: Sequence["_EnvelopeWriter"],
    ) -> None:
        total = int(info.n_samples)
        fs = float(info.fs)
        offset = 0

        while offset < total:
            samples = min(chunk_len, total - offset)
            t0 = offset / fs
            t1 = (offset + samples) / fs
            _, data = loader.read(idx, t0, t1)
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            array[offset : offset + data.size] = data
            for writer in envelope_writers:
                writer.ingest(data)
            offset += data.size
            self._report_progress(data.size)

        for writer in envelope_writers:
            writer.finalize()

    def _report_progress(self, delta: int) -> None:
        if not self.progress_callback:
            return
        increment = max(0, delta)
        self._progress_done = min(self._progress_total, self._progress_done + increment)
        self.progress_callback(self._progress_done, max(1, self._progress_total))

    # ------------------------------------------------------------------

    def _normalized_lod_durations(self) -> tuple[float, ...]:
        seen: dict[float, float] = {}
        for duration in self.lod_durations:
            value = float(duration)
            if not math.isfinite(value) or value <= 0:
                continue
            key = round(value, 9)
            if key not in seen:
                seen[key] = value
        return tuple(seen[key] for key in sorted(seen))

    def _create_lod_writers(
        self,
        lod_root: zarr.Group,
        idx: int,
        info,
        lod_levels: Sequence[float],
    ) -> list["_EnvelopeWriter"]:
        if int(info.n_samples) <= 0:
            return []
        ch_group = lod_root.create_group(str(idx))
        writers: list[_EnvelopeWriter] = []
        fs = float(info.fs)
        total_samples = int(info.n_samples)
        for duration in lod_levels:
            bin_size = max(1, int(round(fs * duration)))
            if bin_size <= 0:
                continue
            bins = int(math.ceil(total_samples / bin_size))
            if bins <= 0:
                continue
            chunk_bins = bins
            if self.max_chunk_samples > 0:
                approx = self.max_chunk_samples // max(1, bin_size)
                if approx <= 0:
                    approx = 1
                chunk_bins = min(bins, approx)
            dataset = ch_group.create_dataset(
                self._lod_dataset_name(duration),
                shape=(bins, 2),
                chunks=(chunk_bins, 2),
                dtype="float32",
                overwrite=True,
            )
            dataset.attrs["bin_duration_s"] = float(duration)
            dataset.attrs["bin_size"] = int(bin_size)
            dataset.attrs["columns"] = ["min", "max"]
            writers.append(_EnvelopeWriter(dataset, bin_size, bins))
        return writers

    @staticmethod
    def _lod_dataset_name(duration: float) -> str:
        text = format(duration, "g").replace("-", "neg").replace(".", "p")
        return f"sec_{text}"


__all__ = ["EdfToZarr", "DEFAULT_PROCESSED_DIR", "resolve_output_path"]


class _EnvelopeWriter:
    """Incrementally computes min/max envelopes for fixed-size bins."""

    def __init__(self, dataset: zarr.Array, bin_size: int, expected_bins: int) -> None:
        self._dataset = dataset
        self._bin_size = max(1, int(bin_size))
        self._expected_bins = max(0, int(expected_bins))
        self._buffer = np.empty(0, dtype=np.float32)
        self._write_idx = 0

    def ingest(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if self._buffer.size:
            data = np.concatenate((self._buffer, data))
            self._buffer = np.empty(0, dtype=np.float32)
        bin_size = self._bin_size
        full_bins = data.size // bin_size
        if full_bins:
            trimmed = data[: full_bins * bin_size]
            reshaped = trimmed.reshape(full_bins, bin_size)
            mins = reshaped.min(axis=1)
            maxs = reshaped.max(axis=1)
            out = np.stack((mins, maxs), axis=1)
            self._dataset[self._write_idx : self._write_idx + full_bins] = out
            self._write_idx += full_bins
            data = data[full_bins * bin_size :]
        if data.size:
            self._buffer = data.copy()

    def finalize(self) -> None:
        if self._buffer.size:
            mins = float(self._buffer.min())
            maxs = float(self._buffer.max())
            self._dataset[self._write_idx] = (mins, maxs)
            self._write_idx += 1
            self._buffer = np.empty(0, dtype=np.float32)

        remaining = self._expected_bins - self._write_idx
        if remaining > 0:
            fill = np.full((remaining, 2), np.nan, dtype=np.float32)
            self._dataset[self._write_idx : self._write_idx + remaining] = fill
            self._write_idx += remaining
