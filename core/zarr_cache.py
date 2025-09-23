from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import zarr
from zarr.storage import BaseStore

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
            store = zarr.DirectoryStore(str(output_path))

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

    def _write_channels(self, group: zarr.Group, loader: EdfLoader) -> None:
        channels_group = group.create_group("channels")

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

            self._stream_channel(loader, idx, info, array, chunk_len)

        # ensure callback sees completion state
        self._progress_done = self._progress_total
        self._report_progress(0)

    def _compute_chunk_len(self, fs: float) -> int:
        if fs <= 0:
            return 1
        samples = int(round(fs * self.chunk_duration))
        samples = max(1, samples)
        return min(self.max_chunk_samples, samples)

    def _stream_channel(self, loader: EdfLoader, idx: int, info, array: zarr.Array, chunk_len: int) -> None:
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
            offset += data.size
            self._report_progress(data.size)

    def _report_progress(self, delta: int) -> None:
        if not self.progress_callback:
            return
        increment = max(0, delta)
        self._progress_done = min(self._progress_total, self._progress_done + increment)
        self.progress_callback(self._progress_done, max(1, self._progress_total))


__all__ = ["EdfToZarr", "DEFAULT_PROCESSED_DIR", "resolve_output_path"]
