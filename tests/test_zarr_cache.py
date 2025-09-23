from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import core.zarr_cache as zc
from core.zarr_loader import ZarrLoader


class FakeEdfLoader:
    def __init__(self, path: str, *, duration_s=4.0, start=None):
        self.path = path
        self.duration_s = duration_s
        self.start_dt = start or datetime(2024, 1, 1, 0, 0, 0)
        self.n_channels = 2
        self.channels = ["C3", "O2"]
        self._fs = [100.0, 50.0]
        self.info = [
            SimpleNamespace(name="C3", fs=self._fs[0], n_samples=int(duration_s * self._fs[0]), unit="uV"),
            SimpleNamespace(name="O2", fs=self._fs[1], n_samples=int(duration_s * self._fs[1]), unit="uV"),
        ]
        t0 = np.arange(self.info[0].n_samples) / self._fs[0]
        t1 = np.arange(self.info[1].n_samples) / self._fs[1]
        self._data = [np.sin(2 * np.pi * 1.0 * t0), np.cos(2 * np.pi * 0.5 * t1)]
        self._closed = False

    def fs(self, idx: int) -> float:
        return self._fs[idx]

    def read(self, idx: int, t0: float, t1: float):
        info = self.info[idx]
        s0 = max(0, int(np.floor(t0 * info.fs)))
        s1 = min(info.n_samples, int(np.ceil(t1 * info.fs)))
        x = self._data[idx][s0:s1].astype(np.float32)
        t = np.arange(s0, s1) / info.fs
        return t, x

    def close(self):
        self._closed = True


@pytest.fixture(autouse=True)
def patch_loader(monkeypatch):
    monkeypatch.setattr(zc, "EdfLoader", FakeEdfLoader)
    yield


def test_edf_to_zarr_builds_channel_arrays():
    store = zarr.storage.MemoryStore()
    builder = zc.EdfToZarr(
        "study.edf",
        out_path="study.zarr",
        chunk_duration=1.0,
        store_factory=lambda path: store,
    )

    group = builder.build()
    assert isinstance(group, zarr.Group)

    assert group.attrs["duration_s"] == pytest.approx(4.0)
    assert group.attrs["start_dt"] == "2024-01-01T00:00:00"
    assert group.attrs["channels"] == ["C3", "O2"]

    ch_group = group["channels"]
    c3 = ch_group["0"]
    o2 = ch_group["1"]

    # chunk duration 1s at 100 Hz -> 100 samples per chunk
    assert c3.chunks == (100,)
    assert o2.chunks == (50,)
    baseline = FakeEdfLoader("", duration_s=4.0)
    np.testing.assert_allclose(c3[:10], baseline._data[0][:10], atol=1e-6)
    np.testing.assert_allclose(o2[:10], baseline._data[1][:10], atol=1e-6)

    assert c3.attrs["name"] == "C3"
    assert c3.attrs["fs"] == 100.0
    assert c3.attrs["unit"] == "uV"


def test_progress_callback_tracks_samples():
    store = zarr.storage.MemoryStore()
    progress = []

    builder = zc.EdfToZarr(
        "study.edf",
        out_path="study.zarr",
        chunk_duration=1.0,
        store_factory=lambda path: store,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    builder.build()

    assert progress  # callback invoked
    totals = {t for _, t in progress}
    assert len(totals) == 1
    final_done, total = progress[-1]
    assert final_done == total
    assert total == sum(info.n_samples for info in FakeEdfLoader("", duration_s=4.0).info)


def test_chunk_size_capped():
    fast_loader = FakeEdfLoader("fast.edf", duration_s=2.0)
    fast_loader._fs[0] = 20000.0
    fast_loader.info[0].fs = 20000.0
    fast_loader.info[0].n_samples = int(20000.0 * fast_loader.duration_s)
    fast_loader._data[0] = np.arange(fast_loader.info[0].n_samples, dtype=np.float32)

    store = zarr.storage.MemoryStore()
    builder = zc.EdfToZarr(
        "fast.edf",
        out_path="fast.zarr",
        chunk_duration=5.0,
        max_chunk_samples=2048,
        store_factory=lambda path: store,
        loader_factory=lambda path: fast_loader,
    )

    group = builder.build()
    c3 = group["channels"]["0"]
    assert c3.chunks == (2048,)

def test_builder_does_not_close_external_loader():
    loader = FakeEdfLoader("study.edf")

    store = zarr.storage.MemoryStore()
    builder = zc.EdfToZarr(
        "study.edf",
        out_path="study.zarr",
        store_factory=lambda path: store,
        loader_factory=lambda path: loader,
        owns_loader=False,
    )
    builder.build()

    assert loader._closed is False


def test_zarr_loader_parity_with_edf(tmp_path):
    store_path = tmp_path / "study.zarr"

    # Build Zarr store using fake EDF loader
    builder = zc.EdfToZarr(
        "study.edf",
        out_path=str(store_path),
    )
    builder.build()

    loader = zc.EdfLoader("study.edf")
    zarr_loader = ZarrLoader(store_path)

    try:
        for idx in range(loader.n_channels):
            t0, t1 = 0.5, 1.7
            t_e, x_e = loader.read(idx, t0, t1)
            t_z, x_z = zarr_loader.read(idx, t0, t1)
            np.testing.assert_allclose(t_z, t_e)
            np.testing.assert_allclose(x_z, x_e, atol=1e-6)
    finally:
        loader.close()
        zarr_loader.close()
