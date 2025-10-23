from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import core.edf_loader as edf_module
import core.annotations as annotations_module


class FakeEdfReader:
    def __init__(self, path: str):  # noqa: D401 - mimic pyedflib signature
        self.path = path
        self.file_duration = 5.0
        self._start = datetime(2024, 1, 1, 0, 0, 0)
        self.signals_in_file = 3
        self._labels = ["chan_a", "chan_b", "EDF Annotations"]
        self._fs = [100.0, 50.0, 0.0]
        self._data = [
            np.arange(int(self._fs[0] * self.file_duration), dtype=np.float64) / self._fs[0],
            np.linspace(-1.0, 1.0, int(self._fs[1] * self.file_duration), endpoint=False, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        ]
        self._units = ["uV", "a.u.", ""]
        self._closed = False
        self._read_signal_calls = 0
        self._read_digital_calls = 0

        self._digital = []
        self._digital_min = []
        self._digital_max = []
        self._physical_min = []
        self._physical_max = []
        self._slope = [0.01, 0.02, 1.0]
        self._offset = [0.0, -1.0, 0.0]
        for idx, arr in enumerate(self._data):
            slope = self._slope[idx]
            offset = self._offset[idx]
            if arr.size == 0:
                digital = np.zeros(0, dtype=np.int16)
                self._digital.append(digital)
                self._digital_min.append(0)
                self._digital_max.append(0)
                self._physical_min.append(0.0)
                self._physical_max.append(0.0)
                continue
            digital = np.round((arr - offset) / slope).astype(np.int16)
            self._digital.append(digital)
            self._digital_min.append(int(digital.min()))
            self._digital_max.append(int(digital.max()))
            self._physical_min.append(float(digital.min() * slope + offset))
            self._physical_max.append(float(digital.max() * slope + offset))

    # --- pyedflib API we rely on -------------------------------------------------
    def getStartdatetime(self):
        return self._start

    def getSignalLabels(self):
        return list(self._labels)

    def getSampleFrequency(self, idx: int) -> float:
        return self._fs[idx]

    def getNSamples(self):
        return [arr.size for arr in self._data]

    def getPhysicalDimension(self, idx: int) -> str:
        return self._units[idx]

    def readSignal(self, idx: int, start: int, n: int):
        self._read_signal_calls += 1
        arr = self._data[idx]
        if start >= arr.size:
            return np.zeros(0, dtype=np.float64)
        end = min(start + n, arr.size)
        return arr[start:end]

    def read_digital_signal(self, idx: int):
        self._read_digital_calls += 1
        return self._digital[idx]

    def getDigitalMinimum(self, idx: int) -> int:
        return self._digital_min[idx]

    def getDigitalMaximum(self, idx: int) -> int:
        return self._digital_max[idx]

    def getPhysicalMinimum(self, idx: int) -> float:
        return self._physical_min[idx]

    def getPhysicalMaximum(self, idx: int) -> float:
        return self._physical_max[idx]

    def readAnnotations(self):
        return ([0.0, 2.5], [1.0, 0.0], ["A", "B"])

    def close(self):
        self._closed = True

    # Allow tests to assert the reader was closed.
    @property
    def closed(self):
        return self._closed

    @property
    def read_signal_calls(self):
        return self._read_signal_calls

    @property
    def read_digital_calls(self):
        return self._read_digital_calls


@pytest.fixture(autouse=True)
def patch_pyedflib(monkeypatch):
    dummy_module = SimpleNamespace(EdfReader=FakeEdfReader)
    monkeypatch.setattr(edf_module, "pyedflib", dummy_module)
    yield


def test_loader_metadata_and_timebase():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        assert loader.duration_s == pytest.approx(5.0)
        assert loader.start_dt == datetime(2024, 1, 1, 0, 0, 0)
        assert loader.n_channels == 2
        assert loader.channels == ["chan_a", "chan_b"]
        assert loader.fs(0) == 100.0
        assert loader.timebase.duration_s == 5.0
        assert loader.timebase.start_dt == loader.start_dt
        assert loader.info[0].raw_index == 0
        assert loader.info[1].raw_index == 1
    finally:
        loader.close()


def test_read_basic_window_returns_time_vector_and_float32():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        t, x = loader.read(0, 1.0, 2.0)
    finally:
        loader.close()

    assert t[0] == pytest.approx(1.0)
    assert t[-1] < 2.0
    assert x.dtype == np.float32
    # Source data is monotonic increasing seconds, so window should match indices
    np.testing.assert_allclose(x[:5], np.array([1.0, 1.01, 1.02, 1.03, 1.04], dtype=np.float32), atol=1e-6)


def test_read_clamps_to_duration_and_handles_tail():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        t, x = loader.read(1, 4.8, 6.0)
    finally:
        loader.close()

    assert t[0] == pytest.approx(4.8, abs=1e-6)
    assert t[-1] <= 5.0 + 1e-6
    assert x.size == 10  # 0.2 s * 50 Hz


def test_read_after_end_returns_empty():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        t, x = loader.read(0, 6.0, 7.0)
    finally:
        loader.close()

    assert t.size == 0
    assert x.size == 0


def test_read_annotations_from_edfplus():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        annotations = loader.annotations()
        assert isinstance(annotations, annotations_module.Annotations)
        assert annotations.size == 2
        assert annotations.data["start_s"].tolist() == [0.0, 2.5]
        assert loader.annotations() is annotations
        onsets, durations, labels = loader.read_annotations()
    finally:
        loader.close()

    assert onsets == [0.0, 2.5]
    assert durations == [1.0, 0.0]
    assert labels == ["A", "B"]


def test_read_respects_max_window():
    loader = edf_module.EdfLoader("dummy.edf", max_window_s=0.5)
    try:
        t, x = loader.read(0, 0.0, 5.0)
    finally:
        loader.close()

    assert t.size == x.size
    assert t.size <= int(np.ceil(loader.fs(0) * 0.5)) + 1
    assert t[-1] - t[0] <= 0.5 + 1 / loader.fs(0)


def test_build_int16_cache_and_cached_reads_bypass_pyedflib(monkeypatch):
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        reader = loader._r
        baseline_t, baseline_x = loader.read(0, 1.0, 1.2)
        assert reader.read_signal_calls > 0
        reader._read_signal_calls = 0

        built = loader.build_int16_cache(max_mb=1.0, prefer_memmap=False)
        assert built is True
        assert reader.read_digital_calls == reader.signals_in_file

        def boom(*_args, **_kwargs):  # pragma: no cover - we want this unused
            raise AssertionError("pyedflib readSignal should not be called when cache is present")

        monkeypatch.setattr(reader, "readSignal", boom)

        t, x = loader.read(0, 1.0, 1.2)
        assert reader.read_signal_calls == 0
        assert x.dtype == np.float32
        np.testing.assert_allclose(t, baseline_t, atol=1e-6)
        np.testing.assert_allclose(x, baseline_x, atol=1e-6)
    finally:
        loader.close()


def test_lod_levels_available_after_cache_build():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        assert loader.build_int16_cache(max_mb=10.0, prefer_memmap=False)
        levels = loader.lod_levels(0)
        assert levels
        duration = levels[-1]
        chunk = loader.read_lod_window(0, 0.0, 5.0, duration)
        assert chunk.lod_duration_s == pytest.approx(duration, rel=1e-6)
        t, x = chunk
        assert t.size == x.size
        assert t.size > 0
        assert np.isfinite(x).all()
    finally:
        loader.close()


def test_read_lod_window_requires_cache():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        with pytest.raises(KeyError):
            loader.read_lod_window(0, 0.0, 1.0, 1.0)
    finally:
        loader.close()


def test_build_int16_cache_size_guard_and_streaming():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        reader = loader._r
        needed_bytes = sum(arr.nbytes for arr in reader._digital)
        max_mb = (needed_bytes - 1) / (1024 * 1024)

        built = loader.build_int16_cache(max_mb=max_mb, prefer_memmap=False)
        assert built is False
        assert loader.has_cache() is False
        assert loader.cache_bytes() == 0
        # Only the first channel is materialized before the guard fails
        assert reader.read_digital_calls == 1

        reader._read_signal_calls = 0
        t, x = loader.read(0, 0.0, 0.25)
        assert t.size == x.size
        assert x.dtype == np.float32
        assert reader.read_signal_calls > 0
    finally:
        loader.close()


def test_prefetch_cache_tiles_use_cached_loader(monkeypatch):
    from core.prefetch import PrefetchCache, PrefetchConfig

    loader = edf_module.EdfLoader("dummy.edf")
    try:
        reader = loader._r
        assert loader.build_int16_cache(max_mb=1.0, prefer_memmap=False) is True

        def boom(*_args, **_kwargs):  # pragma: no cover - should be bypassed
            raise AssertionError("readSignal should not run while cache is active")

        reader._read_signal_calls = 0
        monkeypatch.setattr(reader, "readSignal", boom)

        calls: list[tuple[int, float, float]] = []
        original_read = loader.read

        def tracking_read(channel: int, start: float, end: float):
            calls.append((channel, start, end))
            return original_read(channel, start, end)

        monkeypatch.setattr(loader, "read", tracking_read)

        cache = PrefetchCache(loader.read, PrefetchConfig(tile_duration=0.25, max_tiles=4))

        tile_t, tile_x = cache.get_tile(0, 0.75, 0.25)
        assert calls == [(0, 0.75, 1.0)]
        assert tile_x.dtype == np.float32
        expected = reader._data[0][75:100].astype(np.float32)
        np.testing.assert_allclose(tile_x, expected, atol=1e-6)
        expected_t = np.arange(75, 100, dtype=np.float64) / reader._fs[0]
        np.testing.assert_allclose(tile_t, expected_t, atol=1e-6)

        calls.clear()
        tile_t2, tile_x2 = cache.get_tile(1, 2.0, 0.5)
        assert calls == [(1, 2.0, 2.5)]
        assert tile_x2.dtype == np.float32
        expected2 = (
            reader._digital[1][100:125].astype(np.float32) * reader._slope[1]
            + reader._offset[1]
        )
        np.testing.assert_allclose(tile_x2, expected2, atol=1e-6)
        # time vector for channel 1 uses 50 Hz
        expected_t2 = np.arange(100, 125) / reader._fs[1]
        np.testing.assert_allclose(tile_t2, expected_t2, atol=1e-6)

        # Second request should hit PrefetchCache without invoking loader.read again
        calls.clear()
        tile_t_cached, tile_x_cached = cache.get_tile(0, 0.75, 0.25)
        assert calls == []
        np.testing.assert_allclose(tile_t_cached, tile_t, atol=1e-6)
        np.testing.assert_allclose(tile_x_cached, tile_x, atol=1e-6)
        assert reader.read_signal_calls == 0
    finally:
        loader.close()


def test_edf_loader_exposes_lod_envelopes():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        assert loader.build_int16_cache(max_mb=1.0, prefer_memmap=False) is True
        durations = loader.lod_durations(0)
        assert durations
        chunk = loader.read_lod_window(0, 0.0, 5.0, durations[-1])
    finally:
        loader.close()

    assert hasattr(chunk, "lod_duration_s")
    assert chunk.lod_duration_s == pytest.approx(durations[-1], rel=1e-3, abs=1e-6)
    t, x = chunk
    assert t.size > 0
    assert x.size == t.size
