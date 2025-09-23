from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import core.edf_loader as edf_module


class FakeEdfReader:
    def __init__(self, path: str):  # noqa: D401 - mimic pyedflib signature
        self.path = path
        self.file_duration = 5.0
        self._start = datetime(2024, 1, 1, 0, 0, 0)
        self.signals_in_file = 2
        self._labels = ["chan_a", "chan_b"]
        self._fs = [100.0, 50.0]
        self._data = [
            np.arange(int(self._fs[0] * self.file_duration), dtype=np.float64) / self._fs[0],
            np.linspace(-1.0, 1.0, int(self._fs[1] * self.file_duration), endpoint=False, dtype=np.float64),
        ]
        self._units = ["uV", "a.u."]
        self._closed = False

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
        arr = self._data[idx]
        if start >= arr.size:
            return np.zeros(0, dtype=np.float64)
        end = min(start + n, arr.size)
        return arr[start:end]

    def readAnnotations(self):
        return ([0.0, 2.5], [1.0, 0.0], ["A", "B"])

    def close(self):
        self._closed = True

    # Allow tests to assert the reader was closed.
    @property
    def closed(self):
        return self._closed


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


def test_read_annotations_passthrough():
    loader = edf_module.EdfLoader("dummy.edf")
    try:
        onsets, durations, labels = loader.read_annotations()
    finally:
        loader.close()

    assert onsets == [0.0, 2.5]
    assert durations == [1.0, 0.0]
    assert labels == ["A", "B"]
