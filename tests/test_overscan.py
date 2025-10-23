import numpy as np
import pytest

try:
    from ui.main_window import _OverscanRequest, _OverscanTile, _OverscanWorker
except ImportError as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(f"Qt dependencies unavailable: {exc}", allow_module_level=True)

from core.overscan import (
    SignalChunk,
    choose_lod_duration,
    envelope_to_series,
    select_lod_duration,
    slice_and_decimate,
)


def test_slice_and_decimate_basic():
    t = np.linspace(0.0, 10.0, 1001)
    x = np.sin(t).astype(np.float32)
    sub_t, sub_x = slice_and_decimate(t, x, 2.0, 4.0, pixels=100)
    assert sub_t[0] >= 2.0 - 1e-6
    assert sub_t[-1] <= 4.0 + 1e-6
    assert sub_t.size == sub_x.size


def test_slice_and_decimate_empty_when_no_overlap():
    t = np.linspace(0.0, 1.0, 100)
    x = np.cos(t).astype(np.float32)
    sub_t, sub_x = slice_and_decimate(t, x, 2.0, 3.0, pixels=50)
    assert sub_t.size == 0
    assert sub_x.size == 0


def test_slice_and_decimate_respects_pixel_budget():
    t = np.linspace(0.0, 1.0, 10000)
    x = np.sin(50 * t).astype(np.float32)
    sub_t, sub_x = slice_and_decimate(t, x, 0.2, 0.8, pixels=100)
    assert sub_t.size <= 200
    assert sub_t.size == sub_x.size


def test_select_lod_duration_prefers_longest():
    durations = [1.0, 5.0, 30.0]
    assert select_lod_duration(120.0, durations, 2.0) == pytest.approx(30.0)
    assert select_lod_duration(40.0, durations, 2.0) == pytest.approx(5.0)
    assert select_lod_duration(5.0, durations, 3.0) == pytest.approx(1.0)
    assert select_lod_duration(1.0, durations, 2.0) is None
    assert select_lod_duration(40.0, durations, 2.0, min_view_duration=60.0) is None


def test_envelope_to_series_clamps_bounds():
    mins = np.array([-1.0, -0.5, -0.25], dtype=np.float32)
    maxs = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    t, x = envelope_to_series(
        mins,
        maxs,
        bin_duration=10.0,
        start_bin=0,
        window_start=5.0,
        window_end=25.0,
    )
    assert t[0] == pytest.approx(5.0)
    assert t[-1] == pytest.approx(25.0)
    assert x.dtype == np.float32
    assert t.size == x.size == 6
    assert x[0] <= x[1]


def test_overscan_worker_prefers_lod(monkeypatch):
    class FakeLoader:
        def __init__(self):
            self.calls: list[tuple[str, float]] = []

        def lod_levels(self, channel: int):
            return [1.0, 10.0]

        def read_lod_window(self, channel: int, start: float, end: float, duration: float):
            self.calls.append(("lod", duration))
            bins = int(np.ceil((end - start) / duration))
            data = np.empty((bins, 2), dtype=np.float32)
            data[:, 0] = -1.0
            data[:, 1] = 1.0
            return data, duration, int(start // duration)

        def read(self, channel: int, start: float, end: float, **_kwargs):
            self.calls.append(("raw", end - start))
            t = np.linspace(start, end, 100, endpoint=False)
            return t, np.sin(t).astype(np.float32)

    loader = FakeLoader()
    worker = _OverscanWorker(loader, lod_enabled=True, lod_min_bin_multiple=2.0)
    captured: dict[str, _OverscanTile] = {}

    def capture(req_id, tile):
        captured["tile"] = tile

    worker.finished.emit = capture  # type: ignore[assignment]
    worker.failed.emit = lambda *_args: None  # type: ignore[assignment]
    request = _OverscanRequest(
        request_id=1,
        start=0.0,
        end=120.0,
        view_start=0.0,
        view_duration=120.0,
        channel_indices=(0,),
        max_samples=None,
    )
    worker.render(request)
    tile = captured["tile"]
    assert loader.calls and loader.calls[0][0] == "lod"
    assert len(tile.raw_channel_data) == 1
    assert isinstance(tile.raw_channel_data[0], SignalChunk)
    assert tile.raw_channel_data[0].lod_duration_s == pytest.approx(10.0)
    assert tile.prepared_mask == [True]
    t_arr, x_arr = tile.channel_data[0]
    assert t_arr.size == x_arr.size > 0
    assert tile.is_final is True


def test_overscan_worker_respects_min_view_duration(monkeypatch):
    class FakeLoader:
        def __init__(self):
            self.calls: list[tuple[str, float]] = []

        def lod_levels(self, channel: int):
            return [1.0, 10.0]

        def read_lod_window(self, channel: int, start: float, end: float, duration: float):
            self.calls.append(("lod", duration))
            bins = int(np.ceil((end - start) / duration))
            data = np.empty((bins, 2), dtype=np.float32)
            data[:, 0] = -1.0
            data[:, 1] = 1.0
            return data, duration, int(start // duration)

        def read(self, channel: int, start: float, end: float, **_kwargs):
            self.calls.append(("raw", end - start))
            t = np.linspace(start, end, 100, endpoint=False)
            return t, np.sin(t).astype(np.float32)

    loader = FakeLoader()
    worker = _OverscanWorker(
        loader,
        lod_enabled=True,
        lod_min_bin_multiple=2.0,
        lod_min_view_duration=200.0,
    )
    captured: dict[str, _OverscanTile] = {}

    def capture(req_id, tile):
        captured["tile"] = tile

    worker.finished.emit = capture  # type: ignore[assignment]
    worker.failed.emit = lambda *_args: None  # type: ignore[assignment]
    request = _OverscanRequest(
        request_id=2,
        start=0.0,
        end=120.0,
        view_start=0.0,
        view_duration=120.0,
        channel_indices=(0,),
        max_samples=None,
    )
    worker.render(request)
    tile = captured["tile"]
    assert loader.calls and loader.calls[0][0] == "raw"
    assert isinstance(tile.raw_channel_data[0], SignalChunk)
    assert tile.raw_channel_data[0].lod_duration_s is None
    assert tile.prepared_mask == [True]
    assert tile.lod_durations == [None]
    assert tile.is_final is True


def test_overscan_worker_ratio_respects_min_view(monkeypatch):
    class FakeLoader:
        def __init__(self):
            self.calls: list[tuple[str, float]] = []

        def lod_levels(self, channel: int):
            return [1.0, 5.0, 30.0]

        def read_lod_window(self, channel: int, start: float, end: float, duration: float):
            self.calls.append(("lod", duration))
            bins = int(np.ceil((end - start) / duration))
            data = np.empty((bins, 2), dtype=np.float32)
            data[:, 0] = -1.0
            data[:, 1] = 1.0
            return data, duration, int(start // duration)

        def read(self, channel: int, start: float, end: float, **_kwargs):
            self.calls.append(("raw", end - start))
            t = np.linspace(start, end, 100, endpoint=False)
            return t, np.sin(t).astype(np.float32)

    loader = FakeLoader()
    worker = _OverscanWorker(
        loader,
        lod_enabled=True,
        lod_min_bin_multiple=2.0,
        lod_min_view_duration=200.0,
        lod_ratio=2.0,
    )
    captured: dict[str, _OverscanTile] = {}

    def capture(req_id, tile):
        captured["tile"] = tile

    worker.finished.emit = capture  # type: ignore[assignment]
    worker.failed.emit = lambda *_args: None  # type: ignore[assignment]
    request = _OverscanRequest(
        request_id=3,
        start=0.0,
        end=60.0,
        view_start=0.0,
        view_duration=60.0,
        channel_indices=(0,),
        max_samples=None,
    )
    worker.render(request)
    tile = captured["tile"]
    assert loader.calls and loader.calls[0][0] == "raw"
    assert isinstance(tile.raw_channel_data[0], SignalChunk)
    assert tile.raw_channel_data[0].lod_duration_s is None
    assert tile.lod_durations == [None]


def test_slice_and_decimate_uses_pre_binned_chunk():
    t = np.linspace(0.0, 100.0, 50)
    x = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    chunk = SignalChunk(t, x, lod_duration_s=10.0)
    sub_t, sub_x = slice_and_decimate(chunk, None, 5.0, 45.0, pixels=10)
    assert sub_t[0] >= 5.0 - 1e-6
    assert sub_t[-1] <= 45.0 + 1e-6
    assert sub_t.size == sub_x.size


def test_choose_lod_duration_prefers_coarsest_available():
    durations = [1.0, 5.0, 30.0]
    assert choose_lod_duration(180.0, durations, ratio=2.0) == 30.0
    assert choose_lod_duration(12.0, durations, ratio=2.0) == 5.0
    assert choose_lod_duration(1.5, durations, ratio=2.0) is None
