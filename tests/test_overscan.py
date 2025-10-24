from collections import OrderedDict
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

try:
    from ui import main_window as main_window_module
    from ui.main_window import _OverscanRequest, _OverscanTile, _OverscanWorker
except ImportError as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(f"Qt dependencies unavailable: {exc}", allow_module_level=True)

from core.overscan import (
    SignalChunk,
    OverscanRenderer,
    choose_lod_duration,
    chunk_from_arrays,
    chunk_from_envelope,
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


class _SyntheticLoader:
    def __init__(self, *, fs: float, lod_durations: Sequence[float]):
        self._fs = float(fs)
        self._lod = tuple(float(v) for v in lod_durations)
        self.duration_s = 600.0
        self.max_window_s = 120.0
        self.info = [SimpleNamespace(fs=self._fs, n_samples=int(self._fs * self.duration_s), unit="u")]
        self.channels = ["synthetic"]
        self.calls: list[tuple[str, float]] = []

    def fs(self, idx: int) -> float:
        assert idx == 0
        return self._fs

    def lod_durations(self, idx: int) -> tuple[float, ...]:
        assert idx == 0
        return self._lod

    def read_lod_window(self, idx: int, start: float, end: float, duration: float) -> SignalChunk:
        assert idx == 0
        self.calls.append(("lod", float(duration)))
        bins = int(np.ceil(max(0.0, end - start) / float(duration)))
        mins = np.full(bins, -1.0, dtype=np.float32)
        maxs = np.full(bins, 1.0, dtype=np.float32)
        return chunk_from_envelope(float(start), float(duration), mins, maxs)

    def read(self, idx: int, start: float, end: float) -> SignalChunk:
        assert idx == 0
        self.calls.append(("raw", float(end - start)))
        sample_count = max(0, int(round((end - start) * self._fs)))
        t = np.linspace(start, end, sample_count, endpoint=False, dtype=np.float64)
        x = np.zeros_like(t, dtype=np.float32)
        return chunk_from_arrays(t, x)


def test_overscan_renderer_chooses_coarsest_envelope():
    loader = _SyntheticLoader(fs=100.0, lod_durations=(0.5, 1.0, 5.0))
    renderer = OverscanRenderer(loader)
    chunk = renderer.render(0, 0.0, 500.0, plot_width_px=100)
    assert isinstance(chunk, SignalChunk)
    assert loader.calls[0] == ("lod", pytest.approx(5.0))


def test_overscan_renderer_falls_back_to_raw_for_zoom():
    loader = _SyntheticLoader(fs=100.0, lod_durations=(0.5, 1.0, 5.0))
    renderer = OverscanRenderer(loader)
    chunk = renderer.render(0, 0.0, 5.0, plot_width_px=1000)
    assert isinstance(chunk, SignalChunk)
    assert loader.calls[0][0] == "raw"


def test_choose_envelope_duration_prefers_stride_threshold():
    loader = _SyntheticLoader(fs=200.0, lod_durations=(0.25, 1.0, 2.0, 4.0))
    renderer = OverscanRenderer(loader)
    duration = renderer.choose_envelope_duration(0, window_duration=120.0, plot_width_px=400)
    # samples per px = 120 * 200 / 400 = 60 -> best bin <= 60 is 0.25 s (50 samples)
    assert duration == pytest.approx(0.25)
    duration_zoomed = renderer.choose_envelope_duration(0, window_duration=120.0, plot_width_px=20)
    # samples per px = 120 * 200 / 20 = 1200 -> allows up to 4.0 s bins (800 samples)
    assert duration_zoomed == pytest.approx(4.0)


def test_prepare_tile_uses_cached_series(monkeypatch):
    window = main_window_module.MainWindow

    class DummyWindow:
        _overscan_factor = 2.0
        _use_gpu_canvas = False
        _gpu_autoswitch_enabled = False

        def _estimate_pixels(self):
            return 100

    dummy = DummyWindow()

    t = np.linspace(0.0, 10.0, 1000)
    x = np.sin(t).astype(np.float32)
    chunk = SignalChunk(t, x)

    tile = _OverscanTile(
        request_id=1,
        start=0.0,
        end=10.0,
        view_start=0.0,
        view_duration=10.0,
        raw_channel_data=[chunk],
        channel_data=[chunk.as_tuple()],
        channel_indices=(0,),
        prepared_mask=[False],
        lod_durations=[None],
        is_final=True,
    )

    call_counter = {"count": 0}
    real_slice = main_window_module.slice_and_decimate

    def counting_slice(*args, **kwargs):
        call_counter["count"] += 1
        return real_slice(*args, **kwargs)

    monkeypatch.setattr(main_window_module, "slice_and_decimate", counting_slice)

    changed_first = window._prepare_tile(dummy, tile)
    assert changed_first is True
    assert call_counter["count"] == 1
    budget = tile.pixel_budget
    assert budget is not None
    assert budget in tile.prepared_cache

    changed_second = window._prepare_tile(dummy, tile)
    assert changed_second is False
    assert call_counter["count"] == 1


def test_request_tile_uses_lru_cache():
    window_cls = main_window_module.MainWindow

    class DummyLoader:
        duration_s = 120.0
        n_channels = 2
        max_window_s = 120.0

    class DummyWindow:
        def __init__(self):
            self.loader = DummyLoader()
            self._overscan_worker = object()
            self._overscan_factor = 2.0
            self._overscan_tile = None
            self._overscan_inflight = None
            self._overscan_request_id = 0
            self._current_tile_id = None
            self._overscan_tile_cache: OrderedDict[
                tuple[tuple[int, ...], int, int], _OverscanTile
            ] = OrderedDict()
            self._overscan_tile_cache_limit = 6
            self._scheduled_refresh = False
            self._applied_tiles: list[_OverscanTile] = []
            self._view_start = 10.0
            self._view_duration = 10.0
            self.overscanRequested = SimpleNamespace(
                calls=[], emit=lambda request: self.overscanRequested.calls.append(request)
            )

        def _estimate_pixels(self):
            return 0

        def _update_tile_view_metadata(self, tile, view_start, view_duration):
            if tile is None:
                return
            tile.view_start = view_start
            tile.view_duration = view_duration

        def _apply_tile_to_curves(self, tile):
            self._applied_tiles.append(tile)

        def _schedule_refresh(self):
            self._scheduled_refresh = True

        def _compute_overscan_bounds(self, view_start, view_duration):
            return window_cls._compute_overscan_bounds(self, view_start, view_duration)

        def _tile_cache_key(self, channels, start, end):
            return (
                tuple(channels),
                int(round(float(start) * 1_000_000)),
                int(round(float(end) * 1_000_000)),
            )

        def _get_cached_tile(self, channels, start, end):
            return window_cls._get_cached_tile(self, channels, start, end)

    dummy = DummyWindow()
    start, end = dummy._compute_overscan_bounds(dummy._view_start, dummy._view_duration)
    channels = tuple(range(dummy.loader.n_channels))
    raw_chunks = [
        SignalChunk(np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float32))
        for _ in channels
    ]
    tile = _OverscanTile(
        request_id=42,
        start=start,
        end=end,
        view_start=dummy._view_start,
        view_duration=dummy._view_duration,
        raw_channel_data=raw_chunks,
        channel_data=[chunk.as_tuple() for chunk in raw_chunks],
        channel_indices=channels,
        prepared_mask=[True] * len(raw_chunks),
        lod_durations=[None] * len(raw_chunks),
        is_final=True,
    )
    cache_key = dummy._tile_cache_key(channels, start, end)
    dummy._overscan_tile_cache[cache_key] = tile

    window_cls._request_overscan_tile(dummy, dummy._view_start, dummy._view_duration)

    assert dummy.overscanRequested.calls == []
    assert dummy._overscan_tile is tile
    assert dummy._overscan_inflight is None
    assert dummy._overscan_request_id == 1
    assert tile.request_id == dummy._overscan_request_id
    assert dummy._applied_tiles[-1] is tile
    assert dummy._scheduled_refresh is True
    assert tile.view_start == pytest.approx(dummy._view_start)
    assert tile.view_duration == pytest.approx(dummy._view_duration)
    assert list(dummy._overscan_tile_cache.values())[-1] is tile
