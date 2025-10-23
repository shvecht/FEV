"""Headless smoke tests for the PySide6 viewer window."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest

try:  # pragma: no cover - environment-dependent import guard
    from PySide6 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - skip when Qt dependencies missing
    pytest.skip(f"PySide6 import failed: {exc}", allow_module_level=True)

from config import ViewerConfig
from core.edf_loader import ChannelInfo
from core.timebase import Timebase
from core.overscan import chunk_from_arrays


# Ensure the tests run with Qt's offscreen platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app():
    """Provide a global QApplication for headless UI tests."""

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
    app.quit()


@dataclass
class _DummyPrefetchCache:
    fetch: Callable[[int, float, float], tuple[np.ndarray, np.ndarray]]
    preview_fetch: Callable[[int, float, float], tuple[np.ndarray, np.ndarray]] | None
    requests: list[tuple[str, int, float, float]]

    def __init__(
        self,
        fetch: Callable[[int, float, float], tuple[np.ndarray, np.ndarray]],
        preview_fetch: Callable[[int, float, float], tuple[np.ndarray, np.ndarray]] | None,
    ):
        self.fetch = fetch
        self.preview_fetch = preview_fetch
        self.requests = []

    def start(self) -> None:  # pragma: no cover - interface compatibility
        return None

    def stop(self) -> None:  # pragma: no cover - interface compatibility
        return None

    def clear(self) -> None:
        self.requests.clear()

    def prefetch_window(self, channel: int, start: float, duration: float) -> None:
        if self.preview_fetch is not None:
            self.requests.append(("preview", channel, start, duration))
        self.requests.append(("final", channel, start, duration))


class _DummyPrefetchService:
    """Replace the threaded prefetcher with a deterministic stub."""

    def __init__(self) -> None:
        self.last_config: tuple[Optional[float], Optional[int], Optional[float]] | None = None
        self.last_cache: _DummyPrefetchCache | None = None

    def configure(
        self,
        *,
        tile_duration: float | None = None,
        max_tiles: int | None = None,
        max_mb: float | None = None,
    ) -> None:
        self.last_config = (tile_duration, max_tiles, max_mb)

    def create_cache(
        self,
        fetch: Callable[[int, float, float], tuple[np.ndarray, np.ndarray]],
        *,
        preview_fetch: Callable[[int, float, float], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> _DummyPrefetchCache:
        cache = _DummyPrefetchCache(fetch, preview_fetch)
        self.last_cache = cache
        return cache


class FakeHeadlessLoader:
    """Deterministic loader that mimics the EDF loader API for tests."""

    def __init__(self, *, duration_s: float = 30.0, base_fs: float = 50.0, n_channels: int = 3):
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if base_fs <= 0:
            raise ValueError("base_fs must be positive")
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")

        self.duration_s = float(duration_s)
        self.max_window_s = 120.0
        self.path = str(Path("tests") / "data" / "fake_headless.edf")
        self._timebase = Timebase(datetime(2024, 1, 1, 0, 0, 0), self.duration_s)
        self._fs = [base_fs + idx * 5.0 for idx in range(n_channels)]
        self.info: list[ChannelInfo] = []
        self._t_arrays: list[np.ndarray] = []
        self._x_arrays: list[np.ndarray] = []

        for idx in range(n_channels):
            fs = self._fs[idx]
            total_samples = int(np.ceil(self.duration_s * fs))
            t = np.arange(total_samples, dtype=np.float64) / fs
            phase = 0.25 * idx
            freq = 0.5 + idx * 0.25
            waveform = np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)
            self.info.append(
                ChannelInfo(
                    name=f"Synthetic {idx + 1}",
                    fs=fs,
                    n_samples=total_samples,
                    unit="µV",
                    raw_index=idx,
                )
            )
            self._t_arrays.append(t)
            self._x_arrays.append(waveform)

        self.n_channels = len(self.info)
        self.channel_count = self.n_channels

    @property
    def timebase(self) -> Timebase:
        return self._timebase

    def read(self, channel: int, start: float, end: float, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        if channel < 0 or channel >= self.n_channels:
            raise IndexError("channel out of range")
        t0, t1 = self._timebase.clamp_window(start, end)
        fs = self._fs[channel]
        s0, n = Timebase.sec_to_idx(t0, t1, fs)
        total = self.info[channel].n_samples
        if s0 >= total or n <= 0:
            empty_t = np.zeros(0, dtype=np.float64)
            empty_x = np.zeros(0, dtype=np.float32)
            return empty_t, empty_x
        s1 = min(total, s0 + n)
        t_slice = self._t_arrays[channel][s0:s1]
        x_slice = self._x_arrays[channel][s0:s1]
        max_samples = kwargs.get("max_samples")
        if isinstance(max_samples, (int, float)) and max_samples > 0 and x_slice.size > max_samples:
            step = max(1, int(np.ceil(x_slice.size / float(max_samples))))
            t_slice = t_slice[::step][: int(max_samples)]
            x_slice = x_slice[::step][: int(max_samples)]
        return chunk_from_arrays(t_slice, x_slice.astype(np.float32)).as_tuple()

    def read_annotations(self):  # pragma: no cover - unused, defensive stub
        return ([], [], [])

    def lod_levels(self, channel: int):  # pragma: no cover - no LOD data
        return []


def _process_events(app: QtWidgets.QApplication) -> None:
    app.processEvents(QtCore.QEventLoop.AllEvents, 100)


def test_main_window_headless_smoke(qt_app, monkeypatch, tmp_path):
    """Exercise the viewer in headless mode and capture a screenshot."""

    from ui import main_window as mw

    dummy_service = _DummyPrefetchService()
    monkeypatch.setattr(mw, "prefetch_service", dummy_service)

    loader = FakeHeadlessLoader(duration_s=24.0, base_fs=40.0, n_channels=3)
    config = ViewerConfig(
        prefetch_tile_s=2.0,
        prefetch_max_tiles=4,
        prefetch_max_mb=1.0,
        controls_collapsed=False,
        canvas_backend="pyqtgraph",
        lod_enabled=False,
    )

    window = mw.MainWindow(loader, config=config)
    window.show()
    _process_events(qt_app)

    window._shutdown_overscan_worker()

    initial_start = window._view_start
    initial_duration = window._view_duration

    refresh_durations: list[float] = []

    def _refresh_and_measure() -> None:
        start = QtCore.QElapsedTimer()
        start.start()
        window.refresh()
        _process_events(qt_app)
        refresh_durations.append(start.elapsed() / 1000.0)

    _refresh_and_measure()

    window.zoomOutBtn.click()
    _process_events(qt_app)
    _refresh_and_measure()

    window.panRightBtn.click()
    _process_events(qt_app)
    _refresh_and_measure()

    window.zoomInBtn.click()
    _process_events(qt_app)
    _refresh_and_measure()

    window.panLeftBtn.click()
    _process_events(qt_app)
    _refresh_and_measure()

    assert window._view_duration >= initial_duration
    assert window._view_start <= initial_start + window._view_duration

    cache = dummy_service.last_cache
    assert cache is not None
    assert cache.requests, "prefetch requests should be recorded"

    pixmap = window.plotLayout.grab()
    output_path = tmp_path / "headless_smoke.png"
    assert pixmap.save(str(output_path))
    assert output_path.exists() and output_path.stat().st_size > 0

    assert refresh_durations and max(refresh_durations) < 1.0

    window.close()
    _process_events(qt_app)


def test_main_window_overscan_preview_then_final(qt_app, monkeypatch):
    from ui import main_window as mw

    dummy_service = _DummyPrefetchService()
    monkeypatch.setattr(mw, "prefetch_service", dummy_service)

    class PreviewLoader(FakeHeadlessLoader):
        def __init__(self):
            super().__init__(duration_s=60.0, base_fs=60.0, n_channels=1)

        def lod_levels(self, channel: int):
            return [1.0, 8.0]

        def read_lod_window(self, channel: int, start: float, end: float, duration: float):
            bins = int(np.ceil((end - start) / duration))
            data = np.empty((bins, 2), dtype=np.float32)
            data[:, 0] = -0.25
            data[:, 1] = 0.25
            return data, duration, int(start // duration)

        def read(self, channel: int, start: float, end: float, **kwargs):
            # Simulate heavier raw reads so previews arrive first.
            time.sleep(0.05)
            return super().read(channel, start, end, **kwargs)

    loader = PreviewLoader()
    config = ViewerConfig(
        prefetch_tile_s=1.0,
        prefetch_max_tiles=2,
        prefetch_max_mb=1.0,
        controls_collapsed=False,
        canvas_backend="pyqtgraph",
        lod_enabled=True,
        lod_min_bin_multiple=2.0,
        lod_min_view_duration_s=0.0,
    )

    window = mw.MainWindow(loader, config=config)
    window.show()

    stages: dict[int, list[bool]] = {}
    sample_counts: dict[int, list[int]] = {}

    def capture(request_id: int, tile):
        stages.setdefault(request_id, []).append(bool(getattr(tile, "is_final", True)))
        if tile.channel_data:
            count = int(tile.channel_data[0][0].size)
        else:
            count = 0
        sample_counts.setdefault(request_id, []).append(count)

    window._overscan_worker.finished.connect(capture)  # type: ignore[arg-type]

    timer = QtCore.QElapsedTimer()
    timer.start()
    matched_id: int | None = None

    while timer.elapsed() < 4000:
        _process_events(qt_app)
        for req_id, seen in stages.items():
            if any(not flag for flag in seen) and any(flag for flag in seen):
                matched_id = req_id
                break
        if matched_id is not None:
            break
    assert matched_id is not None, "Timed out waiting for preview and final overscan tiles"

    seen_stages = stages[matched_id]
    assert seen_stages[0] is False, "Preview should arrive before refinement"
    assert seen_stages[-1] is True
    counts = sample_counts[matched_id]
    assert counts and counts[-1] >= counts[0]

    window._overscan_worker.finished.disconnect(capture)  # type: ignore[arg-type]
    window._shutdown_overscan_worker()
    window.close()
    _process_events(qt_app)
