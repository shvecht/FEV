# ui/main_window.py
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from collections import Counter, OrderedDict
from functools import partial
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui

from ui.time_axis import TimeAxis
from ui.widgets import CollapsibleSection
from ui.themes import DEFAULT_THEME, THEMES, ThemeDefinition
from ui.hover_utils import _sample_at_time
from ui.gpu_canvas import VispyChannelCanvas
from config import ViewerConfig
from core.decimate import min_max_bins
from core.overscan import (
    SignalChunk,
    chunk_from_arrays,
    envelope_to_series,
    select_lod_duration,
    slice_and_decimate,
    choose_lod_duration,
)
from core.prefetch import prefetch_service
from core.view_window import WindowLimits, clamp_window, pan_window, zoom_window
from core.edf_loader import EdfLoader
from core.zarr_cache import EdfToZarr, resolve_output_path
from core.zarr_loader import ZarrLoader
from core import annotations as annotation_core


LOG = logging.getLogger(__name__)

STAGE_COLORS: dict[str, str] = {
    "Wake": "#c47c00",
    "N1": "#c1a146",
    "N2": "#3f8e5a",
    "N3": "#2c7f8d",
    "REM": "#c2556f",
}
DEFAULT_STAGE_COLOR = "#4c5d73"

STAGE_TEXT_MARGIN = 26.0
PANEL_MAX_WIDTH = 400  # hard cap for the controls sidebar width


class _ZarrIngestWorker(QtCore.QObject):
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, edf_path: str, out_path: Path, loader=None):
        super().__init__()
        self._edf_path = edf_path
        self._out_path = Path(out_path)
        self._loader = loader

    @QtCore.Slot()
    def run(self):
        try:
            kwargs = {
                "edf_path": self._edf_path,
                "out_path": str(self._out_path),
                "progress_callback": self._handle_progress,
            }
            if self._loader is not None:
                kwargs["loader_factory"] = lambda _path: self._loader
                kwargs["owns_loader"] = False
            builder = EdfToZarr(**kwargs)
            builder.build()

            # Simple parity assertion: ensure loader can open and read metadata
            z_loader = ZarrLoader(self._out_path)
            try:
                assert z_loader.n_channels > 0
            finally:
                z_loader.close()

            self.finished.emit(str(self._out_path))
        except Exception as exc:  # pragma: no cover - UI feedback
            self.failed.emit(str(exc))

    def _handle_progress(self, done: int, total: int):
        self.progress.emit(done, total)


@dataclass(frozen=True)
class _OverscanRequest:
    request_id: int
    start: float
    end: float
    view_start: float
    view_duration: float
    channel_indices: tuple[int, ...]
    max_samples: Optional[int]


@dataclass
class _OverscanTile:
    request_id: int
    start: float
    end: float
    view_start: float
    view_duration: float
    raw_channel_data: list[SignalChunk]
    channel_data: list[tuple[np.ndarray, np.ndarray]]
    channel_indices: tuple[int, ...] = field(default_factory=tuple)
    vertex_data: list[np.ndarray] = field(default_factory=list)
    max_samples: Optional[int] = None
    pixel_budget: Optional[int] = None
    prepared_mask: list[bool] = field(default_factory=list)
    lod_durations: list[Optional[float]] = field(default_factory=list)
    prepared_cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    vertex_cache: dict[int, list[np.ndarray]] = field(default_factory=dict)
    is_final: bool = True

    def contains(self, window_start: float, window_end: float) -> bool:
        return window_start >= self.start and window_end <= self.end

    def prepared_for_budget(
        self, budget: int
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        if budget is None:
            return None
        return self.prepared_cache.get(int(budget))

    def cache_prepared(
        self,
        budget: int,
        series: list[tuple[np.ndarray, np.ndarray]],
        vertices: list[np.ndarray] | None,
    ) -> None:
        self.prepared_cache[int(budget)] = series
        if vertices is not None:
            self.vertex_cache[int(budget)] = vertices

    def prepared_vertices(self, budget: int) -> list[np.ndarray] | None:
        if budget is None:
            return None
        return self.vertex_cache.get(int(budget))

    def clear_prepared_caches(self) -> None:
        self.prepared_cache.clear()
        self.vertex_cache.clear()


class _OverscanWorker(QtCore.QObject):
    finished = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(
        self,
        loader,
        *,
        lod_enabled: bool = True,
        lod_min_bin_multiple: float = 2.0,
        lod_min_view_duration: float | None = None,
        lod_ratio: float | None = None,
    ):
        super().__init__()
        self._loader = loader
        self._lod_enabled = bool(lod_enabled)
        self._lod_min_bin_multiple = max(1.0, float(lod_min_bin_multiple))
        if lod_min_view_duration is None:
            self._lod_min_view_duration: float | None = None
        else:
            try:
                min_view = float(lod_min_view_duration)
            except (TypeError, ValueError):
                min_view = 0.0
            self._lod_min_view_duration = min_view if min_view > 0.0 else None
        self._lod_ratio = float(lod_ratio or 0.0)
        self._preview_sample_cap = 4096

    @QtCore.Slot(object)
    def render(self, request_obj):
        if not isinstance(request_obj, _OverscanRequest):
            return
        req: _OverscanRequest = request_obj
        preview_tile: _OverscanTile | None = None
        try:
            preview_tile = self._build_tile(req, preview=True)
        except Exception as exc:  # pragma: no cover - preview errors stay silent
            LOG.debug("Overscan preview failed: %s", exc)
            preview_tile = None
        if preview_tile is not None:
            self.finished.emit(req.request_id, preview_tile)

        try:
            final_tile = self._build_tile(req, preview=False)
        except Exception as exc:  # pragma: no cover - worker error propagated to UI
            self.failed.emit(req.request_id, str(exc))
            return
        self.finished.emit(req.request_id, final_tile)

    def _read_channel(
        self,
        channel: int,
        start: float,
        end: float,
        view_duration: float,
        max_samples: Optional[int],
    ) -> tuple[SignalChunk, Optional[float]]:
        lod_chunk = self._try_read_lod(channel, start, end, view_duration)
        if lod_chunk is not None:
            return lod_chunk, lod_chunk.lod_duration_s
        return self._read_raw_channel(channel, start, end, max_samples), None

    def _build_tile(self, req: _OverscanRequest, *, preview: bool) -> _OverscanTile | None:
        raw_chunks: list[SignalChunk] = []
        payloads: list[tuple[np.ndarray, np.ndarray]] = []
        lod_durations: list[Optional[float]] = []
        has_samples = False
        for ch in req.channel_indices:
            if preview:
                chunk = self._read_preview_channel(
                    ch,
                    req.start,
                    req.end,
                    req.view_duration,
                )
                lod_duration = chunk.lod_duration_s if chunk is not None else None
                if chunk is None:
                    empty_t = np.zeros(0, dtype=np.float64)
                    empty_x = np.zeros(0, dtype=np.float32)
                    chunk = chunk_from_arrays(
                        empty_t,
                        empty_x,
                        source_start=req.start,
                        source_end=req.end,
                    )
            else:
                chunk, lod_duration = self._read_channel(
                    ch,
                    req.start,
                    req.end,
                    req.view_duration,
                    req.max_samples,
                )
            raw_chunks.append(chunk)
            payload = chunk.as_tuple()
            payloads.append(payload)
            lod_durations.append(lod_duration)
            if payload[1].size > 0:
                has_samples = True

        if preview and not has_samples:
            return None

        tile = _OverscanTile(
            request_id=req.request_id,
            start=req.start,
            end=req.end,
            view_start=req.view_start,
            view_duration=req.view_duration,
            raw_channel_data=raw_chunks,
            channel_data=payloads,
            channel_indices=req.channel_indices,
            max_samples=req.max_samples,
            prepared_mask=[False] * len(raw_chunks),
            lod_durations=lod_durations,
            is_final=not preview,
        )
        return tile

    def _lod_durations_for_channel(self, channel: int) -> tuple[float, ...]:
        durations: Sequence[float] | None = None
        levels_fn = getattr(self._loader, "lod_levels", None)
        if callable(levels_fn):
            try:
                levels = levels_fn(channel)
            except Exception:
                levels = None
            if levels:
                if isinstance(levels, dict):
                    durations = tuple(float(k) for k in levels.keys())
                else:
                    durations = tuple(float(v) for v in levels)
        if not durations:
            durations_fn = getattr(self._loader, "lod_durations", None)
            if callable(durations_fn):
                try:
                    durations = tuple(float(v) for v in durations_fn(channel))
                except Exception:
                    durations = None
        if not durations:
            return tuple()
        filtered = [float(v) for v in durations if float(v) > 0.0]
        return tuple(sorted(filtered))

    def _read_preview_channel(
        self,
        channel: int,
        start: float,
        end: float,
        view_duration: float,
    ) -> SignalChunk | None:
        durations = self._lod_durations_for_channel(channel)
        if durations:
            min_view = self._lod_min_view_duration
            if min_view is None or view_duration >= min_view:
                coarse = durations[-1]
                chunk = self._read_lod_chunk(channel, start, end, coarse)
                if chunk is not None and chunk.x.size > 0:
                    return chunk
        return self._read_raw_channel(
            channel,
            start,
            end,
            min(self._preview_sample_cap, self._safe_max_samples(view_duration, channel)),
        )

    def _safe_max_samples(self, view_duration: float, channel: int) -> int:
        fs = getattr(self._loader, "channel_fs", None)
        if callable(fs):
            try:
                rate = float(fs(channel))
            except Exception:
                rate = 0.0
        else:
            rate = float(getattr(self._loader, "fs", 0.0) or 0.0)
        if rate <= 0.0:
            return self._preview_sample_cap
        return int(min(self._preview_sample_cap, max(1, int(math.ceil(rate * view_duration)))))

    def _read_raw_channel(
        self,
        channel: int,
        start: float,
        end: float,
        max_samples: Optional[int],
    ) -> SignalChunk:
        try:
            if max_samples is not None:
                result = self._loader.read(channel, start, end, max_samples=max_samples)
            else:
                result = self._loader.read(channel, start, end)
        except TypeError:
            result = self._loader.read(channel, start, end)
        if isinstance(result, SignalChunk):
            return result
        t_arr, x_arr = result
        t_np = np.asarray(t_arr, dtype=np.float64)
        x_np = np.asarray(x_arr, dtype=np.float32)
        source_start = float(t_np[0]) if t_np.size else start
        source_end = float(t_np[-1]) if t_np.size else end
        return chunk_from_arrays(
            t_np,
            x_np,
            source_start=source_start,
            source_end=source_end,
        )

    def _try_read_lod(
        self,
        channel: int,
        start: float,
        end: float,
        view_duration: float,
    ) -> SignalChunk | None:
        if not self._lod_enabled:
            return None
        lod_duration = self._select_lod(channel, view_duration)
        if lod_duration is None:
            return None
        return self._read_lod_chunk(channel, start, end, lod_duration)

    def _select_lod(self, channel: int, view_duration: float) -> Optional[float]:
        durations = self._lod_durations_for_channel(channel)
        if not durations:
            return None
        min_view = self._lod_min_view_duration
        if min_view is not None and view_duration < min_view:
            return None
        ratio = self._lod_ratio
        if ratio > 0 and np.isfinite(ratio):
            selected = choose_lod_duration(view_duration, durations, ratio=ratio)
            if selected is not None:
                return selected
        return select_lod_duration(
            view_duration,
            durations,
            self._lod_min_bin_multiple,
            min_view_duration=self._lod_min_view_duration,
        )

    def _read_lod_chunk(
        self,
        channel: int,
        start: float,
        end: float,
        duration: float,
    ) -> SignalChunk | None:
        read_window = getattr(self._loader, "read_lod_window", None)
        if not callable(read_window):
            return None
        try:
            result = read_window(channel, start, end, duration)
        except KeyError:
            return None
        except Exception:
            return None
        if isinstance(result, SignalChunk):
            chunk = result
        else:
            data, bin_duration, start_bin = result
            if getattr(data, "size", 0) == 0:
                return None
            mins = np.asarray(data[:, 0], dtype=np.float32)
            maxs = np.asarray(data[:, 1], dtype=np.float32)
            t_vals, x_vals = envelope_to_series(
                mins,
                maxs,
                bin_duration=float(bin_duration),
                start_bin=int(start_bin),
                window_start=start,
                window_end=end,
            )
            if t_vals.size == 0:
                return None
            chunk = chunk_from_arrays(
                np.asarray(t_vals, dtype=np.float64),
                np.asarray(x_vals, dtype=np.float32),
                lod_duration_s=float(bin_duration),
                source_start=start,
                source_end=end,
            )
        if chunk.lod_duration_s is None:
            chunk = chunk_from_arrays(
                np.asarray(chunk.t, dtype=np.float64),
                np.asarray(chunk.x, dtype=np.float32),
                lod_duration_s=float(duration),
                source_start=chunk.source_start,
                source_end=chunk.source_end,
            )
        return chunk


class MainWindow(QtWidgets.QMainWindow):
    overscanRequested = QtCore.Signal(object)
    def __init__(self, loader, *, config: ViewerConfig | None = None):
        super().__init__()
        self.loader = loader
        self._config = config or ViewerConfig()
        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []
        self.channel_labels: list[pg.LabelItem] = []
        self._cpu_plot_widget: pg.GraphicsLayoutWidget | None = None
        self._plot_scroll: QtWidgets.QScrollArea | None = None
        self._plot_container: QtWidgets.QWidget | None = None
        self._renderer_badge: QtWidgets.QLabel | None = None
        self._annotation_control_states: dict[QtWidgets.QWidget, bool] = {}
        self._annotation_control_tooltips: dict[QtWidgets.QWidget, str] = {}
        self._plot_viewport_filter_installed = False
        self._ingest_thread: QtCore.QThread | None = None
        self._ingest_worker: _ZarrIngestWorker | None = None
        self._zarr_path: Path | None = None
        self._pending_loader: object | None = None
        self._primary_viewbox = None
        self._splitter: QtWidgets.QSplitter | None = None
        self._control_wrapper: QtWidgets.QWidget | None = None
        self._control_scroll: QtWidgets.QScrollArea | None = None
        self._control_rail: QtWidgets.QWidget | None = None
        self._limits = WindowLimits(
            duration_min=0.25,
            duration_max=float(getattr(loader, "max_window_s", 120.0)),
        )
        self._view_start, self._view_duration = clamp_window(
            0.0,
            min(30.0, loader.duration_s),
            total=loader.duration_s,
            limits=self._limits,
        )
        self._updating_viewbox = False
        self._maybe_build_int16_cache()
        prefetch_service.configure(
            tile_duration=self._config.prefetch_tile_s,
            max_tiles=self._config.prefetch_max_tiles,
            max_mb=self._config.prefetch_max_mb,
        )
        self._prefetch = prefetch_service.create_cache(
            self._fetch_tile, preview_fetch=self._fetch_tile_preview
        )
        self._prefetch.start()

        theme_key = getattr(self._config, "theme", DEFAULT_THEME)
        if theme_key not in THEMES:
            theme_key = DEFAULT_THEME
        self._active_theme_key = theme_key
        self._theme: ThemeDefinition = THEMES[theme_key]
        self._config.theme = theme_key
        self._controls_collapsed = bool(getattr(self._config, "controls_collapsed", False))

        self._gpu_probe = VispyChannelCanvas.capability_probe()
        self._gpu_autoswitch_enabled = bool(self._gpu_probe.available)
        self._gpu_failure_reason: str | None = (
            self._gpu_probe.reason if not self._gpu_probe.available else None
        )
        base_budget = self._gpu_probe.vertex_budget or VispyChannelCanvas.DEFAULT_VERTEX_BUDGET
        self._gpu_vertex_promote_threshold = max(300_000, int(base_budget * 0.55))
        self._gpu_vertex_demote_threshold = max(150_000, int(self._gpu_vertex_promote_threshold * 0.55))
        if self._gpu_vertex_demote_threshold >= self._gpu_vertex_promote_threshold:
            self._gpu_vertex_demote_threshold = max(150_000, int(self._gpu_vertex_promote_threshold * 0.5))
        self._gpu_promote_counter = 0
        self._gpu_demote_counter = 0
        self._gpu_last_vertex_count = 0
        self._gpu_switch_in_progress = False

        requested_backend = str(getattr(self._config, "canvas_backend", "pyqtgraph"))
        self._requested_canvas_backend = requested_backend.lower()
        self._gpu_forced = self._requested_canvas_backend in {"vispy", "gpu", "gpu-vispy"}
        self._gpu_canvas: VispyChannelCanvas | None = None
        self._gpu_init_error: str | None = None
        self._use_gpu_canvas = False
        if self._gpu_forced:
            if self._ensure_gpu_canvas_created():
                self._use_gpu_canvas = True
                self._config.canvas_backend = "vispy"
                self._gpu_failure_reason = None
            else:
                self._config.canvas_backend = "pyqtgraph"
                self._gpu_autoswitch_enabled = False
        else:
            if not self._gpu_probe.available and self._gpu_probe.reason:
                self._gpu_failure_reason = self._gpu_probe.reason

        pg.setConfigOptions(antialias=True)

        self.setWindowTitle("EDF Viewer — Multi-channel")
        self.time_axis = TimeAxis(orientation="bottom", timebase=loader.timebase)
        if self.loader.timebase.start_dt is not None:
            self.time_axis.set_mode("absolute")

        self._primary_plot = None
        self._hover_line: pg.InfiniteLine | None = None
        self._hover_label: pg.TextItem | None = None
        self._hover_plot: pg.PlotItem | None = None
        self._hover_label_anchor: tuple[float, float] = (0.0, 1.0)
        self._plot_viewport: QtWidgets.QWidget | None = None
        self._overscan_factor = 2.0  # windows per side
        self._overscan_zoom_reuse_ratio = 0.7
        self._overscan_tile: _OverscanTile | None = None
        self._overscan_request_id = 0
        self._overscan_inflight: Optional[int] = None
        self._current_tile_id: Optional[int] = None
        self._overscan_tile_cache: OrderedDict[
            tuple[tuple[int, ...], int, int], _OverscanTile
        ] = OrderedDict()
        self._overscan_tile_cache_limit = 6
        self._lod_enabled = bool(getattr(self._config, "lod_enabled", True))
        self._lod_min_bin_multiple = max(
            1.0, float(getattr(self._config, "lod_min_bin_multiple", 2.0) or 2.0)
        )
        raw_min_view = getattr(self._config, "lod_min_view_duration_s", 240.0)
        try:
            min_view = float(raw_min_view)
        except (TypeError, ValueError):
            min_view = 0.0
        self._lod_min_view_duration = min_view if min_view > 0.0 else None
        self._config.lod_min_view_duration_s = min_view if min_view > 0.0 else 0.0
        self._overscan_thread: QtCore.QThread | None = None
        self._overscan_worker: _OverscanWorker | None = None
        self._init_overscan_worker()

        self._hidden_channels: set[int] = set(getattr(self._config, "hidden_channels", ()))
        self._auto_hide_annotation_channels()
        hidden_ann = getattr(self._config, "hidden_annotation_channels", ())
        self._hidden_annotation_channels: set[str] = {
            str(name).strip() for name in hidden_ann if str(name).strip()
        }
        self._hidden_annotation_channels.discard(annotation_core.STAGE_CHANNEL)
        self._manual_annotation_paths: dict[str, Path] = {}
        self._annotations_index: annotation_core.AnnotationIndex | None = None
        self._annotation_lines: list[pg.InfiniteLine] = []
        self._annotations_enabled = False
        self._annotation_rects: list[QtWidgets.QGraphicsRectItem] = []
        self._stage_label_items: list[QtWidgets.QGraphicsSimpleTextItem] = []
        self._all_event_records: list[dict[str, float | str | int]] = []
        self._event_records: list[dict[str, float | str | int]] = []
        self._current_event_index: int = -1
        self._current_event_id: Optional[int] = None
        self._event_color_cache: dict[str, QtGui.QColor] = {}
        self._selected_event_channel: str | None = None
        self._event_label_filter: str = ""
        self._annotation_channel_toggles: dict[str, QtWidgets.QCheckBox] = {}

        self.hypnogramPlot: pg.PlotItem | None = None
        self._hypnogram_label: pg.LabelItem | None = None
        self.hypnogramRegion: pg.LinearRegionItem | None = None
        self._hypnogram_outline: pg.PlotDataItem | None = None
        self._hypnogram_fill_curves: dict[str, pg.PlotDataItem] = {}
        self._stage_curve_cache: dict[str, object] | None = None
        self._updating_hypnogram_region = False

        self._build_ui()
        self._update_annotation_channel_toggles()
        self._apply_theme(self._active_theme_key, persist=False)
        focus_only_pref = bool(getattr(self._config, "annotation_focus_only", False))
        self.annotationFocusOnly.setChecked(focus_only_pref)
        self._connect_signals()
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(60)
        self._debounce_timer.timeout.connect(self.refresh)
        self._event_filter_timer = QtCore.QTimer(self)
        self._event_filter_timer.setSingleShot(True)
        self._event_filter_timer.setInterval(120)
        self._event_filter_timer.timeout.connect(self._apply_event_filters)
        self._refresh_limits()
        self._update_controls_from_state()
        self.refresh()
        self._update_data_source_label()
        self._manual_annotation_paths.clear()
        self._start_zarr_ingest()
        self._load_companion_annotations()
        QtCore.QTimer.singleShot(0, self._ensure_overscan_for_view)
        if self._gpu_init_error:
            QtCore.QTimer.singleShot(
                0,
                lambda: QtWidgets.QMessageBox.warning(
                    self,
                    "GPU canvas unavailable",
                    "Falling back to CPU renderer: " + str(self._gpu_init_error),
                ),
            )

    # ----- UI construction -------------------------------------------------

    def _build_ui(self):
        self.startSpin = QtWidgets.QDoubleSpinBox()
        self.startSpin.setDecimals(3)
        self.startSpin.setSingleStep(0.5)
        self.startSpin.setSuffix(" s")

        self.windowSpin = QtWidgets.QDoubleSpinBox()
        self.windowSpin.setDecimals(1)
        self.windowSpin.setRange(1.0, 120.0)
        self.windowSpin.setSingleStep(1.0)
        self.windowSpin.setValue(30.0)
        self.windowSpin.setSuffix(" s")

        self.absoluteRange = QtWidgets.QLabel("--:--:-- – --:--:--")
        self.absoluteRange.setObjectName("absoluteRange")
        self.windowSummary = QtWidgets.QLabel("Window: 30.0 s")
        self.windowSummary.setObjectName("windowSummary")
        self.stageSummaryLabel = QtWidgets.QLabel("Stage: -- | Position: -- | Events: 0")
        self.stageSummaryLabel.setObjectName("stageSummary")
        self.stageSummaryLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.sourceLabel = QtWidgets.QLabel("Source: EDF (live)")
        self.sourceLabel.setObjectName("sourceLabel")
        self.annotationToggle = QtWidgets.QCheckBox("Show annotations")
        self.annotationToggle.setChecked(True)
        self.annotationToggle.setEnabled(False)
        self.annotationFocusOnly = QtWidgets.QCheckBox("Show only selected event")
        self.annotationFocusOnly.setChecked(False)
        self.annotationFocusOnly.setEnabled(False)
        self.annotationPositionToggle = QtWidgets.QCheckBox("Show body position")
        self.annotationPositionToggle.setChecked(False)
        self.annotationPositionToggle.setEnabled(False)
        self.annotationImportBtn = QtWidgets.QPushButton("Import annotations…")
        self.annotationImportBtn.setEnabled(True)
        self.eventChannelFilter = QtWidgets.QComboBox()
        self.eventChannelFilter.setEnabled(False)
        self.eventChannelFilter.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.eventChannelFilter.addItem("All channels", userData=None)
        self.eventSearchEdit = QtWidgets.QLineEdit()
        self.eventSearchEdit.setPlaceholderText("Search labels…")
        self.eventSearchEdit.setClearButtonEnabled(True)
        self.eventSearchEdit.setEnabled(False)
        self.eventList = QtWidgets.QListWidget()
        self.eventList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.eventList.setEnabled(False)
        self.eventPrevBtn = QtWidgets.QPushButton("Prev")
        self.eventPrevBtn.setEnabled(False)
        self.eventNextBtn = QtWidgets.QPushButton("Next")
        self.eventNextBtn.setEnabled(False)

        control = QtWidgets.QFrame()
        control.setObjectName("controlPanel")
        control.setMinimumWidth(50)
        control.setMaximumWidth(PANEL_MAX_WIDTH)
        control.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        controlLayout = QtWidgets.QVBoxLayout(control)
        controlLayout.setContentsMargins(18, 18, 18, 18)
        controlLayout.setSpacing(14)

        self.panelCollapseBtn = QtWidgets.QToolButton()
        self.panelCollapseBtn.setObjectName("panelCollapseBtn")
        self.panelCollapseBtn.setAutoRaise(True)
        self.panelCollapseBtn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.panelCollapseBtn.setArrowType(QtCore.Qt.LeftArrow)
        self.panelCollapseBtn.setCursor(QtCore.Qt.PointingHandCursor)
        self.panelCollapseBtn.setToolTip("Collapse controls")
        headerLayout = QtWidgets.QHBoxLayout()
        headerLayout.setContentsMargins(0, 0, 0, 0)
        headerLayout.setSpacing(6)
        headerLayout.addStretch(1)
        headerLayout.addWidget(self.panelCollapseBtn)
        controlLayout.addLayout(headerLayout)
        self.controlPanel = control

        navLayout = QtWidgets.QHBoxLayout()
        navLayout.setSpacing(6)
        self.panLeftBtn = QtWidgets.QToolButton()
        self.panLeftBtn.setText("◀")
        self.panRightBtn = QtWidgets.QToolButton()
        self.panRightBtn.setText("▶")
        self.zoomInBtn = QtWidgets.QToolButton()
        self.zoomInBtn.setText("+")
        self.zoomOutBtn = QtWidgets.QToolButton()
        self.zoomOutBtn.setText("−")
        self.fullViewBtn = QtWidgets.QToolButton()
        self.fullViewBtn.setText("All")
        self.resetViewBtn = QtWidgets.QToolButton()
        self.resetViewBtn.setText("Reset")
        for btn in (self.panLeftBtn, self.panRightBtn, self.zoomInBtn, self.zoomOutBtn, self.resetViewBtn, self.fullViewBtn):
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            navLayout.addWidget(btn)
        navLayout.addStretch(1)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.addWidget(QtWidgets.QLabel("Start (s)"), 0, 0)
        form.addWidget(self.startSpin, 0, 1)
        form.addWidget(QtWidgets.QLabel("Duration"), 1, 0)
        form.addWidget(self.windowSpin, 1, 1)

        self.fileButton = QtWidgets.QPushButton("Open EDF…")
        self.fileButton.setObjectName("fileSelectButton")
        self.fileButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.fileButton.setCursor(QtCore.Qt.PointingHandCursor)

        primaryControls = QtWidgets.QGroupBox("Viewing Controls")
        primaryControls.setObjectName("primaryControls")
        primaryLayout = QtWidgets.QVBoxLayout(primaryControls)
        primaryLayout.setContentsMargins(14, 16, 14, 12)
        primaryLayout.setSpacing(12)
        primaryLayout.addLayout(navLayout)
        fileRow = QtWidgets.QHBoxLayout()
        fileRow.addWidget(self.fileButton)
        fileRow.addStretch(1)
        primaryLayout.addLayout(fileRow)
        primaryLayout.addLayout(form)

        channelContent = QtWidgets.QWidget()
        channelContentLayout = QtWidgets.QVBoxLayout(channelContent)
        channelContentLayout.setContentsMargins(0, 0, 0, 0)
        channelContentLayout.setSpacing(6)
        self.channel_checkboxes: list[QtWidgets.QCheckBox] = []
        self._channel_list_layout = channelContentLayout
        channelContentLayout.addStretch(1)
        self.channelSection = CollapsibleSection("Channels", channelContent, expanded=True)
        self.channelSection.setObjectName("channelSection")

        telemetryBar = QtWidgets.QFrame()
        telemetryBar.setObjectName("telemetryBar")
        telemetryBar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        telemetryLayout = QtWidgets.QHBoxLayout(telemetryBar)
        telemetryLayout.setContentsMargins(12, 6, 12, 6)
        telemetryLayout.setSpacing(8)
        telemetryLayout.addWidget(self.absoluteRange)
        telemetryLayout.addWidget(self.windowSummary)
        telemetryLayout.addStretch(1)
        telemetryLayout.addWidget(self.stageSummaryLabel)

        annotationSection = QtWidgets.QGroupBox("Annotations")
        annotationSection.setObjectName("annotationSection")
        annotationSection.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        annotationLayout = QtWidgets.QVBoxLayout(annotationSection)
        annotationLayout.setContentsMargins(14, 16, 14, 12)
        annotationLayout.setSpacing(10)
        annotationLayout.addWidget(self.annotationToggle)
        annotationLayout.addWidget(self.annotationFocusOnly)
        annotationLayout.addWidget(self.annotationPositionToggle)
        annotationLayout.addWidget(self.annotationImportBtn)
        filterRow = QtWidgets.QHBoxLayout()
        filterRow.setSpacing(6)
        filterRow.addWidget(self.eventChannelFilter)
        filterRow.addWidget(self.eventSearchEdit)
        annotationLayout.addLayout(filterRow)
        annotationLayout.addWidget(self.eventList)
        eventNav = QtWidgets.QHBoxLayout()
        eventNav.setSpacing(8)
        eventNav.addWidget(self.eventPrevBtn)
        eventNav.addWidget(self.eventNextBtn)
        annotationLayout.addLayout(eventNav)

        if self._use_gpu_canvas:
            note = "Annotation overlays are unavailable in GPU mode"
            for widget in (
                self.annotationToggle,
                self.annotationFocusOnly,
                self.annotationPositionToggle,
            ):
                widget.setEnabled(False)
                widget.setToolTip(note)

        controlLayout.addWidget(primaryControls)
        controlLayout.addWidget(self.channelSection)
        controlLayout.addWidget(telemetryBar)
        controlLayout.addWidget(self.sourceLabel)
        controlLayout.addWidget(annotationSection)

        self._annotation_channel_toggles = {
            annotation_core.POSITION_CHANNEL: self.annotationPositionToggle,
        }

        appearanceContent = QtWidgets.QWidget()
        appearanceLayout = QtWidgets.QVBoxLayout(appearanceContent)
        appearanceLayout.setContentsMargins(0, 0, 0, 0)
        appearanceLayout.setSpacing(10)
        themeRow = QtWidgets.QHBoxLayout()
        themeRow.setSpacing(6)
        themeLabel = QtWidgets.QLabel("Theme")
        self.themeCombo = QtWidgets.QComboBox()
        self.themeCombo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        for key, theme_def in THEMES.items():
            self.themeCombo.addItem(theme_def.name, userData=key)
        with QtCore.QSignalBlocker(self.themeCombo):
            idx = self.themeCombo.findData(self._active_theme_key)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.themeCombo.setCurrentIndex(idx)
        themeRow.addWidget(themeLabel)
        themeRow.addWidget(self.themeCombo, 1)
        appearanceLayout.addLayout(themeRow)

        self.themePreviewWidget = QtWidgets.QWidget()
        previewLayout = QtWidgets.QHBoxLayout(self.themePreviewWidget)
        previewLayout.setContentsMargins(0, 0, 0, 0)
        previewLayout.setSpacing(6)
        appearanceLayout.addWidget(self.themePreviewWidget)
        appearanceLayout.addStretch(1)

        self.appearanceSection = CollapsibleSection(
            "Appearance",
            appearanceContent,
            expanded=False,
        )
        self.appearanceSection.setObjectName("appearanceSection")
        controlLayout.addWidget(self.appearanceSection)

        prefetchContent = QtWidgets.QWidget()
        prefetchLayout = QtWidgets.QGridLayout(prefetchContent)
        prefetchLayout.setContentsMargins(0, 0, 0, 0)
        prefetchLayout.setHorizontalSpacing(8)
        prefetchLayout.setVerticalSpacing(6)
        self.prefetchTileSpin = QtWidgets.QDoubleSpinBox()
        self.prefetchTileSpin.setRange(0.5, 300.0)
        self.prefetchTileSpin.setDecimals(2)
        self.prefetchTileSpin.setValue(self._config.prefetch_tile_s)
        self.prefetchMaxTilesSpin = QtWidgets.QSpinBox()
        self.prefetchMaxTilesSpin.setRange(1, 4096)
        self.prefetchMaxTilesSpin.setValue(self._config.prefetch_max_tiles or 64)
        self.prefetchMaxMbSpin = QtWidgets.QDoubleSpinBox()
        self.prefetchMaxMbSpin.setRange(1.0, 4096.0)
        self.prefetchMaxMbSpin.setDecimals(1)
        self.prefetchMaxMbSpin.setValue(self._config.prefetch_max_mb or 16.0)
        self.prefetchApplyBtn = QtWidgets.QPushButton("Apply Prefetch")
        prefetchLayout.addWidget(QtWidgets.QLabel("Tile (s)"), 0, 0)
        prefetchLayout.addWidget(self.prefetchTileSpin, 0, 1)
        prefetchLayout.addWidget(QtWidgets.QLabel("Max tiles"), 1, 0)
        prefetchLayout.addWidget(self.prefetchMaxTilesSpin, 1, 1)
        prefetchLayout.addWidget(QtWidgets.QLabel("Max MB"), 2, 0)
        prefetchLayout.addWidget(self.prefetchMaxMbSpin, 2, 1)
        prefetchLayout.addWidget(self.prefetchApplyBtn, 3, 0, 1, 2)
        self.prefetchSection = CollapsibleSection(
            "Prefetch",
            prefetchContent,
            expanded=not getattr(self._config, "prefetch_collapsed", False),
        )
        self.prefetchSection.setObjectName("prefetchSection")
        controlLayout.addWidget(self.prefetchSection)
        self.ingestBar = QtWidgets.QProgressBar()
        self.ingestBar.setObjectName("ingestBar")
        self.ingestBar.setRange(0, 100)
        self.ingestBar.setValue(0)
        self.ingestBar.setFormat("Caching EDF → Zarr: %p%")
        self.ingestBar.setTextVisible(True)
        self.ingestBar.hide()
        controlLayout.addWidget(self.ingestBar)
        controlLayout.addStretch(1)

        if self._use_gpu_canvas and self._ensure_gpu_canvas_created():
            self.plotLayout = self._gpu_canvas
            self.plotLayout.setMinimumSize(0, 0)
            self.plotLayout.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            self._plot_viewport = self._gpu_canvas
            self._hover_line = None
            self._hover_label = None
            self._hover_plot = None
            self._configure_gpu_canvas()
        else:
            self._use_gpu_canvas = False
            self.plotLayout = self._ensure_cpu_canvas()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMinimumSize(0, 0)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        scroll.setWidget(self.plotLayout)
        self._plot_scroll = scroll

        plot_container = QtWidgets.QWidget()
        container_layout = QtWidgets.QGridLayout(plot_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(scroll, 0, 0)
        self._renderer_badge = QtWidgets.QLabel("")
        self._renderer_badge.setObjectName("rendererBadge")
        self._renderer_badge.setAlignment(QtCore.Qt.AlignCenter)
        self._renderer_badge.setStyleSheet(
            "padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600;"
        )
        container_layout.addWidget(
            self._renderer_badge, 0, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight
        )
        self._plot_container = plot_container

        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        control_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding,
        )
        control_scroll.setMinimumWidth(0)
        control_scroll.setMaximumWidth(PANEL_MAX_WIDTH)
        control_scroll.setWidget(control)
        self._control_scroll = control_scroll

        self._control_rail = QtWidgets.QFrame()
        self._control_rail.setObjectName("controlRail")
        self._control_rail.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Expanding,
        )
        self._control_rail.setMinimumWidth(48)
        self._control_rail.setMaximumWidth(48)
        railLayout = QtWidgets.QVBoxLayout(self._control_rail)
        railLayout.setContentsMargins(4, 8, 4, 8)
        railLayout.setSpacing(6)

        def _make_icon_button(pixmap: QtWidgets.QStyle.StandardPixmap, tooltip: str) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
            btn.setIcon(self.style().standardIcon(pixmap))
            btn.setIconSize(QtCore.QSize(20, 20))
            btn.setAutoRaise(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            return btn

        self.railOpenBtn = _make_icon_button(QtWidgets.QStyle.SP_DialogOpenButton, "Open EDF…")
        self.railOpenBtn.clicked.connect(self._prompt_open_file)
        railLayout.addWidget(self.railOpenBtn)

        self.railAnnotationBtn = _make_icon_button(QtWidgets.QStyle.SP_DialogYesButton, "Toggle annotations")
        self.railAnnotationBtn.setCheckable(True)
        self.railAnnotationBtn.setChecked(self.annotationToggle.isChecked())
        self.railAnnotationBtn.toggled.connect(self.annotationToggle.setChecked)
        self.annotationToggle.toggled.connect(self.railAnnotationBtn.setChecked)
        railLayout.addWidget(self.railAnnotationBtn)

        railLayout.addSpacing(12)

        self.railPanLeftBtn = _make_icon_button(QtWidgets.QStyle.SP_ArrowBack, "Pan window left")
        self.railPanLeftBtn.clicked.connect(self.panLeftBtn.click)
        railLayout.addWidget(self.railPanLeftBtn)

        self.railPanRightBtn = _make_icon_button(QtWidgets.QStyle.SP_ArrowForward, "Pan window right")
        self.railPanRightBtn.clicked.connect(self.panRightBtn.click)
        railLayout.addWidget(self.railPanRightBtn)

        self.railZoomInBtn = _make_icon_button(QtWidgets.QStyle.SP_ArrowUp, "Zoom in")
        self.railZoomInBtn.clicked.connect(self.zoomInBtn.click)
        railLayout.addWidget(self.railZoomInBtn)

        self.railZoomOutBtn = _make_icon_button(QtWidgets.QStyle.SP_ArrowDown, "Zoom out")
        self.railZoomOutBtn.clicked.connect(self.zoomOutBtn.click)
        railLayout.addWidget(self.railZoomOutBtn)

        railLayout.addStretch(1)

        self.controlToggleBtn = QtWidgets.QToolButton()
        self.controlToggleBtn.setAutoRaise(True)
        self.controlToggleBtn.setCheckable(True)
        self.controlToggleBtn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.controlToggleBtn.setCursor(QtCore.Qt.PointingHandCursor)
        self.controlToggleBtn.setToolTip("Collapse controls")
        railLayout.addWidget(self.controlToggleBtn)

        control_wrapper = QtWidgets.QWidget()
        wrapperLayout = QtWidgets.QHBoxLayout(control_wrapper)
        wrapperLayout.setContentsMargins(0, 0, 0, 0)
        wrapperLayout.setSpacing(0)
        wrapperLayout.addWidget(self._control_rail)
        wrapperLayout.addWidget(control_scroll)
        self._control_wrapper = control_wrapper

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(control_wrapper)
        splitter.addWidget(plot_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([160, 1040])
        self._splitter = splitter

        central = QtWidgets.QWidget()
        centralLayout = QtWidgets.QVBoxLayout(central)
        centralLayout.setContentsMargins(0, 0, 0, 0)
        centralLayout.setSpacing(0)
        centralLayout.addWidget(splitter)
        self.setCentralWidget(central)

        with QtCore.QSignalBlocker(self.controlToggleBtn):
            self.controlToggleBtn.setChecked(self._controls_collapsed)
        self._update_control_toggle_icon(self._controls_collapsed)
        self._apply_control_panel_state(self._controls_collapsed)
        QtCore.QTimer.singleShot(0, lambda: self._apply_control_panel_state(self._controls_collapsed))
        self._update_annotation_controls_for_renderer()
        self._update_renderer_indicator()

    def _ensure_gpu_canvas_created(self) -> bool:
        if self._gpu_canvas is not None:
            return True
        try:
            self._gpu_canvas = VispyChannelCanvas()
        except Exception as exc:  # pragma: no cover - optional GPU path
            self._gpu_init_error = str(exc)
            self._gpu_canvas = None
            self._gpu_autoswitch_enabled = False
            self._gpu_failure_reason = str(exc)
            LOG.warning("GPU canvas unavailable: %s", exc)
            return False
        return True

    def _ensure_cpu_canvas(self) -> pg.GraphicsLayoutWidget:
        if self._cpu_plot_widget is None:
            widget = pg.GraphicsLayoutWidget()
            widget.setMinimumSize(0, 0)
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            widget.ci.layout.setSpacing(0)
            widget.ci.layout.setContentsMargins(0, 0, 0, 0)

            self._hover_line = pg.InfiniteLine(angle=90, movable=False)
            self._hover_line.setZValue(1000)
            self._hover_line.setVisible(False)
            self._hover_line.setPen(
                pg.mkPen((240, 240, 240, 220), width=1.0, style=QtCore.Qt.DashLine)
            )
            self._hover_label_anchor = (0.0, 1.0)
            self._hover_label = pg.TextItem(anchor=self._hover_label_anchor)
            self._hover_label.setZValue(1001)
            self._hover_label.setColor((240, 240, 240))
            self._hover_label.setVisible(False)
            self._hover_plot = None

            scene = widget.scene()
            scene.sigMouseMoved.connect(self._update_hover_indicator)
            mouse_exited = getattr(scene, "sigMouseExited", None)
            if mouse_exited is not None:
                mouse_exited.connect(self._on_plot_scene_mouse_exited)

            self._cpu_plot_widget = widget
            self.plots = []
            self.curves = []
            self.channel_labels = []
            self._plot_viewport_filter_installed = False
        else:
            widget = self._cpu_plot_widget

        self._plot_viewport = widget.viewport()
        if self._plot_viewport is not None and not self._plot_viewport_filter_installed:
            self._plot_viewport.installEventFilter(self)
            self._plot_viewport_filter_installed = True

        self.plotLayout = widget
        self._ensure_plot_rows(self.loader.n_channels)
        self._configure_plots()
        return widget

    def _update_annotation_controls_for_renderer(self) -> None:
        widgets = [
            getattr(self, "annotationToggle", None),
            getattr(self, "annotationFocusOnly", None),
            getattr(self, "annotationPositionToggle", None),
        ]
        widgets = [w for w in widgets if w is not None]
        if self._use_gpu_canvas:
            note = "Annotation overlays are unavailable in GPU mode"
            for widget in widgets:
                self._annotation_control_states[widget] = widget.isEnabled()
                self._annotation_control_tooltips[widget] = widget.toolTip()
                widget.setEnabled(False)
                widget.setToolTip(note)
            rail_toggle = getattr(self, "railAnnotationBtn", None)
            if rail_toggle is not None:
                self._annotation_control_states[rail_toggle] = rail_toggle.isEnabled()
                self._annotation_control_tooltips[rail_toggle] = rail_toggle.toolTip()
                rail_toggle.setEnabled(False)
                rail_toggle.setToolTip(note)
        else:
            for widget in widgets:
                previous_enabled = self._annotation_control_states.pop(widget, widget.isEnabled())
                widget.setEnabled(previous_enabled)
                previous_tip = self._annotation_control_tooltips.pop(widget, "")
                if previous_tip:
                    widget.setToolTip(previous_tip)
            rail_toggle = getattr(self, "railAnnotationBtn", None)
            if rail_toggle is not None:
                previous_enabled = self._annotation_control_states.pop(
                    rail_toggle, rail_toggle.isEnabled()
                )
                rail_toggle.setEnabled(previous_enabled)
                previous_tip = self._annotation_control_tooltips.pop(rail_toggle, "")
                if previous_tip:
                    rail_toggle.setToolTip(previous_tip)
            self._annotation_control_states.clear()
            self._annotation_control_tooltips.clear()

    def _update_renderer_indicator(self) -> None:
        if self._renderer_badge is None:
            return
        if self._use_gpu_canvas:
            self._renderer_badge.setText("GPU")
            style = (
                "color: #d6e4ff; background-color: rgba(70, 122, 216, 0.4);"
                "padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600;"
            )
            tooltip = "Active renderer: VisPy (GPU)"
        else:
            self._renderer_badge.setText("CPU")
            style = (
                "color: #f4d4aa; background-color: rgba(120, 86, 48, 0.45);"
                "padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600;"
            )
            tooltip = "Active renderer: pyqtgraph (CPU)"
        self._renderer_badge.setStyleSheet(style)
        if self._gpu_failure_reason:
            tooltip += f"\nGPU unavailable: {self._gpu_failure_reason}"
        elif not self._gpu_autoswitch_enabled and not self._use_gpu_canvas and self._gpu_probe.reason:
            tooltip += f"\nGPU disabled: {self._gpu_probe.reason}"
        if self._gpu_last_vertex_count:
            tooltip += f"\nLast vertex load: {self._gpu_last_vertex_count:,} vertices"
        self._renderer_badge.setToolTip(tooltip)

    def _switch_to_gpu(self) -> bool:
        if self._use_gpu_canvas:
            return True
        if not self._gpu_autoswitch_enabled and not self._gpu_forced:
            return False
        if not self._ensure_gpu_canvas_created():
            return False
        if self._plot_scroll is None:
            return False
        self._gpu_switch_in_progress = True
        try:
            current = self._plot_scroll.widget()
            if current is not None and current is not self._gpu_canvas:
                current.setParent(None)
            self._plot_scroll.setWidget(self._gpu_canvas)
            self.plotLayout = self._gpu_canvas
            self._plot_viewport = self._gpu_canvas
            self._use_gpu_canvas = True
            self._config.canvas_backend = "vispy"
            self._gpu_failure_reason = None
            self._update_annotation_controls_for_renderer()
            self._configure_gpu_canvas()
            self._update_renderer_indicator()
            self._update_data_source_label()
            return True
        finally:
            self._gpu_switch_in_progress = False

    def _switch_to_cpu(self) -> bool:
        if not self._use_gpu_canvas:
            return True
        if self._plot_scroll is None:
            return False
        widget = self._ensure_cpu_canvas()
        self._gpu_switch_in_progress = True
        try:
            current = self._plot_scroll.widget()
            if current is not None and current is not widget:
                current.setParent(None)
            self._plot_scroll.setWidget(widget)
            self.plotLayout = widget
            self._plot_viewport = widget.viewport()
            self._use_gpu_canvas = False
            if not self._gpu_forced:
                self._config.canvas_backend = "pyqtgraph"
            self._sync_hover_overlay_target()
            self._update_annotation_controls_for_renderer()
            self._update_renderer_indicator()
            self._update_data_source_label()
            return True
        finally:
            self._gpu_switch_in_progress = False

    def _maybe_autoswitch_renderer(self, vertex_count: int) -> None:
        self._gpu_last_vertex_count = max(0, int(vertex_count))
        if not self._gpu_autoswitch_enabled or self._gpu_switch_in_progress:
            return
        if self._use_gpu_canvas:
            self._gpu_promote_counter = 0
            if self._gpu_forced:
                return
            if vertex_count <= self._gpu_vertex_demote_threshold:
                self._gpu_demote_counter += 1
            else:
                self._gpu_demote_counter = 0
            if self._gpu_demote_counter >= 2:
                if self._switch_to_cpu():
                    self._gpu_demote_counter = 0
        else:
            self._gpu_demote_counter = 0
            if vertex_count >= self._gpu_vertex_promote_threshold:
                self._gpu_promote_counter += 1
            else:
                self._gpu_promote_counter = 0
            if self._gpu_promote_counter >= 2:
                if self._switch_to_gpu():
                    self._gpu_promote_counter = 0

    @staticmethod
    def _make_vertex_payload(t_arr: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
        if t_arr.size == 0 or x_arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        vertices = np.empty((t_arr.size, 2), dtype=np.float32)
        vertices[:, 0] = t_arr.astype(np.float32, copy=False)
        vertices[:, 1] = x_arr.astype(np.float32, copy=False)
        return vertices

    def _connect_signals(self):
        self.startSpin.valueChanged.connect(self._on_start_spin_changed)
        self.windowSpin.valueChanged.connect(self._on_duration_spin_changed)
        self.fileButton.clicked.connect(self._prompt_open_file)
        self.panLeftBtn.clicked.connect(lambda: self._pan_fraction(-0.25))
        self.panRightBtn.clicked.connect(lambda: self._pan_fraction(0.25))
        self.zoomInBtn.clicked.connect(lambda: self._zoom_factor(0.5))
        self.zoomOutBtn.clicked.connect(lambda: self._zoom_factor(2.0))
        self.fullViewBtn.clicked.connect(self._full_view)
        self.resetViewBtn.clicked.connect(self._reset_view)
        self.prefetchApplyBtn.clicked.connect(self._apply_prefetch_settings)
        self.prefetchSection.toggled.connect(self._on_prefetch_section_toggled)
        self.themeCombo.currentIndexChanged.connect(self._on_theme_changed)
        self.annotationToggle.toggled.connect(self._on_annotation_toggle)
        self.annotationFocusOnly.toggled.connect(self._on_annotation_focus_only_changed)
        self.annotationImportBtn.clicked.connect(self._prompt_import_annotations)
        self.eventChannelFilter.currentIndexChanged.connect(self._on_event_channel_changed)
        self.eventSearchEdit.textChanged.connect(self._on_event_search_changed)
        self.eventList.itemSelectionChanged.connect(self._on_event_selection_changed)
        self.eventList.itemDoubleClicked.connect(self._on_event_activated)
        self.eventPrevBtn.clicked.connect(lambda: self._step_event(-1))
        self.eventNextBtn.clicked.connect(lambda: self._step_event(1))
        self.controlToggleBtn.toggled.connect(self._on_control_toggle)
        self.panelCollapseBtn.clicked.connect(
            lambda: self._set_controls_collapsed(True, persist=True)
        )

        for channel, checkbox in self._annotation_channel_toggles.items():
            if checkbox is not None:
                checkbox.toggled.connect(
                    partial(self._on_annotation_channel_toggle, channel)
                )

        self._shortcuts: list[QtGui.QShortcut] = []
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, activated=lambda: self._pan_fraction(-0.1)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=lambda: self._pan_fraction(0.1)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Minus), self, activated=lambda: self._zoom_factor(2.0)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Equal), self, activated=lambda: self._zoom_factor(0.5)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence("F"), self, activated=self._full_view))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self._reset_view))
        self._shortcuts.append(
            QtGui.QShortcut(
                QtGui.QKeySequence("Ctrl+Shift+C"),
                self,
                activated=lambda: self._set_controls_collapsed(not self._controls_collapsed, persist=True),
            )
        )

    def _on_control_toggle(self, collapsed: bool) -> None:
        self._set_controls_collapsed(bool(collapsed), persist=True)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._plot_viewport and event.type() == QtCore.QEvent.Leave:
            self._hide_hover_indicator()
        return super().eventFilter(obj, event)

    def _set_controls_collapsed(self, collapsed: bool, *, persist: bool) -> None:
        collapsed = bool(collapsed)
        if self._controls_collapsed == collapsed:
            self._apply_control_panel_state(collapsed)
        else:
            self._controls_collapsed = collapsed
            self._apply_control_panel_state(collapsed)
        if persist:
            self._config.controls_collapsed = collapsed
            self._config.save()

    def _apply_control_panel_state(self, collapsed: bool) -> None:
        if self._control_wrapper is None or self._control_scroll is None or self._control_rail is None:
            return
        collapsed = bool(collapsed)
        self._control_scroll.setVisible(not collapsed)
        rail_width = self._control_rail.sizeHint().width()
        self._control_rail.setVisible(collapsed)
        self.controlToggleBtn.setVisible(collapsed)
        if self.panelCollapseBtn is not None:
            self.panelCollapseBtn.setVisible(not collapsed)
        panel_min = (
            self.controlPanel.minimumWidth() if hasattr(self, "controlPanel") else 200
        )
        panel_max = (
            self.controlPanel.maximumWidth() if hasattr(self, "controlPanel") else PANEL_MAX_WIDTH
        )
        if collapsed:
            self._control_wrapper.setMinimumWidth(rail_width)
            self._control_wrapper.setMaximumWidth(rail_width)
        else:
            # Let the splitter manage the live width; just bound the range.
            self._control_wrapper.setMinimumWidth(panel_min)
            self._control_wrapper.setMaximumWidth(panel_max)
            # Nudge the initial size toward a sensible value without locking it.
            scroll_hint = (
                self._control_scroll.sizeHint().width()
                if self._control_scroll is not None
                else panel_min
            )
            target = max(panel_min, min(scroll_hint, panel_max))
            # If the current width is far from target (e.g., after theme/layout change), push splitter sizes once.
            if self._splitter is not None:
                sizes = self._splitter.sizes()
                total = sum(sizes) or max(self.width(), panel_max + 600)
                first = int(target)
                second = max(1, total - first)
                self._splitter.setSizes([first, second])
        if self.controlToggleBtn.isChecked() != collapsed:
            with QtCore.QSignalBlocker(self.controlToggleBtn):
                self.controlToggleBtn.setChecked(collapsed)
        self._update_control_toggle_icon(collapsed)
        if self._splitter is not None:
            sizes = self._splitter.sizes()
            total = sum(sizes)
            if total <= 0:
                total = max(self.width(), rail_width + 600)
            if collapsed:
                first = rail_width
                second = max(1, total - first)
                self._splitter.setSizes([first, second])

    def _update_control_toggle_icon(self, collapsed: bool) -> None:
        arrow = QtCore.Qt.RightArrow if collapsed else QtCore.Qt.LeftArrow
        self.controlToggleBtn.setArrowType(arrow)
        self.controlToggleBtn.setToolTip("Expand controls" if collapsed else "Collapse controls")

    def _on_theme_changed(self, index: int) -> None:
        data = self.themeCombo.itemData(index)
        if not data:
            return
        self._apply_theme(str(data), persist=True)

    def _apply_theme(self, key: str, *, persist: bool) -> None:
        resolved_key = key if key in THEMES else DEFAULT_THEME
        theme = THEMES.get(resolved_key, THEMES[DEFAULT_THEME])
        self._active_theme_key = resolved_key
        self._theme = theme

        if hasattr(self, "themeCombo"):
            idx = self.themeCombo.findData(resolved_key)
            if idx >= 0 and self.themeCombo.currentIndex() != idx:
                with QtCore.QSignalBlocker(self.themeCombo):
                    self.themeCombo.setCurrentIndex(idx)

        self._config.theme = resolved_key

        pg.setConfigOption("background", theme.pg_background)
        pg.setConfigOption("foreground", theme.pg_foreground)

        self.setStyleSheet(theme.stylesheet)

        if self._use_gpu_canvas and self._gpu_canvas is not None:
            colors = theme.curve_colors or ("#5f8bff",)
            self._gpu_canvas.set_theme(
                background=theme.pg_background,
                curve_colors=colors,
                label_color=theme.pg_foreground,
            )
        else:
            self.plotLayout.setBackground(theme.pg_background)
            for plot in self.plots:
                for axis_name in ("bottom", "left"):
                    axis = plot.getAxis(axis_name)
                    if axis is not None:
                        pen = pg.mkPen(theme.pg_foreground)
                        axis.setPen(pen)
                        axis.setTextPen(theme.pg_foreground)

        if self.hypnogramPlot is not None:
            axis = self.hypnogramPlot.getAxis("bottom")
            if axis is not None:
                pen = pg.mkPen(theme.pg_foreground)
                axis.setPen(pen)
                axis.setTextPen(theme.pg_foreground)
        if self._hypnogram_outline is not None:
            outline_color = QtGui.QColor(theme.pg_foreground)
            outline_color.setAlpha(200)
            self._hypnogram_outline.setPen(pg.mkPen(outline_color, width=1.0))
        if self.hypnogramRegion is not None:
            brush_color = QtGui.QColor(theme.pg_foreground)
            brush_color.setAlpha(60)
            self.hypnogramRegion.setBrush(pg.mkBrush(brush_color))
            region_pen = pg.mkPen(theme.pg_foreground, width=1.0)
            for line in getattr(self.hypnogramRegion, "lines", ()):  # type: ignore[attr-defined]
                try:
                    line.setPen(region_pen)
                except AttributeError:
                    continue

        self.time_axis.setPen(pg.mkPen(theme.pg_foreground))
        self.time_axis.setTextPen(theme.pg_foreground)

        self._apply_curve_pens()
        self._refresh_channel_label_styles()
        self._update_theme_preview(resolved_key)

        if persist:
            self._write_persistent_state()

    def _curve_color(self, idx: int) -> str:
        colors = self._theme.curve_colors or ("#5f8bff",)
        return colors[idx % len(colors)]

    def _apply_curve_pens(self) -> None:
        if self._use_gpu_canvas and self._gpu_canvas is not None:
            for idx in range(self.loader.n_channels):
                color = self._curve_color(idx)
                self._gpu_canvas.set_curve_color(idx, color)
            return

        for idx, curve in enumerate(self.curves):
            color = self._curve_color(idx)
            curve.setPen(pg.mkPen(color, width=1.2))

    def _refresh_channel_label_styles(self) -> None:
        if self._use_gpu_canvas and self._gpu_canvas is not None:
            infos = getattr(self.loader, "info", [])
            for idx, meta in enumerate(infos):
                hidden = idx in self._hidden_channels
                self._gpu_canvas.set_channel_label(idx, self._format_label(meta, hidden=hidden))
            return

        if not self.channel_labels:
            return
        n = self.loader.n_channels
        for idx, label_item in enumerate(self.channel_labels):
            if idx >= n:
                continue
            meta = self.loader.info[idx]
            hidden = idx in self._hidden_channels
            label_item.setText(self._format_label(meta, hidden=hidden))

    def _update_theme_preview(self, key: str) -> None:
        if not hasattr(self, "themePreviewWidget"):
            return
        layout = self.themePreviewWidget.layout()
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        theme = THEMES.get(key, THEMES[DEFAULT_THEME])
        colors = theme.preview_colors or theme.curve_colors[:3]
        for color in colors:
            swatch = QtWidgets.QFrame()
            swatch.setFixedSize(18, 18)
            swatch.setStyleSheet(
                "QFrame { background-color: %s; border: 1px solid rgba(0, 0, 0, 70); border-radius: 4px; }"
                % color
            )
            layout.addWidget(swatch)
        layout.addStretch(1)
        self.themePreviewWidget.setVisible(bool(colors))

    def _init_overscan_worker(self):
        self._shutdown_overscan_worker()
        thread = QtCore.QThread(self)
        lod_ratio = float(getattr(self._config, "lod_envelope_ratio", 0.0) or 0.0)
        worker = _OverscanWorker(
            self.loader,
            lod_enabled=self._lod_enabled,
            lod_min_bin_multiple=self._lod_min_bin_multiple,
            lod_min_view_duration=self._lod_min_view_duration,
            lod_ratio=lod_ratio,
        )
        worker.moveToThread(thread)
        worker.finished.connect(self._handle_overscan_finished)
        worker.failed.connect(self._handle_overscan_failed)
        thread.finished.connect(worker.deleteLater)
        thread.start()
        self._overscan_thread = thread
        self._overscan_worker = worker
        self.overscanRequested.connect(worker.render)
        self._overscan_request_id = 0
        self._overscan_inflight = None
        self._current_tile_id = None

    def _shutdown_overscan_worker(self):
        thread = self._overscan_thread
        worker = self._overscan_worker
        self._overscan_thread = None
        self._overscan_worker = None
        if worker is not None:
            try:
                self.overscanRequested.disconnect(worker.render)
            except (TypeError, RuntimeError):
                pass
        if thread is not None:
            thread.quit()
            thread.wait()
        if worker is not None:
            try:
                worker.deleteLater()
            except RuntimeError:
                pass

    def _available_lod_durations(self, channel: int) -> tuple[float, ...]:
        durations: list[float] = []
        levels_fn = getattr(self.loader, "lod_levels", None)
        if callable(levels_fn):
            try:
                levels = levels_fn(channel)
            except Exception:
                levels = None
            if levels:
                if isinstance(levels, dict):
                    durations.extend(float(key) for key in levels.keys())
                else:
                    durations.extend(float(val) for val in levels)
        if not durations:
            durations_fn = getattr(self.loader, "lod_durations", None)
            if callable(durations_fn):
                try:
                    extra = durations_fn(channel)
                except Exception:
                    extra = None
                if extra:
                    durations.extend(float(val) for val in extra)
        filtered = [float(val) for val in durations if float(val) > 0.0]
        return tuple(sorted(filtered))

    def _expected_lod_duration_for_channel(
        self, channel: int, view_duration: float
    ) -> Optional[float]:
        if not self._lod_enabled:
            return None
        durations = self._available_lod_durations(channel)
        if not durations:
            return None
        return select_lod_duration(
            view_duration,
            durations,
            self._lod_min_bin_multiple,
            min_view_duration=self._lod_min_view_duration,
        )

    # ----- Behaviors -------------------------------------------------------

    def _refresh_limits(self):
        duration_cap = min(self._limits.duration_max, self.loader.duration_s)
        self.windowSpin.blockSignals(True)
        self.windowSpin.setRange(self._limits.duration_min, max(self._limits.duration_min, duration_cap))
        self.windowSpin.blockSignals(False)

        max_start = max(0.0, self.loader.duration_s - self._view_duration)
        self.startSpin.blockSignals(True)
        self.startSpin.setRange(0.0, max_start)
        self.startSpin.blockSignals(False)

    def _update_limits_from_loader(self):
        cap = getattr(self.loader, "max_window_s", self._limits.duration_max)
        if isinstance(self.loader, ZarrLoader):
            cap = max(float(cap), self.loader.duration_s)
        self._limits = WindowLimits(
            duration_min=self._limits.duration_min,
            duration_max=float(cap),
        )

    def _update_controls_from_state(self):
        self.startSpin.blockSignals(True)
        self.windowSpin.blockSignals(True)
        self.startSpin.setValue(self._view_start)
        self.windowSpin.setValue(self._view_duration)
        self.startSpin.blockSignals(False)
        self.windowSpin.blockSignals(False)

    def _update_viewbox_from_state(self):
        if self._use_gpu_canvas and self._gpu_canvas is not None:
            self._gpu_canvas.set_view(self._view_start, self._view_duration)
            return
        if not self._primary_plot:
            return
        self._updating_viewbox = True
        try:
            self._primary_plot.setXRange(
                self._view_start,
                self._view_start + self._view_duration,
                padding=0,
            )
        finally:
            self._updating_viewbox = False

    def _schedule_refresh(self):
        self._debounce_timer.start()

    def _on_start_spin_changed(self, value: float):
        self._set_view(float(value), self._view_duration, sender="controls")

    def _on_duration_spin_changed(self, value: float):
        self._set_view(self._view_start, float(value), sender="controls")

    @QtCore.Slot()
    def refresh(self):
        if hasattr(self, "_debounce_timer"):
            self._debounce_timer.stop()
        t0 = self._view_start
        duration = self._view_duration
        t1 = min(self.loader.duration_s, t0 + duration)

        pixels = self._estimate_pixels()
        tile = self._overscan_tile
        used_tile = tile is not None and tile.contains(t0, t1)

        if used_tile:
            tile_updated = self._prepare_tile(tile)
            if self._current_tile_id != tile.request_id or tile_updated:
                self._apply_tile_to_curves(tile)
        else:
            self._current_tile_id = None
            for i in range(self.loader.n_channels):
                if i in self._hidden_channels:
                    if self._use_gpu_canvas and self._gpu_canvas is not None:
                        self._gpu_canvas.clear_channel(i)
                    elif i < len(self.curves):
                        self.curves[i].setData([], [])
                    continue
                t, x = self.loader.read(i, t0, t1)
                if pixels and x.size > pixels * 2:
                    t, x = min_max_bins(t, x, pixels)
                if self._use_gpu_canvas and self._gpu_canvas is not None:
                    self._gpu_canvas.set_channel_data(i, t, x)
                elif i < len(self.curves):
                    self.curves[i].setData(t, x)

        self._update_viewbox_from_state()
        self._update_time_labels(t0, t1)

        if not used_tile and self._overscan_inflight is None:
            self._ensure_overscan_for_view()

        self._update_annotation_overlays(t0, t1)
        if not self._use_gpu_canvas:
            self._update_hypnogram(t0, t1)

    def _update_time_labels(self, t0, t1):
        tb = self.loader.timebase
        start_dt = tb.to_datetime(t0)
        end_dt = tb.to_datetime(t1)
        try:
            same_day = start_dt.date() == end_dt.date()
        except AttributeError:
            same_day = True
        if same_day:
            self.absoluteRange.setText(f"{start_dt:%Y-%m-%d %H:%M:%S} – {end_dt:%H:%M:%S}")
        else:
            self.absoluteRange.setText(
                f"{start_dt:%Y-%m-%d %H:%M:%S} – {end_dt:%Y-%m-%d %H:%M:%S}"
            )
        self.windowSummary.setText(f"Window: {t1 - t0:.1f} s")

    def _prompt_open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open EDF file",
            filter="EDF Files (*.edf *.EDF);;All Files (*)",
        )
        if not path:
            return

        self._load_new_file(path)

    def _load_new_file(self, path: str):
        self._cleanup_ingest_thread(wait=True)
        self.ingestBar.hide()
        self.ingestBar.setValue(0)

        old_loader = self.loader
        old_path = getattr(old_loader, "path", None)
        same_path = old_path == path

        if same_path:
            old_loader.close()

        try:
            new_loader = self._create_loader_for_path(path)
        except Exception as exc:  # pragma: no cover - UI feedback
            QtWidgets.QMessageBox.critical(self, "Failed to open", str(exc))
            # attempt to restore previous loader if possible
            if old_path:
                try:
                    restored = type(old_loader)(old_path)
                except Exception:
                    pass
                else:
                    self.loader = restored
                    self._update_data_source_label()
            return

        if not same_path:
            old_loader.close()

        self.loader = new_loader
        if isinstance(self.loader, ZarrLoader):
            setattr(self.loader, "max_window_s", self.loader.duration_s)
        self._update_data_source_label()

        self.startSpin.blockSignals(True)
        self.windowSpin.blockSignals(True)
        try:
            self.time_axis.set_timebase(self.loader.timebase)
            if self.loader.timebase.start_dt is not None:
                self.time_axis.set_mode("absolute")
            self._update_limits_from_loader()
            self._view_start, self._view_duration = clamp_window(
                0.0,
                min(30.0, self.loader.duration_s),
                total=self.loader.duration_s,
                limits=self._limits,
            )
        finally:
            self.startSpin.blockSignals(False)
            self.windowSpin.blockSignals(False)

        self._ensure_plot_rows(self.loader.n_channels)
        self._configure_plots()
        self._refresh_limits()
        self._update_controls_from_state()
        self._invalidate_stage_curve_cache()
        self.refresh()
        self._manual_annotation_paths.clear()
        self._load_companion_annotations()
        self._invalidate_overscan_tile_cache()
        self._init_overscan_worker()
        self._ensure_overscan_for_view()
        self._start_zarr_ingest()
        self._prefetch.clear()
        self._schedule_prefetch()

    def _create_loader_for_path(self, path: str) -> object:
        candidate = Path(path)
        if candidate.is_dir() or candidate.suffix.lower() == ".zarr":
            return ZarrLoader(candidate, max_window_s=float("inf"))
        return EdfLoader(path)

    def _start_zarr_ingest(self):
        if isinstance(self.loader, ZarrLoader):
            return

        if not getattr(self.loader, "path", None):
            return

        zarr_path = resolve_output_path(self.loader.path)
        self._zarr_path = zarr_path

        if zarr_path.exists() and not zarr_path.is_dir():
            LOG.warning("Conflicting file at %s; attempting to remove", zarr_path)
            try:
                zarr_path.unlink()
            except Exception as exc:  # pragma: no cover - user-facing warning
                QtWidgets.QMessageBox.warning(
                    self,
                    "Zarr cache conflict",
                    f"Existing file blocks cache directory:\n{zarr_path}\n{exc}",
                )
                return

        if zarr_path.exists():
            self.ingestBar.hide()
            if not isinstance(self.loader, ZarrLoader):
                try:
                    self._pending_loader = ZarrLoader(zarr_path, max_window_s=float("inf"))
                except Exception:
                    self._pending_loader = None
                else:
                    QtCore.QTimer.singleShot(0, self._swap_in_pending_loader)
            else:
                self._update_data_source_label()
            return

        total_samples = sum(int(info.n_samples) for info in self.loader.info)
        self.ingestBar.setRange(0, max(1, total_samples))
        self.ingestBar.setValue(0)
        self.ingestBar.setFormat("Caching EDF → Zarr: %p%")
        self.ingestBar.show()

        self._ingest_thread = QtCore.QThread(self)
        self._ingest_worker = _ZarrIngestWorker(self.loader.path, zarr_path, loader=self.loader)
        self._ingest_worker.moveToThread(self._ingest_thread)
        self._ingest_thread.started.connect(self._ingest_worker.run)
        self._ingest_worker.progress.connect(self._handle_ingest_progress)
        self._ingest_worker.finished.connect(self._handle_ingest_finished)
        self._ingest_worker.failed.connect(self._handle_ingest_error)
        self._ingest_worker.finished.connect(self._ingest_thread.quit)
        self._ingest_worker.failed.connect(self._ingest_thread.quit)
        self._ingest_thread.finished.connect(self._cleanup_ingest_thread)
        self._ingest_thread.start()

    @QtCore.Slot(int, int)
    def _handle_ingest_progress(self, done: int, total: int):
        total = max(1, total)
        if self.ingestBar.maximum() != total:
            self.ingestBar.setRange(0, total)
        self.ingestBar.setValue(done)

    @QtCore.Slot(str)
    def _handle_ingest_finished(self, path: str):
        self.ingestBar.setValue(self.ingestBar.maximum())
        self.ingestBar.setFormat("Zarr cache ready ✓")
        QtCore.QTimer.singleShot(2000, self.ingestBar.hide)

        self._pending_loader = ZarrLoader(path, max_window_s=float("inf"))
        QtCore.QTimer.singleShot(0, self._swap_in_pending_loader)

    @QtCore.Slot(str)
    def _handle_ingest_error(self, message: str):
        self.ingestBar.setFormat("Zarr cache failed")
        QtWidgets.QMessageBox.warning(self, "Zarr ingest failed", message)
        QtCore.QTimer.singleShot(2000, self.ingestBar.hide)

    def _cleanup_ingest_thread(self, wait: bool = False):
        thread = self._ingest_thread
        worker = self._ingest_worker
        self._ingest_thread = None
        self._ingest_worker = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            if wait and thread.isRunning():
                thread.quit()
                thread.wait()
            if thread.isFinished():
                thread.deleteLater()

    @QtCore.Slot()
    def _swap_in_pending_loader(self):
        pending = self._pending_loader
        if pending is None:
            return

        self._pending_loader = None

        old_loader = self.loader
        carried_annotations = None
        if hasattr(old_loader, "annotations"):
            try:
                carried = old_loader.annotations()
            except Exception:
                carried = None
            else:
                if carried and getattr(carried, "size", 0):
                    carried_annotations = carried
        self.loader = pending
        if isinstance(self.loader, ZarrLoader):
            setattr(self.loader, "max_window_s", self.loader.duration_s)
            if carried_annotations is not None:
                try:
                    self.loader.set_annotations(carried_annotations)
                except Exception:
                    pass
        self._update_limits_from_loader()
        self._view_start, self._view_duration = clamp_window(
            self._view_start,
            self._view_duration,
            total=self.loader.duration_s,
            limits=self._limits,
        )

        self.time_axis.set_timebase(self.loader.timebase)
        if self.loader.timebase.start_dt is not None:
            self.time_axis.set_mode("absolute")

        self._ensure_plot_rows(self.loader.n_channels)
        self._configure_plots()
        self._refresh_limits()
        self._invalidate_overscan_tile_cache()
        self._init_overscan_worker()
        self._manual_annotation_paths.clear()
        self._invalidate_stage_curve_cache()
        self.refresh()
        self._load_companion_annotations()
        self._update_data_source_label()
        self._prefetch.clear()
        self._schedule_prefetch()
        self._ensure_overscan_for_view()

        if hasattr(old_loader, "close") and not isinstance(old_loader, ZarrLoader):
            old_loader.close()

    def _set_view(self, start: float, duration: float, *, sender: str | None = None):
        old_duration = self._view_duration
        start_new, duration_new = clamp_window(
            start,
            duration,
            total=self.loader.duration_s,
            limits=self._limits,
        )
        if (
            abs(start_new - self._view_start) < 1e-6
            and abs(duration_new - self._view_duration) < 1e-6
        ):
            if sender == "controls":
                self._schedule_refresh()
            return

        duration_changed = abs(duration_new - old_duration) >= 1e-6
        self._view_start = start_new
        self._view_duration = duration_new
        self._refresh_limits()
        if sender != "controls":
            self._update_controls_from_state()
        if sender != "viewbox":
            self._update_viewbox_from_state()
        if sender != "hypnogram":
            self._update_hypnogram_region(
                self._view_start,
                self._view_start + self._view_duration,
            )
        if duration_changed:
            self._invalidate_overscan_tile_cache()
        self._schedule_refresh()
        self._schedule_prefetch()
        self._ensure_overscan_for_view()

    def _pan_fraction(self, fraction: float):
        delta = fraction * self._view_duration
        start, duration = pan_window(
            self._view_start,
            self._view_duration,
            delta=delta,
            total=self.loader.duration_s,
            limits=self._limits,
        )
        self._set_view(start, duration, sender="buttons")

    def _apply_prefetch_settings(self):
        tile = self.prefetchTileSpin.value()
        max_tiles = self.prefetchMaxTilesSpin.value()
        max_mb_val = self.prefetchMaxMbSpin.value()
        max_mb = max_mb_val if max_mb_val > 0 else None
        self._config.prefetch_tile_s = tile
        self._config.prefetch_max_tiles = max_tiles
        self._config.prefetch_max_mb = max_mb
        prefetch_service.configure(tile_duration=tile, max_tiles=max_tiles, max_mb=max_mb)
        if self._prefetch is not None:
            self._prefetch.stop()
        self._prefetch = prefetch_service.create_cache(
            self._fetch_tile, preview_fetch=self._fetch_tile_preview
        )
        self._prefetch.start()
        self._schedule_prefetch()
        self._config.save()

    def _on_prefetch_section_toggled(self, expanded: bool) -> None:
        self._config.prefetch_collapsed = not expanded
        self._config.save()

    def _on_annotation_toggle(self, checked: bool):
        self._annotations_enabled = bool(checked)
        self._invalidate_stage_curve_cache()
        self._update_annotation_overlays(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )
        self._update_hypnogram(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )

    def _on_annotation_focus_only_changed(self, checked: bool):
        self._write_persistent_state()
        self._update_annotation_overlays(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )

    def _write_persistent_state(self) -> None:
        if not hasattr(self, "annotationFocusOnly"):
            return
        self._config.annotation_focus_only = bool(self.annotationFocusOnly.isChecked())
        if hasattr(self, "themeCombo"):
            data = self.themeCombo.currentData()
            if isinstance(data, str):
                self._config.theme = data
        self._config.controls_collapsed = bool(self._controls_collapsed)
        self._config.save()

    def _schedule_prefetch(self):
        total = self.loader.duration_s
        for ch in range(self.loader.n_channels):
            if ch in self._hidden_channels:
                continue
            start = max(0.0, self._view_start - self._view_duration)
            start = min(start, total)
            duration = min(self._view_duration * 3, max(0.0, total - start))
            if duration <= 0:
                continue
            self._prefetch.prefetch_window(ch, start, duration)
    def _zoom_factor(self, factor: float):
        anchor = self._view_start + self._view_duration * 0.5
        start, duration = zoom_window(
            self._view_start,
            self._view_duration,
            factor=factor,
            anchor=anchor,
            total=self.loader.duration_s,
            limits=self._limits,
        )
        self._set_view(start, duration, sender="buttons")

    def _reset_view(self):
        duration = min(30.0, self.loader.duration_s)
        start, duration = clamp_window(
            0.0,
            duration,
            total=self.loader.duration_s,
            limits=self._limits,
        )
        self._set_view(start, duration, sender="buttons")

    def _schedule_prefetch(self):
        for ch in range(self.loader.n_channels):
            if ch in self._hidden_channels:
                continue
            start = max(0.0, self._view_start - self._view_duration)
            duration = self._view_duration * 3
            self._prefetch.prefetch_window(ch, start, duration)

    def _full_view(self):
        self._set_view(0.0, self.loader.duration_s, sender="buttons")

    def _fetch_tile(self, channel: int, start: float, end: float):
        loader = self.loader
        start, duration = clamp_window(start, end - start, total=loader.duration_s, limits=self._limits)
        result = loader.read(channel, start, start + duration)
        if isinstance(result, SignalChunk):
            return result.as_tuple()
        return result

    def _fetch_tile_preview(self, channel: int, start: float, end: float):
        loader = self.loader
        start, duration = clamp_window(start, end - start, total=loader.duration_s, limits=self._limits)
        if duration <= 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty.astype(np.float32)
        window_end = start + duration
        durations = self._available_lod_durations(channel)
        read_window = getattr(loader, "read_lod_window", None)
        if durations and callable(read_window):
            coarse = durations[-1]
            try:
                result = read_window(channel, start, window_end, coarse)
            except Exception:
                result = None
            else:
                chunk = None
                if isinstance(result, SignalChunk):
                    chunk = result
                elif result is not None:
                    data, bin_duration, start_bin = result
                    if getattr(data, "size", 0) > 0:
                        mins = np.asarray(data[:, 0], dtype=np.float32)
                        maxs = np.asarray(data[:, 1], dtype=np.float32)
                        t_vals, x_vals = envelope_to_series(
                            mins,
                            maxs,
                            bin_duration=float(bin_duration),
                            start_bin=int(start_bin),
                            window_start=start,
                            window_end=window_end,
                        )
                        if t_vals.size > 0:
                            chunk = chunk_from_arrays(
                                np.asarray(t_vals, dtype=np.float64),
                                np.asarray(x_vals, dtype=np.float32),
                            )
                if chunk is not None and chunk.x.size > 0:
                    return chunk.as_tuple()

        sample_cap = getattr(self._overscan_worker, "_preview_sample_cap", 2048)
        try:
            result = loader.read(
                channel,
                start,
                window_end,
                max_samples=int(sample_cap),
            )
        except TypeError:
            result = loader.read(channel, start, window_end)
        if isinstance(result, SignalChunk):
            return result.as_tuple()
        t_arr, x_arr = result
        return np.asarray(t_arr, dtype=np.float64), np.asarray(x_arr, dtype=np.float32)

    def _ensure_overscan_for_view(self):
        if self._overscan_worker is None or self.loader.n_channels == 0:
            return
        window_start = self._view_start
        window_end = min(self.loader.duration_s, window_start + self._view_duration)
        tile = self._overscan_tile
        if tile is not None and tile.contains(window_start, window_end):
            needs_refresh = False
            desired_cache: dict[int, Optional[float]] = {}
            for idx, tile_duration in enumerate(tile.lod_durations):
                if idx not in desired_cache:
                    desired_cache[idx] = self._expected_lod_duration_for_channel(
                        idx, self._view_duration
                    )
                desired = desired_cache[idx]
                if tile_duration is None and desired is None:
                    continue
                if tile_duration is None or desired is None:
                    needs_refresh = True
                    break
                if not math.isclose(
                    float(tile_duration), float(desired), rel_tol=1e-6, abs_tol=1e-6
                ):
                    needs_refresh = True
                    break
            if needs_refresh:
                self._request_overscan_tile(window_start, self._view_duration)
                self._update_tile_view_metadata(tile, window_start, self._view_duration)
                return
            ratio = 1.0
            if tile.view_duration > 0:
                ratio = self._view_duration / tile.view_duration
            if ratio < self._overscan_zoom_reuse_ratio:
                self._request_overscan_tile(window_start, self._view_duration)
                self._update_tile_view_metadata(tile, window_start, self._view_duration)
                return
            self._update_tile_view_metadata(tile, window_start, self._view_duration)
            return
        self._request_overscan_tile(window_start, self._view_duration)

    @staticmethod
    def _cache_span_key(value: float) -> int:
        return int(round(float(value) * 1_000_000))

    def _tile_cache_key(
        self, channels: tuple[int, ...], start: float, end: float
    ) -> tuple[tuple[int, ...], int, int]:
        return (channels, self._cache_span_key(start), self._cache_span_key(end))

    def _get_cached_tile(
        self, channels: tuple[int, ...], start: float, end: float
    ) -> _OverscanTile | None:
        key = self._tile_cache_key(channels, start, end)
        tile = self._overscan_tile_cache.get(key)
        if tile is not None:
            self._overscan_tile_cache.move_to_end(key)
        return tile

    def _cache_tile(self, tile: _OverscanTile) -> None:
        channels = tuple(tile.channel_indices)
        if not channels:
            return
        key = self._tile_cache_key(channels, tile.start, tile.end)
        self._overscan_tile_cache[key] = tile
        self._overscan_tile_cache.move_to_end(key)
        limit = max(1, int(self._overscan_tile_cache_limit or 0))
        while len(self._overscan_tile_cache) > limit:
            self._overscan_tile_cache.popitem(last=False)

    def _invalidate_overscan_tile_cache(self) -> None:
        self._overscan_tile_cache.clear()
        self._overscan_tile = None
        self._overscan_inflight = None
        self._current_tile_id = None
        self._overscan_request_id += 1

    def _request_overscan_tile(self, window_start: float, window_duration: float):
        if self._overscan_worker is None or self.loader.n_channels == 0:
            return
        self._update_tile_view_metadata(self._overscan_tile, window_start, window_duration)
        start, end = self._compute_overscan_bounds(window_start, window_duration)
        if end <= start:
            return
        channels = tuple(range(self.loader.n_channels))
        cached_tile = self._get_cached_tile(channels, start, end)
        req_id = self._overscan_request_id + 1
        if cached_tile is not None:
            self._overscan_request_id = req_id
            self._overscan_inflight = None
            cached_tile.request_id = req_id
            cached_tile.is_final = True
            self._overscan_tile = cached_tile
            self._update_tile_view_metadata(cached_tile, window_start, window_duration)
            self._current_tile_id = None
            self._apply_tile_to_curves(cached_tile)
            try:
                self._schedule_refresh()
            except AttributeError:
                pass
            return
        req_id = self._overscan_request_id + 1
        self._overscan_request_id = req_id
        self._overscan_inflight = req_id
        request = _OverscanRequest(
            request_id=req_id,
            start=start,
            end=end,
            view_start=window_start,
            view_duration=window_duration,
            channel_indices=channels,
            max_samples=None,
        )
        self.overscanRequested.emit(request)

    def _compute_overscan_bounds(self, view_start: float, view_duration: float) -> tuple[float, float]:
        total = self.loader.duration_s
        left_desired = self._overscan_factor * view_duration
        right_desired = self._overscan_factor * view_duration
        left = min(left_desired, view_start)
        right = min(right_desired, max(0.0, total - (view_start + view_duration)))
        span = left + view_duration + right
        max_span = getattr(self.loader, "max_window_s", None)
        if max_span:
            max_span = float(max_span)
            if span > max_span:
                excess = span - max_span
                reduce_left = min(left, excess / 2.0)
                left -= reduce_left
                excess -= reduce_left
                if excess > 0:
                    right -= min(right, excess)
                left = max(left, 0.0)
                right = max(right, 0.0)
                span = left + view_duration + right
        start = max(0.0, view_start - left)
        end = min(total, view_start + view_duration + right)
        return start, end

    def _handle_overscan_finished(self, request_id: int, tile_obj):
        if not isinstance(tile_obj, _OverscanTile):
            return
        is_final = bool(getattr(tile_obj, "is_final", True))
        if request_id != self._overscan_request_id:
            return
        if is_final:
            self._overscan_inflight = None
        else:
            self._overscan_inflight = request_id
        self._overscan_tile = tile_obj
        self._update_tile_view_metadata(self._overscan_tile, self._view_start, self._view_duration)
        self._current_tile_id = None
        self._apply_tile_to_curves(tile_obj)
        if is_final:
            self._cache_tile(tile_obj)
        self._schedule_refresh()

    def _update_tile_view_metadata(
        self, tile: _OverscanTile | None, view_start: float, view_duration: float
    ) -> None:
        if tile is None:
            return
        tile.view_start = view_start
        tile.view_duration = view_duration

    def _handle_overscan_failed(self, request_id: int, message: str):
        if request_id != self._overscan_request_id:
            return
        self._overscan_inflight = None
        LOG.warning("Overscan render failed: %s", message)
        self._update_annotation_overlays(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )

    # ----- Annotations -----------------------------------------------------

    def _load_companion_annotations(self):
        self._annotations_index = None
        self.annotationToggle.setEnabled(False)
        self.annotationFocusOnly.setEnabled(False)
        self._annotations_enabled = False
        self.stageSummaryLabel.setText("Stage: -- | Position: -- | Events: 0")
        self._clear_annotation_lines()
        self._clear_annotation_rects()
        self._populate_event_list(clear=True)
        self._update_annotation_channel_toggles()
        self._update_annotation_summary()
        self._invalidate_stage_curve_cache()

        path = getattr(self.loader, "path", None)
        t0 = self._view_start
        t1 = min(self.loader.duration_s, self._view_start + self._view_duration)
        if not path:
            self._update_hypnogram(t0, t1)
            return

        ann_sets: list[annotation_core.Annotations] = []
        found = annotation_core.discover_annotation_files(path)
        found.update(self._manual_annotation_paths)
        start_dt = getattr(getattr(self.loader, "timebase", None), "start_dt", None)

        loader_ann: annotation_core.Annotations | None = None
        if hasattr(self.loader, "annotations"):
            try:
                loader_ann = self.loader.annotations()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOG.warning("Failed to load EDF+ annotations: %s", exc)
            else:
                if loader_ann.size:
                    ann_sets.append(loader_ann)

        events_path = found.get("events")
        if events_path:
            try:
                mapping = annotation_core.CsvEventMapping(default_channel="Events")
                ann_sets.append(
                    annotation_core.from_csv_events(events_path, mapping, start_dt=start_dt)
                )
            except Exception as exc:  # pragma: no cover - file parse issues logged
                LOG.warning("Failed to load events CSV %s: %s", events_path, exc)

        stages_path = found.get("stages")
        if stages_path:
            try:
                ann_sets.append(annotation_core.from_csv_stages(stages_path))
            except Exception as exc:  # pragma: no cover
                LOG.warning("Failed to load stage CSV %s: %s", stages_path, exc)

        positions_path = found.get("positions")
        if positions_path:
            try:
                ann_sets.append(annotation_core.from_csv_positions(positions_path))
            except Exception as exc:  # pragma: no cover
                LOG.warning("Failed to load position CSV %s: %s", positions_path, exc)

        if not ann_sets:
            self._update_hypnogram(t0, t1)
            return

        self._annotations_index = annotation_core.AnnotationIndex(ann_sets)
        self.annotationToggle.setEnabled(True)
        self._annotations_enabled = self.annotationToggle.isChecked()
        self._update_annotation_channel_toggles()
        self._rebuild_all_event_records()
        self._update_event_channel_options()
        self._reset_event_filters()
        self._populate_event_list()
        self._update_annotation_summary()
        self._update_annotation_overlays(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )
        self._update_hypnogram(t0, t1)

    def _stage_annotations(self) -> np.ndarray:
        if not self._annotations_index or self._annotations_index.is_empty():
            return np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)
        if annotation_core.STAGE_CHANNEL in self._hidden_annotation_channels:
            return np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)
        data = self._annotations_index.data
        if data.size == 0:
            return data
        mask = np.array(
            [str(chan) == annotation_core.STAGE_CHANNEL for chan in data["chan"]],
            dtype=bool,
        )
        return data[mask]

    def _position_annotations(self) -> np.ndarray:
        if not self._annotations_index or self._annotations_index.is_empty():
            return np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)
        if annotation_core.POSITION_CHANNEL in self._hidden_annotation_channels:
            return np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)
        data = self._annotations_index.data
        if data.size == 0:
            return data
        mask = np.array(
            [str(chan) == annotation_core.POSITION_CHANNEL for chan in data["chan"]],
            dtype=bool,
        )
        return data[mask]

    def _label_at_time(self, annotations: np.ndarray, timestamp: float) -> Optional[str]:
        if annotations.size == 0:
            return None
        starts = annotations["start_s"]
        idx = int(np.searchsorted(starts, timestamp, side="right") - 1)
        if idx < 0 or idx >= annotations.size:
            return None
        if float(annotations[idx]["end_s"]) <= timestamp:
            return None
        label = str(annotations[idx]["label"]).strip()
        return label or None

    def _current_stage_position_labels(self) -> tuple[Optional[str], Optional[str]]:
        center_time = self._view_start + self._view_duration * 0.5
        stage_label = self._label_at_time(self._stage_annotations(), center_time)
        position_label = self._label_at_time(self._position_annotations(), center_time)
        return stage_label, position_label

    def _compose_status_text(
        self,
        stage_label: Optional[str],
        position_label: Optional[str],
        events_summary: Optional[str],
    ) -> str:
        parts = [f"Stage: {stage_label or '--'}", f"Position: {position_label or '--'}"]
        if events_summary:
            parts.append(events_summary)
        return " | ".join(parts)

    def _prompt_import_annotations(self):
        options = QtWidgets.QFileDialog.Options()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import annotations",
            str(Path(self.loader.path).parent if getattr(self.loader, "path", None) else Path.cwd()),
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if not path:
            return
        path_obj = Path(path)
        stem_upper = path_obj.stem.upper()
        if stem_upper.endswith("STAGE"):
            key = "stages"
        elif stem_upper.endswith("POSITION"):
            key = "positions"
        else:
            key = "events"
        self._manual_annotation_paths[key] = path_obj
        self._load_companion_annotations()

    def _schedule_event_filter_refresh(self):
        if self._event_filter_timer.isActive():
            self._event_filter_timer.stop()
        self._event_filter_timer.start()

    def _apply_event_filters(self):
        self._populate_event_list()

    def _on_event_channel_changed(self):
        data = self.eventChannelFilter.currentData()
        self._selected_event_channel = str(data) if data else None
        self._schedule_event_filter_refresh()

    def _on_event_search_changed(self, text: str):
        normalized = text.strip().lower()
        if normalized == self._event_label_filter:
            return
        self._event_label_filter = normalized
        self._schedule_event_filter_refresh()

    def _reset_event_filters(self):
        if self._event_filter_timer.isActive():
            self._event_filter_timer.stop()
        self._selected_event_channel = None
        self._event_label_filter = ""
        self.eventChannelFilter.blockSignals(True)
        self.eventChannelFilter.setCurrentIndex(0)
        self.eventChannelFilter.blockSignals(False)
        self.eventSearchEdit.blockSignals(True)
        self.eventSearchEdit.clear()
        self.eventSearchEdit.blockSignals(False)

    def _update_event_channel_options(self):
        channels = sorted({str(rec["chan"]) for rec in self._all_event_records if rec.get("chan")})
        current = self._selected_event_channel
        self.eventChannelFilter.blockSignals(True)
        self.eventChannelFilter.clear()
        self.eventChannelFilter.addItem("All channels", userData=None)
        for chan in channels:
            self.eventChannelFilter.addItem(chan, userData=chan)
        if current and current in channels:
            index = self.eventChannelFilter.findData(current)
            if index >= 0:
                self.eventChannelFilter.setCurrentIndex(index)
            else:
                self.eventChannelFilter.setCurrentIndex(0)
                self._selected_event_channel = None
        else:
            self.eventChannelFilter.setCurrentIndex(0)
            self._selected_event_channel = None
        self.eventChannelFilter.setEnabled(bool(self._all_event_records))
        self.eventChannelFilter.blockSignals(False)

    def _update_annotation_channel_toggles(self) -> None:
        if not self._annotation_channel_toggles:
            return

        available: set[str] = set()
        if self._annotations_index and not self._annotations_index.is_empty():
            available = {str(chan) for chan in self._annotations_index.channel_set}

        for channel, checkbox in self._annotation_channel_toggles.items():
            if checkbox is None:
                continue
            is_available = channel in available
            blocker = QtCore.QSignalBlocker(checkbox)
            checkbox.setEnabled(is_available)
            checkbox.setChecked(
                is_available and channel not in self._hidden_annotation_channels
            )
            del blocker

    def _event_count_summary(self) -> str:
        total = len(self._all_event_records)
        filtered = len(self._event_records)
        if total and filtered != total:
            return f"Events: {filtered}/{total}"
        return f"Events: {filtered}"

    def _update_annotation_summary(self):
        summary = self._event_count_summary()
        stage_label, position_label = self._current_stage_position_labels()
        self.stageSummaryLabel.setText(
            self._compose_status_text(stage_label, position_label, summary)
        )

    def _ensure_annotation_line_pool(self, count: int):
        if not self._primary_plot:
            return
        while len(self._annotation_lines) < count:
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff9f1c", width=1.0))
            line.setVisible(False)
            self._primary_plot.addItem(line)
            self._annotation_lines.append(line)

    def _ensure_annotation_rect_pool(self, count: int):
        scene = self.plotLayout.scene()
        while len(self._annotation_rects) < count:
            rect_item = QtWidgets.QGraphicsRectItem()
            rect_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 169, 77, 60)))
            rect_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            rect_item.setZValue(-10)
            rect_item.setVisible(False)
            scene.addItem(rect_item)
            self._annotation_rects.append(rect_item)

    def _ensure_stage_label_pool(self, count: int):
        scene = self.plotLayout.scene()
        while len(self._stage_label_items) < count:
            label_item = QtWidgets.QGraphicsSimpleTextItem("")
            font = label_item.font()
            font.setBold(True)
            label_item.setFont(font)
            label_item.setZValue(-5)
            label_item.setVisible(False)
            scene.addItem(label_item)
            self._stage_label_items.append(label_item)

    def _stage_color_for_label(self, label: str) -> QtGui.QColor:
        return QtGui.QColor(STAGE_COLORS.get(label, DEFAULT_STAGE_COLOR))

    def _clear_stage_labels(self):
        for item in self._stage_label_items:
            item.setVisible(False)

    def _clear_annotation_lines(self):
        for line in self._annotation_lines:
            line.setVisible(False)
        self._clear_annotation_rects()
        self._clear_stage_labels()

    def _clear_annotation_rects(self):
        for rect in self._annotation_rects:
            rect.setVisible(False)

    def _dominant_position_for_segment(
        self,
        position_events: np.ndarray,
        start: float,
        end: float,
    ) -> Optional[str]:
        if position_events is None or getattr(position_events, "size", 0) == 0:
            return None
        mask = (position_events["start_s"] < end) & (position_events["end_s"] > start)
        if not np.any(mask):
            return None
        best_label: Optional[str] = None
        best_overlap = -1.0
        for entry in position_events[mask]:
            overlap_start = max(start, float(entry["start_s"]))
            overlap_end = min(end, float(entry["end_s"]))
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap >= best_overlap:
                best_overlap = overlap
                best_label = str(entry["label"]) or None
        return best_label

    def _update_stage_labels(
        self,
        stage_events: np.ndarray,
        position_events: np.ndarray,
        t0: float,
        t1: float,
    ) -> None:
        if not self._primary_plot:
            self._clear_stage_labels()
            return
        if stage_events is None or getattr(stage_events, "size", 0) == 0:
            self._clear_stage_labels()
            return

        vb = self._primary_plot.getViewBox()
        if vb is None:
            self._clear_stage_labels()
            return

        scene_rect = self.plotLayout.ci.mapRectToScene(self.plotLayout.ci.boundingRect())
        if scene_rect is None:
            self._clear_stage_labels()
            return

        segments: list[tuple[float, float, str]] = []
        for ev in stage_events:
            start = max(float(ev["start_s"]), t0)
            end = min(float(ev["end_s"]), t1)
            if end <= start:
                continue
            label = str(ev["label"]) or "--"
            segments.append((start, end, label))

        if not segments:
            self._clear_stage_labels()
            return

        self._ensure_stage_label_pool(len(segments))

        for idx, (start, end, label) in enumerate(segments):
            label_item = self._stage_label_items[idx]
            pos_label = self._dominant_position_for_segment(position_events, start, end)
            text = f"{label} | {pos_label}" if pos_label else label
            label_item.setText(text)
            label_item.setBrush(QtGui.QBrush(self._stage_color_for_label(label)))

            p1 = vb.mapViewToScene(QtCore.QPointF(start, 0))
            p2 = vb.mapViewToScene(QtCore.QPointF(end, 0))
            x1, x2 = p1.x(), p2.x()
            left = min(x1, x2)
            width = max(2.0, abs(x2 - x1))

            text_rect = label_item.boundingRect()
            label_width = text_rect.width()
            label_height = text_rect.height()
            if width < label_width + 12:
                label_item.setVisible(False)
                continue
            x_mid = left + (width - label_width) * 0.5
            y_pos = scene_rect.bottom() - STAGE_TEXT_MARGIN - label_height * 0.5
            label_item.setPos(x_mid, y_pos)
            label_item.setVisible(True)

        for idx in range(len(segments), len(self._stage_label_items)):
            self._stage_label_items[idx].setVisible(False)

    def _set_annotation_channel_visible(
        self, channel: str, visible: bool, *, persist: bool = True
    ) -> None:
        key = str(channel).strip()
        if not key:
            return

        currently_visible = key not in self._hidden_annotation_channels
        if bool(visible) == currently_visible:
            return

        if visible:
            self._hidden_annotation_channels.discard(key)
        else:
            self._hidden_annotation_channels.add(key)

        if key == annotation_core.STAGE_CHANNEL:
            self._invalidate_stage_curve_cache()

        if persist:
            ordered = list(dict.fromkeys(self._config.hidden_annotation_channels))
            if visible:
                if key in ordered:
                    ordered.remove(key)
            else:
                if key not in ordered:
                    ordered.append(key)
            self._config.hidden_annotation_channels = tuple(ordered)
            self._config.save()

        self._update_annotation_channel_toggles()
        self._rebuild_all_event_records()
        self._update_event_channel_options()
        self._populate_event_list()
        self._update_annotation_summary()
        if self.loader is not None:
            self._update_annotation_overlays(
                self._view_start,
                min(self.loader.duration_s, self._view_start + self._view_duration),
            )
            self._update_hypnogram(
                self._view_start,
                min(self.loader.duration_s, self._view_start + self._view_duration),
            )

    def _on_annotation_channel_toggle(self, channel: str, checked: bool) -> None:
        self._set_annotation_channel_visible(channel, bool(checked))

    def _rebuild_all_event_records(self):
        self._all_event_records = []
        if not self._annotations_index or self._annotations_index.is_empty():
            return

        duration = getattr(self.loader, "duration_s", 0.0)
        events, ids = self._annotations_index.between(
            0.0,
            duration,
            channels=None,
            return_indices=True,
        )

        data = np.array(events, copy=False)
        ids = np.asarray(ids, dtype=int)
        if data.size == 0 or ids.size == 0:
            return

        hidden_channels = {str(ch) for ch in self._hidden_annotation_channels}
        if hidden_channels:
            mask = np.array(
                [str(chan) not in hidden_channels for chan in data["chan"]],
                dtype=bool,
            )
            data = data[mask]
            ids = ids[mask]
        if data.size == 0:
            return

        records: list[dict[str, float | str | int]] = []
        for entry, idx in zip(data, ids):
            start = float(entry["start_s"])
            end = float(entry["end_s"])
            records.append(
                {
                    "start": start,
                    "end": end,
                    "label": str(entry["label"]) or "event",
                    "chan": str(entry["chan"]) or "Events",
                    "id": int(idx),
                }
            )

        records.sort(key=lambda r: (r["start"], r["end"], r["label"]))
        self._all_event_records = records

    def _populate_event_list(self, clear: bool = False):
        self.eventList.blockSignals(True)
        self.eventList.clear()
        if clear or not self._annotations_index or self._annotations_index.is_empty():
            self.eventList.setEnabled(False)
            self.eventPrevBtn.setEnabled(False)
            self.eventNextBtn.setEnabled(False)
            self.annotationFocusOnly.setEnabled(False)
            self._all_event_records = []
            self._event_records = []
            self._current_event_index = -1
            self._current_event_id = None
            self.eventChannelFilter.blockSignals(True)
            self.eventChannelFilter.clear()
            self.eventChannelFilter.addItem("All channels", userData=None)
            self.eventChannelFilter.setEnabled(False)
            self.eventChannelFilter.blockSignals(False)
            self.eventSearchEdit.blockSignals(True)
            self.eventSearchEdit.clear()
            self.eventSearchEdit.setEnabled(False)
            self.eventSearchEdit.blockSignals(False)
            self.eventList.blockSignals(False)
            self._update_annotation_summary()
            return

        if not self._all_event_records:
            self.eventList.setEnabled(False)
            self.eventPrevBtn.setEnabled(False)
            self.eventNextBtn.setEnabled(False)
            self.annotationFocusOnly.setEnabled(False)
            self._event_records = []
            self._current_event_index = -1
            self._current_event_id = None
            self.eventChannelFilter.setEnabled(False)
            self.eventSearchEdit.setEnabled(False)
            self.eventList.blockSignals(False)
            self._update_annotation_summary()
            return

        channel_filter = self._selected_event_channel
        label_filter = self._event_label_filter
        previous_id = self._current_event_id
        filtered: list[dict[str, float | str | int]] = []
        restored_index = -1
        for record in self._all_event_records:
            if channel_filter and str(record.get("chan")) != channel_filter:
                continue
            if label_filter:
                label = str(record.get("label", "")).lower()
                if label_filter not in label:
                    continue
            idx = len(filtered)
            filtered.append(record)
            if previous_id is not None and int(record.get("id", -1)) == previous_id:
                restored_index = idx

        self._event_records = filtered
        total = len(filtered)
        self.eventList.setEnabled(total > 0)
        self.eventPrevBtn.setEnabled(total > 0)
        self.eventNextBtn.setEnabled(total > 0)
        self.annotationFocusOnly.setEnabled(total > 0)
        self.eventSearchEdit.setEnabled(bool(self._all_event_records))
        self.eventChannelFilter.setEnabled(bool(self._all_event_records))

        for rec in filtered:
            label = rec["label"]
            chan = rec["chan"]
            ts = self._format_clock(rec["start"])
            duration_s = rec["end"] - rec["start"]
            text = f"{ts} — {label} ({duration_s:.1f} s) [{chan}]"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, rec)
            self.eventList.addItem(item)

        if restored_index >= 0:
            self.eventList.setCurrentRow(restored_index)
            self._current_event_index = restored_index
            self._current_event_id = previous_id
        else:
            self._current_event_index = -1
            self._current_event_id = None
            self.eventList.clearSelection()
        self.eventList.blockSignals(False)
        self._update_annotation_summary()

    def _format_clock(self, seconds: float) -> str:
        tb = getattr(self.loader, "timebase", None)
        if tb is not None and getattr(tb, "start_dt", None) is not None:
            try:
                dt = tb.to_datetime(seconds)
                return dt.strftime("%H:%M:%S")
            except Exception:
                pass
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _on_event_selection_changed(self):
        if not self._event_records:
            return
        row = self.eventList.currentRow()
        if row < 0 or row >= len(self._event_records):
            return
        self._jump_to_event(row, from_selection=True)

    def _on_event_activated(self, item: QtWidgets.QListWidgetItem):
        row = self.eventList.row(item)
        if row >= 0:
            self._jump_to_event(row, from_selection=True)

    def _step_event(self, delta: int):
        if not self._event_records:
            return
        if self._current_event_index == -1:
            index = 0 if delta >= 0 else len(self._event_records) - 1
        else:
            index = max(0, min(len(self._event_records) - 1, self._current_event_index + delta))
        self._jump_to_event(index)

    def _jump_to_event(self, index: int, *, from_selection: bool = False):
        if index < 0 or index >= len(self._event_records):
            return
        record = self._event_records[index]
        self._current_event_index = index
        self._current_event_id = record["id"]
        if not from_selection:
            self.eventList.blockSignals(True)
            self.eventList.setCurrentRow(index)
            self.eventList.blockSignals(False)

        event_start = float(record["start"])
        event_end = float(record["end"])
        duration = max(event_end - event_start, 0.0)
        view_start = event_start - max(0.0, (self._view_duration - duration) * 0.5)
        max_start = max(0.0, self.loader.duration_s - self._view_duration)
        view_start = max(0.0, min(view_start, max_start))
        self._set_view(view_start, self._view_duration, sender="events")

    def _color_for_event(self, label: str) -> QtGui.QColor:
        base = self._event_color_cache.get(label)
        if base is None:
            base = pg.intColor(len(self._event_color_cache), hues=16, values=200)
            self._event_color_cache[label] = base
        return QtGui.QColor(base)

    def _update_annotation_overlays(self, t0: float, t1: float):
        if self._use_gpu_canvas:
            summary = self._event_count_summary()
            stage_label, position_label = self._current_stage_position_labels()
            self.stageSummaryLabel.setText(
                self._compose_status_text(stage_label, position_label, summary)
            )
            return
        if not self._primary_plot:
            return
        if not self._annotations_index or self._annotations_index.is_empty():
            self._clear_annotation_lines()
            summary = self._event_count_summary()
            self.stageSummaryLabel.setText(
                self._compose_status_text(None, None, summary)
            )
            return

        if not self._annotations_enabled:
            self._clear_annotation_lines()
            summary = self._event_count_summary()
            stage_label, position_label = self._current_stage_position_labels()
            self.stageSummaryLabel.setText(
                self._compose_status_text(stage_label, position_label, summary)
            )
            return

        for line in self._annotation_lines:
            line.setVisible(False)

        hidden_channels = {str(ch) for ch in self._hidden_annotation_channels}
        event_channels = [
            c for c in self._annotations_index.channel_set if str(c) not in hidden_channels
        ]
        events, ids = self._annotations_index.between(
            t0,
            t1,
            channels=event_channels or None,
            return_indices=True,
        )

        focus_only_checkbox = getattr(self, "annotationFocusOnly", None)
        focus_only = bool(focus_only_checkbox and focus_only_checkbox.isChecked())
        selected_id = self._current_event_id

        events = np.array(events, copy=False)
        ids = np.asarray(ids, dtype=int)
        if focus_only:
            if selected_id is None or ids.size == 0:
                mask = np.zeros_like(ids, dtype=bool)
            else:
                mask = ids == int(selected_id)
            events = events[mask]
            ids = ids[mask]

        self._clear_annotation_rects()
        if events.size:
            self._ensure_annotation_rect_pool(len(events))
            scene_rect = self.plotLayout.ci.mapRectToScene(self.plotLayout.ci.boundingRect())
            vb = self._primary_plot.getViewBox()
            for idx, (ev, ev_id) in enumerate(zip(events, ids)):
                start = float(ev["start_s"])
                end = float(ev["end_s"])
                if end <= start:
                    end = start + 0.5
                p1 = vb.mapViewToScene(QtCore.QPointF(start, 0))
                p2 = vb.mapViewToScene(QtCore.QPointF(end, 0))
                x1, x2 = p1.x(), p2.x()
                rect = QtCore.QRectF(min(x1, x2), scene_rect.top(), max(2.0, abs(x2 - x1)), scene_rect.height())
                color = self._color_for_event(str(ev["label"]))
                color.setAlpha(140 if ev_id == selected_id else 70)
                brush = QtGui.QBrush(color)
                item = self._annotation_rects[idx]
                item.setRect(rect)
                item.setBrush(brush)
                duration = end - start
                item.setToolTip(f"{ev['label']} ({duration:.1f}s)")
                item.setVisible(True)
            for idx in range(len(events), len(self._annotation_rects)):
                self._annotation_rects[idx].setVisible(False)
        else:
            for idx in range(len(self._annotation_rects)):
                self._annotation_rects[idx].setVisible(False)

        if annotation_core.STAGE_CHANNEL in hidden_channels:
            stage_events = np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)
        else:
            stage_events = self._annotations_index.between(
                t0, t1, channels=[annotation_core.STAGE_CHANNEL]
            )
            if isinstance(stage_events, tuple):
                stage_events = stage_events[0]
            stage_events = np.array(stage_events, copy=False)

        if annotation_core.POSITION_CHANNEL in hidden_channels:
            position_events = np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)
        else:
            position_events = self._annotations_index.between(
                t0, t1, channels=[annotation_core.POSITION_CHANNEL]
            )
            if isinstance(position_events, tuple):
                position_events = position_events[0]
            position_events = np.array(position_events, copy=False)

        self._update_stage_labels(stage_events, position_events, t0, t1)

        stage_display = None
        if getattr(stage_events, "size", 0):
            counts = Counter(stage_events["label"])
            dominant, count = counts.most_common(1)[0]
            stage_display = f"{dominant} ({count})"

        position_display = None
        if getattr(position_events, "size", 0):
            p_counts = Counter(position_events["label"])
            pos_label, p_count = p_counts.most_common(1)[0]
            position_display = f"{pos_label} ({p_count})"

        summary = self._event_count_summary()
        include_summary = (not focus_only) or bool(events.size)
        self.stageSummaryLabel.setText(
            self._compose_status_text(
                stage_display,
                position_display,
                summary if include_summary else None,
            )
        )

    def _prepare_tile(self, tile: _OverscanTile) -> bool:
        pixels = self._estimate_pixels() or 0
        overscan_span = 2 * self._overscan_factor + 1
        budget = int(max(200, pixels * overscan_span * 2)) if pixels else 2000
        want_vertices = self._use_gpu_canvas or self._gpu_autoswitch_enabled
        changed = False

        cached_series = tile.prepared_for_budget(budget)
        if cached_series is not None:
            tile.channel_data = cached_series
            tile.pixel_budget = budget
            tile.prepared_mask = [True] * len(cached_series)
            if want_vertices:
                cached_vertices = tile.prepared_vertices(budget)
                if cached_vertices is None or len(cached_vertices) != len(cached_series):
                    cached_vertices = [
                        self._make_vertex_payload(t_arr, x_arr)
                        for (t_arr, x_arr) in cached_series
                    ]
                    tile.vertex_cache[int(budget)] = cached_vertices
                    changed = True
                tile.vertex_data = cached_vertices
            else:
                if tile.vertex_data:
                    changed = True
                tile.vertex_data = []
            return changed

        same_budget = tile.pixel_budget == budget
        if same_budget and tile.channel_data and tile.prepared_mask and all(tile.prepared_mask):
            if want_vertices:
                if tile.vertex_data and len(tile.vertex_data) == len(tile.channel_data):
                    tile.vertex_cache[int(budget)] = tile.vertex_data
                else:
                    vertices = [
                        self._make_vertex_payload(t_arr, x_arr)
                        for (t_arr, x_arr) in tile.channel_data
                    ]
                    tile.vertex_data = vertices
                    tile.vertex_cache[int(budget)] = vertices
                    changed = True
            else:
                if tile.vertex_data:
                    changed = True
                tile.vertex_data = []
            tile.cache_prepared(
                budget,
                tile.channel_data,
                tile.vertex_data if want_vertices else None,
            )
            tile.pixel_budget = budget
            return changed

        prepared_series: list[tuple[np.ndarray, np.ndarray]] = []
        mask_len = len(tile.prepared_mask)
        existing_vertices = tile.vertex_data if tile.vertex_data else []
        vertex_series: list[np.ndarray] = []

        for idx, raw in enumerate(tile.raw_channel_data):
            chunk = raw
            if not isinstance(chunk, SignalChunk):
                t_arr, x_arr = raw
                chunk = SignalChunk(
                    np.asarray(t_arr, dtype=np.float64),
                    np.asarray(x_arr, dtype=np.float32),
                )
                tile.raw_channel_data[idx] = chunk

            needs_prepare = not same_budget or idx >= mask_len or not tile.prepared_mask[idx]
            if not needs_prepare and idx < len(tile.channel_data):
                series = tile.channel_data[idx]
                prepared_series.append(series)
                if want_vertices:
                    if idx < len(existing_vertices):
                        vertex_series.append(existing_vertices[idx])
                    else:
                        vertex_series.append(self._make_vertex_payload(*series))
                        changed = True
                continue

            t_slice, x_slice = slice_and_decimate(chunk, None, tile.start, tile.end, budget)
            prepared_series.append((t_slice, x_slice))
            if want_vertices:
                vertex_series.append(self._make_vertex_payload(t_slice, x_slice))
            if idx < mask_len:
                tile.prepared_mask[idx] = True
            else:
                tile.prepared_mask.append(True)
                mask_len += 1
            changed = True

        tile.channel_data = prepared_series
        tile.pixel_budget = budget
        if want_vertices:
            tile.vertex_data = vertex_series
        else:
            tile.vertex_data = []
        if prepared_series:
            tile.prepared_mask = [True] * len(prepared_series)
        else:
            tile.prepared_mask = []
        tile.cache_prepared(
            budget,
            prepared_series,
            vertex_series if want_vertices else None,
        )
        return changed

    def _apply_tile_to_curves(self, tile: _OverscanTile) -> None:
        self._prepare_tile(tile)
        is_final = bool(getattr(tile, "is_final", True))

        if tile.vertex_data:
            visible_vertex_count = sum(
                verts.shape[0]
                for idx, verts in enumerate(tile.vertex_data)
                if idx not in self._hidden_channels
            )
        else:
            visible_vertex_count = sum(
                x_arr.size
                for idx, (_t_arr, x_arr) in enumerate(tile.channel_data)
                if idx not in self._hidden_channels
            )

        self._maybe_autoswitch_renderer(visible_vertex_count)

        if self._use_gpu_canvas and self._gpu_canvas is not None:
            vertices = tile.vertex_data
            if not vertices or len(vertices) < len(tile.channel_data):
                vertices = [
                    self._make_vertex_payload(t_arr, x_arr)
                    for (t_arr, x_arr) in tile.channel_data
                ]
                tile.vertex_data = vertices
            self._gpu_canvas.apply_tile_data(
                tile.request_id,
                tile.channel_data,
                vertices,
                self._hidden_channels,
                final=is_final,
            )
            self._current_tile_id = tile.request_id
            self._update_renderer_indicator()
            return

        for idx, series in enumerate(tile.channel_data):
            t_arr, x_arr = series
            if idx < len(self.curves):
                if idx in self._hidden_channels:
                    self.curves[idx].setData([], [])
                    continue
                if not is_final and x_arr.size == 0:
                    continue
                self.curves[idx].setData(t_arr, x_arr)
        self._current_tile_id = tile.request_id
        self._update_renderer_indicator()

    def closeEvent(self, event):
        self._cleanup_ingest_thread(wait=True)
        self._prefetch.stop()
        self._shutdown_overscan_worker()
        super().closeEvent(event)

    def _update_data_source_label(self):
        renderer = "GPU (VisPy)" if self._use_gpu_canvas else "CPU (pyqtgraph)"
        if isinstance(self.loader, ZarrLoader):
            self.sourceLabel.setText(f"Source: Zarr cache | Renderer: {renderer}")
            self.sourceLabel.setStyleSheet("color: #7fb57d; font-style: italic;")
        elif getattr(self.loader, "has_cache", None) and self.loader.has_cache():
            self.sourceLabel.setText(f"Source: EDF (RAM cache) | Renderer: {renderer}")
            self.sourceLabel.setStyleSheet("color: #d7c77b; font-style: italic;")
        else:
            self.sourceLabel.setText(f"Source: EDF (live) | Renderer: {renderer}")
            self.sourceLabel.setStyleSheet("color: #9ba9bf; font-style: italic;")

    def _maybe_build_int16_cache(self) -> None:
        cache_enabled = getattr(self._config, "int16_cache_enabled", False)
        if not cache_enabled:
            return

        build_fn = getattr(self.loader, "build_int16_cache", None)
        if build_fn is None:
            LOG.warning("Int16 cache requested but loader does not support it; skipping.")
            return

        limit_mb = float(getattr(self._config, "int16_cache_max_mb", 0.0) or 0.0)
        if limit_mb <= 0:
            LOG.warning("Int16 cache enabled but max_mb is <= 0; skipping cache build.")
            return

        try:
            size_bytes = self._estimate_source_bytes()
        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.warning("Int16 cache skipped: failed to inspect EDF size (%s)", exc)
            return

        if size_bytes is None or size_bytes <= 0:
            LOG.warning("Int16 cache skipped: unable to estimate EDF size.")
            return

        max_bytes = int(limit_mb * 1024 * 1024)
        if size_bytes > max_bytes:
            LOG.warning(
                "Int16 cache skipped: EDF size %.1f MiB exceeds configured cap %.1f MiB.",
                size_bytes / (1024 * 1024),
                limit_mb,
            )
            return

        prefer_memmap = bool(getattr(self._config, "int16_cache_memmap", False))
        try:
            built = build_fn(limit_mb, prefer_memmap=prefer_memmap)
        except Exception as exc:  # pragma: no cover - log & continue without cache
            LOG.warning("Int16 cache build failed: %s", exc)
            return

        if not built:
            LOG.warning("Int16 cache skipped: loader declined to build within %.1f MiB cap.", limit_mb)

    def _estimate_source_bytes(self) -> int | None:
        path = getattr(self.loader, "path", None)
        if path:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            if size > 0:
                return size

        info = getattr(self.loader, "info", None)
        if info is None:
            return None

        try:
            total_samples = sum(getattr(ch, "n_samples", 0) for ch in info)
        except TypeError:
            return None
        if total_samples <= 0:
            return None
        return int(total_samples * np.dtype(np.int16).itemsize)

    def _estimate_pixels(self) -> int:
        if self._use_gpu_canvas and self._gpu_canvas is not None:
            return max(0, self._gpu_canvas.estimate_pixels())
        if not self._primary_plot:
            return 0
        vb = self._primary_plot.getViewBox()
        if vb is None:
            return 0
        width = int(vb.width())
        return max(0, width)

    # ----- Plot helpers ------------------------------------------------------

    def _ensure_plot_rows(self, count: int):
        if self._use_gpu_canvas:
            return
        while len(self.plots) < count:
            idx = len(self.plots)
            label = self.plotLayout.addLabel(row=idx, col=0, text="", justify="right")
            self.channel_labels.append(label)

            plot = self.plotLayout.addPlot(row=idx, col=1)
            plot.showAxis("bottom", show=False)
            plot.showAxis("left", show=False)
            plot.showAxis("right", show=False)
            plot.showAxis("top", show=False)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=True, y=False)
            plot.showGrid(x=False, y=True, alpha=0.15)
            curve_color = self._curve_color(idx)
            curve = plot.plot([], [], pen=pg.mkPen(curve_color, width=1.2))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")

            self.plots.append(plot)
            self.curves.append(curve)

    def _ensure_hypnogram_plot(self) -> None:
        if self._use_gpu_canvas:
            return
        row = len(self.plots)
        if self.hypnogramPlot is None:
            label = self.plotLayout.addLabel(row=row, col=0, text="Hypnogram", justify="right")
            self._hypnogram_label = label
            plot = self.plotLayout.addPlot(row=row, col=1)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.showAxis("left", show=False)
            plot.showAxis("right", show=False)
            plot.showAxis("top", show=False)
            vb = plot.getViewBox()
            if vb is not None:
                vb.enableAutoRange(x=False, y=False)
            plot.setMinimumHeight(72)
            plot.setXRange(self._view_start, self._view_start + self._view_duration, padding=0)
            outline = plot.plot([], [], pen=pg.mkPen("#f0f4ff", width=1.0))
            outline.setZValue(5)
            outline.setVisible(False)
            self._hypnogram_outline = outline

            region = pg.LinearRegionItem(
                values=(self._view_start, self._view_start + self._view_duration),
                movable=True,
            )
            region.setZValue(20)
            region.sigRegionChanged.connect(self._on_hypnogram_region_changed)
            plot.addItem(region)

            self.hypnogramPlot = plot
            self.hypnogramRegion = region
        else:
            try:
                self.plotLayout.removeItem(self.hypnogramPlot)
            except (KeyError, ValueError):
                pass
            if self._hypnogram_label is not None:
                try:
                    self.plotLayout.removeItem(self._hypnogram_label)
                except (KeyError, ValueError):
                    pass
            if self._hypnogram_label is not None:
                self.plotLayout.addItem(self._hypnogram_label, row=row, col=0)
            self.plotLayout.addItem(self.hypnogramPlot, row=row, col=1)

        self._update_hypnogram_bounds()
        self._update_hypnogram_region(
            self._view_start,
            self._view_start + self._view_duration,
            force=True,
        )

    def _update_hypnogram_axis(self) -> None:
        if self._use_gpu_canvas:
            return
        plot = self.hypnogramPlot
        if plot is None:
            return
        plot.setAxisItems({"bottom": self.time_axis})
        plot.showAxis("bottom", show=True)
        plot.showAxis("left", show=False)
        plot.showAxis("right", show=False)
        plot.showAxis("top", show=False)
        if self._primary_plot is not None:
            plot.setXLink(self._primary_plot)
        else:
            plot.setXLink(None)
        plot.setLimits(xMin=0.0, xMax=self.loader.duration_s)
        vb = plot.getViewBox()
        if vb is not None:
            vb.setMouseEnabled(x=False, y=False)
            vb.enableAutoRange(x=False, y=False)

    def _update_hypnogram_bounds(self) -> None:
        if self._use_gpu_canvas:
            return
        total = getattr(self.loader, "duration_s", None)
        region = self.hypnogramRegion
        if region is not None and total is not None:
            region.setBounds((0.0, float(total)))

    def _update_hypnogram_region(self, start: float, end: float, *, force: bool = False) -> None:
        if self._use_gpu_canvas:
            return
        region = self.hypnogramRegion
        if region is None:
            return
        total = float(getattr(self.loader, "duration_s", 0.0) or 0.0)
        start = max(0.0, min(start, total))
        end = max(start, min(end, total))
        current = region.getRegion()
        if not force and current and abs(current[0] - start) < 1e-6 and abs(current[1] - end) < 1e-6:
            return
        self._updating_hypnogram_region = True
        try:
            region.setRegion((start, end))
        finally:
            self._updating_hypnogram_region = False

    def _set_hypnogram_visible(self, visible: bool) -> None:
        if self._use_gpu_canvas:
            return
        plot = self.hypnogramPlot
        if plot is None:
            return
        plot.setVisible(visible)
        if self._hypnogram_label is not None:
            self._hypnogram_label.setVisible(visible)
        region = self.hypnogramRegion
        if region is not None:
            region.setVisible(visible)
        if not visible and self._hypnogram_outline is not None:
            self._hypnogram_outline.setVisible(False)
        if not visible:
            for item in self._hypnogram_fill_curves.values():
                item.setVisible(False)

    def _ensure_hypnogram_curves(
        self, labels: Sequence[str], colors: dict[str, QtGui.QColor]
    ) -> None:
        if self._use_gpu_canvas:
            return
        plot = self.hypnogramPlot
        if plot is None:
            return
        for label in labels:
            if label not in self._hypnogram_fill_curves:
                base_color = QtGui.QColor(colors.get(label, QtGui.QColor(DEFAULT_STAGE_COLOR)))
                fill_color = QtGui.QColor(base_color)
                fill_color.setAlpha(140)
                pen_color = QtGui.QColor(base_color)
                pen_color.setAlpha(200)
                item = pg.PlotDataItem()
                item.setZValue(-10)
                item.setPen(pg.mkPen(pen_color, width=1.0))
                item.setBrush(pg.mkBrush(fill_color))
                item.setVisible(False)
                plot.addItem(item)
                self._hypnogram_fill_curves[label] = item
        for label, item in list(self._hypnogram_fill_curves.items()):
            if label not in labels:
                item.clear()
                item.setVisible(False)

    def _invalidate_stage_curve_cache(self) -> None:
        self._stage_curve_cache = None

    def _stage_curve_data(self) -> dict[str, object]:
        if self._stage_curve_cache is not None:
            return self._stage_curve_cache

        stages = self._stage_annotations()
        if stages.size == 0:
            cache = {
                "step_x": np.zeros(0, dtype=float),
                "step_y": np.zeros(0, dtype=float),
                "label_data": {},
                "labels": [],
                "level_map": {},
                "colors": {},
                "max_level": -1,
            }
            self._stage_curve_cache = cache
            return cache

        stage_order: list[str] = []
        if self._annotations_index is not None:
            for meta in getattr(self._annotations_index, "sources", []):
                if not isinstance(meta, dict):
                    continue
                stage_map = meta.get("stage_map") if isinstance(meta, dict) else None
                if isinstance(stage_map, dict):
                    for value in stage_map.values():
                        label = str(value)
                        if label and label not in stage_order:
                            stage_order.append(label)

        for label in STAGE_COLORS:
            if label not in stage_order:
                stage_order.append(label)

        for raw_label in stages["label"]:
            label = str(raw_label)
            if label and label not in stage_order:
                stage_order.append(label)

        level_map = {label: idx for idx, label in enumerate(stage_order)}
        color_map = {label: QtGui.QColor(STAGE_COLORS.get(label, DEFAULT_STAGE_COLOR)) for label in stage_order}

        step_x: list[float] = []
        step_y: list[float] = []
        label_data: dict[str, dict[str, object]] = {}
        prev_end: float | None = None

        segments: list[dict[str, object]] = []
        for entry in stages:
            start = float(entry["start_s"])
            end = float(entry["end_s"])
            label = str(entry["label"])
            level = level_map[label]
            if prev_end is not None and start - prev_end > 1e-6:
                step_x.append(float("nan"))
                step_y.append(float("nan"))
            step_x.extend([start, end])
            step_y.extend([level, level])
            prev_end = end

            if segments and segments[-1]["label"] == label and abs(float(segments[-1]["end"]) - start) < 1e-6:
                segments[-1]["end"] = end
            else:
                segments.append({"start": start, "end": end, "label": label, "level": level})

        for segment in segments:
            label = str(segment["label"])
            level = int(segment["level"])
            payload = label_data.setdefault(
                label,
                {"x": [], "top": [], "fill": level - 0.45},
            )
            xs: list[float] = payload["x"]  # type: ignore[assignment]
            ys: list[float] = payload["top"]  # type: ignore[assignment]
            if xs:
                last_val = xs[-1]
                if not np.isnan(last_val) and float(segment["start"]) - last_val > 1e-6:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
            xs.extend([float(segment["start"]), float(segment["end"])])
            top_value = level + 0.45
            ys.extend([top_value, top_value])

        for payload in label_data.values():
            payload["x"] = np.asarray(payload["x"], dtype=float)
            payload["top"] = np.asarray(payload["top"], dtype=float)

        cache = {
            "step_x": np.asarray(step_x, dtype=float),
            "step_y": np.asarray(step_y, dtype=float),
            "label_data": label_data,
            "labels": stage_order,
            "level_map": level_map,
            "colors": color_map,
            "max_level": max(level_map.values()) if level_map else -1,
        }
        self._stage_curve_cache = cache
        return cache

    def _update_hypnogram(self, t0: float, t1: float) -> None:
        plot = self.hypnogramPlot
        if plot is None:
            return
        has_stages = (
            self._annotations_enabled
            and self._annotations_index is not None
            and not self._annotations_index.is_empty()
            and annotation_core.STAGE_CHANNEL not in self._hidden_annotation_channels
        )
        if not has_stages:
            self._set_hypnogram_visible(False)
            self._update_hypnogram_region(t0, t1)
            return

        data = self._stage_curve_data()
        step_x = data["step_x"]
        step_y = data["step_y"]
        if step_x.size == 0:
            self._set_hypnogram_visible(False)
            self._update_hypnogram_region(t0, t1)
            return

        self._set_hypnogram_visible(True)
        self._ensure_hypnogram_curves(data["labels"], data["colors"])

        max_level = max(0, int(data.get("max_level", 0)))
        plot.setYRange(-0.6, max_level + 0.6, padding=0)
        plot.setLimits(yMin=-0.6, yMax=max_level + 0.6)

        if self._hypnogram_outline is not None:
            self._hypnogram_outline.setData(step_x, step_y, connect="finite")
            self._hypnogram_outline.setVisible(True)

        label_data: dict[str, dict[str, object]] = data["label_data"]
        for label, item in self._hypnogram_fill_curves.items():
            payload = label_data.get(label)
            if not payload:
                item.clear()
                item.setVisible(False)
                continue
            x_vals = payload.get("x")
            y_vals = payload.get("top")
            if isinstance(x_vals, np.ndarray) and isinstance(y_vals, np.ndarray) and x_vals.size:
                item.setData(x_vals, y_vals, connect="finite")
                fill_level = float(payload.get("fill", 0.0))
                item.setFillLevel(fill_level)
                item.setVisible(True)
            else:
                item.clear()
                item.setVisible(False)

        self._update_hypnogram_region(t0, t1)

    def _configure_plots(self):
        n = self.loader.n_channels
        # Trim hidden channels to valid range
        self._hidden_channels = {idx for idx in self._hidden_channels if 0 <= idx < n}
        self._sync_channel_controls()

        if self._use_gpu_canvas and self._gpu_canvas is not None:
            self._configure_gpu_canvas()
            self._primary_plot = None
            return

        old_primary = self._primary_plot
        self._ensure_plot_rows(n)

        # Reset previous primary axis if needed
        if self._primary_plot and self._primary_plot not in self.plots[:n]:
            self._primary_plot.setAxisItems({"bottom": pg.AxisItem(orientation="bottom")})
            self._primary_plot.showAxis("bottom", show=False)

        for idx, plot in enumerate(self.plots):
            active = idx < n
            if not active:
                plot.hide()
                self.channel_labels[idx].setText("")
                self.channel_labels[idx].setVisible(False)
                self.curves[idx].setData([], [])
                continue

            meta = self.loader.info[idx]
            visible = idx not in self._hidden_channels
            self._apply_channel_visible(
                idx,
                visible,
                sync_checkbox=False,
                persist=False,
            )

            self.curves[idx].setPen(pg.mkPen(self._curve_color(idx), width=1.2))
            plot.showAxis("bottom", show=False)

        if n == 0:
            self._primary_plot = None
            self._ensure_hypnogram_plot()
            self._update_hypnogram_axis()
            return

        new_primary = self.plots[n - 1]
        if self._primary_plot and self._primary_plot is not new_primary:
            self._primary_plot.setAxisItems({"bottom": pg.AxisItem(orientation="bottom")})
            self._primary_plot.showAxis("bottom", show=False)

        if old_primary and old_primary is not new_primary:
            for line in self._annotation_lines:
                try:
                    old_primary.removeItem(line)
                except Exception:
                    pass

        self._primary_plot = new_primary
        self._connect_primary_viewbox()

        self._ensure_hypnogram_plot()
        self._update_hypnogram_axis()

        if self._annotation_lines:
            for line in self._annotation_lines:
                new_primary.addItem(line)

        for idx, plot in enumerate(self.plots[:n]):
            if plot is new_primary:
                plot.setXLink(None)
            else:
                plot.showAxis("bottom", show=False)
                plot.setXLink(new_primary)

        for plot in self.plots[n:]:
            plot.setXLink(None)
            plot.showAxis("bottom", show=False)
            if plot.getAxis("bottom") is self.time_axis:
                plot.setAxisItems({"bottom": pg.AxisItem(orientation="bottom")})

        self._sync_hover_overlay_target()

    def _configure_gpu_canvas(self) -> None:
        if not (self._use_gpu_canvas and self._gpu_canvas is not None):
            return
        infos = getattr(self.loader, "info", [])
        hidden = {idx for idx in self._hidden_channels if 0 <= idx < len(infos)}
        self._gpu_canvas.configure_channels(infos=infos, hidden_indices=hidden)
        self._gpu_canvas.set_view(self._view_start, self._view_duration)
        theme = self._theme
        colors = theme.curve_colors or ("#5f8bff",)
        self._gpu_canvas.set_theme(
            background=theme.pg_background,
            curve_colors=colors,
            label_color=theme.pg_foreground,
        )
        self._gpu_canvas.set_hover_enabled(False)

    def _sync_channel_controls(self) -> None:
        if not hasattr(self, "channelSection"):
            return
        n = self.loader.n_channels
        self.channelSection.setVisible(n > 0)
        layout = getattr(self, "_channel_list_layout", None)
        if layout is None:
            return

        while len(self.channel_checkboxes) < n:
            checkbox = QtWidgets.QCheckBox()
            checkbox.setCursor(QtCore.Qt.PointingHandCursor)
            idx = len(self.channel_checkboxes)
            checkbox.toggled.connect(partial(self._on_channel_checkbox_toggled, idx))
            layout.insertWidget(max(0, layout.count() - 1), checkbox)
            self.channel_checkboxes.append(checkbox)

        while len(self.channel_checkboxes) > n:
            checkbox = self.channel_checkboxes.pop()
            checkbox.hide()
            checkbox.deleteLater()

        hidden = self._hidden_channels
        for idx, checkbox in enumerate(self.channel_checkboxes):
            meta = self.loader.info[idx]
            label = meta.name
            if getattr(meta, "unit", ""):
                label = f"{label} [{meta.unit}]"
            checkbox.blockSignals(True)
            checkbox.setText(label)
            checkbox.setChecked(idx not in hidden)
            checkbox.blockSignals(False)

    def _apply_channel_visible(
        self,
        idx: int,
        visible: bool,
        *,
        sync_checkbox: bool,
        persist: bool,
    ) -> None:
        prev_hidden = idx in self._hidden_channels
        state_changed = False
        if self._use_gpu_canvas and self._gpu_canvas is not None:
            n = self.loader.n_channels
            if idx >= n:
                self._hidden_channels.discard(idx)
                return
            meta = self.loader.info[idx]
            if visible:
                self._hidden_channels.discard(idx)
            else:
                self._hidden_channels.add(idx)
                self._gpu_canvas.clear_channel(idx)
            state_changed = (idx in self._hidden_channels) != prev_hidden
            self._gpu_canvas.set_channel_visibility(idx, visible)
            self._gpu_canvas.set_channel_label(idx, self._format_label(meta, hidden=not visible))
            if sync_checkbox and idx < len(self.channel_checkboxes):
                checkbox = self.channel_checkboxes[idx]
                checkbox.blockSignals(True)
                checkbox.setChecked(visible)
                checkbox.blockSignals(False)
            if persist:
                self._config.hidden_channels = tuple(sorted(self._hidden_channels))
                self._config.save()
                if state_changed:
                    self._invalidate_overscan_tile_cache()
            return
        if idx >= len(self.plots):
            return

        n = self.loader.n_channels
        plot = self.plots[idx]
        label_item = self.channel_labels[idx]
        curve = self.curves[idx]
        meta = self.loader.info[idx] if idx < n else None

        if idx >= n:
            plot.hide()
            label_item.setVisible(False)
            curve.setData([], [])
            self._hidden_channels.discard(idx)
            return

        if visible:
            plot.show()
            self._hidden_channels.discard(idx)
        else:
            plot.hide()
            self._hidden_channels.add(idx)
            curve.setData([], [])
        state_changed = (idx in self._hidden_channels) != prev_hidden

        if meta is not None:
            label_item.setVisible(visible)
            label_item.setText(self._format_label(meta, hidden=not visible))

        if sync_checkbox and idx < len(self.channel_checkboxes):
            checkbox = self.channel_checkboxes[idx]
            checkbox.blockSignals(True)
            checkbox.setChecked(visible)
            checkbox.blockSignals(False)

        if persist:
            self._config.hidden_channels = tuple(sorted(self._hidden_channels))
            self._config.save()
            if state_changed:
                self._invalidate_overscan_tile_cache()

    def _attach_hover_overlay(self, plot: pg.PlotItem | None) -> None:
        if self._use_gpu_canvas:
            return
        if self._hover_line is None or self._hover_label is None:
            return
        if plot is self._hover_plot:
            return
        if self._hover_plot is not None:
            try:
                self._hover_plot.removeItem(self._hover_line)
            except Exception:
                pass
            try:
                self._hover_plot.removeItem(self._hover_label)
            except Exception:
                pass
        self._hover_plot = plot
        if plot is not None:
            plot.addItem(self._hover_line, ignoreBounds=True)
            plot.addItem(self._hover_label, ignoreBounds=True)

    def _sync_hover_overlay_target(self) -> None:
        if self._use_gpu_canvas:
            return
        if self._hover_line is None or self._hover_label is None:
            return
        plot = self._hover_plot
        if plot is None:
            return
        if plot not in self.plots[: self.loader.n_channels] or not plot.isVisible():
            self._hide_hover_indicator(detach=True)
            return
        target_vb = plot.getViewBox()
        get_viewbox = getattr(self._hover_line, "getViewBox", None)
        line_vb = get_viewbox() if callable(get_viewbox) else None
        if target_vb is not None and line_vb is not target_vb:
            self._attach_hover_overlay(plot)

    def _hide_hover_indicator(self, *, detach: bool = False) -> None:
        if self._use_gpu_canvas:
            return
        if self._hover_line is not None:
            self._hover_line.setVisible(False)
        if self._hover_label is not None:
            self._hover_label.setVisible(False)
        if detach:
            self._attach_hover_overlay(None)

    @QtCore.Slot()
    def _on_plot_scene_mouse_exited(self) -> None:
        if self._use_gpu_canvas:
            return
        self._hide_hover_indicator()

    @QtCore.Slot(QtCore.QPointF)
    def _update_hover_indicator(self, scene_pos: QtCore.QPointF) -> None:
        if self._use_gpu_canvas:
            return
        if self._hover_line is None or self._hover_label is None:
            return
        if self._view_duration > 300 or self._primary_plot is None:
            self._hide_hover_indicator(detach=True)
            return

        active_plot: pg.PlotItem | None = None
        active_idx: int | None = None
        for idx, plot in enumerate(self.plots[: self.loader.n_channels]):
            if idx in self._hidden_channels:
                continue
            if not plot.isVisible():
                continue
            vb = plot.vb if hasattr(plot, "vb") else plot.getViewBox()
            if vb is None:
                continue
            if not vb.sceneBoundingRect().contains(scene_pos):
                continue
            active_plot = plot
            active_idx = idx
            break

        if active_plot is None or active_idx is None:
            self._hide_hover_indicator(detach=True)
            return

        vb = active_plot.vb if hasattr(active_plot, "vb") else active_plot.getViewBox()
        if vb is None:
            self._hide_hover_indicator(detach=True)
            return

        view_point = vb.mapSceneToView(scene_pos)
        t_val = float(view_point.x())
        curve = self.curves[active_idx]
        sample = _sample_at_time(curve, t_val)
        if sample is None:
            self._hide_hover_indicator(detach=True)
            return

        sample_time, sample_value = sample
        meta = self.loader.info[active_idx]
        unit = getattr(meta, "unit", "") or ""
        value_text = f"{sample_value:.3f}"
        if unit:
            value_text = f"{value_text} {unit}"

        self._attach_hover_overlay(active_plot)
        self._hover_line.setValue(sample_time)
        self._hover_line.setVisible(True)

        y_min, y_max = vb.viewRange()[1]
        anchor_y = 1.0
        span = abs(y_max - y_min)
        if span > 0 and (y_max - sample_value) < 0.1 * span:
            anchor_y = 0.0
        new_anchor = (0.0, anchor_y)
        if new_anchor != self._hover_label_anchor:
            self._hover_label.setAnchor(new_anchor)
            self._hover_label_anchor = new_anchor
        self._hover_label.setText(value_text)
        self._hover_label.setPos(sample_time, sample_value)
        self._hover_label.setVisible(True)

    def _on_channel_checkbox_toggled(self, idx: int, checked: bool) -> None:
        self._set_channel_visible(idx, bool(checked))

    @QtCore.Slot(int, bool)
    def _set_channel_visible(self, idx: int, visible: bool) -> None:
        self._apply_channel_visible(idx, bool(visible), sync_checkbox=True, persist=True)
        self.refresh()

    def _format_label(self, meta, *, hidden: bool = False) -> str:
        unit = f" [{meta.unit}]" if getattr(meta, "unit", "") else ""
        text = f"{meta.name}{unit}"
        if hidden:
            text = f"{text} (hidden)"

        if self._use_gpu_canvas and self._gpu_canvas is not None:
            return text

        theme = getattr(self, "_theme", THEMES[DEFAULT_THEME])
        if hidden:
            color = theme.channel_label_hidden
            extra = "font-style: italic; opacity:0.7;"
        else:
            color = theme.channel_label_active
            extra = "font-weight:600;"
        return (
            "<span style='color:" + color + ";" + extra + "font-size:11pt;padding-right:12px;'>"
            f"{text}"
            "</span>"
        )

    def _auto_hide_annotation_channels(self) -> None:
        info = getattr(self.loader, "info", None)
        if not info:
            return
        for idx, meta in enumerate(info):
            name = getattr(meta, "name", "")
            if not name:
                continue
            lowered = str(name).strip().lower()
            if not lowered:
                continue
            sanitized = re.sub(r"[^a-z0-9]", "", lowered)
            tokens = [tok for tok in re.split(r"[^a-z0-9]+", lowered) if tok]
            token_set = set(tokens)

            stage_hit = (
                "hypnogram" in token_set
                or ("sleep" in token_set and "stage" in token_set)
                or sanitized in {"stage", "stages", "sleepstage", "sleepstages"}
            )
            position_hit = (
                ("body" in token_set and "position" in token_set)
                or "posture" in token_set
                or sanitized in {"bodyposition", "bodypos", "positionbody"}
            )

            if stage_hit or position_hit:
                self._hidden_channels.add(idx)

    def _connect_primary_viewbox(self):
        if self._use_gpu_canvas:
            return
        if self._primary_viewbox is not None:
            try:
                self._primary_viewbox.sigXRangeChanged.disconnect(self._on_viewbox_range)
            except (TypeError, RuntimeError):
                pass
        self._primary_viewbox = None
        if self._primary_plot is None:
            return
        vb = self._primary_plot.getViewBox()
        if vb is None:
            return
        self._primary_viewbox = vb
        vb.sigXRangeChanged.connect(self._on_viewbox_range)
        vb.setMouseEnabled(x=True, y=False)
        vb.enableAutoRange(y=True)
        self._update_viewbox_from_state()

    def _on_viewbox_range(self, viewbox, xrange):
        if self._use_gpu_canvas:
            return
        if viewbox is not self._primary_viewbox or self._updating_viewbox:
            return
        if not xrange or len(xrange) != 2:
            return
        start = float(xrange[0])
        end = float(xrange[1])
        duration = max(self._limits.duration_min, end - start)
        self._set_view(start, duration, sender="viewbox")

    @QtCore.Slot()
    def _on_hypnogram_region_changed(self):
        if self._updating_hypnogram_region or self.hypnogramRegion is None:
            return
        region = self.hypnogramRegion.getRegion()
        if not region or len(region) != 2:
            return
        start, end = region
        duration = max(self._limits.duration_min, end - start)
        self._set_view(start, duration, sender="hypnogram")
