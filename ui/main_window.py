# ui/main_window.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui

from ui.time_axis import TimeAxis
from config import ViewerConfig
from core.decimate import min_max_bins
from core.overscan import slice_and_decimate
from core.prefetch import prefetch_service
from core.view_window import WindowLimits, clamp_window, pan_window, zoom_window
from core.zarr_cache import EdfToZarr, resolve_output_path
from core.zarr_loader import ZarrLoader
from core import annotations as annotation_core


LOG = logging.getLogger(__name__)


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
    raw_channel_data: list[tuple[np.ndarray, np.ndarray]]
    channel_data: list[tuple[np.ndarray, np.ndarray]]
    max_samples: Optional[int]
    pixel_budget: Optional[int] = None

    def contains(self, window_start: float, window_end: float) -> bool:
        return window_start >= self.start and window_end <= self.end


class _OverscanWorker(QtCore.QObject):
    finished = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(self, loader):
        super().__init__()
        self._loader = loader

    @QtCore.Slot(object)
    def render(self, request_obj):
        if not isinstance(request_obj, _OverscanRequest):
            return
        req: _OverscanRequest = request_obj
        try:
            data: list[tuple[np.ndarray, np.ndarray]] = []
            for ch in req.channel_indices:
                data.append(self._read_channel(ch, req.start, req.end, req.max_samples))
        except Exception as exc:  # pragma: no cover - worker error propagated to UI
            self.failed.emit(req.request_id, str(exc))
            return

        tile = _OverscanTile(
            request_id=req.request_id,
            start=req.start,
            end=req.end,
            view_start=req.view_start,
            view_duration=req.view_duration,
            raw_channel_data=data,
            channel_data=list(data),
            max_samples=req.max_samples,
        )
        self.finished.emit(req.request_id, tile)

    def _read_channel(self, channel: int, start: float, end: float, max_samples: Optional[int]):
        try:
            if max_samples is not None:
                return self._loader.read(channel, start, end, max_samples=max_samples)
        except TypeError:
            pass
        return self._loader.read(channel, start, end)


class MainWindow(QtWidgets.QMainWindow):
    overscanRequested = QtCore.Signal(object)
    def __init__(self, loader, *, config: ViewerConfig | None = None):
        super().__init__()
        self.loader = loader
        self._config = config or ViewerConfig()
        self._ingest_thread: QtCore.QThread | None = None
        self._ingest_worker: _ZarrIngestWorker | None = None
        self._zarr_path: Path | None = None
        self._pending_loader: object | None = None
        self._primary_viewbox = None
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
        self._prefetch = prefetch_service.create_cache(self._fetch_tile)
        self._prefetch.start()

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#10131d")
        pg.setConfigOption("foreground", "#e3e7f3")

        self.setWindowTitle("EDF Viewer — Multi-channel")
        self.time_axis = TimeAxis(orientation="bottom", timebase=loader.timebase)
        if self.loader.timebase.start_dt is not None:
            self.time_axis.set_mode("absolute")

        self._primary_plot = None
        self._overscan_factor = 2.0  # windows per side
        self._overscan_tile: _OverscanTile | None = None
        self._overscan_request_id = 0
        self._overscan_inflight: Optional[int] = None
        self._current_tile_id: Optional[int] = None
        self._overscan_thread: QtCore.QThread | None = None
        self._overscan_worker: _OverscanWorker | None = None
        self._init_overscan_worker()

        self._manual_annotation_paths: dict[str, Path] = {}
        self._annotations_index: annotation_core.AnnotationIndex | None = None
        self._annotation_lines: list[pg.InfiniteLine] = []
        self._annotations_enabled = False
        self._annotation_rects: list[QtWidgets.QGraphicsRectItem] = []
        self._event_records: list[dict[str, float | str | int]] = []
        self._current_event_index: int = -1
        self._current_event_id: Optional[int] = None
        self._event_color_cache: dict[str, QtGui.QColor] = {}
        self.stage_plot: pg.PlotItem | None = None
        self._stage_curve: pg.PlotDataItem | None = None
        self._stage_label_item: pg.LabelItem | None = None

        self._build_ui()
        self._connect_signals()
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(60)
        self._debounce_timer.timeout.connect(self.refresh)
        self._refresh_limits()
        self._update_controls_from_state()
        self.refresh()
        self._update_data_source_label()
        self._manual_annotation_paths.clear()
        self._start_zarr_ingest()
        self._load_companion_annotations()
        QtCore.QTimer.singleShot(0, self._ensure_overscan_for_view)

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
        self.stageSummaryLabel = QtWidgets.QLabel("Stage: --")
        self.stageSummaryLabel.setObjectName("stageSummary")
        self.sourceLabel = QtWidgets.QLabel("Source: EDF (live)")
        self.sourceLabel.setObjectName("sourceLabel")
        self.annotationToggle = QtWidgets.QCheckBox("Show annotations")
        self.annotationToggle.setChecked(True)
        self.annotationToggle.setEnabled(False)
        self.annotationImportBtn = QtWidgets.QPushButton("Import annotations…")
        self.annotationImportBtn.setEnabled(True)
        eventHeader = QtWidgets.QLabel("Annotations")
        eventHeader.setObjectName("annotationsHeader")
        self.eventList = QtWidgets.QListWidget()
        self.eventList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.eventList.setEnabled(False)
        self.eventPrevBtn = QtWidgets.QPushButton("Prev")
        self.eventPrevBtn.setEnabled(False)
        self.eventNextBtn = QtWidgets.QPushButton("Next")
        self.eventNextBtn.setEnabled(False)

        control = QtWidgets.QFrame()
        control.setObjectName("controlPanel")
        controlLayout = QtWidgets.QVBoxLayout(control)
        controlLayout.setContentsMargins(18, 18, 18, 18)
        controlLayout.setSpacing(16)

        header = QtWidgets.QLabel("Viewing Window")
        header.setObjectName("panelTitle")

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

        controlLayout.addWidget(header)
        controlLayout.addLayout(navLayout)
        self.fileButton = QtWidgets.QPushButton("Open EDF…")
        self.fileButton.setObjectName("fileSelectButton")
        self.fileButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.fileButton.setCursor(QtCore.Qt.PointingHandCursor)
        controlLayout.addWidget(self.fileButton)
        controlLayout.addLayout(form)
        controlLayout.addWidget(self.absoluteRange)
        controlLayout.addWidget(self.windowSummary)
        controlLayout.addWidget(self.stageSummaryLabel)
        controlLayout.addWidget(self.sourceLabel)
        controlLayout.addWidget(self.annotationToggle)
        controlLayout.addWidget(self.annotationImportBtn)
        controlLayout.addWidget(eventHeader)
        controlLayout.addWidget(self.eventList)
        eventNav = QtWidgets.QHBoxLayout()
        eventNav.addWidget(self.eventPrevBtn)
        eventNav.addWidget(self.eventNextBtn)
        controlLayout.addLayout(eventNav)
        prefetchBox = QtWidgets.QGroupBox("Prefetch")
        prefetchLayout = QtWidgets.QGridLayout(prefetchBox)
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
        controlLayout.addWidget(prefetchBox)
        self.ingestBar = QtWidgets.QProgressBar()
        self.ingestBar.setObjectName("ingestBar")
        self.ingestBar.setRange(0, 100)
        self.ingestBar.setValue(0)
        self.ingestBar.setFormat("Caching EDF → Zarr: %p%")
        self.ingestBar.setTextVisible(True)
        self.ingestBar.hide()
        controlLayout.addWidget(self.ingestBar)
        controlLayout.addStretch(1)

        self.plotLayout = pg.GraphicsLayoutWidget()
        self.plotLayout.ci.layout.setSpacing(0)
        self.plotLayout.ci.layout.setContentsMargins(0, 0, 0, 0)

        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []
        self.channel_labels: list[pg.LabelItem] = []

        self._ensure_plot_rows(self.loader.n_channels)
        self._configure_plots()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.plotLayout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(control)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 940])

        central = QtWidgets.QWidget()
        centralLayout = QtWidgets.QVBoxLayout(central)
        centralLayout.setContentsMargins(0, 0, 0, 0)
        centralLayout.setSpacing(0)
        centralLayout.addWidget(splitter)
        self.setCentralWidget(central)

        self.setStyleSheet(
            """
            QMainWindow { background-color: #0b111c; color: #e6ebf5; }
            QLabel { font-size: 13px; color: #e6ebf5; }
            QLabel#panelTitle { font-size: 15px; font-weight: 600; color: #f3f6ff; }
            QLabel#absoluteRange, QLabel#windowSummary { color: #9ba9bf; }
            QLabel#sourceLabel { color: #9ba9bf; font-style: italic; }
            QDoubleSpinBox {
                background-color: #121a2a;
                border: 1px solid #1f2a3d;
                border-radius: 6px;
                padding: 6px 8px;
                color: #f0f4ff;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: transparent;
                border: none;
            }
            QFrame#controlPanel {
                background-color: #131b2b;
                border-right: 1px solid #1f2a3d;
            }
            QPushButton#fileSelectButton {
                background-color: #1a2436;
                border: 1px solid #27324a;
                border-radius: 6px;
                padding: 8px 12px;
                color: #f3f6ff;
                font-weight: 600;
            }
            QPushButton#fileSelectButton:hover {
                background-color: #22304a;
                border-color: #39507a;
            }
           QPushButton#fileSelectButton:pressed {
               background-color: #182235;
           }
            QGroupBox QPushButton {
                background-color: #1c273a;
                border: 1px solid #2b3850;
                border-radius: 6px;
                padding: 6px 10px;
                color: #e1e9ff;
            }
            QGroupBox QPushButton:hover {
                background-color: #263755;
            }
            QGroupBox QPushButton:pressed {
                background-color: #142033;
            }
            QGroupBox {
                margin-top: 10px;
                border: 1px solid #1f2a3d;
                border-radius: 6px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #a7b4cf;
            }
            QToolButton {
                background-color: #1a2333;
                border: 1px solid #263247;
                border-radius: 4px;
                padding: 4px 8px;
                color: #dfe7ff;
            }
            QToolButton:hover {
                background-color: #25314a;
            }
            QToolButton:pressed {
                background-color: #172132;
            }
            QProgressBar#ingestBar {
                background-color: #121a24;
                border: 1px solid #1f2a3d;
                border-radius: 6px;
                padding: 3px;
                color: #e6ebf5;
            }
            QProgressBar#ingestBar::chunk {
                background-color: #3d6dff;
                border-radius: 4px;
            }
            QScrollArea { background-color: #0b111c; }
        """
        )

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
        self.annotationToggle.toggled.connect(self._on_annotation_toggle)
        self.annotationImportBtn.clicked.connect(self._prompt_import_annotations)
        self.eventList.itemSelectionChanged.connect(self._on_event_selection_changed)
        self.eventList.itemDoubleClicked.connect(self._on_event_activated)
        self.eventPrevBtn.clicked.connect(lambda: self._step_event(-1))
        self.eventNextBtn.clicked.connect(lambda: self._step_event(1))

        self._shortcuts: list[QtGui.QShortcut] = []
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, activated=lambda: self._pan_fraction(-0.1)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=lambda: self._pan_fraction(0.1)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Minus), self, activated=lambda: self._zoom_factor(2.0)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Equal), self, activated=lambda: self._zoom_factor(0.5)))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence("F"), self, activated=self._full_view))
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self._reset_view))

    def _init_overscan_worker(self):
        self._shutdown_overscan_worker()
        thread = QtCore.QThread(self)
        worker = _OverscanWorker(self.loader)
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
                t, x = self.loader.read(i, t0, t1)
                if pixels and x.size > pixels * 2:
                    t, x = min_max_bins(t, x, pixels)
                self.curves[i].setData(t, x)

        if self._primary_plot is not None:
            self._update_viewbox_from_state()
        self._update_time_labels(t0, t1)

        if not used_tile and self._overscan_inflight is None:
            self._ensure_overscan_for_view()

        self._update_annotation_overlays(t0, t1)

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
            new_loader = type(old_loader)(path)
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
        self.refresh()
        self._manual_annotation_paths.clear()
        self._load_companion_annotations()
        self._overscan_tile = None
        self._overscan_inflight = None
        self._current_tile_id = None
        self._init_overscan_worker()
        self._ensure_overscan_for_view()
        self._start_zarr_ingest()
        self._prefetch.clear()
        self._schedule_prefetch()

    def _start_zarr_ingest(self):
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
        self.loader = pending
        if isinstance(self.loader, ZarrLoader):
            setattr(self.loader, "max_window_s", self.loader.duration_s)
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
        self._overscan_tile = None
        self._overscan_inflight = None
        self._current_tile_id = None
        self._init_overscan_worker()
        self._manual_annotation_paths.clear()
        self.refresh()
        self._load_companion_annotations()
        self._update_data_source_label()
        self._prefetch.clear()
        self._schedule_prefetch()
        self._ensure_overscan_for_view()

        if hasattr(old_loader, "close") and not isinstance(old_loader, ZarrLoader):
            old_loader.close()

    def _set_view(self, start: float, duration: float, *, sender: str | None = None):
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

        self._view_start = start_new
        self._view_duration = duration_new
        self._refresh_limits()
        if sender != "controls":
            self._update_controls_from_state()
        if sender != "viewbox":
            self._update_viewbox_from_state()
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
        self._config.prefetch_max_mb = max_mb_val if max_mb_val > 0 else None
        prefetch_service.configure(tile_duration=tile, max_tiles=max_tiles, max_mb=max_mb)
        if self._prefetch is not None:
            self._prefetch.stop()
        self._prefetch = prefetch_service.create_cache(self._fetch_tile)
        self._prefetch.start()
        self._schedule_prefetch()
        self._config.save()

    def _on_annotation_toggle(self, checked: bool):
        self._annotations_enabled = bool(checked)
        self._update_annotation_overlays(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )

    def _schedule_prefetch(self):
        total = self.loader.duration_s
        for ch in range(self.loader.n_channels):
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
            start = max(0.0, self._view_start - self._view_duration)
            duration = self._view_duration * 3
            self._prefetch.prefetch_window(ch, start, duration)

    def _full_view(self):
        self._set_view(0.0, self.loader.duration_s, sender="buttons")

    def _fetch_tile(self, channel: int, start: float, end: float):
        loader = self.loader
        start, duration = clamp_window(start, end - start, total=loader.duration_s, limits=self._limits)
        return loader.read(channel, start, start + duration)

    def _ensure_overscan_for_view(self):
        if self._overscan_worker is None or self.loader.n_channels == 0:
            return
        window_start = self._view_start
        window_end = min(self.loader.duration_s, window_start + self._view_duration)
        tile = self._overscan_tile
        if tile is not None and tile.contains(window_start, window_end):
            return
        self._request_overscan_tile(window_start, self._view_duration)

    def _request_overscan_tile(self, window_start: float, window_duration: float):
        if self._overscan_worker is None or self.loader.n_channels == 0:
            return
        start, end = self._compute_overscan_bounds(window_start, window_duration)
        if end <= start:
            return
        req_id = self._overscan_request_id + 1
        self._overscan_request_id = req_id
        self._overscan_inflight = req_id
        channels = tuple(range(self.loader.n_channels))
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
        if request_id != self._overscan_request_id:
            return
        if not isinstance(tile_obj, _OverscanTile):
            return
        self._overscan_inflight = None
        self._overscan_tile = tile_obj
        self._current_tile_id = None
        self._apply_tile_to_curves(tile_obj)
        self._schedule_refresh()

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
        self._annotations_enabled = False
        self.stageSummaryLabel.setText("Stage: --")
        self._clear_annotation_lines()
        self._clear_annotation_rects()
        self._populate_event_list(clear=True)
        self._update_stage_plot_data()
        self._update_annotation_summary()

        path = getattr(self.loader, "path", None)
        if not path:
            return

        ann_sets: list[annotation_core.Annotations] = []
        found = annotation_core.discover_annotation_files(path)
        found.update(self._manual_annotation_paths)
        start_dt = getattr(getattr(self.loader, "timebase", None), "start_dt", None)

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

        if not ann_sets:
            return

        self._annotations_index = annotation_core.AnnotationIndex(ann_sets)
        self.annotationToggle.setEnabled(True)
        self._annotations_enabled = self.annotationToggle.isChecked()
        self._populate_event_list()
        self._update_annotation_summary()
        self._update_stage_plot_data()
        self._update_annotation_overlays(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )

    def _update_stage_plot_data(self):
        if self.stage_plot is None:
            self._ensure_stage_plot()
        if self._stage_curve is None:
            return
        if not self._annotations_index or self._annotations_index.is_empty():
            self._stage_curve.setData([], [])
            self.stage_plot.hide()
            return

        stage_data = self._annotations_index.between(
            0.0,
            self.loader.duration_s,
            channels=["stage"],
        )
        if isinstance(stage_data, tuple):
            stage_data = stage_data[0]
        if stage_data.size == 0:
            self._stage_curve.setData([], [])
            self.stage_plot.hide()
            return

        mapping = {
            "N3": 0.0,
            "N4": 0.0,  # treat legacy N4 like N3
            "N2": 1.0,
            "N1": 2.0,
            "REM": 3.0,
            "Wake": 4.0,
        }

        # Build step-mode arrays: for stepMode=True, pyqtgraph expects
        # len(x) == len(y) + 1, where x are edge times and y are bin values.
        xs: list[float] = []
        ys: list[float] = []

        # Optionally fill a gap before the first stage with the first value
        first_edge: float | None = None
        for idx, entry in enumerate(stage_data):
            start = float(entry["start_s"])
            end = float(entry["end_s"])
            label = str(entry["label"])  # e.g., N2, REM, Wake
            value = mapping.get(label, 2.0)

            if idx == 0:
                # first edge
                first_edge = start
                xs.append(start)
            else:
                # if there is a gap from the previous edge, extend edges but keep last value
                prev_end = xs[-1]
                if start > prev_end:
                    xs.append(start)  # gap edge; y stays the same (implicit)
            # push the end edge of this stage and its value
            xs.append(end)
            ys.append(value)

        # Safety: ensure the invariant for stepMode=True
        if not xs or len(xs) != len(ys) + 1:
            # Fallback to hiding the curve if shapes are inconsistent
            self._stage_curve.setData([], [])
        else:
            self._stage_curve.setData(xs, ys)
        self.stage_plot.show()

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
        key = "stages" if path_obj.stem.upper().endswith("STAGE") else "events"
        self._manual_annotation_paths[key] = path_obj
        self._load_companion_annotations()

    def _update_annotation_summary(self):
        total_events = len(self._event_records)
        if not self._annotations_index or self._annotations_index.is_empty():
            self.stageSummaryLabel.setText(f"Stage: -- | Events: {total_events}")
            return
        self.stageSummaryLabel.setText(f"Stage: -- | Events: {total_events}")

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

    def _clear_annotation_lines(self):
        for line in self._annotation_lines:
            line.setVisible(False)
        self._clear_annotation_rects()

    def _clear_annotation_rects(self):
        for rect in self._annotation_rects:
            rect.setVisible(False)

    def _populate_event_list(self, clear: bool = False):
        self.eventList.blockSignals(True)
        self.eventList.clear()
        if clear or not self._annotations_index or self._annotations_index.is_empty():
            self.eventList.setEnabled(False)
            self.eventPrevBtn.setEnabled(False)
            self.eventNextBtn.setEnabled(False)
            self._event_records = []
            self._current_event_index = -1
            self._current_event_id = None
            self.eventList.blockSignals(False)
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
        if data.size == 0:
            self.eventList.setEnabled(False)
            self.eventPrevBtn.setEnabled(False)
            self.eventNextBtn.setEnabled(False)
            self._event_records = []
            self._current_event_index = -1
            self._current_event_id = None
            self.eventList.blockSignals(False)
            return

        mask = np.array([str(chan) != "stage" for chan in data["chan"]], dtype=bool)
        data = data[mask]
        ids = ids[mask]
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

        records.sort(key=lambda r: r["start"])
        self._event_records = records
        total = len(records)
        self.eventList.setEnabled(total > 0)
        self.eventPrevBtn.setEnabled(total > 0)
        self.eventNextBtn.setEnabled(total > 0)

        for rec in records:
            label = rec["label"]
            chan = rec["chan"]
            ts = self._format_clock(rec["start"])
            duration_s = rec["end"] - rec["start"]
            text = f"{ts} — {label} ({duration_s:.1f} s) [{chan}]"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, rec)
            self.eventList.addItem(item)

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
        if not self._primary_plot:
            return
        if not self._annotations_index or not self._annotations_enabled:
            self._clear_annotation_lines()
            if self._annotations_index is None:
                self.stageSummaryLabel.setText("Stage: --")
            return

        for line in self._annotation_lines:
            line.setVisible(False)

        event_channels = [c for c in self._annotations_index.channel_set if c != "stage"]
        events, ids = self._annotations_index.between(
            t0,
            t1,
            channels=event_channels or None,
            return_indices=True,
        )

        events = np.array(events, copy=False)
        ids = np.asarray(ids, dtype=int)
        self._clear_annotation_rects()
        if events.size:
            self._ensure_annotation_rect_pool(len(events))
            scene_rect = self.plotLayout.ci.mapRectToScene(self.plotLayout.ci.boundingRect())
            vb = self._primary_plot.getViewBox()
            selected_id = self._current_event_id
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

        stage_events = self._annotations_index.between(t0, t1, channels=["stage"])
        if isinstance(stage_events, tuple):
            stage_events = stage_events[0]
        total_events = len(self._event_records)
        if stage_events.size:
            counts = Counter(stage_events["label"])
            dominant, count = counts.most_common(1)[0]
            self.stageSummaryLabel.setText(f"Stage: {dominant} ({count}) | Events: {total_events}")
        else:
            self.stageSummaryLabel.setText(f"Stage: -- | Events: {total_events}")

    def _prepare_tile(self, tile: _OverscanTile) -> bool:
        pixels = self._estimate_pixels() or 0
        overscan_span = 2 * self._overscan_factor + 1
        budget = int(max(200, pixels * overscan_span * 2)) if pixels else 2000
        if tile.pixel_budget == budget and tile.channel_data:
            return False
        prepared: list[tuple[np.ndarray, np.ndarray]] = []
        for t_arr, x_arr in tile.raw_channel_data:
            t_slice, x_slice = slice_and_decimate(t_arr, x_arr, tile.start, tile.end, budget)
            prepared.append((t_slice, x_slice))
        tile.channel_data = prepared
        tile.pixel_budget = budget
        return True

    def _apply_tile_to_curves(self, tile: _OverscanTile) -> None:
        self._prepare_tile(tile)
        for idx, (t_arr, x_arr) in enumerate(tile.channel_data):
            if idx < len(self.curves):
                self.curves[idx].setData(t_arr, x_arr)
        self._current_tile_id = tile.request_id

    def closeEvent(self, event):
        self._cleanup_ingest_thread(wait=True)
        self._prefetch.stop()
        self._shutdown_overscan_worker()
        super().closeEvent(event)

    def _update_data_source_label(self):
        if isinstance(self.loader, ZarrLoader):
            self.sourceLabel.setText("Source: Zarr cache")
            self.sourceLabel.setStyleSheet("color: #7fb57d; font-style: italic;")
        elif getattr(self.loader, "has_cache", None) and self.loader.has_cache():
            self.sourceLabel.setText("Source: EDF (RAM cache)")
            self.sourceLabel.setStyleSheet("color: #d7c77b; font-style: italic;")
        else:
            self.sourceLabel.setText("Source: EDF (live)")
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
        if not self._primary_plot:
            return 0
        vb = self._primary_plot.getViewBox()
        if vb is None:
            return 0
        width = int(vb.width())
        return max(0, width)

    # ----- Plot helpers ------------------------------------------------------

    def _ensure_plot_rows(self, count: int):
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
            curve = plot.plot([], [], pen=pg.mkPen(pg.intColor(idx, values=220), width=1.2))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")

            self.plots.append(plot)
            self.curves.append(curve)

    def _configure_plots(self):
        n = self.loader.n_channels
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
                self.curves[idx].setData([], [])
                continue

            plot.show()
            meta = self.loader.info[idx]
            self.channel_labels[idx].setText(self._format_label(meta))

            pen = pg.mkPen(pg.intColor(idx, hues=max(1, n), values=220), width=1.2)
            self.curves[idx].setPen(pen)

        if n == 0:
            self._primary_plot = None
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

        new_primary.setAxisItems({"bottom": self.time_axis})
        new_primary.showAxis("bottom", show=True)
        self._primary_plot = new_primary
        self._connect_primary_viewbox()

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

        self._ensure_stage_plot()

    @staticmethod
    def _format_label(meta) -> str:
        unit = f" [{meta.unit}]" if getattr(meta, "unit", "") else ""
        text = f"{meta.name}{unit}"
        return (
            "<span style='color:#dfe7ff;font-weight:600;font-size:11pt;padding-right:12px;'>"
            f"{text}"
            "</span>"
        )

    def _ensure_stage_plot(self):
        if self.stage_plot is None:
            row = len(self.plots)
            self._stage_label_item = self.plotLayout.addLabel(
                row=row, col=0, text="Stage", justify="right"
            )
            plot = self.plotLayout.addPlot(row=row, col=1)
            plot.setMaximumHeight(90)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=False, y=True, alpha=0.15)
            plot.hideAxis("right")
            plot.hideAxis("top")
            plot.getAxis("left").setStyle(showValues=True)
            ticks = [
                (0.0, "N3"),
                (1.0, "N2"),
                (2.0, "N1"),
                (3.0, "REM"),
                (4.0, "Wake"),
            ]
            plot.setYRange(-0.5, 4.5)
            plot.getAxis("left").setTicks([ticks])
            plot.showAxis("bottom", show=True)
            self.stage_plot = plot
            self._stage_curve = plot.plot([], [], stepMode=True, fillLevel=None, pen=pg.mkPen("#5f8bff", width=2))

        if self._primary_plot is not None and self.stage_plot is not None:
            self.stage_plot.setXLink(self._primary_plot)

    def _connect_primary_viewbox(self):
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
        if viewbox is not self._primary_viewbox or self._updating_viewbox:
            return
        if not xrange or len(xrange) != 2:
            return
        start = float(xrange[0])
        end = float(xrange[1])
        duration = max(self._limits.duration_min, end - start)
        self._set_view(start, duration, sender="viewbox")
