# ui/main_window.py
from __future__ import annotations

import logging
from dataclasses import dataclass
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
        self._start_zarr_ingest()
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
        self.sourceLabel = QtWidgets.QLabel("Source: EDF (live)")
        self.sourceLabel.setObjectName("sourceLabel")

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
        controlLayout.addWidget(self.sourceLabel)
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
        self.refresh()
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
        else:
            self.sourceLabel.setText("Source: EDF (live)")
            self.sourceLabel.setStyleSheet("color: #9ba9bf; font-style: italic;")

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

        new_primary.setAxisItems({"bottom": self.time_axis})
        new_primary.showAxis("bottom", show=True)
        self._primary_plot = new_primary
        self._connect_primary_viewbox()

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

    @staticmethod
    def _format_label(meta) -> str:
        unit = f" [{meta.unit}]" if getattr(meta, "unit", "") else ""
        text = f"{meta.name}{unit}"
        return (
            "<span style='color:#dfe7ff;font-weight:600;font-size:11pt;padding-right:12px;'>"
            f"{text}"
            "</span>"
        )

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
