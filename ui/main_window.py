# ui/main_window.py
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from collections import Counter
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
from config import ViewerConfig
from core.decimate import min_max_bins
from core.overscan import slice_and_decimate
from core.prefetch import prefetch_service
from core.view_window import WindowLimits, clamp_window, pan_window, zoom_window
from core.zarr_cache import EdfToZarr, resolve_output_path
from core.zarr_loader import ZarrLoader
from core import annotations as annotation_core


LOG = logging.getLogger(__name__)

STAGE_COLORS: dict[str, str] = {
    "Wake": "#f6c87c",
    "N1": "#f3dda3",
    "N2": "#9fd7a5",
    "N3": "#6ec2d0",
    "REM": "#f59db7",
}
DEFAULT_STAGE_COLOR = "#b9c1cf"
STAGE_LABEL_COLOR = "#0f1a2b"
STAGE_TEXT_MARGIN = 26.0


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
        self._splitter: QtWidgets.QSplitter | None = None
        self._control_wrapper: QtWidgets.QWidget | None = None
        self._sidebar_stack: QtWidgets.QStackedWidget | None = None
        self.sidebarList: QtWidgets.QListWidget | None = None
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

        theme_key = getattr(self._config, "theme", DEFAULT_THEME)
        if theme_key not in THEMES:
            theme_key = DEFAULT_THEME
        self._active_theme_key = theme_key
        self._theme: ThemeDefinition = THEMES[theme_key]
        self._config.theme = theme_key
        self._controls_collapsed = bool(getattr(self._config, "controls_collapsed", False))

        pg.setConfigOptions(antialias=True)

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

        self._hidden_channels: set[int] = set(getattr(self._config, "hidden_channels", ()))
        self._auto_hide_annotation_channels()
        hidden_ann = getattr(self._config, "hidden_annotation_channels", ("stage", "position"))
        self._hidden_annotation_channels: set[str] = {
            str(name).strip() for name in hidden_ann if str(name).strip()
        }
        self._manual_annotation_paths: dict[str, Path] = {}
        self._annotations_index: annotation_core.AnnotationIndex | None = None
        self._annotation_lines: list[pg.InfiniteLine] = []
        self._annotations_enabled = False
        self._annotation_rects: list[QtWidgets.QGraphicsRectItem] = []
        self._stage_rects: list[QtWidgets.QGraphicsRectItem] = []
        self._stage_label_items: list[QtWidgets.QGraphicsSimpleTextItem] = []
        self._all_event_records: list[dict[str, float | str | int]] = []
        self._event_records: list[dict[str, float | str | int]] = []
        self._current_event_index: int = -1
        self._current_event_id: Optional[int] = None
        self._event_color_cache: dict[str, QtGui.QColor] = {}
        self._selected_event_channel: str | None = None
        self._event_label_filter: str = ""
        self._stage_label_item: pg.LabelItem | None = None
        self._stage_info_widget: QtWidgets.QTableWidget | None = None
        self._stage_info_proxy: QtWidgets.QGraphicsProxyWidget | None = None
        self._annotation_channel_toggles: dict[str, QtWidgets.QCheckBox] = {}

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
        self.annotationStageToggle = QtWidgets.QCheckBox("Show sleep stages")
        self.annotationStageToggle.setChecked(False)
        self.annotationStageToggle.setEnabled(False)
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

        primaryControls = QtWidgets.QGroupBox("Viewing Controls")
        primaryControls.setObjectName("primaryControls")
        primaryLayout = QtWidgets.QVBoxLayout(primaryControls)
        primaryLayout.setContentsMargins(14, 16, 14, 12)
        primaryLayout.setSpacing(12)
        primaryLayout.addLayout(navLayout)
        primaryLayout.addLayout(form)

        self.fileButton = QtWidgets.QPushButton("Open EDF…")
        self.fileButton.setObjectName("fileSelectButton")
        self.fileButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.fileButton.setCursor(QtCore.Qt.PointingHandCursor)
        self.exportButton = QtWidgets.QPushButton("Export…")
        self.exportButton.setObjectName("exportButton")
        self.exportButton.setCursor(QtCore.Qt.PointingHandCursor)

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
        annotationLayout.addWidget(self.annotationStageToggle)
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

        self._annotation_channel_toggles = {
            annotation_core.STAGE_CHANNEL: self.annotationStageToggle,
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
        self.ingestBar = QtWidgets.QProgressBar()
        self.ingestBar.setObjectName("ingestBar")
        self.ingestBar.setRange(0, 100)
        self.ingestBar.setValue(0)
        self.ingestBar.setFormat("Caching EDF → Zarr: %p%")
        self.ingestBar.setTextVisible(True)
        self.ingestBar.hide()

        viewingPage = QtWidgets.QWidget()
        viewingLayout = QtWidgets.QVBoxLayout(viewingPage)
        viewingLayout.setContentsMargins(18, 18, 18, 18)
        viewingLayout.setSpacing(14)
        viewingLayout.addWidget(primaryControls)
        viewingLayout.addWidget(telemetryBar)
        viewingLayout.addWidget(self.sourceLabel)
        viewingLayout.addStretch(1)

        channelsPage = QtWidgets.QWidget()
        channelsLayout = QtWidgets.QVBoxLayout(channelsPage)
        channelsLayout.setContentsMargins(18, 18, 18, 18)
        channelsLayout.setSpacing(14)
        channelsLayout.addWidget(self.channelSection)
        channelsLayout.addStretch(1)

        annotationsPage = QtWidgets.QWidget()
        annotationsLayout = QtWidgets.QVBoxLayout(annotationsPage)
        annotationsLayout.setContentsMargins(18, 18, 18, 18)
        annotationsLayout.setSpacing(14)
        annotationsLayout.addWidget(annotationSection)
        annotationsLayout.addStretch(1)

        appearancePage = QtWidgets.QWidget()
        appearancePageLayout = QtWidgets.QVBoxLayout(appearancePage)
        appearancePageLayout.setContentsMargins(18, 18, 18, 18)
        appearancePageLayout.setSpacing(14)
        appearancePageLayout.addWidget(self.appearanceSection)
        appearancePageLayout.addStretch(1)

        prefetchPage = QtWidgets.QWidget()
        prefetchPageLayout = QtWidgets.QVBoxLayout(prefetchPage)
        prefetchPageLayout.setContentsMargins(18, 18, 18, 18)
        prefetchPageLayout.setSpacing(14)
        prefetchPageLayout.addWidget(self.prefetchSection)
        prefetchPageLayout.addWidget(self.ingestBar)
        prefetchPageLayout.addStretch(1)

        self.sidebarList = QtWidgets.QListWidget()
        self.sidebarList.setObjectName("sidebarList")
        self.sidebarList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sidebarList.addItem("Viewing")
        self.sidebarList.addItem("Channels")
        self.sidebarList.addItem("Annotations")
        self.sidebarList.addItem("Appearance")
        self.sidebarList.addItem("Prefetch")
        self.sidebarList.setCurrentRow(0)

        sidebarStack = QtWidgets.QStackedWidget()
        sidebarStack.addWidget(viewingPage)
        sidebarStack.addWidget(channelsPage)
        sidebarStack.addWidget(annotationsPage)
        sidebarStack.addWidget(appearancePage)
        sidebarStack.addWidget(prefetchPage)
        sidebarStack.setCurrentIndex(0)
        self._sidebar_stack = sidebarStack

        sidebarFrame = QtWidgets.QFrame()
        sidebarFrame.setObjectName("controlPanel")
        sidebarFrame.setMinimumWidth(220)
        sidebarFrame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        sidebarLayout = QtWidgets.QVBoxLayout(sidebarFrame)
        sidebarLayout.setContentsMargins(0, 0, 0, 0)
        sidebarLayout.setSpacing(0)
        sidebarLayout.addWidget(self.sidebarList)
        sidebarLayout.addWidget(sidebarStack)
        self._control_wrapper = sidebarFrame

        self.plotLayout = pg.GraphicsLayoutWidget()
        self.plotLayout.setMinimumSize(0, 0)
        self.plotLayout.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
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
        scroll.setMinimumSize(0, 0)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        scroll.setWidget(self.plotLayout)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(sidebarFrame)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 980])
        self._splitter = splitter

        self.controlToggleBtn = QtWidgets.QToolButton()
        self.controlToggleBtn.setObjectName("sidebarToggleBtn")
        self.controlToggleBtn.setAutoRaise(True)
        self.controlToggleBtn.setCheckable(True)
        self.controlToggleBtn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.controlToggleBtn.setCursor(QtCore.Qt.PointingHandCursor)
        self.controlToggleBtn.setToolTip("Hide controls")

        header = QtWidgets.QWidget()
        header.setObjectName("mainHeader")
        headerLayout = QtWidgets.QHBoxLayout(header)
        headerLayout.setContentsMargins(12, 10, 12, 10)
        headerLayout.setSpacing(8)
        headerLayout.addWidget(self.fileButton)
        headerLayout.addWidget(self.exportButton)
        headerLayout.addStretch(1)
        headerLayout.addWidget(self.controlToggleBtn)

        central = QtWidgets.QWidget()
        centralLayout = QtWidgets.QVBoxLayout(central)
        centralLayout.setContentsMargins(0, 0, 0, 0)
        centralLayout.setSpacing(0)
        centralLayout.addWidget(header)
        centralLayout.addWidget(splitter)
        self.setCentralWidget(central)

        with QtCore.QSignalBlocker(self.controlToggleBtn):
            self.controlToggleBtn.setChecked(self._controls_collapsed)
        self._update_control_toggle_icon(self._controls_collapsed)
        self._apply_control_panel_state(self._controls_collapsed)
        QtCore.QTimer.singleShot(0, lambda: self._apply_control_panel_state(self._controls_collapsed))

    def _connect_signals(self):
        self.startSpin.valueChanged.connect(self._on_start_spin_changed)
        self.windowSpin.valueChanged.connect(self._on_duration_spin_changed)
        self.fileButton.clicked.connect(self._prompt_open_file)
        self.exportButton.clicked.connect(self._on_export_clicked)
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
        if self.sidebarList is not None:
            self.sidebarList.currentRowChanged.connect(self._on_sidebar_page_changed)

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
        if self._control_wrapper is None or self._splitter is None:
            return
        collapsed = bool(collapsed)
        if self._sidebar_stack is not None:
            self._sidebar_stack.setVisible(not collapsed)
        if self.sidebarList is not None:
            self.sidebarList.setVisible(not collapsed)
        self._control_wrapper.setVisible(not collapsed)
        panel_min = 220
        if collapsed:
            self._control_wrapper.setMinimumWidth(0)
            self._control_wrapper.setMaximumWidth(0)
        else:
            expanded_width = max(panel_min, self._control_wrapper.sizeHint().width())
            self._control_wrapper.setMinimumWidth(expanded_width)
            self._control_wrapper.setMaximumWidth(16777215)
        if self.controlToggleBtn.isChecked() != collapsed:
            with QtCore.QSignalBlocker(self.controlToggleBtn):
                self.controlToggleBtn.setChecked(collapsed)
        self._update_control_toggle_icon(collapsed)

        sizes = self._splitter.sizes()
        total = sum(sizes)
        if total <= 0:
            total = max(self.width(), panel_min + 600)
        if collapsed:
            self._splitter.setSizes([0, max(1, total)])
        else:
            panel_width = max(panel_min, self._control_wrapper.sizeHint().width())
            second = max(1, total - panel_width)
            self._splitter.setSizes([panel_width, second])

    def _update_control_toggle_icon(self, collapsed: bool) -> None:
        arrow = QtCore.Qt.RightArrow if collapsed else QtCore.Qt.LeftArrow
        self.controlToggleBtn.setArrowType(arrow)
        self.controlToggleBtn.setToolTip("Show controls" if collapsed else "Hide controls")

    def _on_sidebar_page_changed(self, index: int) -> None:
        if self._sidebar_stack is None:
            return
        if index < 0 or index >= self._sidebar_stack.count():
            return
        self._sidebar_stack.setCurrentIndex(index)

    def _on_export_clicked(self) -> None:
        LOG.info("Export requested (placeholder)")

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
        self.plotLayout.setBackground(theme.pg_background)
        for plot in self.plots:
            for axis_name in ("bottom", "left"):
                axis = plot.getAxis(axis_name)
                if axis is not None:
                    pen = pg.mkPen(theme.pg_foreground)
                    axis.setPen(pen)
                    axis.setTextPen(theme.pg_foreground)

        self.time_axis.setPen(pg.mkPen(theme.pg_foreground))
        self.time_axis.setTextPen(theme.pg_foreground)

        if self._stage_info_widget is not None:
            self._stage_info_widget.viewport().setAutoFillBackground(False)
            header = self._stage_info_widget.horizontalHeader()
            header.setStyleSheet("")

        self._apply_curve_pens()
        self._refresh_channel_label_styles()
        self._update_stage_label_style()
        self._update_theme_preview(resolved_key)

        if persist:
            self._write_persistent_state()

    def _curve_color(self, idx: int) -> str:
        colors = self._theme.curve_colors or ("#5f8bff",)
        return colors[idx % len(colors)]

    def _apply_curve_pens(self) -> None:
        for idx, curve in enumerate(self.curves):
            color = self._curve_color(idx)
            curve.setPen(pg.mkPen(color, width=1.2))

    def _refresh_channel_label_styles(self) -> None:
        if not self.channel_labels:
            return
        n = self.loader.n_channels
        for idx, label_item in enumerate(self.channel_labels):
            if idx >= n:
                continue
            meta = self.loader.info[idx]
            hidden = idx in self._hidden_channels
            label_item.setText(self._format_label(meta, hidden=hidden))

    def _update_stage_label_style(self) -> None:
        if self._stage_label_item is not None:
            self._stage_label_item.setText(self._stage_label_markup())

    def _stage_label_markup(self) -> str:
        color = getattr(self._theme, "channel_label_active", "#ffffff")
        return (
            "<span style='color:" + color + ";font-weight:600;font-size:11pt;padding-right:12px;'>Stage</span>"
        )

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
                if i in self._hidden_channels:
                    self.curves[i].setData([], [])
                    continue
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
        self._update_stage_info()

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
        self._config.prefetch_max_mb = max_mb
        prefetch_service.configure(tile_duration=tile, max_tiles=max_tiles, max_mb=max_mb)
        if self._prefetch is not None:
            self._prefetch.stop()
        self._prefetch = prefetch_service.create_cache(self._fetch_tile)
        self._prefetch.start()
        self._schedule_prefetch()
        self._config.save()

    def _on_prefetch_section_toggled(self, expanded: bool) -> None:
        self._config.prefetch_collapsed = not expanded
        self._config.save()

    def _on_annotation_toggle(self, checked: bool):
        self._annotations_enabled = bool(checked)
        self._update_annotation_overlays(
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
        self.annotationFocusOnly.setEnabled(False)
        self._annotations_enabled = False
        self.stageSummaryLabel.setText("Stage: -- | Position: -- | Events: 0")
        self._clear_annotation_lines()
        self._clear_annotation_rects()
        self._populate_event_list(clear=True)
        self._update_annotation_channel_toggles()
        self._update_stage_info()
        self._update_annotation_summary()

        path = getattr(self.loader, "path", None)
        if not path:
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
        self._update_stage_info()
        self._update_annotation_overlays(
            self._view_start,
            min(self.loader.duration_s, self._view_start + self._view_duration),
        )

    def _update_stage_info(self):
        if self._stage_info_widget is None:
            self._ensure_stage_info_widget()
        table = self._stage_info_widget
        proxy = self._stage_info_proxy
        label_item = self._stage_label_item
        if table is None:
            return
        if table.rowCount() != 3:
            table.setRowCount(3)
        entries = [None, None, None]
        if not self._annotations_index or self._annotations_index.is_empty():
            if proxy is not None:
                proxy.setVisible(False)
            if label_item is not None:
                label_item.setVisible(False)
            for row in range(3):
                for col in range(table.columnCount()):
                    self._set_stage_table_item(table, row, col, "--")
            table.clearSelection()
            return

        stages = self._stage_annotations()
        if stages.size == 0:
            if proxy is not None:
                proxy.setVisible(False)
            if label_item is not None:
                label_item.setVisible(False)
            for row in range(3):
                for col in range(table.columnCount()):
                    self._set_stage_table_item(table, row, col, "--")
            table.clearSelection()
            return

        center_time = self._view_start + self._view_duration * 0.5
        starts = stages["start_s"]
        idx = int(np.searchsorted(starts, center_time, side="right") - 1)
        current_idx = idx if idx >= 0 and stages[idx]["end_s"] > center_time else -1
        prev_idx = current_idx - 1 if current_idx >= 0 else idx
        next_idx = current_idx + 1 if current_idx >= 0 else idx + 1

        current_entry = stages[current_idx] if current_idx >= 0 else None
        prev_entry = stages[prev_idx] if prev_idx is not None and prev_idx >= 0 else None
        next_entry = stages[next_idx] if next_idx is not None and next_idx < stages.size else None

        entries = [prev_entry, current_entry, next_entry]

        if proxy is not None:
            proxy.setVisible(True)
        if label_item is not None:
            label_item.setVisible(True)

        for row, entry in enumerate(entries):
            if entry is None:
                self._set_stage_table_item(table, row, 0, "--")
                self._set_stage_table_item(table, row, 1, "--")
                self._set_stage_table_item(table, row, 2, "--")
            else:
                label = str(entry["label"]) or "--"
                start_text = self._format_clock(float(entry["start_s"]))
                end_text = self._format_clock(float(entry["end_s"]))
                self._set_stage_table_item(table, row, 0, label)
                self._set_stage_table_item(table, row, 1, start_text)
                self._set_stage_table_item(table, row, 2, end_text)

        if entries[1] is not None:
            table.selectRow(1)
        else:
            table.clearSelection()

    def _set_stage_table_item(
        self, table: QtWidgets.QTableWidget, row: int, column: int, text: str
    ) -> None:
        item = table.item(row, column)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            table.setItem(row, column, item)
        item.setText(text)

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

    def _ensure_stage_rect_pool(self, count: int):
        scene = self.plotLayout.scene()
        while len(self._stage_rects) < count:
            rect_item = QtWidgets.QGraphicsRectItem()
            rect_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            rect_item.setZValue(-40)
            rect_item.setVisible(False)
            scene.addItem(rect_item)

            label_item = QtWidgets.QGraphicsSimpleTextItem("")
            font = label_item.font()
            font.setBold(True)
            label_item.setFont(font)
            label_item.setBrush(QtGui.QBrush(QtGui.QColor(STAGE_LABEL_COLOR)))
            label_item.setZValue(-39)
            label_item.setVisible(False)
            scene.addItem(label_item)

            self._stage_rects.append(rect_item)
            self._stage_label_items.append(label_item)

    def _stage_color_for_label(self, label: str) -> QtGui.QColor:
        base = STAGE_COLORS.get(label, DEFAULT_STAGE_COLOR)
        color = QtGui.QColor(base)
        color.setAlpha(90)
        return color

    def _clear_stage_rects(self):
        for rect in self._stage_rects:
            rect.setVisible(False)
        for label_item in self._stage_label_items:
            label_item.setVisible(False)

    def _clear_annotation_lines(self):
        for line in self._annotation_lines:
            line.setVisible(False)
        self._clear_annotation_rects()
        self._clear_stage_rects()

    def _clear_annotation_rects(self):
        for rect in self._annotation_rects:
            rect.setVisible(False)

    def _update_stage_background(self, stage_events: np.ndarray, t0: float, t1: float):
        if not self._primary_plot:
            self._clear_stage_rects()
            return
        if stage_events is None or getattr(stage_events, "size", 0) == 0:
            self._clear_stage_rects()
            return

        vb = self._primary_plot.getViewBox()
        if vb is None:
            self._clear_stage_rects()
            return

        scene_rect = self.plotLayout.ci.mapRectToScene(self.plotLayout.ci.boundingRect())
        if scene_rect is None:
            self._clear_stage_rects()
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
            self._clear_stage_rects()
            return

        self._ensure_stage_rect_pool(len(segments))
        height = scene_rect.height()

        for idx, (start, end, label) in enumerate(segments):
            p1 = vb.mapViewToScene(QtCore.QPointF(start, 0))
            p2 = vb.mapViewToScene(QtCore.QPointF(end, 0))
            x1, x2 = p1.x(), p2.x()
            left = min(x1, x2)
            width = max(2.0, abs(x2 - x1))
            rect = QtCore.QRectF(left, scene_rect.top(), width, height)

            rect_item = self._stage_rects[idx]
            rect_item.setRect(rect)
            rect_item.setBrush(QtGui.QBrush(self._stage_color_for_label(label)))
            rect_item.setVisible(True)

            label_item = self._stage_label_items[idx]
            label_item.setText(label)
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

        for idx in range(len(segments), len(self._stage_rects)):
            self._stage_rects[idx].setVisible(False)
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
        self._update_stage_info()
        if self.loader is not None:
            self._update_annotation_overlays(
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

            self._clear_stage_rects()
        else:
            stage_events = self._annotations_index.between(
                t0, t1, channels=[annotation_core.STAGE_CHANNEL]
            )
            if isinstance(stage_events, tuple):
                stage_events = stage_events[0]
            stage_events = np.array(stage_events, copy=False)
            self._update_stage_background(stage_events, t0, t1)
        if annotation_core.STAGE_CHANNEL in hidden_channels:
            stage_events = np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)

        if annotation_core.POSITION_CHANNEL in hidden_channels:
            position_events = np.zeros(0, dtype=annotation_core.ANNOT_DTYPE)
        else:
            position_events = self._annotations_index.between(
                t0, t1, channels=[annotation_core.POSITION_CHANNEL]
            )
            if isinstance(position_events, tuple):
                position_events = position_events[0]

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
                if idx in self._hidden_channels:
                    self.curves[idx].setData([], [])
                    continue
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
            curve_color = self._curve_color(idx)
            curve = plot.plot([], [], pen=pg.mkPen(curve_color, width=1.2))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")

            self.plots.append(plot)
            self.curves.append(curve)

    def _configure_plots(self):
        n = self.loader.n_channels
        old_primary = self._primary_plot
        self._ensure_plot_rows(n)

        # Trim hidden channels to valid range
        self._hidden_channels = {idx for idx in self._hidden_channels if 0 <= idx < n}
        self._sync_channel_controls()

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

        self._ensure_stage_info_widget()

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

    def _on_channel_checkbox_toggled(self, idx: int, checked: bool) -> None:
        self._set_channel_visible(idx, bool(checked))

    @QtCore.Slot(int, bool)
    def _set_channel_visible(self, idx: int, visible: bool) -> None:
        self._apply_channel_visible(idx, bool(visible), sync_checkbox=True, persist=True)
        self.refresh()

    def _format_label(self, meta, *, hidden: bool = False) -> str:
        unit = f" [{meta.unit}]" if getattr(meta, "unit", "") else ""
        text = f"{meta.name}{unit}"
        theme = getattr(self, "_theme", THEMES[DEFAULT_THEME])
        if hidden:
            color = theme.channel_label_hidden
            extra = "font-style: italic; opacity:0.7;"
            text = f"{text} (hidden)"
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

    def _ensure_stage_info_widget(self):
        if self._stage_info_widget is None:
            row = len(self.plots)
            self._stage_label_item = self.plotLayout.addLabel(
                row=row, col=0, text="Stage", justify="right"
            )
            self._stage_label_item.setText(self._stage_label_markup())
            self._stage_label_item.setVisible(False)
            table = QtWidgets.QTableWidget(3, 3)
            table.setHorizontalHeaderLabels(["Stage", "Start", "End"])
            table.setVerticalHeaderLabels(["Previous", "Current", "Next"])
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            table.setFocusPolicy(QtCore.Qt.NoFocus)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setDefaultSectionSize(26)
            table.setAlternatingRowColors(True)
            table.setMaximumHeight(140)
            table.setMinimumHeight(110)
            proxy = QtWidgets.QGraphicsProxyWidget()
            proxy.setWidget(table)
            proxy.setVisible(False)
            self.plotLayout.addItem(proxy, row=row, col=1)
            self._stage_info_widget = table
            self._stage_info_proxy = proxy

        if self._stage_label_item is not None:
            self._stage_label_item.setText(self._stage_label_markup())

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
