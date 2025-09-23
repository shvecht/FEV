# ui/main_window.py
from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ui.time_axis import TimeAxis
from core.zarr_cache import EdfToZarr, resolve_output_path
from core.zarr_loader import ZarrLoader


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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader
        self._ingest_thread: QtCore.QThread | None = None
        self._ingest_worker: _ZarrIngestWorker | None = None
        self._zarr_path: Path | None = None
        self._pending_loader: object | None = None

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "#10131d")
        pg.setConfigOption("foreground", "#e3e7f3")

        self.setWindowTitle("EDF Viewer — Multi-channel")
        self.time_axis = TimeAxis(orientation="bottom", timebase=loader.timebase)
        if self.loader.timebase.start_dt is not None:
            self.time_axis.set_mode("absolute")

        self._primary_plot = None
        self._build_ui()
        self._connect_signals()
        self._refresh_limits()
        self.refresh()
        self._update_data_source_label()
        self._start_zarr_ingest()

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

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.addWidget(QtWidgets.QLabel("Start (s)"), 0, 0)
        form.addWidget(self.startSpin, 0, 1)
        form.addWidget(QtWidgets.QLabel("Duration"), 1, 0)
        form.addWidget(self.windowSpin, 1, 1)

        controlLayout.addWidget(header)
        self.fileButton = QtWidgets.QPushButton("Open EDF…")
        self.fileButton.setObjectName("fileSelectButton")
        self.fileButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.fileButton.setCursor(QtCore.Qt.PointingHandCursor)
        controlLayout.addWidget(self.fileButton)
        controlLayout.addLayout(form)
        controlLayout.addWidget(self.absoluteRange)
        controlLayout.addWidget(self.windowSummary)
        controlLayout.addWidget(self.sourceLabel)
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
        self.startSpin.valueChanged.connect(self.refresh)
        self.windowSpin.valueChanged.connect(self._on_window_changed)
        self.fileButton.clicked.connect(self._prompt_open_file)

    # ----- Behaviors -------------------------------------------------------

    def _refresh_limits(self):
        max_start = max(0.0, self.loader.duration_s - self.windowSpin.value())
        self.startSpin.setRange(0.0, max_start)
        if self.startSpin.value() > max_start:
            self.startSpin.setValue(max_start)

    def _on_window_changed(self, value):
        _ = value  # unused, keeps slot signature
        self._refresh_limits()
        self.refresh()

    @QtCore.Slot()
    def refresh(self):
        t0 = self.startSpin.value()
        duration = self.windowSpin.value()
        t1 = min(self.loader.duration_s, t0 + duration)

        for i in range(self.loader.n_channels):
            t, x = self.loader.read(i, t0, t1)
            self.curves[i].setData(t, x)

        if self._primary_plot is not None:
            self._primary_plot.setXRange(t0, t1, padding=0)
        self._update_time_labels(t0, t1)

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

            self.startSpin.setValue(0.0)
            self.windowSpin.setValue(min(30.0, self.loader.duration_s))
            self._refresh_limits()
        finally:
            self.startSpin.blockSignals(False)
            self.windowSpin.blockSignals(False)

        self._ensure_plot_rows(self.loader.n_channels)
        self._configure_plots()
        self.refresh()
        self._start_zarr_ingest()

    def _start_zarr_ingest(self):
        if not getattr(self.loader, "path", None):
            return

        zarr_path = resolve_output_path(self.loader.path)
        self._zarr_path = zarr_path

        if zarr_path.exists():
            self.ingestBar.hide()
            if not isinstance(self.loader, ZarrLoader):
                try:
                    self._pending_loader = ZarrLoader(zarr_path)
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

        self._pending_loader = ZarrLoader(path)
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

        self.time_axis.set_timebase(self.loader.timebase)
        if self.loader.timebase.start_dt is not None:
            self.time_axis.set_mode("absolute")

        self._ensure_plot_rows(self.loader.n_channels)
        self._configure_plots()
        self._refresh_limits()
        self.refresh()
        self._update_data_source_label()

        if hasattr(old_loader, "close") and not isinstance(old_loader, ZarrLoader):
            old_loader.close()

    def closeEvent(self, event):
        self._cleanup_ingest_thread(wait=True)
        super().closeEvent(event)

    def _update_data_source_label(self):
        if isinstance(self.loader, ZarrLoader):
            self.sourceLabel.setText("Source: Zarr cache")
            self.sourceLabel.setStyleSheet("color: #7fb57d; font-style: italic;")
        else:
            self.sourceLabel.setText("Source: EDF (live)")
            self.sourceLabel.setStyleSheet("color: #9ba9bf; font-style: italic;")

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
