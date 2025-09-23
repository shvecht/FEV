# ui/main_window.py
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ui.time_axis import TimeAxis


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

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
        old_loader = self.loader
        try:
            new_loader = type(old_loader)(path)
        except Exception as exc:  # pragma: no cover - UI feedback
            QtWidgets.QMessageBox.critical(self, "Failed to open", str(exc))
            return

        self.loader = new_loader
        old_loader.close()

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
