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
        controlLayout.addLayout(form)
        controlLayout.addWidget(self.absoluteRange)
        controlLayout.addWidget(self.windowSummary)
        controlLayout.addStretch(1)

        self.plotLayout = pg.GraphicsLayoutWidget()
        self.plotLayout.ci.layout.setSpacing(0)
        self.plotLayout.ci.layout.setContentsMargins(0, 0, 0, 0)

        self.plots = []
        self.curves = []
        self.channel_labels = []
        for idx, meta in enumerate(self.loader.info):
            label_text = meta.name if not meta.unit else f"{meta.name} [{meta.unit}]"
            label_html = (
                "<span style='color:#dfe7ff;font-weight:600;font-size:11pt;padding-right:12px;'>"
                f"{label_text}"
                "</span>"
            )
            label = self.plotLayout.addLabel(
                row=idx,
                col=0,
                text=label_html,
                justify="right",
            )
            self.channel_labels.append(label)

            axisItems = {"bottom": self.time_axis} if idx == self.loader.n_channels - 1 else None
            plot = self.plotLayout.addPlot(row=idx, col=1, axisItems=axisItems)
            if idx != self.loader.n_channels - 1:
                plot.showAxis("bottom", show=False)
            plot.showAxis("left", show=False)
            plot.showAxis("right", show=False)
            plot.showAxis("top", show=False)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=True, y=False)
            plot.showGrid(x=False, y=True, alpha=0.15)
            pen = pg.mkPen(pg.intColor(idx, hues=self.loader.n_channels, values=220), width=1.2)
            curve = plot.plot([], [], pen=pen)
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self.plots.append(plot)
            self.curves.append(curve)

        self._primary_plot = self.plots[-1]
        for plot in self.plots[:-1]:
            plot.setXLink(self._primary_plot)

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
            QScrollArea { background-color: #0b111c; }
        """
        )

    def _connect_signals(self):
        self.startSpin.valueChanged.connect(self.refresh)
        self.windowSpin.valueChanged.connect(self._on_window_changed)

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

        for i, curve in enumerate(self.curves):
            t, x = self.loader.read(i, t0, t1)
            curve.setData(t, x)

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
