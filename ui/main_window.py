# ui/main_window.py
import numpy as np, pyqtgraph as pg
from PySide6 import QtWidgets, QtCore
from ui.time_axis import TimeAxis

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader
        self.setWindowTitle("EDF Viewer — Phase 2")

        axis = TimeAxis(orientation='bottom')
        self.plot = pg.PlotWidget(axisItems={'bottom': axis})
        self.curve = self.plot.plot([], [], pen='w')
        self.plot.showGrid(x=True, y=True)

        self.chanBox = QtWidgets.QComboBox()
        self.chanBox.addItems(loader.channels)
        self.winSpin = QtWidgets.QDoubleSpinBox()
        self.winSpin.setRange(5, 120); self.winSpin.setValue(30); self.winSpin.setSuffix(" s")
        self.t0Spin = QtWidgets.QDoubleSpinBox()
        self.t0Spin.setRange(0, max(0.0, loader.duration_s - self.winSpin.value()))
        self.t0Spin.setDecimals(3); self.t0Spin.setSingleStep(1.0)

        toolbar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(toolbar)
        for w in (QtWidgets.QLabel("Channel:"), self.chanBox,
                  QtWidgets.QLabel("Start:"), self.t0Spin,
                  QtWidgets.QLabel("Window:"), self.winSpin):
            layout.addWidget(w)
        layout.addStretch(1)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addWidget(toolbar); v.addWidget(self.plot)
        self.setCentralWidget(central)

        self.chanBox.currentIndexChanged.connect(self.refresh)
        self.winSpin.valueChanged.connect(self._update_limits)
        self.t0Spin.valueChanged.connect(self.refresh)
        self._update_limits(); self.refresh()

    def _update_limits(self):
        max_start = max(0.0, self.loader.duration_s - self.winSpin.value())
        self.t0Spin.setMaximum(max_start)

    @QtCore.Slot()
    def refresh(self):
        i = self.chanBox.currentIndex()
        t0 = self.t0Spin.value()
        t1 = t0 + self.winSpin.value()
        t, x = self.loader.read(i, t0, t1)
        self.curve.setData(t, x)
        self.plot.setXRange(t0, t1, padding=0)