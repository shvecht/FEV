# ui/time_axis.py
import pyqtgraph as pg

from ui.time_axis_formatter import TimeTickFormatter


class TimeAxis(pg.AxisItem):
    """Format tick labels as HH:MM:SS, either relative or absolute."""

    def __init__(self, *, timebase=None, mode: str = "relative", **kwargs):
        super().__init__(**kwargs)
        self._formatter = TimeTickFormatter(timebase=timebase, mode=mode)
        self._timebase = timebase
        self._mode = self._formatter.mode

    def set_timebase(self, timebase):
        self._timebase = timebase
        self._formatter.set_timebase(timebase)
        self._refresh()

    def set_mode(self, mode: str):
        self._formatter.set_mode(mode)
        self._mode = self._formatter.mode
        self._refresh()

    def _refresh(self):
        view = self.linkedView()
        if view is not None:
            self.linkedViewChanged(view, None)
        else:
            self.update()

    def tickStrings(self, values, scale, spacing):
        return self._formatter.format_ticks(values)
