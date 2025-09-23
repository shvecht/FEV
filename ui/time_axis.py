# ui/time_axis.py
import pyqtgraph as pg


class TimeAxis(pg.AxisItem):
    """Format tick labels as HH:MM:SS, either relative or absolute."""

    def __init__(self, *, timebase=None, mode: str = "relative", **kwargs):
        super().__init__(**kwargs)
        self._timebase = timebase
        self._mode = mode  # 'relative' or 'absolute'

    def set_timebase(self, timebase):
        self._timebase = timebase
        self._refresh()

    def set_mode(self, mode: str):
        self._mode = mode
        self._refresh()

    def _refresh(self):
        view = self.linkedView()
        if view is not None:
            self.linkedViewChanged(view, None)
        else:
            self.update()

    def tickStrings(self, values, scale, spacing):
        out = []
        if self._mode == "absolute" and self._timebase is not None:
            for v in values:
                dt = self._timebase.to_datetime(float(v))
                out.append(dt.strftime("%H:%M:%S"))
            return out

        for v in values:
            s = int(round(v))
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            out.append(f"{h:02d}:{m:02d}:{s:02d}")
        return out
