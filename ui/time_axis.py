# ui/time_axis.py
import pyqtgraph as pg
class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            s = int(round(v))
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            out.append(f"{h:02d}:{m:02d}:{s:02d}")
        return out