"""Shared channel canvas backend protocol and CPU implementation."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets


SeriesPayload = Sequence[tuple[np.ndarray, np.ndarray]]


class ChannelCanvasBackend(Protocol):
    """Contract implemented by concrete channel rendering backends."""

    @property
    def widget(self) -> QtWidgets.QWidget:  # pragma: no cover - Qt accessor
        """Return the Qt widget hosting the canvas."""

    def configure_channels(
        self,
        *,
        infos: Sequence[object],
        hidden_indices: set[int],
    ) -> None:
        """Ensure backend state matches loader metadata."""

    def set_view(self, start: float, duration: float) -> None:
        """Inform the backend of the active time window."""

    def set_theme(
        self,
        *,
        background: str,
        curve_colors: Sequence[str],
        label_active: str,
        label_hidden: str | None,
        axis_color: str | None,
    ) -> None:
        """Apply colors and palette tweaks."""

    def set_timebase(self, timebase: object | None) -> None:
        """Provide the active timebase (if any)."""

    def set_time_mode(self, mode: str) -> None:
        """Switch between relative/absolute time modes."""

    def set_channel_label(self, idx: int, text: str, *, hidden: bool) -> None:
        """Update an individual channel label."""

    def set_hover_enabled(self, enabled: bool) -> None:
        """Toggle hover overlays."""

    def apply_series(
        self,
        request_id: int,
        series: SeriesPayload,
        hidden_indices: Iterable[int],
        *,
        final: bool,
        vertices: Sequence[np.ndarray] | None = None,
    ) -> None:
        """Apply prepared channel series to the canvas."""

    def estimate_pixels(self) -> int:
        """Estimate horizontal pixel budget for decimation heuristics."""


class PyqtgraphChannelBackend(ChannelCanvasBackend):
    """CPU renderer backed by pyqtgraph plots."""

    def __init__(self, layout: pg.GraphicsLayoutWidget) -> None:
        self._layout = layout
        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []
        self.channel_labels: list[pg.LabelItem] = []
        self._label_hidden: list[bool] = []
        self._label_active_color = "#dfe7ff"
        self._label_hidden_color = "#6c788f"

    # ChannelCanvasBackend -------------------------------------------------
    @property
    def widget(self) -> QtWidgets.QWidget:  # pragma: no cover - trivial
        return self._layout

    def configure_channels(
        self,
        *,
        infos: Sequence[object],
        hidden_indices: set[int],
    ) -> None:
        count = len(infos)
        self._ensure_rows(count)
        for idx, label in enumerate(self.channel_labels):
            if idx >= count:
                label.setText("")
                label.setVisible(False)
                continue
            meta = infos[idx]
            name = getattr(meta, "name", f"Ch {idx + 1}")
            unit = getattr(meta, "unit", "")
            label_text = name if not unit else f"{name} [{unit}]"
            label.setText(label_text)
            label.setVisible(True)
            hidden_flag = idx in hidden_indices
            if idx < len(self._label_hidden):
                self._label_hidden[idx] = hidden_flag
            else:
                self._label_hidden.append(hidden_flag)
            label.setAttr(
                "color",
                self._label_hidden_color if hidden_flag else self._label_active_color,
            )
        for idx, plot in enumerate(self.plots):
            active = idx < count
            if not active:
                plot.hide()
                if idx < len(self.curves):
                    self.curves[idx].setData([], [])
                if idx < len(self._label_hidden):
                    self._label_hidden[idx] = False
                continue
            plot.show()
            if idx < len(self.curves) and idx in hidden_indices:
                self.curves[idx].setData([], [])

    def set_view(self, start: float, duration: float) -> None:  # noqa: ARG002
        return

    def set_theme(
        self,
        *,
        background: str,
        curve_colors: Sequence[str],
        label_active: str,
        label_hidden: str | None,
        axis_color: str | None,
    ) -> None:
        self._layout.setBackground(background)
        for idx, curve in enumerate(self.curves):
            color = curve_colors[idx % len(curve_colors)] if curve_colors else "#5f8bff"
            curve.setPen(pg.mkPen(color, width=1.2))
        self._label_active_color = label_active
        self._label_hidden_color = label_hidden or label_active
        for idx, label in enumerate(self.channel_labels):
            hidden = self._label_hidden[idx] if idx < len(self._label_hidden) else False
            label.setAttr("color", self._label_hidden_color if hidden else self._label_active_color)
        if axis_color:
            axis_pen = pg.mkPen(axis_color)
            for plot in self.plots:
                axis = plot.getAxis("bottom")
                if axis is not None:
                    axis.setPen(axis_pen)
                    axis.setTextPen(axis_color)
                left_axis = plot.getAxis("left")
                if left_axis is not None:
                    left_axis.setPen(axis_pen)
                    left_axis.setTextPen(axis_color)

    def set_timebase(self, timebase: object | None) -> None:  # noqa: ARG002
        return

    def set_time_mode(self, mode: str) -> None:  # noqa: ARG002
        return

    def set_channel_label(self, idx: int, text: str, *, hidden: bool) -> None:
        if idx >= len(self.channel_labels):
            return
        label = self.channel_labels[idx]
        label.setText(text)
        label.setVisible(True)
        if idx < len(self._label_hidden):
            self._label_hidden[idx] = bool(hidden)
        else:
            self._label_hidden.append(bool(hidden))
        label.setAttr("color", self._label_hidden_color if hidden else self._label_active_color)

    def set_hover_enabled(self, enabled: bool) -> None:  # noqa: ARG002
        return

    def apply_series(
        self,
        request_id: int,  # noqa: ARG002
        series: SeriesPayload,
        hidden_indices: Iterable[int],
        *,
        final: bool,
        vertices: Sequence[np.ndarray] | None = None,  # noqa: ARG002
    ) -> None:
        hidden = set(hidden_indices)
        for idx, data in enumerate(series):
            if idx >= len(self.curves):
                break
            curve = self.curves[idx]
            if idx in hidden:
                curve.setData([], [])
                continue
            t_arr, x_arr = data
            if not final and x_arr.size == 0:
                continue
            curve.setData(t_arr, x_arr)

    def estimate_pixels(self) -> int:
        if not self.plots:
            return 0
        primary = self.plots[-1]
        vb = primary.getViewBox()
        if vb is None:
            return 0
        width = int(vb.width())
        return max(0, width)

    # Internal helpers ----------------------------------------------------
    def _ensure_rows(self, count: int) -> None:
        while len(self.plots) < count:
            idx = len(self.plots)
            label = self._layout.addLabel(row=idx, col=0, text="", justify="right")
            self.channel_labels.append(label)
            plot = self._layout.addPlot(row=idx, col=1)
            plot.showAxis("bottom", show=False)
            plot.showAxis("left", show=False)
            plot.showAxis("right", show=False)
            plot.showAxis("top", show=False)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=True, y=False)
            plot.showGrid(x=False, y=True, alpha=0.15)
            curve = plot.plot([], [], pen=pg.mkPen("#5f8bff", width=1.2))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self.plots.append(plot)
            self.curves.append(curve)
            self._label_hidden.append(False)

