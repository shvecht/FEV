"""Experimental GPU-backed channel canvas using VisPy."""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Sequence

import contextlib

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

LOG = logging.getLogger(__name__)


try:  # pragma: no cover - import guarded for optional dependency
    from vispy import app as _vispy_app
    from vispy import scene
    from vispy.scene import transforms
    from vispy.color import Color
except Exception:  # pragma: no cover - handled by MainWindow fallback
    _vispy_app = None
    scene = None  # type: ignore
    Color = None  # type: ignore


@dataclass(slots=True)
class _ChannelState:
    t: np.ndarray
    x: np.ndarray


@dataclass(frozen=True, slots=True)
class VispyCapability:
    """Result of probing VisPy/OpenGL readiness."""

    available: bool
    reason: str | None = None
    vertex_budget: int = 0
    vendor: str | None = None
    renderer: str | None = None
    backend: str | None = None


class VispyChannelCanvas(QtWidgets.QWidget):
    """Stacked multichannel canvas rendered via VisPy.

    This widget mirrors the CPU-backed pyqtgraph grid but draws each channel
    using VisPy's GPU line visuals. The widget purposefully keeps its API
    narrow so the caller (``MainWindow``) controls loader interactions and
    application state.
    """

    hoverMoved = QtCore.Signal(float, float, int)
    hoverExited = QtCore.Signal()

    DEFAULT_VERTEX_BUDGET = 900_000

    @classmethod
    def capability_probe(cls) -> VispyCapability:
        """Attempt to create a minimal canvas to gauge readiness."""

        if _vispy_app is None or scene is None:
            return VispyCapability(False, reason="VisPy import failed")

        try:
            _vispy_app.use_app("pyside6")
        except RuntimeError:
            # Already initialised – safe to ignore.
            pass

        try:
            canvas = scene.SceneCanvas(show=False, size=(4, 4), bgcolor="#000000")
        except Exception as exc:  # pragma: no cover - headless or driver issues
            LOG.debug("VisPy capability probe failed to create canvas: %s", exc)
            return VispyCapability(False, reason=str(exc))

        try:
            context = getattr(canvas, "context", None)
            info_dict = {}
            backend = renderer = vendor = None
            if context is not None:
                info_dict = getattr(context, "gl_info", {}) or {}
                if isinstance(info_dict, dict):
                    backend = info_dict.get("backend")
                    renderer = info_dict.get("renderer")
                    vendor = info_dict.get("vendor")

            vertex_budget = 0
            shared = getattr(context, "shared", None)
            limits_dict = None
            if shared is not None:
                parser = getattr(shared, "parser", None)
                limits_dict = getattr(parser, "_limits", None)
            if isinstance(limits_dict, dict):
                for key in (
                    "max_elements_vertices",
                    "max_vertices",
                    "GL_MAX_ELEMENTS_VERTICES",
                    "GL_MAX_ELEMENTS_INDICES",
                ):
                    value = limits_dict.get(key)
                    if isinstance(value, (int, float)) and value > 0:
                        vertex_budget = int(value)
                        break

            if not vertex_budget:
                vertex_budget = cls.DEFAULT_VERTEX_BUDGET

            return VispyCapability(
                True,
                reason=None,
                vertex_budget=int(vertex_budget),
                vendor=vendor,
                renderer=renderer,
                backend=backend,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug("VisPy capability probe failed: %s", exc)
            return VispyCapability(False, reason=str(exc))
        finally:
            with contextlib.suppress(Exception):
                canvas.close()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        if _vispy_app is None or scene is None or Color is None:
            raise RuntimeError("VisPy is not available")

        try:
            _vispy_app.use_app("pyside6")
        except RuntimeError:
            # Already initialised – safe to ignore.
            pass

        self._canvas = scene.SceneCanvas(
            keys=None,
            show=False,
            bgcolor="#10141a",
            vsync=True,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas.native)

        self.setMinimumSize(0, 0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        native = self._canvas.native
        native.setMinimumSize(0, 0)
        native.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        native.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._grid = self._canvas.central_widget.add_grid(margin=0, bgcolor=None)
        self._grid.spacing = 0

        # We'll insert an AxisWidget at the bottom row when channels exist.
        self._x_axis: scene.AxisWidget | None = None

        self._views: list[scene.widgets.ViewBox] = []
        self._lines: list[scene.visuals.Line] = []
        self._label_nodes: list[scene.visuals.Text] = []
        self._grid_lines: list[scene.visuals.GridLines] = []
        self._channel_states: list[_ChannelState] = []
        self._channel_visible: list[bool] = []
        self._channel_colors: list[Color] = []
        self._view_y_ranges: list[tuple[float, float]] = []
        self._label_hidden: list[bool] = []

        self._background_color = Color("#10141a")
        self._label_active_color = Color("#dfe7ff")
        self._label_hidden_color = Color("#6c788f")
        self._grid_line_color = Color((0.45, 0.52, 0.68, 0.2))
        self._axis_formatter_installed = False
        self._timebase = None
        self._time_mode = "relative"

        self._hover_line = scene.visuals.Line(
            pos=self._empty_vertices(),
            color=Color("#d0d4ff"),
            width=1.0,
            method="gl",
        )
        self._hover_line.visible = False
        self._hover_marker = scene.visuals.Text(
            text="",
            color="white",
            anchor_x="left",
            anchor_y="top",
        )
        self._hover_marker.visible = False
        self._hover_enabled = False
        self._hover_channel: int | None = None

        self._view_start = 0.0
        self._view_duration = 30.0

        # Connect canvas mouse events to drive hover feedback.
        self._canvas.events.mouse_move.connect(self._on_mouse_move)
        mouse_leave_emitter = getattr(self._canvas.events, "mouse_leave", None)
        if mouse_leave_emitter is not None:
            mouse_leave_emitter.connect(self._on_mouse_leave)
        else:  # pragma: no cover - depends on VisPy backend
            LOG.debug("VisPy canvas has no mouse_leave event; hover will persist until next move")

    def sizeHint(self) -> QtCore.QSize:  # pragma: no cover - simple geometry hint
        return QtCore.QSize(960, 720)

    def minimumSizeHint(self) -> QtCore.QSize:  # pragma: no cover - simple geometry hint
        return QtCore.QSize(320, 240)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # pragma: no cover - GUI only
        super().showEvent(event)
        self._sync_canvas_geometry()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # pragma: no cover - GUI only
        super().resizeEvent(event)
        self._sync_canvas_geometry(event.size())

    def event(self, event: QtCore.QEvent) -> bool:  # pragma: no cover - GUI only
        event_type = event.type()
        dpi_events: tuple[QtCore.QEvent.Type, ...] = tuple(
            t
            for t in (
                getattr(QtCore.QEvent.Type, "DevicePixelRatioChange", None),
                getattr(QtCore.QEvent.Type, "DpiChange", None),
                getattr(QtCore.QEvent.Type, "ScreenChangeInternal", None),
            )
            if t is not None
        )
        if dpi_events and event_type in dpi_events:
            QtCore.QTimer.singleShot(0, self._sync_canvas_geometry)
        return super().event(event)

    def _sync_canvas_geometry(self, size: QtCore.QSize | None = None) -> None:
        if size is None:
            size = self.size()
        width = max(1, size.width())
        height = max(1, size.height())
        native = self._canvas.native
        device_ratio = 1.0
        ratio_getter = getattr(native, "devicePixelRatioF", None)
        if callable(ratio_getter):
            try:
                device_ratio = float(ratio_getter())
            except Exception:
                device_ratio = 1.0
        elif hasattr(native, "devicePixelRatio"):
            try:
                device_ratio = float(native.devicePixelRatio())
            except Exception:
                device_ratio = 1.0
        device_ratio = max(device_ratio, 1.0)
        physical_width = max(1, int(round(width * device_ratio)))
        physical_height = max(1, int(round(height * device_ratio)))
        try:
            self._canvas.size = (physical_width, physical_height)
            with contextlib.suppress(AttributeError, TypeError):
                self._canvas.pixel_scale = device_ratio
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public control surface

    def set_theme(
        self,
        *,
        background: str,
        curve_colors: Sequence[str],
        label_active: str,
        label_hidden: str | None = None,
        axis_color: str | None = None,
    ) -> None:
        """Update canvas palette."""

        self._background_color = Color(background)
        self._canvas.bgcolor = self._background_color
        self._channel_colors = [Color(code) for code in curve_colors]
        self._label_active_color = Color(label_active)
        if label_hidden:
            self._label_hidden_color = Color(label_hidden)
        else:
            rgba = np.array(self._label_active_color.rgba)
            rgba[3] = max(0.25, float(rgba[3]) * 0.6)
            self._label_hidden_color = Color(rgba)
        self._grid_line_color = self._compute_grid_color()
        for idx in range(len(self._label_nodes)):
            if idx < len(self._channel_colors):
                self._lines[idx].set_data(color=self._channel_colors[idx])
            self._apply_label_style(idx)
        if self._x_axis is not None:
            axis_color_obj = Color(axis_color or label_active)
            # Update available color properties without assigning new attributes.
            with contextlib.suppress(AttributeError):
                self._x_axis.axis.tick_color = axis_color_obj
            with contextlib.suppress(AttributeError):
                self._x_axis.axis.text_color = axis_color_obj
            with contextlib.suppress(AttributeError):
                self._x_axis.axis.axis_color = axis_color_obj
            with contextlib.suppress(AttributeError):
                self._x_axis.axis.tick_width = 1.0
            with contextlib.suppress(AttributeError):
                self._x_axis.axis.axis_width = 1.0
            with contextlib.suppress(AttributeError):
                self._x_axis.axis.tick_font_size = 11
            with contextlib.suppress(AttributeError):
                self._x_axis.bgcolor = self._background_color
            self._install_axis_formatter()
            self._request_axis_update()
        self._update_grid_palette()

    def configure_channels(
        self,
        *,
        infos: Sequence[object],
        hidden_indices: set[int],
    ) -> None:
        """Ensure canvas rows match the loader metadata."""

        count = len(infos)
        self._ensure_rows(count)
        self._channel_visible = [idx not in hidden_indices for idx in range(count)]
        for idx, meta in enumerate(infos):
            name = getattr(meta, "name", f"Ch {idx + 1}")
            unit = getattr(meta, "unit", "")
            label = f"{name}"
            if unit:
                label = f"{label} [{unit}]"
            self._label_nodes[idx].text = label
            visible = self._channel_visible[idx]
            self._views[idx].visible = visible
            self._lines[idx].visible = visible
            if idx < len(self._grid_lines):
                self._grid_lines[idx].visible = visible
            if idx < len(self._label_hidden):
                self._label_hidden[idx] = not visible
            self._apply_label_style(idx)

        for idx in range(count, len(self._views)):
            self._views[idx].visible = False
            self._lines[idx].visible = False
            self._label_nodes[idx].text = ""
            if idx < len(self._grid_lines):
                self._grid_lines[idx].visible = False
            if idx < len(self._label_hidden):
                self._label_hidden[idx] = False
                self._apply_label_style(idx)

    def set_channel_visibility(self, idx: int, visible: bool) -> None:
        if idx >= len(self._views):
            return
        self._channel_visible[idx] = visible
        self._views[idx].visible = visible
        self._lines[idx].visible = visible
        if idx < len(self._grid_lines):
            self._grid_lines[idx].visible = visible
        if idx < len(self._label_hidden):
            self._label_hidden[idx] = not visible
            self._apply_label_style(idx)
        if not visible:
            self._lines[idx].set_data(pos=np.zeros((0, 2), dtype=np.float32))

    def set_channel_label(self, idx: int, text: str, *, hidden: bool | None = None) -> None:
        if idx < len(self._label_nodes):
            self._label_nodes[idx].text = text
            if hidden is not None and idx < len(self._label_hidden):
                self._label_hidden[idx] = bool(hidden)
            self._apply_label_style(idx)

    def set_curve_color(self, idx: int, color: str) -> None:
        if idx >= len(self._lines):
            return
        if idx >= len(self._channel_colors):
            try:
                self._channel_colors.append(Color(color))
            except Exception:
                self._channel_colors.append(Color("#66aaff"))
        else:
            try:
                self._channel_colors[idx] = Color(color)
            except Exception:
                self._channel_colors[idx] = Color("#66aaff")
        self._lines[idx].set_data(color=self._channel_colors[idx])

    def set_channel_data(self, idx: int, t: np.ndarray, x: np.ndarray) -> None:
        if idx >= len(self._lines):
            return
        if t.size == 0 or x.size == 0:
            self._lines[idx].set_data(pos=self._empty_vertices())
            if idx < len(self._channel_states):
                self._channel_states[idx] = _ChannelState(t=np.array([]), x=np.array([]))
            return

        pos = self._prepare_vertices(t, x)
        if idx < len(self._channel_colors):
            color = self._channel_colors[idx]
        else:
            color = Color("#66aaff")
        self._lines[idx].set_data(pos=pos, color=color, width=1.2)

        state = _ChannelState(t=t.copy(), x=x.copy())
        if idx < len(self._channel_states):
            self._channel_states[idx] = state
        else:
            self._channel_states.append(state)
        self._update_channel_range(idx, state)

    def apply_tile_data(
        self,
        tile_id: int,
        series: Sequence[tuple[np.ndarray, np.ndarray]],
        vertices: Sequence[np.ndarray],
        hidden_indices: set[int],
        *,
        final: bool,
    ) -> None:
        del tile_id  # reserved for future use

        limit = min(len(series), len(self._lines))
        for idx in range(limit):
            hidden = idx in hidden_indices
            if idx < len(self._channel_visible) and not self._channel_visible[idx]:
                hidden = True
            if hidden:
                self.clear_channel(idx)
                continue

            t_arr, x_arr = series[idx]
            if t_arr.size == 0 or x_arr.size == 0:
                self.clear_channel(idx)
                continue
            vertex = vertices[idx] if idx < len(vertices) else None
            if vertex is None:
                vertex = self._prepare_vertices(t_arr, x_arr)
            if vertex is None or vertex.shape[0] < 2:
                self.clear_channel(idx)
                continue

            color = self._channel_colors[idx] if idx < len(self._channel_colors) else Color("#66aaff")
            line_item = self._lines[idx]
            line_item.visible = True
            line_item.set_data(pos=vertex, color=color, width=1.2)

            state = _ChannelState(t=t_arr.copy(), x=x_arr.copy())
            if idx < len(self._channel_states):
                self._channel_states[idx] = state
            else:
                self._channel_states.append(state)
            self._update_channel_range(idx, state)

        for idx in range(limit, len(self._lines)):
            self.clear_channel(idx)

    def clear_channel(self, idx: int) -> None:
        if idx >= len(self._lines):
            return
        line = self._lines[idx]
        line.visible = False
        line.set_data(pos=self._empty_vertices())
        if idx < len(self._channel_states):
            self._channel_states[idx] = _ChannelState(t=np.array([]), x=np.array([]))
        if idx < len(self._view_y_ranges):
            self._view_y_ranges[idx] = (-1.0, 1.0)
            if idx < len(self._views):
                self._views[idx].camera.set_range(y=(-1.0, 1.0))
        if idx < len(self._grid_lines):
            self._grid_lines[idx].visible = False

    def set_view(self, start: float, duration: float) -> None:
        self._view_start = start
        self._view_duration = max(0.001, duration)
        x_range = (self._view_start, self._view_start + self._view_duration)
        for idx, view in enumerate(self._views):
            y_range = (-1.0, 1.0)
            if idx < len(self._view_y_ranges):
                y_range = self._view_y_ranges[idx]
            view.camera.set_range(x=x_range, y=y_range)
        if self._x_axis is not None and self._views:
            # AxisWidget follows the first linked view; ensure linkage stays intact.
            with contextlib.suppress(Exception):
                self._x_axis.link_view(self._views[0])
            self._request_axis_update()

    def estimate_pixels(self) -> int:
        native = self._canvas.native
        try:
            return int(native.width())
        except Exception:  # pragma: no cover - very defensive
            return 0

    def set_hover_enabled(self, enabled: bool) -> None:
        self._hover_enabled = bool(enabled)
        if not enabled:
            self._hover_line.visible = False
            self._hover_marker.visible = False
            self._hover_channel = None

    # ------------------------------------------------------------------
    # Internal helpers

    def _ensure_rows(self, count: int) -> None:
        if count <= len(self._views):
            return

        # Remove existing axis to append below new rows later.
        if self._x_axis is not None:
            self._grid.remove_widget(self._x_axis)
            self._x_axis = None

        while len(self._views) < count:
            row = len(self._views)
            label = scene.visuals.Text(
                text="",
                anchor_x="left",
                anchor_y="top",
                color="white",
                font_size=12,
            )
            label_transform = transforms.STTransform(translate=(4, 4))
            label.transform = label_transform

            view = self._grid.add_view(row=row, col=1, camera="panzoom")
            view.camera.interactive = False
            view.border_color = None
            view.padding = 0.0

            view.add(label)
            grid = scene.visuals.GridLines(color=self._grid_line_color, scale=(1.0, 1.0))
            grid.set_gl_state(depth_test=False)
            view.add(grid)
            line = scene.visuals.Line(method="gl")
            line.set_data(pos=self._empty_vertices())
            line.visible = False
            view.add(line)
            view.add(self._hover_line)
            view.add(self._hover_marker)

            self._views.append(view)
            self._lines.append(line)
            self._label_nodes.append(label)
            self._grid_lines.append(grid)
            self._channel_states.append(_ChannelState(t=np.array([]), x=np.array([])))
            self._channel_visible.append(True)
            self._channel_colors.append(Color("#66aaff"))
            self._view_y_ranges.append((-1.0, 1.0))
            self._label_hidden.append(False)

        # Re-append axis under the populated rows.
        self._x_axis = scene.AxisWidget(orientation="bottom")
        self._x_axis.axis.scale_type = "linear"
        self._x_axis.height_max = 32
        self._x_axis.height_min = 28
        self._grid.add_widget(self._x_axis, row=len(self._views), col=1)
        if self._views:
            self._x_axis.link_view(self._views[0])
        self._install_axis_formatter()
        self._update_axis_theme()
        self.set_view(self._view_start, self._view_duration)

    def _update_channel_range(self, idx: int, state: _ChannelState) -> None:
        if idx >= len(self._views):
            return
        if state.x.size == 0:
            if idx < len(self._view_y_ranges):
                self._view_y_ranges[idx] = (-1.0, 1.0)
            self._views[idx].camera.set_range(y=(-1.0, 1.0))
            return
        min_val = float(np.nanmin(state.x))
        max_val = float(np.nanmax(state.x))
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            min_val, max_val = -1.0, 1.0
        if abs(max_val - min_val) < 1e-6:
            pad = max(1e-3, abs(max_val) * 0.1 + 1e-3)
            min_val -= pad
            max_val += pad
        if idx < len(self._view_y_ranges):
            self._view_y_ranges[idx] = (min_val, max_val)
        self._views[idx].camera.set_range(y=(min_val, max_val))

    # ------------------------------------------------------------------
    # Hover feedback

    def _on_mouse_move(self, event) -> None:  # pragma: no cover - GUI only
        if not self._hover_enabled:
            return
        if event.pos is None:
            self._hide_hover()
            return

        # Map cursor to data coordinates across views.
        for idx, view in enumerate(self._views):
            if idx >= len(self._channel_visible) or not self._channel_visible[idx]:
                continue
            if not view.visible:
                continue
            tr = view.scene.transform
            if tr is None:
                continue
            bounds = view.scene.node_transform(view.scene).map(view.node.bounds(axis=0))
            if bounds is None:
                continue
            try:
                rect = view.camera.rect
            except AttributeError:
                rect = None

            if rect is None:
                rect = view.camera.get_state()["rect"]

            # Rough containment test in framebuffer coordinates.
            if not view.canvas.native.rect().contains(event.pos):
                continue

            # Convert to data coordinates.
            try:
                data_pos = view.camera.transform.imap(event.pos)
            except Exception:
                continue
            if data_pos is None:
                continue
            t_val = float(data_pos[0])
            sample = self._interpolate_sample(idx, t_val)
            if sample is None:
                self._hide_hover()
                return

            sample_t, sample_x = sample
            self._hover_channel = idx
            self._hover_line.set_data(pos=np.array([[sample_t, rect.y0], [sample_t, rect.y1]], dtype=np.float32))
            self._hover_line.visible = True
            self._hover_marker.text = f"{sample_x:.3f}"
            self._hover_marker.pos = sample_t, sample_x
            self._hover_marker.visible = True
            self.hoverMoved.emit(sample_t, sample_x, idx)
            return

        self._hide_hover()

    def _on_mouse_leave(self, event) -> None:  # pragma: no cover - GUI only
        if self._hover_enabled:
            self._hide_hover()

    def _hide_hover(self) -> None:
        if self._hover_channel is not None:
            self.hoverExited.emit()
        self._hover_channel = None
        self._hover_line.visible = False
        self._hover_marker.visible = False

    def _interpolate_sample(self, idx: int, t_val: float):
        if idx >= len(self._channel_states):
            return None
        state = self._channel_states[idx]
        if state.t.size == 0:
            return None
        t_arr = state.t
        x_arr = state.x
        pos = np.searchsorted(t_arr, t_val)
        if pos <= 0:
            return float(t_arr[0]), float(x_arr[0])
        if pos >= t_arr.size:
            return float(t_arr[-1]), float(x_arr[-1])
        t0, t1 = t_arr[pos - 1], t_arr[pos]
        x0, x1 = x_arr[pos - 1], x_arr[pos]
        if t1 == t0:
            return float(t0), float(x0)
        alpha = (t_val - t0) / (t1 - t0)
        value = x0 + alpha * (x1 - x0)
        return float(t_val), float(value)

    @staticmethod
    def _prepare_vertices(t: np.ndarray, x: np.ndarray) -> np.ndarray:
        if t.size == 0 or x.size == 0:
            return VispyChannelCanvas._empty_vertices()
        pos = np.empty((t.size, 2), dtype=np.float32)
        pos[:, 0] = np.asarray(t, dtype=np.float32)
        pos[:, 1] = np.asarray(x, dtype=np.float32)
        return pos

    @staticmethod
    def _empty_vertices() -> np.ndarray:
        # Two nearly coincident vertices keep VisPy bounds finite without rendering a visible segment.
        return np.array([[0.0, 0.0], [1e-6, 0.0]], dtype=np.float32)

    # ------------------------------------------------------------------
    # Palette helpers

    def _apply_label_style(self, idx: int) -> None:
        if idx >= len(self._label_nodes):
            return
        label = self._label_nodes[idx]
        hidden = self._label_hidden[idx] if idx < len(self._label_hidden) else False
        color = self._label_hidden_color if hidden else self._label_active_color
        label.color = color
        label.bold = not hidden
        label.italic = hidden

    def _compute_grid_color(self) -> Color:
        fg = np.array(self._label_active_color.rgba)
        bg = np.array(self._background_color.rgba)
        luminance = float(np.dot(bg[:3], np.array([0.2126, 0.7152, 0.0722])))
        if luminance > 0.5:
            base = np.clip(bg[:3] * 0.4, 0.0, 1.0)
            alpha = 0.25
        else:
            base = np.clip(fg[:3] * 0.7 + 0.1, 0.0, 1.0)
            alpha = 0.2
        return Color((float(base[0]), float(base[1]), float(base[2]), alpha))

    def _update_grid_palette(self) -> None:
        rgba = tuple(self._grid_line_color.rgba)
        for idx, grid in enumerate(self._grid_lines):
            with contextlib.suppress(Exception):
                grid.set_data(color=rgba)
            grid.visible = idx < len(self._channel_visible) and self._channel_visible[idx]

    # ------------------------------------------------------------------
    # Axis helpers

    def set_timebase(self, timebase) -> None:
        self._timebase = timebase
        self._request_axis_update()

    def set_time_mode(self, mode: str) -> None:
        mode_lower = str(mode).lower()
        self._time_mode = "absolute" if mode_lower == "absolute" else "relative"
        self._request_axis_update()

    def _format_tick_label(self, value: float) -> str:
        if not np.isfinite(value):
            return ""
        if self._time_mode == "absolute" and self._timebase is not None:
            try:
                dt = self._timebase.to_datetime(float(value))
            except Exception:
                dt = None
            if dt is not None:
                with contextlib.suppress(Exception):
                    return dt.strftime("%H:%M:%S")
        seconds = float(value)
        sign = "-" if seconds < 0 else ""
        total = int(round(abs(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}"

    def _install_axis_formatter(self) -> None:
        if self._axis_formatter_installed or self._x_axis is None:
            return
        axis = getattr(self._x_axis, "axis", None)
        if axis is None:
            return
        original = getattr(axis, "_get_tick_frac_labels", None)
        if original is None:
            return

        def patched(self_axis):
            major_frac, minor_frac, labels = original()
            if major_frac is None or len(major_frac) == 0:
                return major_frac, minor_frac, labels
            domain = getattr(self_axis, "domain", None)
            if not domain or len(domain) != 2:
                return major_frac, minor_frac, labels
            try:
                start = float(domain[0])
                end = float(domain[1])
            except Exception:
                return major_frac, minor_frac, labels
            span = end - start
            if abs(span) < 1e-12:
                values = [start for _ in major_frac]
            else:
                values = [start + frac * span for frac in major_frac]
            formatted = [self._format_tick_label(val) for val in values]
            return major_frac, minor_frac, formatted

        axis._get_tick_frac_labels = types.MethodType(patched, axis)
        self._axis_formatter_installed = True

    def _update_axis_theme(self) -> None:
        if self._x_axis is None:
            return
        axis = getattr(self._x_axis, "axis", None)
        if axis is None:
            return
        axis.tick_font_size = 11
        axis.axis_width = 1.0
        axis.tick_width = 1.0
        axis.tick_color = self._label_active_color
        axis.text_color = self._label_active_color
        axis.axis_color = self._label_active_color
        self._x_axis.bgcolor = self._background_color

    def _request_axis_update(self) -> None:
        if self._x_axis is None:
            return
        axis = getattr(self._x_axis, "axis", None)
        if axis is not None:
            with contextlib.suppress(Exception):
                axis.update()
        with contextlib.suppress(Exception):
            self._x_axis.update()
