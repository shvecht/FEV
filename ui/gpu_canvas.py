"""Experimental GPU-backed channel canvas using VisPy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import contextlib

import numpy as np
from PySide6 import QtCore, QtWidgets

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
        if _vispy_app is None or scene is None:
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

        self._grid = self._canvas.central_widget.add_grid(margin=0, bgcolor=None)
        self._grid.spacing = 0

        # We'll insert an AxisWidget at the bottom row when channels exist.
        self._x_axis: scene.AxisWidget | None = None

        self._views: list[scene.widgets.ViewBox] = []
        self._lines: list[scene.visuals.Line] = []
        self._label_nodes: list[scene.visuals.Text] = []
        self._channel_states: list[_ChannelState] = []
        self._channel_visible: list[bool] = []
        self._channel_colors: list[Color] = []

        self._hover_line = scene.visuals.Line(
            pos=np.zeros((0, 2), dtype=np.float32),
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
        self._canvas.events.mouse_leave.connect(self._on_mouse_leave)

    # ------------------------------------------------------------------
    # Public control surface

    def set_theme(
        self,
        *,
        background: str,
        curve_colors: Sequence[str],
        label_color: str,
    ) -> None:
        """Update canvas palette."""

        self._canvas.bgcolor = Color(background)
        self._channel_colors = [Color(code) for code in curve_colors]
        for idx, text in enumerate(self._label_nodes):
            text.color = Color(label_color)
            if idx < len(self._channel_colors):
                line = self._lines[idx]
                line.set_data(color=self._channel_colors[idx])
        if self._x_axis is not None:
            axis_color = Color(label_color)
            self._x_axis.axis.color = axis_color
            self._x_axis.axis.tick_color = axis_color

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
            self._views[idx].visible = self._channel_visible[idx]
            self._lines[idx].visible = self._channel_visible[idx]

        for idx in range(count, len(self._views)):
            self._views[idx].visible = False
            self._lines[idx].visible = False
            self._label_nodes[idx].text = ""

    def set_channel_visibility(self, idx: int, visible: bool) -> None:
        if idx >= len(self._views):
            return
        self._channel_visible[idx] = visible
        self._views[idx].visible = visible
        self._lines[idx].visible = visible
        if not visible:
            self._lines[idx].set_data(pos=np.zeros((0, 2), dtype=np.float32))

    def set_channel_label(self, idx: int, text: str) -> None:
        if idx < len(self._label_nodes):
            self._label_nodes[idx].text = text

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
            vertex = vertices[idx] if idx < len(vertices) else None
            if vertex is None:
                vertex = self._prepare_vertices(t_arr, x_arr)

            if not final and vertex.size == 0:
                continue

            color = self._channel_colors[idx] if idx < len(self._channel_colors) else Color("#66aaff")
            self._lines[idx].set_data(pos=vertex, color=color, width=1.2)

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
        self._lines[idx].set_data(pos=self._empty_vertices())
        if idx < len(self._channel_states):
            self._channel_states[idx] = _ChannelState(t=np.array([]), x=np.array([]))

    def set_view(self, start: float, duration: float) -> None:
        self._view_start = start
        self._view_duration = max(0.001, duration)
        x_range = (self._view_start, self._view_start + self._view_duration)
        for view in self._views:
            view.camera.set_range(x=x_range)
        if self._x_axis is not None:
            self._x_axis.domain = x_range

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
            line = scene.visuals.Line(method="gl")
            view.add(line)
            view.add(self._hover_line)
            view.add(self._hover_marker)

            self._views.append(view)
            self._lines.append(line)
            self._label_nodes.append(label)
            self._channel_states.append(_ChannelState(t=np.array([]), x=np.array([])))
            self._channel_visible.append(True)
            self._channel_colors.append(Color("#66aaff"))

        # Re-append axis under the populated rows.
        self._x_axis = scene.AxisWidget(orientation="bottom")
        self._x_axis.axis.scale_type = "linear"
        self._x_axis.height_max = 32
        self._x_axis.height_min = 28
        self._grid.add_widget(self._x_axis, row=len(self._views), col=1)
        if self._views:
            self._x_axis.link_view(self._views[0])
        self.set_view(self._view_start, self._view_duration)

    def _update_channel_range(self, idx: int, state: _ChannelState) -> None:
        if idx >= len(self._views):
            return
        if state.x.size == 0:
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
        return np.zeros((0, 2), dtype=np.float32)

