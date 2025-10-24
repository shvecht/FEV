"""Experimental GPU-backed channel canvas using VisPy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import contextlib

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from ui.time_axis_formatter import TimeTickFormatter

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
    supports_annotations = True

    @property
    def widget(self) -> QtWidgets.QWidget:  # pragma: no cover - trivial accessor
        return self

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

        self._gutter_views: list[scene.widgets.ViewBox] = []
        self._views: list[scene.widgets.ViewBox] = []
        self._lines: list[scene.visuals.Line] = []
        self._label_nodes: list[scene.visuals.Text] = []
        self._grid_lines: list[scene.visuals.GridLines] = []
        self._channel_states: list[_ChannelState] = []
        self._channel_visible: list[bool] = []
        self._channel_colors: list[Color] = []
        self._view_y_ranges: list[tuple[float, float]] = []
        self._label_hidden: list[bool] = []
        self._gutter_axis_placeholder: scene.widgets.Widget | None = None

        self._background_color = Color("#10141a")
        self._label_active_color = Color("#dfe7ff")
        self._label_hidden_color = Color("#6c788f")
        self._grid_line_color = Color((0.45, 0.52, 0.68, 0.2))
        self._axis_formatter = TimeTickFormatter()
        self._timebase = None
        self._time_mode = "relative"

        self._hypnogram_view: scene.widgets.ViewBox | None = None
        self._hypnogram_label_view: scene.widgets.ViewBox | None = None
        self._hypnogram_label_text: scene.visuals.Text | None = None
        self._hypnogram_outline: scene.visuals.Line | None = None
        self._hypnogram_fill_meshes: dict[str, scene.visuals.Mesh] = {}
        self._hypnogram_region_mesh: scene.visuals.Mesh | None = None
        self._hypnogram_region_border: scene.visuals.Line | None = None
        self._hypnogram_visible = False
        self._hypnogram_y_range: tuple[float, float] = (-0.6, 0.6)
        self._annotation_rectangles: list[scene.visuals.Mesh] = []
        self._annotation_marker_lines: list[scene.visuals.Line] = []
        self._annotation_events_visible = False
        self._annotation_faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)

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
            self._canvas.size = (width, height)
        except Exception:
            pass

        backend = getattr(self._canvas, "_backend", None)
        if backend is not None:
            set_physical = getattr(backend, "_vispy_set_physical_size", None)
            if callable(set_physical):
                with contextlib.suppress(Exception):
                    set_physical(physical_width, physical_height)

        with contextlib.suppress(Exception):
            context = getattr(self._canvas, "context", None)
            if context is not None:
                context.set_viewport(0, 0, physical_width, physical_height)
        with contextlib.suppress(Exception):
            self._canvas.update()

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
        for gutter in self._gutter_views:
            with contextlib.suppress(Exception):
                gutter.bgcolor = self._background_color
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
        if self._hypnogram_view is not None:
            self._hypnogram_view.bgcolor = self._background_color
        if self._hypnogram_label_text is not None:
            self._hypnogram_label_text.color = self._label_active_color

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
            if idx < len(self._gutter_views):
                self._gutter_views[idx].visible = visible
            self._views[idx].visible = visible
            self._lines[idx].visible = visible
            if idx < len(self._grid_lines):
                self._grid_lines[idx].visible = visible
            if idx < len(self._label_hidden):
                self._label_hidden[idx] = not visible
            self._apply_label_style(idx)

        for idx in range(count, len(self._views)):
            if idx < len(self._gutter_views):
                self._gutter_views[idx].visible = False
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
        if idx < len(self._gutter_views):
            self._gutter_views[idx].visible = visible
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

    def apply_series(
        self,
        request_id: int,
        series: Sequence[tuple[np.ndarray, np.ndarray]],
        hidden_indices: Sequence[int],
        *,
        final: bool,
        vertices: Sequence[np.ndarray] | None = None,
    ) -> None:
        hidden_set = set(hidden_indices)
        if vertices is None:
            vertices = [self._prepare_vertices(t, x) for (t, x) in series]
        self.apply_tile_data(
            request_id,
            series,
            list(vertices),
            hidden_set,
            final=final,
        )

    def update_hypnogram(
        self,
        payload: dict[str, object] | None,
        *,
        visible: bool,
        view_start: float,
        view_end: float,
    ) -> None:
        if not visible or payload is None:
            self._hypnogram_visible = False
            if self._hypnogram_outline is not None:
                self._hypnogram_outline.visible = False
            self._update_hypnogram_region_visual(view_start, view_end)
            self._update_annotation_lane_visibility()
            return

        self._ensure_hypnogram_row()
        self._hypnogram_visible = True

        step_x = payload.get("step_x")
        step_y = payload.get("step_y")
        outline_points: list[tuple[float, float]] = []
        outline_connect: list[tuple[int, int]] = []
        if isinstance(step_x, np.ndarray) and isinstance(step_y, np.ndarray):
            prev_idx = None
            for x_val, y_val in zip(step_x, step_y):
                if not (np.isfinite(x_val) and np.isfinite(y_val)):
                    prev_idx = None
                    continue
                outline_points.append((float(x_val), float(y_val)))
                curr_idx = len(outline_points) - 1
                if prev_idx is not None:
                    outline_connect.append((prev_idx, curr_idx))
                prev_idx = curr_idx
        if self._hypnogram_outline is not None:
            if outline_points:
                pos_arr = np.asarray(outline_points, dtype=np.float32)
                connect_arr = (
                    np.asarray(outline_connect, dtype=np.uint32)
                    if outline_connect
                    else None
                )
                self._hypnogram_outline.set_data(
                    pos=pos_arr,
                    connect=connect_arr,
                    color=self._label_active_color,
                    width=1.0,
                )
                self._hypnogram_outline.visible = True
            else:
                self._hypnogram_outline.visible = False

        label_data = payload.get("label_data", {})
        colors = payload.get("colors", {})
        active_labels: set[str] = set()
        if isinstance(label_data, dict):
            for label, data in label_data.items():
                x_vals = data.get("x")
                top_vals = data.get("top")
                fill_level = float(data.get("fill", 0.0))
                if not isinstance(x_vals, np.ndarray) or not isinstance(top_vals, np.ndarray):
                    continue
                vertices, faces = self._stage_fill_mesh_data(x_vals, top_vals, fill_level)
                mesh = self._hypnogram_fill_meshes.get(label)
                if vertices is None or faces is None:
                    if mesh is not None:
                        mesh.visible = False
                    continue
                if mesh is None:
                    mesh = scene.visuals.Mesh()
                    mesh.set_gl_state(depth_test=False, blend=True)
                    mesh.visible = False
                    if self._hypnogram_view is not None:
                        self._hypnogram_view.add(mesh)
                    self._hypnogram_fill_meshes[label] = mesh
                color = colors.get(label)
                if isinstance(color, QtGui.QColor):
                    rgba = (
                        float(color.redF()),
                        float(color.greenF()),
                        float(color.blueF()),
                        0.5,
                    )
                else:
                    rgba = Color(color or "#6c788f").rgba
                mesh.set_data(vertices=vertices, faces=faces, color=rgba)
                mesh.visible = True
                active_labels.add(label)
        for label, mesh in list(self._hypnogram_fill_meshes.items()):
            if label not in active_labels:
                mesh.visible = False

        try:
            max_level = float(payload.get("max_level", 0.0) or 0.0)
        except (TypeError, ValueError):
            max_level = 0.0
        self._hypnogram_y_range = (-0.6, max(0.6, max_level + 0.6))
        if self._hypnogram_view is not None:
            self._hypnogram_view.camera.set_range(
                x=(view_start, view_end),
                y=self._hypnogram_y_range,
            )

        self._update_hypnogram_region_visual(view_start, view_end)
        self._update_annotation_lane_visibility()

    def update_annotations(
        self,
        events: Sequence[dict[str, object]] | None,
        *,
        view_start: float,
        view_end: float,
    ) -> None:
        entries = list(events or [])
        if not entries:
            self._annotation_events_visible = False
            for mesh in self._annotation_rectangles:
                mesh.visible = False
            for line in self._annotation_marker_lines:
                line.visible = False
            self._update_annotation_lane_visibility()
            return

        self._ensure_annotation_mesh_pool(len(entries))
        self._ensure_annotation_marker_pool(len(entries))
        self._annotation_events_visible = True

        y0, y1 = self._hypnogram_y_range
        if not (np.isfinite(y0) and np.isfinite(y1)) or y1 <= y0:
            y0, y1 = -0.6, 0.6

        faces = self._annotation_faces
        for idx, entry in enumerate(entries):
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", start))
            if not np.isfinite(start):
                start = 0.0
            if not np.isfinite(end) or end <= start:
                end = start + 0.5
            color = entry.get("color")
            if color is None:
                color = (0.9, 0.6, 0.3, 0.3)
            line_color = entry.get("line_color", color)
            mesh = self._annotation_rectangles[idx]
            vertices = np.array(
                [
                    (start, y0, 0.0),
                    (start, y1, 0.0),
                    (end, y0, 0.0),
                    (end, y1, 0.0),
                ],
                dtype=np.float32,
            )
            mesh.set_data(vertices=vertices, faces=faces, color=color)
            mesh.visible = True

            marker = self._annotation_marker_lines[idx]
            marker_pos = np.array(
                [
                    (start, y0),
                    (start, y1),
                ],
                dtype=np.float32,
            )
            marker.set_data(pos=marker_pos, color=line_color, width=1.2)
            marker.visible = True

        for mesh in self._annotation_rectangles[len(entries) :]:
            mesh.visible = False
        for line in self._annotation_marker_lines[len(entries) :]:
            line.visible = False

        if self._hypnogram_view is not None:
            self._hypnogram_view.camera.set_range(
                x=(view_start, view_end),
                y=self._hypnogram_y_range,
            )
        self._update_hypnogram_region_visual(view_start, view_end)
        self._update_annotation_lane_visibility()

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
        if self._hypnogram_view is not None:
            self._hypnogram_view.camera.set_range(
                x=x_range,
                y=self._hypnogram_y_range,
            )
        self._update_hypnogram_region_visual(*x_range)
        if self._x_axis is not None:
            self._refresh_axis_link()
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
        if self._gutter_axis_placeholder is not None:
            self._grid.remove_widget(self._gutter_axis_placeholder)
            self._gutter_axis_placeholder = None

        while len(self._views) < count:
            row = len(self._views)
            gutter = self._grid.add_view(row=row, col=0, camera="panzoom")
            gutter.camera.interactive = False
            gutter.border_color = None
            gutter.padding = 0.0
            gutter.bgcolor = None
            with contextlib.suppress(Exception):
                gutter.width_min = 72
            with contextlib.suppress(Exception):
                gutter.width_max = 220

            label = scene.visuals.Text(
                text="",
                anchor_x="left",
                anchor_y="top",
                color="white",
                font_size=12,
            )
            label.transform = transforms.STTransform(translate=(8, 8))
            gutter.add(label)

            view = self._grid.add_view(row=row, col=1, camera="panzoom")
            view.camera.interactive = False
            view.border_color = None
            view.padding = 0.0

            grid = scene.visuals.GridLines(color=self._grid_line_color, scale=(1.0, 1.0))
            grid.set_gl_state(depth_test=False)
            view.add(grid)
            line = scene.visuals.Line(method="gl")
            line.set_data(pos=self._empty_vertices())
            line.visible = False
            view.add(line)
            view.add(self._hover_line)
            view.add(self._hover_marker)

            self._gutter_views.append(gutter)
            self._views.append(view)
            self._lines.append(line)
            self._label_nodes.append(label)
            self._grid_lines.append(grid)
            self._channel_states.append(_ChannelState(t=np.array([]), x=np.array([])))
            self._channel_visible.append(True)
            self._channel_colors.append(Color("#66aaff"))
            self._view_y_ranges.append((-1.0, 1.0))
            self._label_hidden.append(False)

        self._install_axis_widget()
        self.set_view(self._view_start, self._view_duration)

    def _install_axis_widget(self) -> None:
        if self._x_axis is not None:
            self._grid.remove_widget(self._x_axis)
            self._x_axis = None
        if self._gutter_axis_placeholder is not None:
            self._grid.remove_widget(self._gutter_axis_placeholder)
            self._gutter_axis_placeholder = None

        total_rows = len(self._views)
        if self._hypnogram_view is not None:
            total_rows += 1

        self._x_axis = scene.AxisWidget(orientation="bottom")
        self._x_axis.axis.scale_type = "linear"
        self._x_axis.height_max = 32
        self._x_axis.height_min = 28
        self._grid.add_widget(self._x_axis, row=total_rows, col=1)
        with contextlib.suppress(Exception):
            placeholder = scene.widgets.Widget()
            placeholder.height_max = self._x_axis.height_max
            placeholder.height_min = self._x_axis.height_min
            placeholder.border_color = None
            placeholder.bgcolor = None
            self._grid.add_widget(placeholder, row=total_rows, col=0)
            self._gutter_axis_placeholder = placeholder
        self._apply_axis_formatter()
        self._update_axis_theme()
        self._refresh_axis_link()

    def _refresh_axis_link(self) -> None:
        if self._x_axis is None:
            return
        target = None
        if (self._hypnogram_view is not None) and (
            self._hypnogram_visible or self._annotation_events_visible
        ):
            target = self._hypnogram_view
        if target is None and self._views:
            target = self._views[0]
        if target is None:
            return
        with contextlib.suppress(Exception):
            self._x_axis.link_view(target)

    def _ensure_hypnogram_row(self) -> None:
        if self._hypnogram_view is not None:
            return

        if self._x_axis is not None:
            self._grid.remove_widget(self._x_axis)
            self._x_axis = None
        if self._gutter_axis_placeholder is not None:
            self._grid.remove_widget(self._gutter_axis_placeholder)
            self._gutter_axis_placeholder = None

        row = len(self._views)
        label_view = self._grid.add_view(row=row, col=0, camera="panzoom")
        label_view.camera.interactive = False
        label_view.border_color = None
        label_view.padding = 0.0
        label_view.bgcolor = None
        with contextlib.suppress(Exception):
            label_view.height_min = 48
            label_view.height_max = 96

        label_text = scene.visuals.Text(
            text="Hypnogram",
            anchor_x="left",
            anchor_y="top",
            color=self._label_active_color,
            font_size=12,
        )
        label_text.transform = transforms.STTransform(translate=(8, 8))
        label_view.add(label_text)

        view = self._grid.add_view(row=row, col=1, camera="panzoom")
        view.camera.interactive = False
        view.border_color = None
        view.padding = 0.0
        with contextlib.suppress(Exception):
            view.height_min = 64
            view.height_max = 140
        view.bgcolor = self._background_color

        outline = scene.visuals.Line(method="gl")
        outline.set_data(pos=self._empty_vertices())
        outline.visible = False
        outline.set_gl_state(depth_test=False, blend=True)
        view.add(outline)

        region_mesh = scene.visuals.Mesh(color=(0.4, 0.6, 0.9, 0.18))
        region_mesh.visible = False
        region_mesh.set_gl_state(depth_test=False, blend=True)
        view.add(region_mesh)

        region_border = scene.visuals.Line(method="gl")
        region_border.visible = False
        region_border.set_gl_state(depth_test=False, blend=True)
        view.add(region_border)

        self._hypnogram_label_view = label_view
        self._hypnogram_label_text = label_text
        self._hypnogram_view = view
        self._hypnogram_outline = outline
        self._hypnogram_region_mesh = region_mesh
        self._hypnogram_region_border = region_border
        self._hypnogram_fill_meshes = {}
        self._annotation_rectangles = []
        self._annotation_marker_lines = []
        self._hypnogram_visible = False
        self._annotation_events_visible = False
        self._hypnogram_y_range = (-0.6, 0.6)

        self._install_axis_widget()
        self.set_view(self._view_start, self._view_duration)

    def _update_annotation_lane_visibility(self) -> None:
        active = self._hypnogram_visible or self._annotation_events_visible
        if self._hypnogram_view is not None:
            self._hypnogram_view.visible = active
        if self._hypnogram_label_view is not None:
            self._hypnogram_label_view.visible = active
        if not active:
            if self._hypnogram_outline is not None:
                self._hypnogram_outline.visible = False
            if self._hypnogram_region_mesh is not None:
                self._hypnogram_region_mesh.visible = False
            if self._hypnogram_region_border is not None:
                self._hypnogram_region_border.visible = False
            for mesh in self._hypnogram_fill_meshes.values():
                mesh.visible = False
            for mesh in self._annotation_rectangles:
                mesh.visible = False
            for line in self._annotation_marker_lines:
                line.visible = False
        self._refresh_axis_link()

    def _stage_fill_mesh_data(
        self, x_vals: np.ndarray, top_vals: np.ndarray, fill_level: float
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if x_vals.size == 0 or top_vals.size == 0:
            return None, None
        valid = np.isfinite(x_vals) & np.isfinite(top_vals)
        if not np.any(valid):
            return None, None

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        start_idx: int | None = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start_idx is None:
                start_idx = idx
            elif not is_valid and start_idx is not None:
                if idx - start_idx >= 2:
                    xs = x_vals[start_idx:idx]
                    ys = top_vals[start_idx:idx]
                    base = len(vertices)
                    start_x = float(xs[0])
                    end_x = float(xs[-1])
                    top_val = float(ys[0])
                    vertices.extend(
                        [
                            (start_x, fill_level, 0.0),
                            (start_x, top_val, 0.0),
                            (end_x, fill_level, 0.0),
                            (end_x, top_val, 0.0),
                        ]
                    )
                    faces.append((base, base + 1, base + 2))
                    faces.append((base + 2, base + 1, base + 3))
                start_idx = None
        if start_idx is not None and x_vals.size - start_idx >= 2:
            xs = x_vals[start_idx:]
            ys = top_vals[start_idx:]
            base = len(vertices)
            start_x = float(xs[0])
            end_x = float(xs[-1])
            top_val = float(ys[0])
            vertices.extend(
                [
                    (start_x, fill_level, 0.0),
                    (start_x, top_val, 0.0),
                    (end_x, fill_level, 0.0),
                    (end_x, top_val, 0.0),
                ]
            )
            faces.append((base, base + 1, base + 2))
            faces.append((base + 2, base + 1, base + 3))

        if not faces:
            return None, None
        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)

    def _update_hypnogram_region_visual(self, view_start: float, view_end: float) -> None:
        if self._hypnogram_region_mesh is None or self._hypnogram_region_border is None:
            return
        if not (self._hypnogram_visible or self._annotation_events_visible):
            self._hypnogram_region_mesh.visible = False
            self._hypnogram_region_border.visible = False
            return
        width = max(0.0, float(view_end) - float(view_start))
        height = max(0.0, self._hypnogram_y_range[1] - self._hypnogram_y_range[0])
        if width <= 0.0 or height <= 0.0:
            self._hypnogram_region_mesh.visible = False
            self._hypnogram_region_border.visible = False
            return
        x0 = float(view_start)
        x1 = x0 + width
        y0, y1 = self._hypnogram_y_range
        vertices = np.array(
            [
                (x0, y0, 0.0),
                (x0, y1, 0.0),
                (x1, y0, 0.0),
                (x1, y1, 0.0),
            ],
            dtype=np.float32,
        )
        faces = self._annotation_faces
        self._hypnogram_region_mesh.set_data(
            vertices=vertices,
            faces=faces,
            color=(0.4, 0.6, 0.9, 0.18),
        )
        self._hypnogram_region_mesh.visible = True
        border_pos = np.array(
            [
                (x0, y0),
                (x0, y1),
                (x1, y1),
                (x1, y0),
                (x0, y0),
            ],
            dtype=np.float32,
        )
        self._hypnogram_region_border.set_data(
            pos=border_pos,
            color=(0.6, 0.7, 0.9, 0.4),
            width=1.0,
        )
        self._hypnogram_region_border.visible = True

    def _ensure_annotation_mesh_pool(self, count: int) -> None:
        self._ensure_hypnogram_row()
        while len(self._annotation_rectangles) < count:
            mesh = scene.visuals.Mesh()
            mesh.visible = False
            mesh.set_gl_state(depth_test=False, blend=True)
            if self._hypnogram_view is not None:
                self._hypnogram_view.add(mesh)
            self._annotation_rectangles.append(mesh)

    def _ensure_annotation_marker_pool(self, count: int) -> None:
        self._ensure_hypnogram_row()
        while len(self._annotation_marker_lines) < count:
            line = scene.visuals.Line(method="gl")
            line.visible = False
            line.set_gl_state(depth_test=False, blend=True)
            if self._hypnogram_view is not None:
                self._hypnogram_view.add(line)
            self._annotation_marker_lines.append(line)

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
        self._axis_formatter.set_timebase(timebase)
        self._request_axis_update()

    def set_time_mode(self, mode: str) -> None:
        mode_lower = str(mode).lower()
        self._time_mode = "absolute" if mode_lower == "absolute" else "relative"
        self._axis_formatter.set_mode(self._time_mode)
        self._request_axis_update()

    def _apply_axis_formatter(self) -> None:
        if self._x_axis is None:
            return
        axis = getattr(self._x_axis, "axis", None)
        if axis is None:
            return
        axis.tick_formatter = self._axis_formatter

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
