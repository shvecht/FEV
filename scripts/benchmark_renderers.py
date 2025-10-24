#!/usr/bin/env python3
"""Benchmark CPU vs GPU redraw performance for the EDF viewer.

The harness drives both render backends through a representative redraw loop
that mimics the production overlays (hypnogram lane + annotation spans).  Each
test batch reports the average frame time together with the peak Python memory
usage observed while the renderer processed the frame payload.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from ui.gpu_canvas import VispyChannelCanvas


@dataclass
class BenchmarkResult:
    backend: str
    channels: int
    samples_per_channel: int
    frame_ms: float
    peak_bytes: int
    extra: dict[str, object]


def _format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units[1:]:
        if value < 1024.0:
            break
        value /= 1024.0
    else:
        unit = units[-1]
    return f"{value:.1f} {unit}"


def _build_synthetic_data(channels: int, duration_s: float, sample_rate: float) -> tuple[np.ndarray, list[np.ndarray]]:
    n_samples = max(1, int(duration_s * sample_rate))
    t = np.linspace(0.0, duration_s, n_samples, dtype=np.float64)
    data: list[np.ndarray] = []
    for idx in range(channels):
        freq = 1.0 + 0.35 * idx
        phase = idx * 0.4
        signal = np.sin(2 * math.pi * freq * t + phase)
        signal += 0.15 * np.sin(2 * math.pi * 0.1 * t + phase * 0.5)
        signal += 0.02 * np.random.default_rng(idx).standard_normal(t.size)
        data.append(signal.astype(np.float32, copy=False))
    return t, data


def _build_stage_payload(duration_s: float) -> dict[str, object]:
    epoch = 30.0
    epochs = max(1, int(duration_s / epoch))
    stage_cycle = ["W", "N1", "N2", "N3", "REM"]
    step_x: list[float] = []
    step_y: list[float] = []
    label_data: dict[str, dict[str, object]] = {}
    level_map = {label: idx for idx, label in enumerate(stage_cycle)}
    prev_end = 0.0
    for idx in range(epochs):
        label = stage_cycle[idx % len(stage_cycle)]
        start = idx * epoch
        end = start + epoch
        if start - prev_end > 1e-6:
            step_x.extend([float("nan"), float("nan")])
            step_y.extend([float("nan"), float("nan")])
        step_x.extend([start, end])
        level = level_map[label]
        step_y.extend([level, level])
        payload = label_data.setdefault(label, {"x": [], "top": [], "fill": level - 0.45})
        xs: list[float] = payload["x"]  # type: ignore[assignment]
        ys: list[float] = payload["top"]  # type: ignore[assignment]
        if xs and not math.isnan(xs[-1]) and abs(xs[-1] - start) > 1e-6:
            xs.append(float("nan"))
            ys.append(float("nan"))
        xs.extend([start, end])
        top = level + 0.45
        ys.extend([top, top])
        prev_end = end

    for payload in label_data.values():
        payload["x"] = np.asarray(payload["x"], dtype=float)
        payload["top"] = np.asarray(payload["top"], dtype=float)

    colors = {label: QtGui.QColor(pg.intColor(idx, hues=len(stage_cycle))) for idx, label in enumerate(stage_cycle)}

    return {
        "step_x": np.asarray(step_x, dtype=float),
        "step_y": np.asarray(step_y, dtype=float),
        "label_data": label_data,
        "labels": stage_cycle,
        "level_map": level_map,
        "colors": colors,
        "max_level": max(level_map.values()) if level_map else -1,
    }


def _build_annotation_events(duration_s: float) -> list[dict[str, object]]:
    rng = np.random.default_rng(42)
    events: list[dict[str, object]] = []
    span = max(5.0, duration_s / 20.0)
    for idx in range(6):
        start = float(idx * span * 0.75)
        if start >= duration_s:
            break
        length = float(span * 0.4 + rng.uniform(0.0, span * 0.3))
        end = min(duration_s, start + length)
        base_color = pg.intColor(idx, hues=8)
        events.append(
            {
                "start": start,
                "end": end,
                "label": f"Event {idx + 1}",
                "color": (
                    base_color.red(),
                    base_color.green(),
                    base_color.blue(),
                    int(0.35 * 255),
                ),
                "line_color": (
                    base_color.red(),
                    base_color.green(),
                    base_color.blue(),
                    255,
                ),
            }
        )
    return events


def _benchmark_cpu(
    app: QtWidgets.QApplication,
    t: np.ndarray,
    payloads: list[np.ndarray],
    *,
    window_s: float,
    iterations: int,
) -> BenchmarkResult:
    widget = pg.GraphicsLayoutWidget()
    widget.setWindowTitle("CPU Benchmark")
    widget.resize(800, 600)
    curves: list[pg.PlotDataItem] = []
    for idx in range(len(payloads)):
        plot = widget.addPlot(row=idx, col=0)
        plot.showAxis("bottom", show=False)
        plot.showAxis("left", show=False)
        curve = plot.plot([], [], pen=pg.mkPen(width=1.0))
        curves.append(curve)
    stage_payload = _build_stage_payload(window_s * 3.0)
    events = _build_annotation_events(window_s * 3.0)
    stage_label = pg.LabelItem(text="Stage")
    widget.addItem(stage_label, row=len(payloads), col=0)
    stage_plot = widget.addPlot(row=len(payloads), col=1)
    stage_plot.setXRange(0.0, window_s, padding=0)
    stage_plot.setYRange(-0.6, float(stage_payload.get("max_level", 0)) + 0.6, padding=0)
    stage_plot.showAxis("left", show=False)
    stage_plot.showAxis("bottom", show=False)
    outline = stage_plot.plot([], [], pen=pg.mkPen("#4d4f59", width=1.0))
    fill_items: dict[str, pg.PlotDataItem] = {}
    for label in stage_payload.get("labels", []):
        item = stage_plot.plot([], [], pen=None, brush=pg.mkBrush(pg.intColor(len(fill_items), hues=8)))
        item.setFillLevel(float(stage_payload["label_data"][label]["fill"]))  # type: ignore[index]
        item.hide()
        fill_items[label] = item
    event_regions: list[pg.LinearRegionItem] = []
    for entry in events:
        region = pg.LinearRegionItem(values=(entry["start"], entry["end"]))  # type: ignore[index]
        region.setZValue(-10)
        region.setBrush(pg.mkBrush(entry["color"]))  # type: ignore[index]
        region.setMovable(False)
        stage_plot.addItem(region)
        event_regions.append(region)

    widget.show()
    app.processEvents()

    series = [(t, payloads[idx]) for idx in range(len(payloads))]
    timings: list[float] = []
    gc.collect()
    statistics_mem: list[int] = []
    import tracemalloc

    tracemalloc.start()
    try:
        for _ in range(iterations):
            start = time.perf_counter()
            for curve, (t_arr, x_arr) in zip(curves, series):
                curve.setData(t_arr, x_arr)
            outline.setData(stage_payload["step_x"], stage_payload["step_y"])  # type: ignore[index]
            for label, item in fill_items.items():
                payload = stage_payload["label_data"].get(label)  # type: ignore[index]
                if not payload:
                    item.hide()
                    continue
                item.setData(payload["x"], payload["top"])  # type: ignore[index]
                item.show()
            for region, entry in zip(event_regions, events):
                region.setRegion((entry["start"], entry["end"]))  # type: ignore[index]
            app.processEvents()
            timings.append(time.perf_counter() - start)
            statistics_mem.append(tracemalloc.get_traced_memory()[1])
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    widget.close()
    app.processEvents()

    peak_bytes = peak if peak else max(statistics_mem or [0])
    vertex_count = len(payloads) * len(t)
    return BenchmarkResult(
        backend="cpu",
        channels=len(payloads),
        samples_per_channel=len(t),
        frame_ms=statistics.mean(timings) * 1000.0,
        peak_bytes=peak_bytes,
        extra={
            "vertices": vertex_count,
            "events": len(events),
            "stages": len(stage_payload.get("labels", [])),
        },
    )


def _benchmark_gpu(
    app: QtWidgets.QApplication,
    t: np.ndarray,
    payloads: list[np.ndarray],
    *,
    window_s: float,
    iterations: int,
) -> tuple[BenchmarkResult | None, str | None]:
    probe = VispyChannelCanvas.capability_probe()
    if not probe.available:
        return None, probe.reason or "VisPy unavailable"

    try:
        canvas = VispyChannelCanvas()
    except Exception as exc:  # pragma: no cover - optional GPU
        return None, str(exc)

    infos = [SimpleNamespace(name=f"Ch {idx + 1}", unit="µV") for idx in range(len(payloads))]
    canvas.configure_channels(infos=infos, hidden_indices=set())
    canvas.set_view(0.0, window_s)
    canvas.set_hover_enabled(False)

    series = [(t, payloads[idx]) for idx in range(len(payloads))]
    vertices = [VispyChannelCanvas._prepare_vertices(t, payloads[idx]) for idx in range(len(payloads))]
    stage_payload = _build_stage_payload(window_s * 3.0)
    events = _build_annotation_events(window_s * 3.0)
    timings: list[float] = []
    gc.collect()
    import tracemalloc

    tracemalloc.start()
    try:
        for _ in range(iterations):
            start = time.perf_counter()
            canvas.apply_tile_data(0, series, vertices, set(), final=True)
            canvas.update_hypnogram(stage_payload, visible=True, view_start=0.0, view_end=window_s)
            canvas.update_annotations(events, view_start=0.0, view_end=window_s)
            app.processEvents()
            timings.append(time.perf_counter() - start)
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    canvas.close()
    app.processEvents()

    vertex_count = int(sum(len(v) for v in vertices))
    peak_bytes = int(peak or current)
    return (
        BenchmarkResult(
            backend=probe.renderer or "VisPy",
            channels=len(payloads),
            samples_per_channel=len(t),
            frame_ms=statistics.mean(timings) * 1000.0,
            peak_bytes=peak_bytes,
            extra={
                "vertices": vertex_count,
                "events": len(events),
                "stages": len(stage_payload.get("labels", [])),
            },
        ),
        probe.renderer or "VisPy",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels",
        type=int,
        action="append",
        help="Number of synthetic channels (repeatable; default: 6, 12, 24)",
    )
    parser.add_argument("--duration", type=float, default=180.0, help="Signal duration in seconds")
    parser.add_argument("--sample-rate", type=float, default=256.0, help="Samples per second")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Redraw iterations per backend"
    )
    parser.add_argument(
        "--window", type=float, default=60.0, help="Window span (seconds) for renderer view"
    )
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU benchmark")
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    channel_sets = args.channels or [6, 12, 24]

    for channel_count in channel_sets:
        t_arr, payloads = _build_synthetic_data(channel_count, args.duration, args.sample_rate)
        cpu_result = _benchmark_cpu(
            app,
            t_arr,
            payloads,
            window_s=args.window,
            iterations=args.iterations,
        )
        vertex_count = cpu_result.extra.get("vertices", channel_count * len(t_arr))
        print(
            "== Channels: {channels} (≈{vertices:,} vertices) ==".format(
                channels=channel_count,
                vertices=int(vertex_count),
            )
        )
        print(
            "CPU renderer: {frame:.2f} ms avg — peak {mem}".format(
                frame=cpu_result.frame_ms,
                mem=_format_bytes(cpu_result.peak_bytes),
            )
        )

        if args.skip_gpu:
            continue

        gpu_result, gpu_label = _benchmark_gpu(
            app,
            t_arr,
            payloads,
            window_s=args.window,
            iterations=args.iterations,
        )
        if gpu_result is None:
            print(f"GPU renderer skipped: {gpu_label or 'unavailable'}")
        else:
            print(
                "GPU renderer ({label}): {frame:.2f} ms avg — peak {mem}".format(
                    label=gpu_result.backend,
                    frame=gpu_result.frame_ms,
                    mem=_format_bytes(gpu_result.peak_bytes),
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
