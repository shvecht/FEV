#!/usr/bin/env python3
"""Benchmark CPU vs GPU redraw performance for the EDF viewer."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from ui.gpu_canvas import VispyChannelCanvas


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


def _benchmark_cpu(
    app: QtWidgets.QApplication,
    t: np.ndarray,
    payloads: list[np.ndarray],
    *,
    iterations: int,
) -> float:
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
    widget.show()
    app.processEvents()

    series = [(t, payloads[idx]) for idx in range(len(payloads))]
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        for curve, (t_arr, x_arr) in zip(curves, series):
            curve.setData(t_arr, x_arr)
        app.processEvents()
        timings.append(time.perf_counter() - start)

    widget.close()
    app.processEvents()
    return statistics.mean(timings)


def _benchmark_gpu(
    app: QtWidgets.QApplication,
    t: np.ndarray,
    payloads: list[np.ndarray],
    *,
    window_s: float,
    iterations: int,
) -> tuple[float | None, str | None]:
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
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        canvas.apply_tile_data(0, series, vertices, set(), final=True)
        app.processEvents()
        timings.append(time.perf_counter() - start)

    canvas.close()
    app.processEvents()
    return statistics.mean(timings), probe.renderer or "VisPy"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=8, help="Number of synthetic channels")
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

    t_arr, payloads = _build_synthetic_data(args.channels, args.duration, args.sample_rate)
    cpu_time = _benchmark_cpu(app, t_arr, payloads, iterations=args.iterations)
    vertex_count = args.channels * len(t_arr)
    print(
        f"CPU renderer: {cpu_time * 1000:.2f} ms per redraw "
        f"({args.channels} channels × {len(t_arr):,} samples)"
    )

    if not args.skip_gpu:
        gpu_time, gpu_label = _benchmark_gpu(
            app, t_arr, payloads, window_s=args.window, iterations=args.iterations
        )
        if gpu_time is None:
            print(f"GPU renderer skipped: {gpu_label or 'unavailable'}")
        else:
            label = gpu_label or "VisPy"
            print(
                f"GPU renderer ({label}): {gpu_time * 1000:.2f} ms per redraw "
                f"for ~{vertex_count:,} vertices"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
