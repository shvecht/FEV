"""Ensure GPU and CPU time axes emit identical tick labels."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

try:
    from PySide6 import QtWidgets
except Exception as exc:  # pragma: no cover - optional dependency may be missing
    pytest.skip(f"PySide6 QtWidgets unavailable: {exc}", allow_module_level=True)

from core.timebase import Timebase
from ui.time_axis import TimeAxis
from ui.time_axis_formatter import TimeTickFormatter


@pytest.fixture(scope="module")
def qapp():  # pragma: no cover - Qt global singleton
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.mark.parametrize(
    "mode, values",
    [
        ("relative", [0.0, 12.3, 59.6, 61.2, 3599.9, 3661.01]),
        ("absolute", [0.0, 30.0, 90.2, 3599.9, 43210.4]),
    ],
)
def test_tick_labels_match_between_backends(qapp, mode, values):
    start = datetime(2024, 1, 1, 22, 30, tzinfo=timezone.utc)
    timebase = Timebase(start, duration_s=12 * 3600)

    axis = TimeAxis(orientation="bottom", timebase=timebase, mode=mode)
    formatter = TimeTickFormatter(timebase=timebase, mode=mode)

    axis_labels = axis.tickStrings(values, scale=1.0, spacing=1.0)
    formatter_labels = formatter.format_ticks(values)

    assert axis_labels == formatter_labels


def test_tick_formatter_handles_nan_values():
    formatter = TimeTickFormatter()
    labels = formatter.format_ticks([np.nan, np.inf, -np.inf, None])
    assert labels == ["", "", "", ""]
