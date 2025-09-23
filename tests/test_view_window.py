import pytest

from core.view_window import WindowLimits, clamp_window, pan_window, zoom_window


LIMITS = WindowLimits(duration_min=0.5, duration_max=60.0)
TOTAL = 120.0


def test_clamp_window_bounds():
    start, duration = clamp_window(-10.0, 5.0, total=TOTAL, limits=LIMITS)
    assert start == 0.0
    assert duration == 5.0

    start, duration = clamp_window(200.0, 5.0, total=TOTAL, limits=LIMITS)
    assert start == TOTAL - 5.0

    # shorter recording than min duration
    start, duration = clamp_window(0.0, 5.0, total=0.2, limits=LIMITS)
    assert start == 0.0
    assert duration == 0.2


def test_pan_window():
    start, duration = pan_window(10.0, 5.0, delta=4.0, total=TOTAL, limits=LIMITS)
    assert start == 14.0
    assert duration == 5.0

    start, _ = pan_window(TOTAL - 5.0, 5.0, delta=10.0, total=TOTAL, limits=LIMITS)
    assert start == TOTAL - 5.0


def test_zoom_window_in_and_out():
    # zoom in around center (factor < 1)
    start, duration = zoom_window(10.0, 10.0, factor=0.5, anchor=15.0, total=TOTAL, limits=LIMITS)
    assert pytest.approx(duration, rel=1e-6) == 5.0
    assert pytest.approx(start + duration / 2, rel=1e-6) == 15.0

    # zoom out capped by max
    start, duration = zoom_window(0.0, 30.0, factor=10.0, anchor=5.0, total=TOTAL, limits=LIMITS)
    assert duration == LIMITS.duration_max
    assert start == 0.0

    # anchor near end
    start, duration = zoom_window(100.0, 10.0, factor=0.5, anchor=109.0, total=TOTAL, limits=LIMITS)
    assert start + duration <= TOTAL + 1e-6


def test_zoom_window_invalid_factor():
    with pytest.raises(ValueError):
        zoom_window(0.0, 10.0, factor=0.0, anchor=0.0, total=TOTAL, limits=LIMITS)
