import numpy as np
from datetime import datetime, timedelta

from core.timebase import Timebase


def make_timebase():
    start = datetime(2024, 1, 2, 3, 4, 5)
    return Timebase(start, duration_s=180.0)


def test_sec_to_idx_basic():
    tb = make_timebase()
    s0, n = Timebase.sec_to_idx(0.5, 2.8, fs=100.0)
    assert s0 == 50
    assert n == 230


def test_sec_to_idx_clamps_negative_start():
    s0, n = Timebase.sec_to_idx(-1.0, 1.0, fs=50.0)
    assert s0 == 0
    assert n == 100


def test_idx_to_time_scalar_and_array():
    fs = 200.0
    assert Timebase.idx_to_time(100, fs) == 0.5
    arr = np.array([0, 200, 400])
    out = Timebase.idx_to_time(arr, fs)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.0]))


def test_time_vector_length_and_values():
    vec = Timebase.time_vector(100, 5, fs=10.0)
    np.testing.assert_allclose(vec, np.array([10.0, 10.1, 10.2, 10.3, 10.4]))


def test_time_vector_empty_when_n_zero():
    vec = Timebase.time_vector(0, 0, fs=100.0)
    assert vec.size == 0


def test_clamp_window_limits():
    tb = make_timebase()
    assert tb.clamp_window(-5.0, 10.0) == (0.0, 10.0)
    assert tb.clamp_window(175.0, 190.0) == (175.0, 180.0)
    assert tb.clamp_window(175.0, 170.0) == (175.0, 175.0)


def test_to_datetime_scalar_and_array():
    tb = make_timebase()
    dt = tb.to_datetime(12.5)
    assert dt == datetime(2024, 1, 2, 3, 4, 17, 500000)

    arr = tb.to_datetime(np.array([0.0, 1.0, 2.5]))
    expected = np.array([
        '2024-01-02T03:04:05.000000000',
        '2024-01-02T03:04:06.000000000',
        '2024-01-02T03:04:07.500000000',
    ], dtype='datetime64[ns]')
    np.testing.assert_array_equal(arr, expected)


def test_to_seconds_inverse():
    tb = make_timebase()
    dt = tb.start_dt + timedelta(seconds=42.25)
    assert tb.to_seconds(dt) == 42.25
