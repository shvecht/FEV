import numpy as np
from core.overscan import slice_and_decimate


def test_slice_and_decimate_basic():
    t = np.linspace(0.0, 10.0, 1001)
    x = np.sin(t).astype(np.float32)
    sub_t, sub_x = slice_and_decimate(t, x, 2.0, 4.0, pixels=100)
    assert sub_t[0] >= 2.0 - 1e-6
    assert sub_t[-1] <= 4.0 + 1e-6
    assert sub_t.size == sub_x.size


def test_slice_and_decimate_empty_when_no_overlap():
    t = np.linspace(0.0, 1.0, 100)
    x = np.cos(t).astype(np.float32)
    sub_t, sub_x = slice_and_decimate(t, x, 2.0, 3.0, pixels=50)
    assert sub_t.size == 0
    assert sub_x.size == 0


def test_slice_and_decimate_respects_pixel_budget():
    t = np.linspace(0.0, 1.0, 10000)
    x = np.sin(50 * t).astype(np.float32)
    sub_t, sub_x = slice_and_decimate(t, x, 0.2, 0.8, pixels=100)
    assert sub_t.size <= 200
    assert sub_t.size == sub_x.size
