import numpy as np
from core.overscan import SignalChunk, slice_and_decimate, select_lod_duration


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


def test_slice_and_decimate_uses_pre_binned_chunk():
    t = np.linspace(0.0, 100.0, 50)
    x = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    chunk = SignalChunk(t, x, lod_duration_s=10.0)
    # Passing a SignalChunk should bypass further decimation
    sub_t, sub_x = slice_and_decimate(chunk, None, 5.0, 45.0, pixels=10)
    assert sub_t[0] >= 5.0 - 1e-6
    assert sub_t[-1] <= 45.0 + 1e-6
    assert sub_t.size == sub_x.size


def test_select_lod_duration_prefers_coarsest_available():
    durations = [1.0, 5.0, 30.0]
    assert select_lod_duration(180.0, durations, ratio=2.0) == 30.0
    assert select_lod_duration(12.0, durations, ratio=2.0) == 5.0
    assert select_lod_duration(1.5, durations, ratio=2.0) is None
