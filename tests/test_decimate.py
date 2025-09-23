import numpy as np
import pytest

from core.decimate import min_max_bins


def test_returns_original_when_small():
    t = np.linspace(0, 1, 10)
    x = np.sin(t)
    out_t, out_x = min_max_bins(t, x, pixels=10)
    assert out_t is t
    assert out_x is x


def test_min_max_envelope_reduces_samples():
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 5 * t)
    out_t, out_x = min_max_bins(t, x, pixels=100)
    assert out_t.size <= 200
    assert np.all(np.diff(out_t) >= 0)
    # Envelope should stay within bounds of original signal
    assert out_x.max() <= 1.0 + 1e-2
    assert out_x.min() >= -1.0 - 1e-2


def test_handles_plateaus_and_monotone():
    t = np.arange(0, 500)
    x = np.concatenate([np.full(250, 1.0), np.full(250, -1.0)])
    out_t, out_x = min_max_bins(t, x, pixels=50)
    assert out_t.size <= 100
    assert set(np.unique(out_x)).issubset({-1.0, 1.0})


def test_pixels_must_be_positive():
    t = np.arange(5)
    x = np.arange(5)
    with pytest.raises(ValueError):
        min_max_bins(t, x, 0)


def test_time_and_signal_length_mismatch():
    with pytest.raises(ValueError):
        min_max_bins(np.arange(5), np.arange(4), 10)
