from pathlib import Path

import numpy as np
import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.hover_utils import _sample_at_time


class _CurveStub:
    def __init__(self, x, y):
        self.xData = np.asarray(x)
        self.yData = np.asarray(y)


def test_sample_at_time_returns_nearest_pair():
    curve = _CurveStub([0.0, 0.5, 1.0, 1.5], [10.0, 12.0, 14.0, 18.0])

    time_val, sample_val = _sample_at_time(curve, 0.6)
    assert time_val == pytest.approx(0.5)
    assert sample_val == pytest.approx(12.0)

    time_val, sample_val = _sample_at_time(curve, 1.6)
    assert time_val == pytest.approx(1.5)
    assert sample_val == pytest.approx(18.0)
