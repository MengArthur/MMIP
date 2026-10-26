"""Ensemble-disagreement uncertainty (routevision/uncertainty.py), on data whose answer is known.

    python -m pytest tests/test_uncertainty.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routevision import uncertainty as U


def test_disagreement_is_member_std():
    s = {m: np.full((8, 8), 0.4, np.float32) for m in U.MEMBERS}
    assert np.allclose(U.disagreement(s), 0) and U.disagreement(s).shape == (8, 8)
    s[U.MEMBERS[0]] = np.full((8, 8), 1.0, np.float32)
    assert np.allclose(U.disagreement(s), np.std([1.0, 0.4, 0.4]), atol=1e-6)


def test_error_capture_and_risk_coverage():
    err = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    perfect = err.astype(float) + np.linspace(0, 0.1, 10)       # errors are the most uncertain patches
    assert U.captured(err, perfect, 0.2) == 1.0 and U.captured(err, -perfect, 0.2) == 0.0
    rc = U.risk_coverage(err, perfect)
    assert rc[0]['error'] == 0 and rc[-1]['error'] == pytest.approx(0.2)
