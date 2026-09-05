from src.metrics import reconstruction_report, retention, safe_r2
import numpy as np


def test_safe_r2_perfect():
    y = np.arange(10, dtype=float)
    assert abs(safe_r2(y, y) - 1.0) < 1e-12


def test_reconstruction_report_identity():
    X = np.random.RandomState(0).normal(size=(20, 3))
    r = reconstruction_report(X, X)
    assert r["global_r2"] > 0.99
    assert r["max_feature_r2"] > 0.99


def test_retention_ratio():
    assert abs(retention(0.9, 1.0) - 0.9) < 1e-12
