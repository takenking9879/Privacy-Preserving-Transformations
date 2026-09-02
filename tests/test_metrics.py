"""Unit tests for metrics helpers."""

import numpy as np

from src.metrics import pairwise_distance_corr, per_feature_r2, reconstruction_report, retention


def test_retention_and_recon():
    assert abs(retention(0.9, 1.0) - 0.9) < 1e-12
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 5))
    rep = reconstruction_report(X, X)
    assert abs(rep["global_r2"] - 1.0) < 1e-8
    assert abs(rep["max_feature_r2"] - 1.0) < 1e-8
    r2j = per_feature_r2(X, X + 50)
    assert np.all(r2j < 0)
    corr = pairwise_distance_corr(X, X, max_n=80, seed=0)
    assert corr > 0.99
