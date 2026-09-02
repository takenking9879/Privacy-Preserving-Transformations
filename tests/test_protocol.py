"""Protocol sanity checks (no test-set leakage, frozen model, quantization)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import MODEL_HYPERPARAMS, TransformSpec
from src.datasets import make_banking_synthetic, split_indices
from src.models import make_model
from src.transforms import PrivacyTransform


def test_model_hyperparameters_are_frozen():
    m0 = make_model(0)
    m1 = make_model(1)
    assert isinstance(m0, HistGradientBoostingRegressor)
    for key, val in MODEL_HYPERPARAMS.items():
        assert getattr(m0, key) == val
        assert getattr(m1, key) == val
    assert m0.random_state == 0
    assert m1.random_state == 1


def test_split_reuse_and_disjoint():
    idx0 = split_indices(1000, seed=0)
    idx1 = split_indices(1000, seed=0)
    for k in ("train", "test", "aux"):
        np.testing.assert_array_equal(idx0[k], idx1[k])
    t = set(idx0["train"]) | set(idx0["test"]) | set(idx0["aux"])
    assert len(t) == 1000
    assert set(idx0["train"]).isdisjoint(idx0["test"])
    assert set(idx0["train"]).isdisjoint(idx0["aux"])
    assert set(idx0["test"]).isdisjoint(idx0["aux"])


def test_transform_fit_on_train_only_does_not_require_test():
    ds = make_banking_synthetic(n=400, d=12, q=4, seed=0)
    idx = split_indices(ds.n, seed=0)
    spec = TransformSpec("gw_q16_rot", gaussianize=True, whiten=True, quantize_levels=16, rotate=True)
    tfm = PrivacyTransform(spec, seed=0)
    X_train = ds.X[idx["train"]]
    tfm.fit(X_train, ds.y[idx["train"]])
    Z_test = tfm.transform(ds.X[idx["test"]])
    assert Z_test.shape[0] == len(idx["test"])
    assert Z_test.shape[1] == 12


def test_quantization_is_many_to_one():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(2000, 6))
    spec = TransformSpec("q8", gaussianize=True, whiten=True, quantize_levels=8)
    Z = PrivacyTransform(spec, seed=0).fit_transform(X, rng.normal(size=2000))
    # Unique rows of Z should be far fewer than unique rows of X.
    uniq_x = len(np.unique(np.round(X, 8), axis=0))
    uniq_z = len(np.unique(np.round(Z, 8), axis=0))
    assert uniq_x == 2000
    assert uniq_z < uniq_x


def test_rotation_is_orthogonal():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(300, 8))
    spec = TransformSpec("rot", gaussianize=True, whiten=True, rotate=True)
    tfm = PrivacyTransform(spec, seed=1)
    tfm.fit(X)
    R = tfm.R
    np.testing.assert_allclose(R.T @ R, np.eye(8), atol=1e-8)


def test_bottleneck_reduces_dimension():
    ds = make_banking_synthetic(n=500, d=16, q=4, seed=2)
    spec = TransformSpec("bn", bottleneck=True, bottleneck_ratio=0.5, rotate=True)
    tfm = PrivacyTransform(spec, seed=2)
    Z = tfm.fit_transform(ds.X, ds.y)
    assert Z.shape == (500, 8)
