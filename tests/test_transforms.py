"""Regression tests for transforms, remapping, and anonymity."""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets import make_banking_mixed
from src.schema import TargetMap
from src.transforms import build_transform, _make_target_map


def test_target_map_regression_roundtrip():
    rng = np.random.RandomState(0)
    y = rng.normal(size=50)
    tm = _make_target_map(y, "regression", rng)
    np.testing.assert_allclose(tm.inverse(tm.transform(y)), y, atol=1e-10)
    assert tm.scale != 1.0 or tm.shift != 0.0


def test_target_map_classification_roundtrip():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 3, size=80)
    tm = _make_target_map(y, "classification", rng)
    out = tm.inverse(tm.transform(y))
    np.testing.assert_array_equal(out, y)


def test_identity_target_is_plaintext():
    table = make_banking_mixed(n=120, seed=0, task="regression")
    idx = np.arange(80)
    tfm = build_transform("identity", seed=0)
    tfm.fit(table, idx)
    assert tfm.target_map.scale == 1.0
    assert tfm.target_map.shift == 0.0


@pytest.mark.parametrize("name", ["gauss", "typed_keyed", "keyed_monotone", "rff"])
def test_transformed_columns_are_anonymous(name):
    table = make_banking_mixed(n=200, seed=3, task="regression")
    idx = np.arange(140)
    tfm = build_transform(name, seed=3)
    tfm.fit(table, idx)
    result = tfm.apply(table, idx)
    assert result.Z.shape[0] == 140
    assert result.Z.ndim == 2
    for cid in result.column_ids:
        assert cid.startswith("c")
        assert cid[1:].isdigit()
    forbidden = {"income", "gender", "email", "zip_code", "credit_score"}
    assert forbidden.isdisjoint(set(result.column_ids))
    assert not any(name.lower() in cid.lower() for name in forbidden for cid in result.column_ids)


def test_typed_keyed_handles_mixed_types():
    table = make_banking_mixed(n=180, seed=4, task="classification")
    idx = np.arange(120)
    tfm = build_transform("typed_keyed", seed=4)
    tfm.fit(table, idx)
    Z = tfm.transform_X(table.predictive_frame().iloc[idx])
    assert np.isfinite(Z).all()
    assert Z.shape[1] >= 8


def test_gauss_is_rank_preserving_on_numeric():
    table = make_banking_mixed(n=160, seed=5, task="regression")
    idx = np.arange(110)
    tfm = build_transform("gauss", seed=5)
    tfm.fit(table, idx)
    frame = table.predictive_frame().iloc[idx]
    Z = tfm.transform_X(frame)
    income = frame["income"].to_numpy()
    # Some transformed column should be comonotonic with income.
    from scipy.stats import spearmanr

    best = max(abs(spearmanr(income, Z[:, k]).correlation) for k in range(Z.shape[1]))
    assert best > 0.85
