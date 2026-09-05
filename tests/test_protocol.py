"""Protocol-level regression tests: utility interface and attackers."""

from __future__ import annotations

import numpy as np

from src.attacks import attacker_a, attacker_b
from src.datasets import make_banking_mixed
from src.encode import PublicEncoder
from src.experiment import run_one, split_indices
from src.models import train_eval
from src.transforms import build_transform


def test_identity_utility_is_finite():
    table = make_banking_mixed(n=220, seed=0, task="regression")
    idx = split_indices(table.n, 0)
    enc = PublicEncoder(table.columns)
    X = enc.fit_transform(table.predictive_frame())
    util, pred = train_eval(
        "linear",
        "regression",
        X[idx["train"]],
        table.y[idx["train"]],
        X[idx["test"]],
        table.y[idx["test"]],
        seed=0,
    )
    assert np.isfinite(util.r2)
    assert pred.shape[0] == len(idx["test"])


def test_run_one_identity_has_anonymous_or_raw_ids():
    table = make_banking_mixed(n=240, seed=1, task="regression")
    idx = split_indices(table.n, 1)
    enc = PublicEncoder(table.columns)
    X = enc.fit_transform(table.predictive_frame())
    row = run_one(table, "identity", 1, idx, {}, {}, enc, X, heavy=False)
    assert row["method"] == "identity"
    assert row["utility"]
    assert "attacker_a" in row and "attacker_b" in row


def test_attacker_a_identity_recovers_with_enough_pairs():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 6))
    Z = X.copy()
    y = X[:, 0] + rng.normal(scale=0.1, size=400)
    out = attacker_a(
        X[:280],
        Z[:280],
        X[280:],
        Z[280:],
        y[:280],
        y[280:],
        np.array([True, False, False, False, False, False]),
        [],
        seed=0,
        heavy=False,
    )
    assert out["summary_recon_r2"] > 0.95
    assert out["known_pair_robustness"] == "collapses_with_enough_pairs"


def test_attacker_a_rotation_needs_pairs():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(500, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    Z = X @ q
    y = X.sum(axis=1)
    out = attacker_a(
        X[:350],
        Z[:350],
        X[350:],
        Z[350:],
        y[:350],
        y[350:],
        None,
        [],
        seed=1,
        heavy=False,
    )
    # With 200 pairs and d=8, ridge should essentially invert the rotation.
    assert out["summary_recon_r2"] > 0.85


def test_attacker_b_does_not_need_schema():
    rng = np.random.RandomState(2)
    Z = rng.normal(size=(300, 5))
    mask = np.zeros(300, dtype=bool)
    mask[:210] = True
    out = attacker_b(
        Z,
        "This dataset comes from a bank and contains customer-related statistics.",
        [],
        [],
        np.zeros(5, dtype=bool),
        X_aux_public=None,
        train_mask=mask,
        seed=2,
    )
    assert out["n_cols"] == 5
    assert "semantic_guesses" in out
    assert out["context_used"].startswith("This dataset comes from a bank")


def test_predictions_remap_after_target_transform():
    table = make_banking_mixed(n=260, seed=6, task="regression")
    idx = np.arange(180)
    tfm = build_transform("gauss", seed=6)
    tfm.fit(table, idx)
    y = table.y[idx]
    y_t = tfm.transform_y(y)
    assert not np.allclose(y_t, y)
    np.testing.assert_allclose(tfm.inverse_y(y_t), y, atol=1e-8)
