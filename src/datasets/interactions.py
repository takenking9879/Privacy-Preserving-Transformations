"""Interaction-stress DGP: product, XOR, and group-specific slopes.

A Gaussian copula only recovers pairwise rank correlations.  In this table
the regression signal is the latent product ``a*b`` (plus a sign-XOR and
group-specific slopes), so ``corr(a, y)`` and ``corr(b, y)`` stay near
zero and a copula should fail TSTR.  Sequential trees can split on ``a``
then ``b`` (and on ``group``) and should recover ``P(y | X)``.

This is the stress test that a *general* synthesizer must capture
interactions — not just elliptical / rank-linear dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "a",
    "b",
    "c",
    "flag",
    "group",
    "xor_like",
    "noise1",
    "noise2",
    "count_x",
)

TARGET_REG = "y"
TARGET_CLF = "y_class"

GROUP_LEVELS: tuple[str, ...] = ("G1", "G2", "G3")
# Unequal mix so a copula cannot treat ``group`` as a balanced dummy.
_GROUP_PROBS: tuple[float, ...] = (0.45, 0.35, 0.20)

# Group-specific slope of ``c`` in the interaction index (mean near 0 so
# it does not create a global linear ``c``–``y`` shortcut that a copula
# could ride, but each segment is strongly sloped).
_SLOPE_C = np.array([1.05, -1.15, 0.40], dtype=np.float64)
# Group-specific slope of the product ``a*b``.  All same sign so a single
# ``a*b`` term still explains y (self-test R²), while trees can refine by
# splitting on ``group``.
_SLOPE_AB = np.array([2.05, 1.50, 1.15], dtype=np.float64)
_INTERCEPT = np.array([0.20, -0.15, 0.05], dtype=np.float64)

_XOR_COEF = 0.65
_FLAG_COEF = 0.30
_NOISE_SIGMA = 0.90
# Logistic intercept: keeps ``y_class`` mixed given Var(index) ≈ 4.
_YCLASS_INTERCEPT = -0.35
_FLAG_P = 0.40


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` rows from the interaction-stress DGP.

    Parameters
    ----------
    n:
        Number of rows. Must be a positive integer.
    seed:
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS + (y, y_class)``.  Latent product / XOR
        index are not returned as extra columns; ``xor_like`` is the
        observed 0/1 encoding of ``(a>0) XOR (b>0)``.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Features.  a, b are i.i.d. mean-zero symmetric so
    # Cov(a, a*b) = E[a²] E[b] = 0 (and likewise Cov(b, a*b) = 0).
    # The *interaction* is the signal; the marginals of a and b are not.
    # ------------------------------------------------------------------
    a = rng.normal(0.0, 1.0, size=n)
    b = rng.normal(0.0, 1.0, size=n)
    c = rng.normal(0.0, 1.0, size=n)
    flag = rng.binomial(1, _FLAG_P, size=n).astype(np.int64)
    group_idx = rng.choice(3, size=n, p=np.asarray(_GROUP_PROBS))
    group = np.asarray(GROUP_LEVELS, dtype=object)[group_idx]

    # Deterministic sign-XOR.  E[xor_like | a] = 1/2 for all a (b is
    # symmetric about 0), so xor_like also induces no corr(a, y).
    xor_like = np.not_equal(a > 0.0, b > 0.0).astype(np.int64)

    # Distractors: unused in y.  A copula that matches their (null)
    # correlations still misses the product.
    noise1 = rng.normal(0.0, 1.0, size=n)
    noise2 = rng.normal(0.0, 1.0, size=n)

    # Count feature: Poisson, depends on flag / group only — not on a, b.
    is_g3 = (group_idx == 2).astype(np.float64)
    lam = np.clip(1.20 + 0.90 * flag.astype(np.float64) + 0.55 * is_g3, 0.15, 12.0)
    count_x = rng.poisson(lam).astype(np.int64)

    # ------------------------------------------------------------------
    # Interaction index → y (continuous) and y_class (logistic).
    # ------------------------------------------------------------------
    slope_ab = _SLOPE_AB[group_idx]
    slope_c = _SLOPE_C[group_idx]
    intercept_g = _INTERCEPT[group_idx]
    product = a * b

    interaction_index = (
        intercept_g
        + slope_ab * product
        + _XOR_COEF * xor_like.astype(np.float64)
        + slope_c * c
        + _FLAG_COEF * flag.astype(np.float64)
    )
    y = interaction_index + _NOISE_SIGMA * rng.normal(0.0, 1.0, size=n)
    y_class = rng.binomial(
        1, _sigmoid(_YCLASS_INTERCEPT + interaction_index), size=n
    ).astype(np.int64)

    return pd.DataFrame(
        {
            "a": a.astype(np.float64),
            "b": b.astype(np.float64),
            "c": c.astype(np.float64),
            "flag": flag,
            "group": pd.Series(group, dtype="string"),
            "xor_like": xor_like,
            "noise1": noise1.astype(np.float64),
            "noise2": noise2.astype(np.float64),
            "count_x": count_x,
            "y": y.astype(np.float64),
            "y_class": y_class,
        }
    )


def get_ground_truth_description() -> str:
    """Return a human-readable summary of the interaction DGP."""
    return (
        "Interaction-stress DGP (copula should fail; sequential trees should "
        "win).  Features: a, b, c ~ i.i.d. N(0,1); flag ~ Bern(0.40); "
        "group ∈ {G1, G2, G3} with probs 0.45 / 0.35 / 0.20; "
        "xor_like = 1{(a>0) XOR (b>0)} (deterministic); "
        "noise1, noise2 ~ N(0,1) (unused in y); "
        "count_x ~ Poisson(1.20 + 0.90·flag + 0.55·I(G3)).  "
        "Latent interaction index η = intercept_g + s_ab(g)·(a·b) + 0.65·xor_like "
        "+ s_c(g)·c + 0.30·flag, where "
        "s_ab = (G1: 2.05, G2: 1.50, G3: 1.15), "
        "s_c = (G1: 1.05, G2: −1.15, G3: 0.40), "
        "intercept = (G1: 0.20, G2: −0.15, G3: 0.05).  "
        "TARGET y = η + 0.90·ε, ε~N(0,1).  "
        "TARGET y_class ~ Bern(σ(−0.35 + η)) — a logistic of the same "
        "interaction index, not a quantile cut on y.  "
        "Because a and b are independent mean-zero symmetric, "
        "Cov(a, a·b) = Cov(b, a·b) = 0 and E[xor_like | a] = 1/2, so "
        "marginal corr(a, y) and corr(b, y) are weak; the product (and XOR) "
        "is the signal.  A Gaussian copula that matches pairwise rank "
        "correlations therefore sees a, b, y as nearly independent and "
        "cannot reconstruct xor_like from (a, b).  Sequential trees can "
        "split on a then b (and on group) and recover both the product "
        "surface and the XOR atom."
    )


SPEC = DatasetSpec(
    name="interactions",
    generate=generate,
    description=get_ground_truth_description(),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=("interactions", "xor", "product", "copula-fail", "trees-win", "mixed"),
)


__all__ = [
    "FEATURE_COLS",
    "TARGET_REG",
    "TARGET_CLF",
    "GROUP_LEVELS",
    "SPEC",
    "generate",
    "get_ground_truth_description",
]
