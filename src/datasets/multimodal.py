"""Mixture-of-clusters DGP: opposite-sign covariances and segment slopes.

Three latent components (never returned) generate the same observed
schema.  A *single* Gaussian copula sees only the pooled rank
correlations, which cancel across clusters; a mixture or a sequential
tree that can split on ``cluster_hint`` (or on location in
``(x1, x2)``) can recover the local joints and ``P(y | X)``.

Observed table
--------------
* ``x1``, ``x2``, ``x3``, ``x4`` — continuous
* ``cluster_hint`` — 3-level categorical, a *noisy* label of the latent
* ``mix_flag`` — Bernoulli 0/1 (cluster-dependent prevalence)
* ``y`` — continuous target
* ``y_class`` — Bernoulli of a logistic of the **same cluster-wise
  index** (not a quantile cut on ``y``)

Latent geometry
---------------
* **A** — ``corr(x1, x2) > 0``, ``y ~ +x1``
* **B** — ``corr(x1, x2) < 0``, ``y ~ -x1``
* **C** — high variance blob, ``y ~ x3²``

Pooled ``corr(x1, x2)`` and ``corr(x1, y)`` stay near zero because A and
B have opposite signs; ``corr(x3, y)`` stays near zero because the C
signal is quadratic and ``x3`` is mean-zero.  That is the copula failure
mode.  ``cluster_hint`` is informative but imperfect, so a synthesizer
cannot treat it as a hard stratum key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "x1",
    "x2",
    "x3",
    "x4",
    "cluster_hint",
    "mix_flag",
)
TARGET_REG = "y"
TARGET_CLF = "y_class"

CLUSTER_LEVELS: tuple[str, ...] = ("A", "B", "C")
# Slightly unequal so a copula cannot treat the mixture as a balanced dummy.
CLUSTER_PROBS: tuple[float, ...] = (0.36, 0.34, 0.30)

# (x1, x2) location — A right, B left, C below.  Between-component
# Cov(x1, x2) is near 0 so the pooled correlation is the cancelled
# within-component average, not a mean-shift artefact.
_MEAN_X12 = np.array(
    [
        [2.85, 1.10],
        [-2.85, 1.00],
        [0.00, -4.10],
    ],
    dtype=np.float64,
)
# Scale: A/B tight; C is the high-variance component (and sits well below
# so the wide blob does not smear A vs B on x1).
_SD_X1 = np.array([0.72, 0.72, 1.60], dtype=np.float64)
_SD_X2 = np.array([0.72, 0.72, 1.75], dtype=np.float64)
_RHO_X12 = np.array([0.82, -0.82, 0.05], dtype=np.float64)

_SD_X3 = np.array([0.95, 0.95, 1.35], dtype=np.float64)
_SD_X4 = np.array([1.00, 1.00, 1.90], dtype=np.float64)

# y = intercept_k + s1_k · x1 + s3_k · x3² + mix_coef · mix_flag + ε_k
# Intercepts cancel E[s1·x1] / E[s3·x3²] so cluster means of y stay close
# and a copula cannot ride a location shortcut.
_SLOPE_X1 = np.array([1.55, -1.55, 0.00], dtype=np.float64)
_SLOPE_X3SQ = np.array([0.00, 0.00, 0.80], dtype=np.float64)
_Y_CENTER = 0.20
_MIX_COEF = 0.25
_NOISE_Y = np.array([0.55, 0.55, 0.80], dtype=np.float64)

# mix_flag prevalence differs by cluster (observed mixed-type association).
_MIX_P = np.array([0.28, 0.42, 0.62], dtype=np.float64)

# P(hint | true cluster).  Diagonal ≈ 0.84 — useful, not a hard key.
_HINT_CONFUSION = np.array(
    [
        [0.84, 0.09, 0.07],
        [0.09, 0.84, 0.07],
        [0.08, 0.08, 0.84],
    ],
    dtype=np.float64,
)

# y_class shares the cluster-wise map (sign of x1 / x3²).  Features are
# cluster-centered so prevalence stays mixed (raw x1 would saturate A/B).
_YCLASS_INTERCEPT = -0.18
_YCLASS_X1 = np.array([1.20, -1.20, 0.00], dtype=np.float64)
_YCLASS_X3SQ = np.array([0.00, 0.00, 0.55], dtype=np.float64)
_YCLASS_MIX = 0.35


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _cluster_intercepts() -> np.ndarray:
    """Intercepts that keep E[η | cluster] near ``_Y_CENTER``."""
    e_x1 = _MEAN_X12[:, 0]
    e_x3sq = _SD_X3**2  # x3 ~ N(0, σ) ⇒ E[x3²] = σ²
    return _Y_CENTER - _SLOPE_X1 * e_x1 - _SLOPE_X3SQ * e_x3sq


def _draw_x12(
    rng: np.random.Generator,
    cluster_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-conditional bivariate Gaussian for ``(x1, x2)``."""
    n = int(cluster_idx.size)
    x1 = np.empty(n, dtype=np.float64)
    x2 = np.empty(n, dtype=np.float64)
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    for k in range(3):
        m = cluster_idx == k
        if not np.any(m):
            continue
        rho = float(_RHO_X12[k])
        rho = float(np.clip(rho, -0.95, 0.95))
        x1[m] = _MEAN_X12[k, 0] + _SD_X1[k] * z1[m]
        x2[m] = _MEAN_X12[k, 1] + _SD_X2[k] * (
            rho * z1[m] + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2[m]
        )
    return x1, x2


def _noisy_hint(rng: np.random.Generator, cluster_idx: np.ndarray) -> np.ndarray:
    """Draw a 3-level hint from the confusion matrix of the true cluster."""
    probs = _HINT_CONFUSION[cluster_idx]
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(cluster_idx.shape[0])
    return (u[:, None] < cdf).argmax(axis=1)


def regression_index(
    cluster_idx: np.ndarray,
    x1: np.ndarray,
    x3: np.ndarray,
    mix_flag: np.ndarray,
) -> np.ndarray:
    """Cluster-wise index shared (up to scale) by ``y`` and ``y_class``."""
    intercept = _cluster_intercepts()[cluster_idx]
    return (
        intercept
        + _SLOPE_X1[cluster_idx] * x1
        + _SLOPE_X3SQ[cluster_idx] * (x3 * x3)
        + _MIX_COEF * mix_flag.astype(np.float64)
    )


def generate(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` rows from the three-cluster multimodal DGP.

    Parameters
    ----------
    n:
        Number of rows. Must be a positive integer.
    seed:
        RNG seed (passed to ``numpy.random.default_rng``).

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS + (y, y_class)``.  The latent cluster
        label is **not** returned; ``cluster_hint`` is a noisy version.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))
    cluster_idx = rng.choice(3, size=n, p=np.asarray(CLUSTER_PROBS))

    x1, x2 = _draw_x12(rng, cluster_idx)
    x3 = rng.normal(0.0, 1.0, size=n) * _SD_X3[cluster_idx]
    x4 = rng.normal(0.0, 1.0, size=n) * _SD_X4[cluster_idx]

    mix_p = _MIX_P[cluster_idx]
    mix_flag = (rng.random(n) < mix_p).astype(np.int64)

    hint_idx = _noisy_hint(rng, cluster_idx)
    cluster_hint = np.asarray(CLUSTER_LEVELS, dtype=object)[hint_idx]

    eta = regression_index(cluster_idx, x1, x3, mix_flag)
    y = eta + _NOISE_Y[cluster_idx] * rng.normal(0.0, 1.0, size=n)

    x1_c = x1 - _MEAN_X12[cluster_idx, 0]
    x3sq_c = (x3 * x3) - (_SD_X3[cluster_idx] ** 2)
    mix_c = mix_flag.astype(np.float64) - _MIX_P[cluster_idx]
    cls_index = (
        _YCLASS_INTERCEPT
        + _YCLASS_X1[cluster_idx] * x1_c
        + _YCLASS_X3SQ[cluster_idx] * x3sq_c
        + _YCLASS_MIX * mix_c
    )
    y_class = rng.binomial(1, _sigmoid(cls_index), size=n).astype(np.int64)

    return pd.DataFrame(
        {
            "x1": x1.astype(np.float64),
            "x2": x2.astype(np.float64),
            "x3": x3.astype(np.float64),
            "x4": x4.astype(np.float64),
            "cluster_hint": pd.Series(cluster_hint, dtype="string"),
            "mix_flag": mix_flag,
            "y": y.astype(np.float64),
            "y_class": y_class,
        }
    )


def get_ground_truth_description() -> str:
    """Return a human-readable summary of the multimodal mixture DGP."""
    return (
        "Multimodal mixture DGP (name=multimodal; copula should fail; "
        "mixture/CART should win).  Latent cluster Z ∈ {A, B, C} with "
        "probs 0.36 / 0.34 / 0.30 (not returned).  "
        "(x1, x2) | A ~ N((2.85, 1.10), ρ=+0.82, σ=0.72); "
        "(x1, x2) | B ~ N((-2.85, 1.00), ρ=−0.82, σ=0.72); "
        "(x1, x2) | C ~ N((0.00, −4.10), ρ=+0.05, σ≈1.6–1.75) (high variance).  "
        "x3 ~ N(0, σ3,Z), σ3=(0.95, 0.95, 1.35); "
        "x4 ~ N(0, σ4,Z), σ4=(1.00, 1.00, 1.90) (unused in y).  "
        "cluster_hint is a 3-level noisy label of Z (P(correct)≈0.84).  "
        "mix_flag ~ Bern(p_Z), p=(0.28, 0.42, 0.62).  "
        "η = c_Z + s1_Z·x1 + s3_Z·x3² + 0.25·mix_flag, "
        "s1=(+1.55, −1.55, 0), s3=(0, 0, 0.80), "
        "c_Z chosen so E[η|Z]≈0.20.  "
        "TARGET y = η + σ_Z·ε, σ=(0.55, 0.55, 0.80), ε~N(0,1).  "
        "TARGET y_class ~ Bern(σ(−0.18 + 1.20·s_sign_Z·(x1−μ1,Z) "
        "+ 0.55·I(C)·(x3²−σ3²) + 0.35·(mix_flag−p_Z))) — logistic of the "
        "same cluster-wise map (cluster-centered so the class is not "
        "saturated), not a quantile cut on y.  "
        "Pooled corr(x1, x2) and corr(x1, y) cancel (A vs B); "
        "corr(x3, y) is weak because C's signal is x3² with mean-zero x3.  "
        "A single Gaussian copula therefore smears the three modes and "
        "sees near-independence.  A mixture or CART that splits on "
        "cluster_hint (or on location) recovers the local sign of "
        "corr(x1, x2) and the segment-specific slope / quadratic."
    )


SPEC = DatasetSpec(
    name="multimodal",
    generate=generate,
    description=get_ground_truth_description(),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=("multimodal", "mixture", "clusters", "copula-fail", "trees-win", "mixed"),
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _assign_nearest_cluster(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Map rows to the nearest design mean in ``(x1, x2)`` (self-test)."""
    loc = np.column_stack([np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)])
    d2 = ((loc[:, None, :] - _MEAN_X12[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def _self_test() -> None:
    """Schema, cancelled pooled correlations, local signs, tree > linear."""
    from sklearn.linear_model import LinearRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.tree import DecisionTreeRegressor

    n = 2800
    seed = 11
    df = generate(n, seed)

    expected_cols = list(FEATURE_COLS) + [TARGET_REG, TARGET_CLF]
    assert list(df.columns) == expected_cols, list(df.columns)
    assert df.shape == (n, len(expected_cols))
    assert int(df.isna().sum().sum()) == 0

    for col in ("x1", "x2", "x3", "x4", TARGET_REG):
        assert pd.api.types.is_float_dtype(df[col]), col
    assert str(df["cluster_hint"].dtype) == "string"
    assert set(df["cluster_hint"].unique()) <= set(CLUSTER_LEVELS)
    assert int(df["cluster_hint"].nunique()) == 3
    assert pd.api.types.is_integer_dtype(df["mix_flag"])
    assert set(df["mix_flag"].unique()) <= {0, 1}
    assert int(df["mix_flag"].nunique()) == 2
    assert pd.api.types.is_integer_dtype(df[TARGET_CLF])
    assert set(df[TARGET_CLF].unique()) == {0, 1}

    # Every hint level and both mix flags appear with usable mass.
    hint_counts = df["cluster_hint"].value_counts()
    for level in CLUSTER_LEVELS:
        assert int(hint_counts.get(level, 0)) > 0.15 * n, level

    y = df[TARGET_REG].to_numpy(dtype=float)
    assert float(np.std(y)) > 0.5
    assert np.unique(y).size == n

    x1 = df["x1"].to_numpy(dtype=float)
    x2 = df["x2"].to_numpy(dtype=float)
    x3 = df["x3"].to_numpy(dtype=float)
    zhat = _assign_nearest_cluster(x1, x2)
    for k in range(3):
        assert int(np.sum(zhat == k)) > 0.12 * n, k

    # Pooled (copula view): opposite clusters cancel.
    r12 = _pearson(x1, x2)
    r1y = _pearson(x1, y)
    r3y = _pearson(x3, y)
    assert abs(r12) < 0.22, f"pooled corr(x1,x2)={r12:.4f} (copula shortcut)"
    assert abs(r1y) < 0.22, f"pooled corr(x1,y)={r1y:.4f} (copula shortcut)"
    assert abs(r3y) < 0.18, f"pooled corr(x3,y)={r3y:.4f} (quadratic leak)"

    # Local geometry on recovered modes (not the noisy hint).
    mask_a = zhat == 0
    mask_b = zhat == 1
    mask_c = zhat == 2

    r12_a = _pearson(x1[mask_a], x2[mask_a])
    r12_b = _pearson(x1[mask_b], x2[mask_b])
    r1y_a = _pearson(x1[mask_a], y[mask_a])
    r1y_b = _pearson(x1[mask_b], y[mask_b])
    r3sq_c = _pearson(x3[mask_c] ** 2, y[mask_c])
    assert r12_a > 0.55, f"corr(x1,x2|A)={r12_a:.4f}"
    assert r12_b < -0.55, f"corr(x1,x2|B)={r12_b:.4f}"
    assert r1y_a > 0.55, f"corr(x1,y|A)={r1y_a:.4f}"
    assert r1y_b < -0.55, f"corr(x1,y|B)={r1y_b:.4f}"
    assert r3sq_c > 0.50, f"corr(x3^2,y|C)={r3sq_c:.4f}"

    # Noisy hint still carries the sign (weaker than the recovered modes).
    hint = df["cluster_hint"].to_numpy()
    r12_ha = _pearson(x1[hint == "A"], x2[hint == "A"])
    r12_hb = _pearson(x1[hint == "B"], x2[hint == "B"])
    assert r12_ha > 0.20, f"corr(x1,x2|hint=A)={r12_ha:.4f}"
    assert r12_hb < -0.20, f"corr(x1,x2|hint=B)={r12_hb:.4f}"

    # C is the high-variance component on (x1, x2) and x4.
    sd_a = float(np.std(x1[mask_a]))
    sd_c = float(np.std(x1[mask_c]))
    assert sd_c > 1.45 * sd_a, f"sd(x1|C)={sd_c:.3f} vs sd(x1|A)={sd_a:.3f}"
    sd4_c = float(np.std(df.loc[mask_c, "x4"].to_numpy(dtype=float)))
    sd4_a = float(np.std(df.loc[mask_a, "x4"].to_numpy(dtype=float)))
    assert sd4_c > 1.35 * sd4_a, f"sd(x4|C)={sd4_c:.3f} vs A={sd4_a:.3f}"

    # cluster_hint is noisy: disagrees with the recovered mode on a real slice.
    zhat_label = np.asarray(CLUSTER_LEVELS)[zhat]
    agree = float(np.mean(zhat_label == hint))
    assert 0.70 <= agree <= 0.93, f"hint vs nearest-mean agree={agree:.3f}"

    # GMM: 3 components beat 1 (modes are real, not one ellipse).
    x12 = np.column_stack([x1, x2])
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(x12)
    gmm3 = GaussianMixture(n_components=3, random_state=0).fit(x12)
    bic1 = float(gmm1.bic(x12))
    bic3 = float(gmm3.bic(x12))
    assert bic3 < bic1 - 40.0, f"GMM BIC 3={bic3:.1f} not << 1={bic1:.1f}"

    # Utility geometry: a single linear map loses; a tree / piecewise
    # linear-by-mode recovers the cancelled slopes (CART / mixture win).
    X_num = df[["x1", "x2", "x3", "x4", "mix_flag"]].to_numpy(dtype=float)
    hint_oh = pd.get_dummies(df["cluster_hint"], prefix="h").to_numpy(dtype=float)
    X_obs = np.column_stack([X_num, hint_oh])
    r2_lin = float(LinearRegression().fit(X_obs, y).score(X_obs, y))
    tree = DecisionTreeRegressor(
        max_depth=8, min_samples_leaf=20, random_state=0
    )
    r2_tree = float(tree.fit(X_obs, y).score(X_obs, y))
    assert r2_lin < 0.42, f"global linear R²={r2_lin:.3f} (should be weak)"
    assert r2_tree > r2_lin + 0.18, (
        f"tree R²={r2_tree:.3f} not enough above linear {r2_lin:.3f}"
    )

    # Piecewise linear on recovered modes (mixture-style oracle).
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = 0.0
    for k in range(3):
        m = zhat == k
        lr = LinearRegression().fit(X_num[m], y[m])
        pred = lr.predict(X_num[m])
        ss_res += float(np.sum((y[m] - pred) ** 2))
    r2_piece = 1.0 - ss_res / ss_tot
    assert r2_piece > r2_lin + 0.20, (
        f"piecewise R²={r2_piece:.3f} vs linear {r2_lin:.3f}"
    )

    # y_class: associated with the index, not a y-quantile cut.
    yc = df[TARGET_CLF].to_numpy(dtype=float)
    prev = float(np.mean(yc))
    assert 0.25 <= prev <= 0.75, f"y_class prevalence={prev:.3f}"
    r_y = _pearson(y, yc)
    assert r_y > 0.15, f"corr(y, y_class)={r_y:.4f}"
    assert not np.array_equal(yc, (y > np.median(y)).astype(float))

    # x4 is a distractor: weak pooled link to y.
    r4y = abs(_pearson(df["x4"].to_numpy(dtype=float), y))
    assert r4y < 0.12, f"corr(x4,y)={r4y:.4f}"

    # Seed stability / sensitivity.
    a = generate(80, 7)
    b = generate(80, 7)
    pd.testing.assert_frame_equal(a, b)
    c = generate(80, 8)
    assert not a.equals(c)

    # n validation + DatasetSpec contract.
    try:
        generate(0, 0)
        raise AssertionError("n=0 should raise")
    except ValueError:
        pass
    assert SPEC.name == "multimodal"
    assert SPEC.target_reg == "y"
    assert SPEC.target_clf == "y_class"
    assert SPEC.feature_cols == FEATURE_COLS
    sampled = SPEC.sample(40, seed=3)
    assert len(sampled) == 40
    assert "y" in sampled.columns

    print(
        "multimodal self-test ok  "
        f"n={n}  "
        f"pooled r(x1,x2)={r12:+.3f}  "
        f"pooled r(x1,y)={r1y:+.3f}  "
        f"r(x1,x2|A)={r12_a:+.3f}  "
        f"r(x1,x2|B)={r12_b:+.3f}  "
        f"r(x1,y|A)={r1y_a:+.3f}  "
        f"r(x1,y|B)={r1y_b:+.3f}  "
        f"r(x3^2,y|C)={r3sq_c:+.3f}  "
        f"R2_lin={r2_lin:.3f}  "
        f"R2_tree={r2_tree:.3f}  "
        f"R2_piece={r2_piece:.3f}  "
        f"GMM_BIC 1={bic1:.0f} 3={bic3:.0f}  "
        f"hint_agree={agree:.3f}  "
        f"y_class_prev={prev:.3f}"
    )


if __name__ == "__main__":
    _self_test()


__all__ = [
    "FEATURE_COLS",
    "TARGET_REG",
    "TARGET_CLF",
    "CLUSTER_LEVELS",
    "CLUSTER_PROBS",
    "SPEC",
    "generate",
    "regression_index",
    "get_ground_truth_description",
]
