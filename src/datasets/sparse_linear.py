"""Sparse linear DGP: four signal features + one dummy; the rest is noise.

Observed table
--------------
* ``x00`` … ``x19`` — i.i.d. standard normals (20 numeric features)
* ``cat_a`` — 3-level unused categorical (dummy factor, independent of y)
* ``bin_a`` — Bernoulli dummy that *does* enter the linear index
* ``y`` — continuous sparse-linear target
* ``y_class`` — Bernoulli of a logistic of the **same** linear index

Structural equation
-------------------
::

    index = 1.2·x00 − 0.8·x03 + 0.6·x07 + 0.5·x12 + 0.9·bin_a
    y     = index + ε,   ε ~ N(0, 1)
    y_class ~ Bern(σ(index))

Only ``x00``, ``x03``, ``x07``, ``x12``, and ``bin_a`` affect ``y``.
``cat_a`` and the other 16 numeric columns are independent noise: they
are generated independently of each other and of the index, so their
population ``corr(·, y)`` and pairwise correlations are zero.

A good *general* synthesizer must:

1. keep the four true ``corr(X, y)`` (and ``corr(bin_a, y)``);
2. not invent fake pairwise correlations among the noise columns;
3. leave ``cat_a`` independent of ``y``.

Matching a dense covariance or hallucinating noise–noise / noise–y
links is a failure mode of this DGP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

NUMERIC_COLS: tuple[str, ...] = tuple(f"x{i:02d}" for i in range(20))
CAT_COL = "cat_a"
BIN_COL = "bin_a"
FEATURE_COLS: tuple[str, ...] = NUMERIC_COLS + (CAT_COL, BIN_COL)
TARGET_REG = "y"
TARGET_CLF = "y_class"

SIGNAL_NUMERIC: tuple[str, ...] = ("x00", "x03", "x07", "x12")
SIGNAL_FEATURES: tuple[str, ...] = SIGNAL_NUMERIC + (BIN_COL,)
NOISE_NUMERIC: tuple[str, ...] = tuple(c for c in NUMERIC_COLS if c not in SIGNAL_NUMERIC)

COEFS: dict[str, float] = {
    "x00": 1.2,
    "x03": -0.8,
    "x07": 0.6,
    "x12": 0.5,
    "bin_a": 0.9,
}

CAT_A_LEVELS: tuple[str, ...] = ("A", "B", "C")
# Unequal margins so a synthesizer must match a non-uniform 3-level factor.
CAT_A_PROBS: tuple[float, ...] = (0.50, 0.30, 0.20)
BIN_A_P = 0.40
Y_NOISE_SCALE = 1.0
Y_CLASS_INTERCEPT = 0.0  # logistic of the same index (no extra shift)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def linear_index(
    x00: np.ndarray,
    x03: np.ndarray,
    x07: np.ndarray,
    x12: np.ndarray,
    bin_a: np.ndarray,
) -> np.ndarray:
    """Return the sparse linear predictor shared by ``y`` and ``y_class``."""
    return (
        COEFS["x00"] * x00
        + COEFS["x03"] * x03
        + COEFS["x07"] * x07
        + COEFS["x12"] * x12
        + COEFS["bin_a"] * bin_a
    )


def generate(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` rows from the sparse-linear DGP.

    Parameters
    ----------
    n:
        Number of rows. Must be a positive integer.
    seed:
        RNG seed (passed to ``numpy.random.default_rng``).

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS + (y, y_class)``. No missing values.
        Latent structure is fully observed (there is no hidden confounder).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # Independent N(0, 1) block — signal and noise columns are exchangeable
    # marginally, so a method cannot pick the true features from scale alone.
    X = rng.normal(0.0, 1.0, size=(n, 20))

    cat_a = rng.choice(CAT_A_LEVELS, size=n, p=CAT_A_PROBS)
    bin_a = rng.binomial(1, BIN_A_P, size=n).astype(np.float64)

    col = {name: X[:, i] for i, name in enumerate(NUMERIC_COLS)}
    index = linear_index(col["x00"], col["x03"], col["x07"], col["x12"], bin_a)
    y = index + rng.normal(0.0, Y_NOISE_SCALE, size=n)
    y_class = rng.binomial(1, _sigmoid(index + Y_CLASS_INTERCEPT), size=n)

    data: dict[str, object] = {
        name: X[:, i].astype(np.float64) for i, name in enumerate(NUMERIC_COLS)
    }
    data[CAT_COL] = pd.Series(cat_a, dtype="string")
    data[BIN_COL] = bin_a.astype(np.int64)
    data[TARGET_REG] = y.astype(np.float64)
    data[TARGET_CLF] = y_class.astype(np.int64)
    return pd.DataFrame(data)


def get_ground_truth_description() -> str:
    """Return a human-readable summary of the sparse-linear equations."""
    return (
        "Sparse linear DGP (name=sparse_linear). "
        "x00..x19 ~ i.i.d. N(0,1); cat_a ∈ {A,B,C} with probs 0.50/0.30/0.20 "
        "(unused dummy factor); bin_a ~ Bern(0.40). "
        "index = 1.2·x00 − 0.8·x03 + 0.6·x07 + 0.5·x12 + 0.9·bin_a; "
        "y = index + N(0,1); y_class ~ Bern(σ(index)). "
        "Only those four numeric columns plus bin_a affect y. "
        "The other 16 numeric columns and cat_a are independent noise "
        "(population corr with y and with each other is 0). "
        "A good general synthesizer must keep the four true corr(X,y) "
        "(and corr(bin_a,y)) and must not invent fake correlations among noise."
    )


SPEC = DatasetSpec(
    name="sparse_linear",
    generate=generate,
    description=get_ground_truth_description(),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=("sparse", "linear", "high-dim", "noise", "mixed"),
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _self_test() -> None:
    """Runtime checks for schema, sparsity, and seed stability."""
    n = 2500
    seed = 11
    df = generate(n, seed)

    expected_cols = list(FEATURE_COLS) + [TARGET_REG, TARGET_CLF]
    assert list(df.columns) == expected_cols, list(df.columns)
    assert df.shape == (n, len(expected_cols))
    assert df.isna().sum().sum() == 0

    for col in NUMERIC_COLS:
        assert pd.api.types.is_float_dtype(df[col]), col
    assert str(df[CAT_COL].dtype) == "string"
    assert set(df[CAT_COL].unique()) <= set(CAT_A_LEVELS)
    assert df[CAT_COL].nunique() == 3
    assert pd.api.types.is_integer_dtype(df[BIN_COL])
    assert set(df[BIN_COL].unique()) == {0, 1}
    assert pd.api.types.is_float_dtype(df[TARGET_REG])
    assert pd.api.types.is_integer_dtype(df[TARGET_CLF])
    assert set(df[TARGET_CLF].unique()) == {0, 1}

    y = df[TARGET_REG].to_numpy(dtype=float)
    assert float(np.std(y)) > 0.5
    assert np.unique(y).size == n  # continuous, no ties at this n

    # True corr(X, y) stay well above the sampling floor.
    min_abs = {"x00": 0.45, "x03": 0.28, "x07": 0.20, "x12": 0.15, "bin_a": 0.12}
    signs = {"x00": 1.0, "x03": -1.0, "x07": 1.0, "x12": 1.0, "bin_a": 1.0}
    for name, floor in min_abs.items():
        r = _pearson(df[name].to_numpy(dtype=float), y)
        assert np.isfinite(r), name
        assert abs(r) >= floor, f"|corr({name}, y)|={r:.4f} < {floor}"
        assert r * signs[name] > 0.0, f"corr({name}, y) has the wrong sign: {r:.4f}"

    # Noise features: near-zero corr with y (do not leak the target).
    noise_xy = [
        abs(_pearson(df[c].to_numpy(dtype=float), y)) for c in NOISE_NUMERIC
    ]
    assert max(noise_xy) < 0.10, f"noise–y |corr| max={max(noise_xy):.4f}"
    assert float(np.mean(noise_xy)) < 0.04, f"noise–y |corr| mean={np.mean(noise_xy):.4f}"

    # Noise–noise pairwise: population identity; sample |r| stays small.
    noise = df.loc[:, list(NOISE_NUMERIC)].to_numpy(dtype=float)
    corr = np.corrcoef(noise, rowvar=False)
    off = np.abs(corr[np.triu_indices_from(corr, k=1)])
    assert float(np.max(off)) < 0.10, f"noise–noise |corr| max={np.max(off):.4f}"
    assert float(np.mean(off)) < 0.03, f"noise–noise |corr| mean={np.mean(off):.4f}"

    # cat_a is a dummy: mean(y | level) stays close to the global mean.
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    for level in CAT_A_LEVELS:
        sub = y[df[CAT_COL].to_numpy() == level]
        assert sub.size > 20, level
        shift = abs(float(np.mean(sub)) - y_mean) / y_std
        assert shift < 0.20, f"cat_a={level} mean(y) shift={shift:.4f} σ"

    # y_class is a noisy logistic of the same index (not a quantile cut on y).
    index = linear_index(
        df["x00"].to_numpy(dtype=float),
        df["x03"].to_numpy(dtype=float),
        df["x07"].to_numpy(dtype=float),
        df["x12"].to_numpy(dtype=float),
        df[BIN_COL].to_numpy(dtype=float),
    )
    yc = df[TARGET_CLF].to_numpy(dtype=float)
    r_idx = _pearson(index, yc)
    r_y = _pearson(y, yc)
    assert r_idx > 0.45, f"corr(index, y_class)={r_idx:.4f}"
    # Associated with y (shared index) but not a deterministic cut.
    assert r_y > 0.35, f"corr(y, y_class)={r_y:.4f}"
    assert not np.array_equal(yc, (y > np.median(y)).astype(float))

    # Prevalence in a usable range for classification utility.
    prev = float(np.mean(yc))
    assert 0.25 <= prev <= 0.75, f"y_class prevalence={prev:.3f}"

    # Seed stability / sensitivity.
    a = generate(80, 7)
    b = generate(80, 7)
    pd.testing.assert_frame_equal(a, b)
    c = generate(80, 8)
    assert not a.equals(c)

    # DatasetSpec contract.
    assert SPEC.name == "sparse_linear"
    assert SPEC.target_reg == "y"
    assert SPEC.target_clf == "y_class"
    assert SPEC.feature_cols == FEATURE_COLS
    sampled = SPEC.sample(40, seed=3)
    assert len(sampled) == 40
    assert "y" in sampled.columns

    print(
        "sparse_linear self-test ok  "
        f"n={n}  "
        f"corr(x00,y)={_pearson(df['x00'].to_numpy(dtype=float), y):+.3f}  "
        f"corr(x03,y)={_pearson(df['x03'].to_numpy(dtype=float), y):+.3f}  "
        f"corr(x07,y)={_pearson(df['x07'].to_numpy(dtype=float), y):+.3f}  "
        f"corr(x12,y)={_pearson(df['x12'].to_numpy(dtype=float), y):+.3f}  "
        f"corr(bin_a,y)={_pearson(df[BIN_COL].to_numpy(dtype=float), y):+.3f}  "
        f"max|noise-y|={max(noise_xy):.3f}  "
        f"y_class_prev={prev:.3f}"
    )


if __name__ == "__main__":
    _self_test()


__all__ = [
    "NUMERIC_COLS",
    "FEATURE_COLS",
    "TARGET_REG",
    "TARGET_CLF",
    "SIGNAL_NUMERIC",
    "SIGNAL_FEATURES",
    "NOISE_NUMERIC",
    "COEFS",
    "CAT_A_LEVELS",
    "SPEC",
    "linear_index",
    "generate",
    "get_ground_truth_description",
]
