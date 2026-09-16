"""Auto-insurance frequency / severity DGP for synthesizer fidelity tests.

Observed table: rating-factor mix (driver, vehicle, territory, exposure)
plus a continuous claim-cost target ``y`` and a binary claim-filed flag
``y_class``.  Latent risk is never returned.

A synthesizer MUST recover:

* heavy-tailed ``y`` (lognormal body, rare jumbo / Pareto losses)
* a ``territory × mileage`` interaction in ``E[log y | X]``
* a young-driver × ``night_driving`` lift in ``y``
* ``y_class`` as a logistic claim-filed draw (prevalence ~15–25%),
  **not** a quantile cut on ``y``
* a deductible threshold that trims paid severity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "driver_age",
    "vehicle_age",
    "mileage",
    "credit_score",
    "territory",
    "vehicle_type",
    "prior_claims",
    "multi_policy",
    "annual_miles",
    "night_driving",
    "deductable",
)
TARGET_REG = "y"
TARGET_CLF = "y_class"

TERRITORY_LEVELS: tuple[str, ...] = ("urban", "suburban", "rural")
VEHICLE_TYPE_LEVELS: tuple[str, ...] = ("sedan", "suv", "truck", "ev")
DEDUCTABLE_LEVELS: tuple[int, ...] = (250, 500, 1000, 1500, 2500)

# Fixed location for log(odometer) so the territory × mileage interaction
# does not depend on sample moments.
_LOG_MILES_CENTER = 10.85  # ≈ log(51_500)
_YOUNG_AGE_CUT = 25.0
_YOUNG_SCALE = 9.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _categorical_from_probs(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Vectorized categorical draw; ``probs`` is (n, k) and rows sum to 1."""
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(probs.shape[0])
    return (u[:, None] < cdf).argmax(axis=1)


def _young_score(driver_age: np.ndarray) -> np.ndarray:
    """Smooth [0, 1] youth score: 1 at age 16, 0 at age 25+."""
    return np.clip((_YOUNG_AGE_CUT - driver_age) / _YOUNG_SCALE, 0.0, 1.0)


def generate(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` personal-auto policies from the insurance DGP.

    Parameters
    ----------
    n:
        Number of rows. Must be a positive integer.
    seed:
        RNG seed (deterministic given ``n``).

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS`` plus ``y`` and ``y_class``.  Latent risk
        is not returned.  No missing values.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Latent risk (not exported).  Fans out into credit, claims, and both
    # targets so residual dependence remains after conditioning on age.
    # ------------------------------------------------------------------
    risk = rng.normal(0.0, 1.0, size=n)

    # ------------------------------------------------------------------
    # Driver age: extra mass on 16–25 so the youth × night cell is populated.
    # ------------------------------------------------------------------
    young_pool = rng.random(n) < 0.22
    age_young = rng.uniform(16.0, 25.0, size=n)
    age_rest = np.clip(rng.normal(46.0, 13.5, size=n), 25.0, 90.0)
    driver_age = np.where(young_pool, age_young, age_rest)
    young = _young_score(driver_age)
    old = np.clip((driver_age - 70.0) / 15.0, 0.0, 1.0)

    # Credit: older / lower-risk drivers score higher (300–850).
    credit_score = np.clip(
        678.0
        + 1.15 * (driver_age - 40.0)
        - 28.0 * risk
        + rng.normal(0.0, 52.0, size=n),
        300.0,
        850.0,
    )

    # ------------------------------------------------------------------
    # Territory: urban / suburban / rural.  Younger drivers tilt urban.
    # Target mix ≈ 0.42 / 0.36 / 0.22.
    # ------------------------------------------------------------------
    terr_logits = np.column_stack(
        [
            0.22 + 0.55 * young + 0.08 * risk,  # urban
            0.10 + 0.08 * (1.0 - young),  # suburban
            -0.35 - 0.55 * young + 0.18 * old,  # rural
        ]
    )
    terr_idx = _categorical_from_probs(rng, _softmax_rows(terr_logits))
    territory = np.asarray(TERRITORY_LEVELS, dtype=object)[terr_idx]
    is_urban = (terr_idx == 0).astype(np.float64)
    is_suburban = (terr_idx == 1).astype(np.float64)
    is_rural = (terr_idx == 2).astype(np.float64)

    # ------------------------------------------------------------------
    # Vehicle type: sedan / suv / truck / ev (rating factor).
    # EV ← urban + credit; truck ← rural.
    # ------------------------------------------------------------------
    vtype_logits = np.column_stack(
        [
            0.55 - 0.12 * is_rural,  # sedan
            0.12 + 0.28 * is_suburban + 0.10 * (1.0 - young),  # suv
            -0.70 + 1.05 * is_rural + 0.18 * old,  # truck
            -1.25
            + 0.85 * is_urban
            + 0.006 * (credit_score - 700.0)
            + 0.35 * young,  # ev
        ]
    )
    vtype_idx = _categorical_from_probs(rng, _softmax_rows(vtype_logits))
    vehicle_type = np.asarray(VEHICLE_TYPE_LEVELS, dtype=object)[vtype_idx]
    is_sedan = (vtype_idx == 0).astype(np.float64)
    is_suv = (vtype_idx == 1).astype(np.float64)
    is_truck = (vtype_idx == 2).astype(np.float64)
    is_ev = (vtype_idx == 3).astype(np.float64)

    # Vehicle age (years).  EVs are newer.
    veh_shape = 2.15 + 0.35 * (1.0 - is_ev)
    vehicle_age = rng.gamma(shape=veh_shape, scale=3.4, size=n)
    vehicle_age = np.where(is_ev > 0.5, vehicle_age * 0.42, vehicle_age)
    vehicle_age = np.clip(vehicle_age, 0.0, 28.0)

    # Annual miles (exposure).  Rural / young drive more.
    log_annual = (
        np.log(11_800.0)
        + 0.18 * is_rural
        - 0.10 * is_urban
        + 0.12 * young
        + 0.06 * is_truck
        + rng.normal(0.0, 0.32, size=n)
    )
    annual_miles = np.clip(np.exp(log_annual), 1_200.0, 48_000.0)

    # Odometer ≈ age × annual miles × wear.  This is the mileage that
    # interacts with territory in y.
    wear = rng.lognormal(mean=0.0, sigma=0.28, size=n)
    mileage = np.clip(vehicle_age * annual_miles * wear, 0.0, 360_000.0)
    log_miles = np.log1p(mileage)
    miles_s = (log_miles - _LOG_MILES_CENTER) / 1.15

    # Night-driving share in [0, 1].  Youth and urban lift the share.
    night_lp = (
        -1.35
        + 1.15 * young
        + 0.32 * is_urban
        + 0.18 * risk
        + rng.normal(0.0, 0.55, size=n)
    )
    night_driving = np.clip(_sigmoid(night_lp), 0.02, 0.95)

    # Prior claims (count).  Youth / urban / night / risk raise the rate.
    claims_lam = np.clip(
        0.22
        + 0.28 * young
        + 0.20 * is_urban
        + 0.35 * night_driving
        + 0.18 * risk
        + 0.04 * vehicle_age / 8.0,
        0.04,
        7.0,
    )
    prior_claims = np.clip(rng.poisson(claims_lam), 0, 12).astype(np.int64)
    prior_f = prior_claims.astype(np.float64)

    # Multi-policy discount (0/1).  Higher credit / older more likely.
    p_multi = _sigmoid(
        -0.35
        + 0.0075 * (credit_score - 650.0)
        + 0.018 * (driver_age - 40.0)
        - 0.15 * young
    )
    multi_policy = rng.binomial(1, p_multi, size=n).astype(np.int64)
    multi_f = multi_policy.astype(np.float64)

    # Deductible (spelled ``deductable``): discrete rating choice.
    # Higher credit / multi-policy → larger retention.
    ded_logits = np.column_stack(
        [
            0.55 - 0.008 * (credit_score - 640.0) - 0.45 * multi_f,  # 250
            0.70 - 0.002 * (credit_score - 660.0),  # 500
            0.15 + 0.004 * (credit_score - 680.0) + 0.35 * multi_f,  # 1000
            -0.55 + 0.006 * (credit_score - 700.0) + 0.45 * multi_f,  # 1500
            -1.25 + 0.007 * (credit_score - 720.0) + 0.55 * multi_f,  # 2500
        ]
    )
    ded_idx = _categorical_from_probs(rng, _softmax_rows(ded_logits))
    deductable = np.asarray(DEDUCTABLE_LEVELS, dtype=np.int64)[ded_idx]
    deductable_f = deductable.astype(np.float64)

    # ------------------------------------------------------------------
    # y — paid claim cost (USD).  Lognormal body + rare jumbo tail.
    # Named interactions: territory × mileage, young × night_driving.
    # Deductible is a threshold: paid = max(ground-up − deductable, floor).
    # ------------------------------------------------------------------
    eta = (
        6.55
        + 0.22 * young
        + 0.14 * night_driving
        + 0.95 * young * night_driving  # young + night elevate y
        + 0.10 * old
        + 0.22 * is_urban
        + 0.06 * is_suburban
        + 0.10 * miles_s
        + 0.52 * is_urban * miles_s  # territory × mileage
        + 0.12 * is_rural * np.maximum(miles_s, 0.0)
        + 0.16 * prior_f
        - 0.16 * multi_f
        - 0.0010 * (credit_score - 680.0)
        + 0.035 * vehicle_age
        + 0.14 * is_truck
        + 0.07 * is_suv
        - 0.08 * is_ev
        - 0.04 * is_sedan
        + 0.000012 * annual_miles
        + 0.11 * risk
    )
    # Heteroscedastic body noise; a small jumbo mixture for the tail.
    sigma = 0.48 * (1.0 + 0.28 * is_urban + 0.18 * young)
    body = rng.normal(0.0, 1.0, size=n)
    jumbo_p = np.clip(0.045 + 0.025 * is_urban + 0.018 * is_truck + 0.012 * young, 0.02, 0.14)
    is_jumbo = rng.random(n) < jumbo_p
    # Pareto(α≈1.4) multipliers on the jumbo slice → rare very large losses.
    pareto_mult = 1.0 + rng.pareto(1.40, size=n)
    log_ground = eta + sigma * body
    log_ground = np.where(is_jumbo, log_ground + np.log(np.clip(pareto_mult, 1.0, 80.0)), log_ground)
    log_ground = np.clip(log_ground, 2.0, 15.5)
    ground_up = np.exp(log_ground)
    # Deductible threshold on paid severity (still strictly positive).
    y = np.maximum(ground_up - deductable_f, 35.0)
    y = np.clip(y, 35.0, 2_000_000.0)

    # ------------------------------------------------------------------
    # y_class — claim filed.  Logistic frequency index (NOT a y quantile).
    # Shares youth, night, urban, mileage, and prior claims with y.
    # Intercept sized so prevalence sits in ~15–25%.
    # ------------------------------------------------------------------
    freq_lp = (
        -2.15
        + 0.90 * young
        + 0.75 * night_driving
        + 0.55 * young * night_driving
        + 0.48 * is_urban
        + 0.12 * is_suburban
        + 0.18 * miles_s
        + 0.22 * is_urban * miles_s
        + 0.26 * prior_f
        - 0.32 * multi_f
        - 0.0018 * (credit_score - 680.0)
        + 0.08 * is_truck
        + 0.22 * risk
    )
    y_class = rng.binomial(1, _sigmoid(freq_lp), size=n).astype(np.int64)

    return pd.DataFrame(
        {
            "driver_age": driver_age.astype(np.float64),
            "vehicle_age": vehicle_age.astype(np.float64),
            "mileage": mileage.astype(np.float64),
            "credit_score": credit_score.astype(np.float64),
            "territory": pd.Series(territory, dtype="string"),
            "vehicle_type": pd.Series(vehicle_type, dtype="string"),
            "prior_claims": prior_claims,
            "multi_policy": multi_policy,
            "annual_miles": annual_miles.astype(np.float64),
            "night_driving": night_driving.astype(np.float64),
            "deductable": deductable,
            "y": y.astype(np.float64),
            "y_class": y_class,
        }
    )


def get_ground_truth_description() -> str:
    """Human-readable ground truth for reports / diagnostics."""
    return (
        "Auto-insurance DGP (name=insurance). "
        "Features: driver_age, vehicle_age, mileage (odometer), credit_score, "
        "territory ∈ {urban, suburban, rural}, vehicle_type ∈ {sedan, suv, truck, ev}, "
        "prior_claims (count), multi_policy (0/1), annual_miles, night_driving "
        "(night-mile share), deductable ∈ {250, 500, 1000, 1500, 2500}. "
        "y is paid claim cost: lognormal body + rare Pareto jumbo losses "
        "(heavy tail; mostly small, rare large), minus a deductible threshold. "
        "E[log y] includes a territory × mileage product (urban mileage is "
        "especially costly) and a young-driver × night_driving lift. "
        "y_class ~ Bern(σ(frequency index)) is claim-filed at ~15–25%, "
        "not a quantile cut on y."
    )


SPEC = DatasetSpec(
    name="insurance",
    generate=generate,
    description=get_ground_truth_description(),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=(
        "insurance",
        "heavy-tail",
        "tweedie",
        "mixed",
        "interaction",
        "imbalanced",
        "deductible",
    ),
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cell_mean_logy(df: pd.DataFrame, mask: np.ndarray) -> float:
    y = df.loc[mask, "y"].to_numpy(dtype=float)
    if y.size < 20:
        return float("nan")
    return float(np.mean(np.log(np.clip(y, 1e-6, None))))


def _self_test(n: int = 4500, seed: int = 7) -> dict[str, float | int | bool]:
    """Runtime checks for schema, tail, interactions, and class balance."""
    df = generate(n, seed)

    expected_cols = list(FEATURE_COLS) + [TARGET_REG, TARGET_CLF]
    assert list(df.columns) == expected_cols, list(df.columns)
    assert df.shape == (n, len(expected_cols))
    assert int(df.isna().sum().sum()) == 0

    assert pd.api.types.is_float_dtype(df["driver_age"])
    assert pd.api.types.is_float_dtype(df["vehicle_age"])
    assert pd.api.types.is_float_dtype(df["mileage"])
    assert pd.api.types.is_float_dtype(df["credit_score"])
    assert pd.api.types.is_float_dtype(df["annual_miles"])
    assert pd.api.types.is_float_dtype(df["night_driving"])
    assert pd.api.types.is_float_dtype(df["y"])
    assert pd.api.types.is_integer_dtype(df["prior_claims"])
    assert pd.api.types.is_integer_dtype(df["multi_policy"])
    assert pd.api.types.is_integer_dtype(df["deductable"])
    assert pd.api.types.is_integer_dtype(df["y_class"])
    assert str(df["territory"].dtype) == "string"
    assert str(df["vehicle_type"].dtype) == "string"

    assert set(df["territory"].unique()) <= set(TERRITORY_LEVELS)
    assert df["territory"].nunique() == 3
    assert set(df["vehicle_type"].unique()) <= set(VEHICLE_TYPE_LEVELS)
    assert df["vehicle_type"].nunique() == 4
    assert set(df["multi_policy"].unique()) <= {0, 1}
    assert set(df["y_class"].unique()) <= {0, 1}
    assert set(df["deductable"].unique()) <= set(DEDUCTABLE_LEVELS)

    age = df["driver_age"].to_numpy(dtype=float)
    assert float(age.min()) >= 15.5
    assert float(age.max()) <= 91.0
    night = df["night_driving"].to_numpy(dtype=float)
    assert float(night.min()) >= 0.0
    assert float(night.max()) <= 1.0
    assert float(df["credit_score"].min()) >= 300.0
    assert float(df["credit_score"].max()) <= 850.0
    assert int(df["prior_claims"].min()) >= 0

    y = df["y"].to_numpy(dtype=float)
    assert np.all(np.isfinite(y))
    assert float(y.min()) > 0.0
    median_y = float(np.median(y))
    mean_y = float(np.mean(y))
    p95 = float(np.quantile(y, 0.95))
    p99 = float(np.quantile(y, 0.99))
    # Heavy tail: mean pulled by rares; upper quantiles ≫ median.
    assert mean_y > 1.6 * median_y, f"mean/median={mean_y / median_y:.3f}"
    assert p95 / median_y > 3.0, f"p95/median={p95 / median_y:.3f}"
    assert p99 / median_y > 6.0, f"p99/median={p99 / median_y:.3f}"
    log_y = np.log(y)
    # Excess skew on the raw scale (body is already lognormal).
    centered = y - mean_y
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    skew = m3 / (m2**1.5 + 1e-12)
    assert skew > 1.5, f"skew(y)={skew:.3f}"

    # Territory × mileage: urban mileage slope exceeds rural on log y.
    miles = df["mileage"].to_numpy(dtype=float)
    high_m = miles >= np.median(miles)
    terr = df["territory"].to_numpy()
    urban_hi = _cell_mean_logy(df, (terr == "urban") & high_m)
    urban_lo = _cell_mean_logy(df, (terr == "urban") & ~high_m)
    rural_hi = _cell_mean_logy(df, (terr == "rural") & high_m)
    rural_lo = _cell_mean_logy(df, (terr == "rural") & ~high_m)
    urban_delta = urban_hi - urban_lo
    rural_delta = rural_hi - rural_lo
    assert np.isfinite(urban_delta) and np.isfinite(rural_delta)
    assert urban_delta > rural_delta + 0.12, (
        f"territory×mileage contrast urban={urban_delta:.3f} rural={rural_delta:.3f}"
    )
    # Direct Pearson on the urban slice should also beat rural.
    r_urban = _pearson(np.log1p(miles[terr == "urban"]), log_y[terr == "urban"])
    r_rural = _pearson(np.log1p(miles[terr == "rural"]), log_y[terr == "rural"])
    assert r_urban > r_rural + 0.05, f"corr urban={r_urban:.3f} rural={r_rural:.3f}"

    # Young drivers + night_driving elevate y (interaction, not just mains).
    is_young = age < _YOUNG_AGE_CUT
    high_n = night >= np.median(night)
    young_hi = _cell_mean_logy(df, is_young & high_n)
    young_lo = _cell_mean_logy(df, is_young & ~high_n)
    old_hi = _cell_mean_logy(df, ~is_young & high_n)
    old_lo = _cell_mean_logy(df, ~is_young & ~high_n)
    young_delta = young_hi - young_lo
    old_delta = old_hi - old_lo
    assert np.isfinite(young_delta) and np.isfinite(old_delta)
    assert young_delta > old_delta + 0.12, (
        f"young×night contrast young={young_delta:.3f} old={old_delta:.3f}"
    )
    yn_mean = float(np.mean(y[is_young & high_n]))
    other_mean = float(np.mean(y[~(is_young & high_n)]))
    assert yn_mean > other_mean, f"young-night mean y={yn_mean:.1f} vs {other_mean:.1f}"

    # y_class prevalence in the requested band; not a y-quantile cut.
    yc = df["y_class"].to_numpy(dtype=float)
    prev = float(np.mean(yc))
    assert 0.15 <= prev <= 0.25, f"y_class prevalence={prev:.3f}"
    cut = (y >= np.quantile(y, 1.0 - prev)).astype(np.float64)
    assert not np.array_equal(yc, cut)
    agree = float(np.mean(yc == cut))
    assert agree < 0.90, f"y_class too close to a y-quantile (agree={agree:.3f})"
    r_yc = _pearson(log_y, yc)
    assert r_yc > 0.05, f"corr(log y, y_class)={r_yc:.3f}"

    # Seed stability / sensitivity.
    a = generate(90, 3)
    b = generate(90, 3)
    pd.testing.assert_frame_equal(a, b)
    c = generate(90, 4)
    assert not a.equals(c)

    try:
        generate(0, 0)
        raise AssertionError("n=0 should raise")
    except ValueError:
        pass

    assert SPEC.name == "insurance"
    assert SPEC.target_reg == "y"
    assert SPEC.target_clf == "y_class"
    assert SPEC.feature_cols == FEATURE_COLS
    sampled = SPEC.sample(40, seed=3)
    assert len(sampled) == 40
    assert "y" in sampled.columns
    assert "y_class" in sampled.columns

    summary = {
        "n": int(n),
        "n_cols": int(df.shape[1]),
        "mean_y": mean_y,
        "median_y": median_y,
        "p95_over_median": p95 / median_y,
        "p99_over_median": p99 / median_y,
        "skew_y": skew,
        "urban_miles_delta": urban_delta,
        "rural_miles_delta": rural_delta,
        "young_night_delta": young_delta,
        "old_night_delta": old_delta,
        "y_class_rate": prev,
        "corr_logy_yclass": r_yc,
        "passed": True,
    }
    return summary


if __name__ == "__main__":
    out = _self_test()
    print("insurance self-test ok")
    for key, val in out.items():
        if isinstance(val, float):
            print(f"  {key}={val:.4f}")
        else:
            print(f"  {key}={val}")


__all__ = [
    "FEATURE_COLS",
    "TARGET_REG",
    "TARGET_CLF",
    "TERRITORY_LEVELS",
    "VEHICLE_TYPE_LEVELS",
    "DEDUCTABLE_LEVELS",
    "SPEC",
    "generate",
    "get_ground_truth_description",
]
