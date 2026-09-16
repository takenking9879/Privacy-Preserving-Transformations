"""Original customer / credit-risk data-generating process (DGP).

The observed table is a mixed-type customer panel with a continuous
expected-loss target ``y`` and a binary high-loss flag ``y_class``.
Two latent variables (never returned) induce the dependence a
synthesizer is meant to recover:

* ``Z`` — continuous financial-stress confounder, ``Z ~ N(0, 1)``.
* ``H`` — rare distressed cluster, ``H ~ Bernoulli(0.04)``.

See ``data/schema.md`` and :func:`get_ground_truth_description`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    "age",
    "income",
    "tenure_months",
    "region",
    "segment",
    "risk_score",
    "num_products",
    "is_premium",
    "usage",
    "engagement",
    "complaint_count",
    "credit_util",
]
TARGET_REG: list[str] = ["y"]
TARGET_CLF: list[str] = ["y_class"]

REGION_LEVELS: tuple[str, ...] = ("Northeast", "Midwest", "South", "West")
SEGMENT_LEVELS: tuple[str, ...] = ("Mass", "Affluent", "Private")

# High-loss class is a logistic of a structural risk index (not a y-quantile).
# Documented in schema.md / get_ground_truth_description().
_Y_CLASS_INTERCEPT = -2.35
_MAR_INCOME_BASE = -3.35
_MAR_ENGAGE_BASE = -3.55


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _negbin(rng: np.random.Generator, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Overdispersed counts: ``Var = mu + alpha * mu^2`` (alpha > 0)."""
    mu = np.clip(mu, 1e-8, None)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return rng.negative_binomial(n, p)


def generate_original(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Simulate ``n`` customers from the ground-truth credit-risk DGP.

    Parameters
    ----------
    n:
        Number of rows. Defaults to 4000.
    seed:
        RNG seed. Defaults to 7.

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS + TARGET_REG + TARGET_CLF``. ``income`` and
        ``engagement`` contain MAR NaNs (impute-ready). Latent ``Z`` / ``H``
        are not returned.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Latent structure (not exported)
    # ------------------------------------------------------------------
    z = rng.normal(0.0, 1.0, size=n)
    h = rng.binomial(1, 0.04, size=n).astype(np.float64)
    # Wealth propensity: mildly anti-correlated with stress.
    wealth = 0.75 * rng.normal(0.0, 1.0, size=n) - 0.25 * z

    # ------------------------------------------------------------------
    # Demographics
    # ------------------------------------------------------------------
    age = np.clip(rng.normal(42.0, 12.5, size=n), 18.0, 79.0)

    # Segment: 3 levels, unequal, wealth- and age-tilted.
    # Target marginals roughly Mass ~ 0.60, Affluent ~ 0.28, Private ~ 0.12.
    seg_logits = np.column_stack(
        [
            1.25 - 0.95 * wealth,
            0.05 + 0.35 * wealth,
            -1.25 + 1.15 * wealth + 0.018 * (age - 40.0),
        ]
    )
    seg_p = _softmax_rows(seg_logits)
    segment_idx = np.array(
        [rng.choice(3, p=seg_p[i]) for i in range(n)], dtype=np.int64
    )
    is_mass = (segment_idx == 0).astype(np.float64)
    is_affluent = (segment_idx == 1).astype(np.float64)
    is_private = (segment_idx == 2).astype(np.float64)
    segment = np.array(SEGMENT_LEVELS, dtype=object)[segment_idx]

    # Region: 4 levels, unequal. Stress slightly shifts West / South.
    # Target marginals ~ NE 0.34, MW 0.24, South 0.30, West 0.12.
    west_boost = 0.28 * z + 0.35 * h
    south_boost = 0.12 * z
    reg_logits = np.column_stack(
        [
            np.full(n, 0.55),
            np.full(n, 0.18),
            0.42 + south_boost,
            -0.55 + west_boost,
        ]
    )
    reg_p = _softmax_rows(reg_logits)
    region_idx = np.array(
        [rng.choice(4, p=reg_p[i]) for i in range(n)], dtype=np.int64
    )
    region = np.array(REGION_LEVELS, dtype=object)[region_idx]

    # ------------------------------------------------------------------
    # Continuous / mixed features
    # ------------------------------------------------------------------
    log_income = (
        10.38
        + 0.52 * wealth
        + 0.38 * is_private
        + 0.16 * is_affluent
        + 0.011 * (age - 40.0)
        - 0.16 * z
        - 0.58 * h
        + rng.normal(0.0, 0.36, size=n)
    )
    income = np.clip(np.exp(log_income), 14_000.0, 420_000.0)

    tenure_mean = np.clip(
        10.0
        + 0.85 * (age - 18.0)
        + 8.0 * is_private
        + 3.5 * is_affluent
        - 6.0 * h
        + 1.5 * wealth,
        2.0,
        300.0,
    )
    tenure_months = np.clip(
        np.rint(rng.gamma(shape=5.0, scale=tenure_mean / 5.0)), 0.0, 360.0
    ).astype(np.int64)

    prem_lp = (
        -2.85
        + 2.25 * is_private
        + 1.05 * is_affluent
        + 0.50 * wealth
        + 0.28 * (log_income - 10.4)
        - 0.45 * h
    )
    is_premium = rng.binomial(1, _sigmoid(prem_lp), size=n).astype(np.int64)

    # Spend-like usage intensity (right-skewed, roughly monthly volume / 1e3).
    log_usage = (
        2.75
        + 0.32 * (log_income - 10.4)
        + 0.28 * is_premium
        + 0.12 * z
        + 0.38 * wealth
        + 0.15 * is_private
        + rng.normal(0.0, 0.50, size=n)
    )
    usage = np.clip(np.exp(log_usage), 0.15, 180.0)

    # Utilization in (0, 1) via logit-normal + distressed shift.
    util_lp = (
        -0.55
        + 0.52 * z
        + 1.55 * h
        - 0.22 * (log_income - 10.4)
        + 0.10 * ((age - 40.0) / 20.0)
        - 0.18 * is_premium
        + rng.normal(0.0, 0.62, size=n)
    )
    credit_util = np.clip(_sigmoid(util_lp), 0.01, 0.99)

    # Near-normal risk score on [5, 99], shifted by Z, H, util.
    risk_score = (
        47.5
        + 7.5 * z
        + 21.0 * h
        + 16.0 * credit_util
        + 0.10 * (age - 40.0)
        - 3.8 * is_premium
        - 2.6 * wealth
        + rng.normal(0.0, 6.8, size=n)
    )
    risk_score = np.clip(risk_score, 5.0, 99.0)

    # Near-normal engagement score on [0, 100].
    engagement = (
        61.0
        - 5.5 * z
        - 15.0 * h
        + 7.5 * is_premium
        + 0.055 * tenure_months
        + 2.4 * (log_income - 10.4)
        + rng.normal(0.0, 9.5, size=n)
    )
    engagement = np.clip(engagement, 0.0, 100.0)

    prod_lam = np.clip(
        1.05
        + 0.95 * is_premium
        + 0.0075 * tenure_months
        + 0.75 * is_private
        + 0.32 * is_affluent
        + 0.018 * usage
        - 0.25 * h,
        0.15,
        12.0,
    )
    num_products = np.clip(rng.poisson(prod_lam), 0, 14).astype(np.int64)

    # Overdispersed complaints (NegBin).
    cmp_mu = np.clip(
        0.22
        + 0.32 * np.maximum(z, 0.0)
        + 1.90 * h
        + 0.95 * credit_util
        + 0.016 * np.maximum(risk_score - 60.0, 0.0)
        + 0.08 * is_mass,
        0.02,
        18.0,
    )
    complaint_count = np.clip(_negbin(rng, cmp_mu, alpha=1.15), 0, 30).astype(
        np.int64
    )

    # ------------------------------------------------------------------
    # Target y = expected_loss (USD, right-skewed)
    # Linear terms + interactions + threshold + segment intercepts
    # + latent effects + heteroscedastic log-normal noise.
    # ------------------------------------------------------------------
    risk_excess = np.maximum(risk_score - 68.0, 0.0)
    log_inc_c = log_income - 10.4
    usage_c = usage / 15.0

    eta = (
        3.55
        + 0.18 * is_affluent
        + 0.40 * is_private
        + 1.10 * credit_util
        + 0.016 * risk_score
        + 0.07 * log_inc_c
        + 0.085 * complaint_count
        - 0.14 * is_premium
        + 0.58 * credit_util * is_premium
        + 0.14 * log_inc_c * usage_c
        - 0.075 * np.sqrt(np.maximum(tenure_months, 0.0))
        + 0.52 * (risk_excess / 10.0)
        + 0.26 * z
        + 0.80 * h
    )
    sigma = 0.26 * (
        1.0
        + 0.95 * credit_util
        + 0.75 * h
        + 0.20 * (risk_score > 68.0).astype(np.float64)
    )
    y = np.exp(eta + sigma * rng.normal(0.0, 1.0, size=n))
    y = np.clip(y, 1.0, 25_000.0)

    # y_class: Bernoulli of a logistic risk index (shares Z, H, util, risk).
    risk_index = (
        _Y_CLASS_INTERCEPT
        + 2.05 * credit_util
        + 0.032 * (risk_score - 50.0)
        + 1.15 * h
        + 0.38 * z
        + 0.14 * complaint_count
        - 0.28 * is_premium
        + 0.45 * (risk_score > 68.0).astype(np.float64)
    )
    y_class = rng.binomial(1, _sigmoid(risk_index), size=n).astype(np.int64)

    # ------------------------------------------------------------------
    # MAR missingness (impute-ready NaNs)
    # income  ~ more missing for Mass + higher risk_score
    # engagement ~ more missing as complaint_count rises
    # ------------------------------------------------------------------
    p_miss_income = _sigmoid(
        _MAR_INCOME_BASE + 0.85 * is_mass + 0.018 * (risk_score - 50.0)
    )
    p_miss_engage = _sigmoid(_MAR_ENGAGE_BASE + 0.38 * complaint_count)
    miss_income = rng.random(n) < p_miss_income
    miss_engage = rng.random(n) < p_miss_engage

    income_obs = income.copy()
    income_obs[miss_income] = np.nan
    engagement_obs = engagement.copy()
    engagement_obs[miss_engage] = np.nan

    frame = pd.DataFrame(
        {
            "age": age.astype(np.float64),
            "income": income_obs.astype(np.float64),
            "tenure_months": tenure_months.astype(np.int64),
            "region": pd.Series(region, dtype="string"),
            "segment": pd.Series(segment, dtype="string"),
            "risk_score": risk_score.astype(np.float64),
            "num_products": num_products.astype(np.int64),
            "is_premium": is_premium.astype(np.int64),
            "usage": usage.astype(np.float64),
            "engagement": engagement_obs.astype(np.float64),
            "complaint_count": complaint_count.astype(np.int64),
            "credit_util": credit_util.astype(np.float64),
            "y": y.astype(np.float64),
            "y_class": y_class.astype(np.int64),
        }
    )
    return frame


def get_ground_truth_description() -> str:
    """Return a human-readable summary of the latent DGP and target equations."""
    return (
        "Original DGP: mixed-type customer credit-risk table. "
        "Latent Z ~ N(0,1) (financial stress) and rare cluster H ~ Bern(0.04) "
        "are never observed but shift income (−), credit_util (+), risk_score (+), "
        "complaint_count (+), engagement (−), and y (+), inducing residual "
        "correlation among those columns. "
        "Wealth propensity (anti-correlated with Z) drives segment, income, "
        "usage, and is_premium. "
        "age ~ truncated N(42, 12.5) on [18, 79] (near-normal). "
        "income = exp(10.38 + 0.52·wealth + segment shifts + 0.011·(age−40) "
        "− 0.16·Z − 0.58·H + N(0, 0.36)), clipped, so right-skewed. "
        "tenure_months ~ rounded Gamma(5, mean/5) with mean rising in age and "
        "Private/Affluent. "
        "region ∈ {Northeast, Midwest, South, West} with unequal frequencies "
        "(≈34/24/30/12) and a mild Z/H tilt toward West/South. "
        "segment ∈ {Mass, Affluent, Private} ≈ 60/28/12 via softmax(wealth, age). "
        "is_premium ~ Bern(σ(−2.85 + 2.25·Private + 1.05·Affluent + 0.50·wealth "
        "+ 0.28·(log income−10.4) − 0.45·H)). "
        "usage = exp(2.75 + 0.32·log-income-c + 0.28·is_premium + 0.12·Z "
        "+ 0.38·wealth + …) (skewed spend-like). "
        "credit_util = σ(−0.55 + 0.52·Z + 1.55·H − 0.22·log-income-c + …) "
        "clipped to [0.01, 0.99]. "
        "risk_score ≈ 47.5 + 7.5·Z + 21·H + 16·credit_util − 3.8·is_premium "
        "+ N(0, 6.8), clipped [5, 99] (near-normal mixture). "
        "engagement ≈ 61 − 5.5·Z − 15·H + 7.5·is_premium + 0.055·tenure + … "
        "(near-normal, clipped [0, 100]). "
        "num_products ~ Poisson(1.05 + 0.95·is_premium + 0.0075·tenure + "
        "segment shifts + 0.018·usage). "
        "complaint_count ~ NegBin(mean = 0.22 + 0.32·max(Z,0) + 1.90·H + "
        "0.95·credit_util + 0.016·max(risk_score−60, 0), α=1.15). "
        "TARGET y = expected_loss (USD) = exp(η + σ·ε), ε~N(0,1), with "
        "η = 3.55 + 0.18·Affluent + 0.40·Private + 1.10·credit_util + "
        "0.016·risk_score + 0.07·log-income-c + 0.085·complaints − 0.14·is_premium "
        "+ 0.58·(credit_util·is_premium) + 0.14·(log-income-c)·(usage/15) "
        "− 0.075·sqrt(tenure) + 0.52·max(risk_score−68, 0)/10 + 0.26·Z + 0.80·H; "
        "heteroscedastic σ = 0.26·(1 + 0.95·credit_util + 0.75·H + 0.20·I(risk>68)). "
        "y_class ~ Bern(σ(risk_index)) where risk_index = −2.35 + 2.05·credit_util "
        "+ 0.032·(risk_score−50) + 1.15·H + 0.38·Z + 0.14·complaints "
        "− 0.28·is_premium + 0.45·I(risk_score>68) — a logistic of a risk index, "
        "not a quantile cut on y. "
        "MAR missingness: P(income missing) = σ(−3.35 + 0.85·Mass + "
        "0.018·(risk_score−50)); P(engagement missing) = σ(−3.55 + "
        "0.38·complaint_count). NaNs are left in place (impute-ready); "
        "complete-case analysis remains valid because missingness is moderate "
        "and fully observed columns identify the MAR mechanism. "
        "A good synthesizer MUST recover: (1) Z-induced residual correlation "
        "among income, credit_util, risk_score, complaints, engagement, y; "
        "(2) the ~4% jointly-extreme distressed cluster; (3) unequal region/"
        "segment margins and segment-specific y intercepts; (4) interactions "
        "credit_util×is_premium and log(income)×usage in E[y]; (5) the "
        "hockey-stick threshold on risk_score>68; (6) heteroscedastic Var(y); "
        "(7) Poisson/NegBin count margins; (8) [0,1] util and skewed income/"
        "usage/y; (9) MAR missingness patterns; (10) y_class as a noisy "
        "logistic of the shared risk index rather than a hard cut on y."
    )


__all__ = [
    "FEATURE_COLS",
    "TARGET_REG",
    "TARGET_CLF",
    "REGION_LEVELS",
    "SEGMENT_LEVELS",
    "generate_original",
    "get_ground_truth_description",
]
