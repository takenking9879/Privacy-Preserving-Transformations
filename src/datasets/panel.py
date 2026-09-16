"""Flattened customer-month panel DGP (no entity id).

Each row is one customer-month snapshot.  There is no ``customer_id``;
serial structure lives in the observed features ``tenure`` and
``customer_segment`` plus the lagged spend / churn-risk columns.

The continuous target is next-month spend:

    y = ρ · lag_spend  +  is_promo × plan-specific lift  +  extras + noise

A general synthesizer must recover the AR-like pair ``(lag_spend, y)``
and the ``is_promo × plan`` interaction from a mixed-type table that
looks i.i.d. once the id is stripped.

See :data:`SPEC`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "customer_segment",
    "tenure",
    "lag_spend",
    "lag_churn_risk",
    "plan",
    "tickets",
    "nps",
    "usage_trend",
    "is_promo",
    "region",
)
TARGET_REG: str = "y"
TARGET_CLF: str = "y_class"

PLAN_LEVELS: tuple[str, ...] = ("basic", "plus", "pro")
REGION_LEVELS: tuple[str, ...] = ("Northeast", "Midwest", "South", "West")
SEGMENT_LEVELS: tuple[str, ...] = ("consumer", "smb", "enterprise")

# Autoregressive coefficient on lag_spend.  Sized so corr(lag_spend, y) > 0.4
# even after promo × plan shocks and heteroscedastic noise.
_AR_RHO = 0.85

# Promo lift by plan (the interaction a copula can miss).
_PROMO_LIFT = np.array([8.0, 28.0, 55.0], dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _gumbel_categorical(rng: np.random.Generator, logits: np.ndarray) -> np.ndarray:
    """Vectorized categorical draw via Gumbel-max."""
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=logits.shape)
    gumbel = -np.log(-np.log(u))
    return np.argmax(logits + gumbel, axis=1).astype(np.int64)


def generate(n: int, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` flattened customer-month rows.

    Parameters
    ----------
    n:
        Number of rows (customer-months).
    seed:
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS + (y, y_class)``.  Latent value /
        dissatisfaction scores are not returned.  No entity id is
        written — AR structure is only in ``lag_spend`` → ``y``.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Latent customer traits (not exported)
    # ------------------------------------------------------------------
    value = rng.normal(0.0, 1.0, size=n)
    dissat = 0.70 * rng.normal(0.0, 1.0, size=n) - 0.30 * value

    # ------------------------------------------------------------------
    # Persistent / serial features (stand in for a panel id)
    # ------------------------------------------------------------------
    # customer_segment: consumer / smb / enterprise, value-tilted.
    # Target-ish marginals: consumer ~0.58, smb ~0.30, enterprise ~0.12.
    seg_logits = np.column_stack(
        [
            1.15 - 0.85 * value,
            0.05 + 0.35 * value,
            -1.35 + 1.20 * value,
        ]
    )
    segment_idx = _gumbel_categorical(rng, seg_logits)
    is_consumer = (segment_idx == 0).astype(np.float64)
    is_smb = (segment_idx == 1).astype(np.float64)
    is_enterprise = (segment_idx == 2).astype(np.float64)
    customer_segment = np.asarray(SEGMENT_LEVELS, dtype=object)[segment_idx]

    # Region: 4 unequal levels.  Mild value tilt on West / South.
    # Approx Northeast 0.34, Midwest 0.24, South 0.28, West 0.14.
    reg_logits = np.column_stack(
        [
            np.full(n, 0.55),
            np.full(n, 0.18),
            0.38 + 0.10 * value,
            -0.50 + 0.22 * value,
        ]
    )
    region_idx = _gumbel_categorical(rng, reg_logits)
    region = np.asarray(REGION_LEVELS, dtype=object)[region_idx]

    # Tenure (months): longer for enterprise / high-value accounts.
    tenure_mean = np.clip(
        14.0 + 10.0 * is_smb + 22.0 * is_enterprise + 4.5 * value,
        3.0,
        90.0,
    )
    tenure = np.clip(
        np.rint(rng.gamma(shape=4.0, scale=tenure_mean / 4.0)),
        1.0,
        120.0,
    ).astype(np.int64)

    # Plan: basic / plus / pro.  Segment + value drive upgrades.
    plan_logits = np.column_stack(
        [
            1.05 - 0.70 * value - 0.55 * is_enterprise + 0.25 * is_consumer,
            0.15 + 0.20 * value + 0.35 * is_smb,
            -1.20 + 0.95 * value + 1.10 * is_enterprise + 0.35 * is_smb,
        ]
    )
    plan_idx = _gumbel_categorical(rng, plan_logits)
    is_basic = (plan_idx == 0).astype(np.float64)
    is_plus = (plan_idx == 1).astype(np.float64)
    is_pro = (plan_idx == 2).astype(np.float64)
    plan = np.asarray(PLAN_LEVELS, dtype=object)[plan_idx]

    # ------------------------------------------------------------------
    # Lagged / month-t features
    # ------------------------------------------------------------------
    # Last-month spend (right-skewed).  Plan, segment, tenure, value.
    log_lag_spend = (
        3.15
        + 0.55 * is_plus
        + 1.05 * is_pro
        + 0.40 * is_smb
        + 0.85 * is_enterprise
        + 0.010 * tenure
        + 0.32 * value
        + rng.normal(0.0, 0.38, size=n)
    )
    lag_spend = np.clip(np.exp(log_lag_spend), 2.0, 2_500.0)

    usage_trend = np.clip(
        0.18 * value
        - 0.28 * dissat
        + 0.006 * (tenure.astype(np.float64) - 18.0)
        + rng.normal(0.0, 0.50, size=n),
        -2.5,
        2.5,
    )

    ticket_mu = np.clip(
        0.35
        + 0.85 * np.maximum(dissat, 0.0)
        + 0.20 * is_enterprise
        + 0.12 * is_pro
        - 0.08 * (tenure.astype(np.float64) / 24.0),
        0.05,
        10.0,
    )
    tickets = np.clip(rng.poisson(ticket_mu), 0, 20).astype(np.int64)

    nps = np.clip(
        7.1
        + 1.25 * value
        - 0.50 * tickets.astype(np.float64)
        - 1.05 * dissat
        + 0.015 * tenure.astype(np.float64)
        + rng.normal(0.0, 1.35, size=n),
        0.0,
        10.0,
    )

    lag_churn_lp = (
        -1.75
        + 0.48 * tickets.astype(np.float64)
        - 0.26 * nps
        - 0.42 * usage_trend
        - 0.018 * tenure.astype(np.float64)
        + 0.65 * dissat
        + rng.normal(0.0, 0.35, size=n)
    )
    lag_churn_risk = np.clip(_sigmoid(lag_churn_lp), 0.01, 0.99)

    # Promos tilt toward basic / plus; only a mild churn-risk target so the
    # structural promo × plan lift is not wiped out by selection.
    promo_lp = (
        -1.15
        + 0.70 * is_basic
        + 0.20 * is_plus
        - 0.35 * is_pro
        + 0.45 * lag_churn_risk
        - 0.10 * is_enterprise
    )
    is_promo = rng.binomial(1, _sigmoid(promo_lp), size=n).astype(np.int64)

    # ------------------------------------------------------------------
    # y = next-month spend: AR(1) on lag_spend + promo × plan
    # ------------------------------------------------------------------
    promo_lift = is_promo.astype(np.float64) * (
        _PROMO_LIFT[0] * is_basic
        + _PROMO_LIFT[1] * is_plus
        + _PROMO_LIFT[2] * is_pro
    )
    sigma = 7.5 + 0.12 * lag_spend
    y = (
        _AR_RHO * lag_spend
        + promo_lift
        + 5.5 * usage_trend
        + 0.10 * tenure.astype(np.float64)
        - 6.0 * lag_churn_risk
        + sigma * rng.normal(0.0, 1.0, size=n)
    )
    y = np.clip(y, 0.5, 3_500.0)

    # y_class: next-month attrition / spend-drop flag.
    # Logistic of shared risk drivers — not a quantile cut on y.
    class_lp = (
        -0.85
        + 2.25 * lag_churn_risk
        - 0.14 * nps
        + 0.12 * tickets.astype(np.float64)
        - 0.40 * usage_trend
        - 0.30 * is_promo.astype(np.float64)
        + 0.25 * is_basic
    )
    y_class = rng.binomial(1, _sigmoid(class_lp), size=n).astype(np.int64)

    return pd.DataFrame(
        {
            "customer_segment": pd.Series(customer_segment, dtype="string"),
            "tenure": tenure.astype(np.int64),
            "lag_spend": lag_spend.astype(np.float64),
            "lag_churn_risk": lag_churn_risk.astype(np.float64),
            "plan": pd.Series(plan, dtype="string"),
            "tickets": tickets.astype(np.int64),
            "nps": nps.astype(np.float64),
            "usage_trend": usage_trend.astype(np.float64),
            "is_promo": is_promo.astype(np.int64),
            "region": pd.Series(region, dtype="string"),
            "y": y.astype(np.float64),
            "y_class": y_class.astype(np.int64),
        }
    )


SPEC = DatasetSpec(
    name="panel",
    generate=generate,
    description=(
        "Flattened customer-month rows (no entity id).  Next-month spend y "
        "is autoregressive in lag_spend plus an is_promo × plan interaction. "
        "customer_segment and tenure carry serial structure in the features. "
        "Tests whether a general synthesizer keeps AR-like dependence without IDs."
    ),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=("panel", "autoregressive", "mixed", "interaction"),
)


def _self_test(n: int = 800, seed: int = 0, min_corr: float = 0.4) -> dict[str, float | int | bool]:
    """Assert corr(lag_spend, y) exceeds ``min_corr`` and return a summary."""
    df = SPEC.sample(n=n, seed=seed)
    lag = np.asarray(df["lag_spend"], dtype=float)
    y = np.asarray(df["y"], dtype=float)
    mask = np.isfinite(lag) & np.isfinite(y)
    corr = float(np.corrcoef(lag[mask], y[mask])[0, 1])
    if not np.isfinite(corr) or corr <= min_corr:
        raise AssertionError(
            f"corr(lag_spend, y)={corr:.4f} is not > {min_corr} (n={n}, seed={seed})"
        )
    return {
        "n": int(len(df)),
        "n_cols": int(df.shape[1]),
        "corr_lag_spend_y": corr,
        "y_class_rate": float(np.mean(df["y_class"])),
        "promo_rate": float(np.mean(df["is_promo"])),
        "passed": True,
    }


if __name__ == "__main__":
    summary = _self_test()
    print(summary)
