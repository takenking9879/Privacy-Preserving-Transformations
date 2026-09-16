"""E-commerce RFM customer data-generating process.

Observed table: mixed-type shoppers with a continuous next-90-day spend
target ``y`` and a binary repeat-purchaser flag ``y_class``.  Latent
loyalty / value scores are never returned; they induce the RFM joints a
synthesizer is meant to recover.

Required structure
------------------
* Frequency and monetary are positively associated (shared latents plus
  a direct ``log(frequency)`` path into monetary).
* Recency is negatively associated with ``y`` (lapsed shoppers spend less
  in the next 90 days).
* Gold-tier subscribers receive an extra interaction boost on ``y``.
* ``returns`` is zero-inflated (structural zero + Poisson).
* ``y_class`` is a logistic repeat-purchase draw, **not** a quantile of
  ``y``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "recency_days",
    "frequency",
    "monetary",
    "discount_rate",
    "channel",
    "tier",
    "items_last_month",
    "is_subscriber",
    "session_time",
    "cart_adds",
    "returns",
    "region",
)
TARGET_REG: str = "y"
TARGET_CLF: str = "y_class"

CHANNEL_LEVELS: tuple[str, ...] = ("web", "app", "store")
TIER_LEVELS: tuple[str, ...] = ("bronze", "silver", "gold")
REGION_LEVELS: tuple[str, ...] = ("Northeast", "Midwest", "South", "West")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Draw one category per row from an ``(n, k)`` probability matrix."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0])
    return (u[:, None] < cdf).argmax(axis=1)


def generate(n: int, seed: int) -> pd.DataFrame:
    """Simulate ``n`` e-commerce customers.

    Parameters
    ----------
    n:
        Number of rows. Must be a positive integer.
    seed:
        RNG seed (deterministic given ``n``).

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS`` plus ``y`` and ``y_class``. Latent
        loyalty / value scores are not returned.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Latents (not exported)
    # ------------------------------------------------------------------
    loyalty = rng.normal(0.0, 1.0, size=n)
    # Value propensity: corr(loyalty, value) ≈ 0.55.
    value = 0.55 * loyalty + 0.835 * rng.normal(0.0, 1.0, size=n)
    digital = rng.normal(0.0, 1.0, size=n)

    # ------------------------------------------------------------------
    # RFM block
    # ------------------------------------------------------------------
    # Recency (days since last order): high loyalty → more recent.
    log_recency = 3.35 - 0.62 * loyalty - 0.18 * value + rng.normal(0.0, 0.48, size=n)
    recency_days = np.clip(np.exp(log_recency), 1.0, 400.0)

    # Frequency (orders in the lookback window).
    lam_freq = np.clip(
        np.exp(1.05 + 0.52 * loyalty + 0.22 * value),
        0.25,
        36.0,
    )
    frequency = np.clip(rng.poisson(lam_freq), 0, 80).astype(np.int64)

    # Monetary (historical spend, right-skewed). Direct log-frequency path
    # plus shared latents → positive frequency–monetary association.
    log_monetary = (
        4.55
        + 0.46 * np.log1p(frequency.astype(np.float64))
        + 0.50 * value
        + 0.16 * loyalty
        + rng.normal(0.0, 0.40, size=n)
    )
    monetary = np.clip(np.exp(log_monetary), 6.0, 28_000.0)

    # ------------------------------------------------------------------
    # Channel, tier, region
    # ------------------------------------------------------------------
    chan_logits = np.column_stack(
        [
            0.35 + 0.18 * digital,  # web
            0.05 + 0.58 * digital + 0.20 * loyalty,  # app
            0.28 - 0.52 * digital + 0.10 * recency_days / 60.0,  # store
        ]
    )
    channel_idx = _categorical(rng, _softmax_rows(chan_logits))
    channel = np.array(CHANNEL_LEVELS, dtype=object)[channel_idx]
    is_web = (channel_idx == 0).astype(np.float64)
    is_app = (channel_idx == 1).astype(np.float64)
    is_store = (channel_idx == 2).astype(np.float64)

    log_freq = np.log1p(frequency.astype(np.float64))
    tier_logits = np.column_stack(
        [
            1.05 - 0.80 * value - 0.22 * log_freq,  # bronze
            0.12 + 0.12 * value + 0.06 * log_freq,  # silver
            -1.40 + 0.95 * value + 0.38 * log_freq,  # gold
        ]
    )
    tier_idx = _categorical(rng, _softmax_rows(tier_logits))
    tier = np.array(TIER_LEVELS, dtype=object)[tier_idx]
    is_bronze = (tier_idx == 0).astype(np.float64)
    is_silver = (tier_idx == 1).astype(np.float64)
    is_gold = (tier_idx == 2).astype(np.float64)

    # Unequal regions; mild loyalty / digital tilt.
    reg_logits = np.column_stack(
        [
            np.full(n, 0.50),
            np.full(n, 0.12),
            0.38 + 0.10 * loyalty,
            -0.48 + 0.22 * digital,
        ]
    )
    region_idx = _categorical(rng, _softmax_rows(reg_logits))
    region = np.array(REGION_LEVELS, dtype=object)[region_idx]
    is_west = (region_idx == 3).astype(np.float64)
    is_south = (region_idx == 2).astype(np.float64)

    # ------------------------------------------------------------------
    # Subscriber, discount, engagement counts
    # ------------------------------------------------------------------
    p_sub = _sigmoid(
        -1.65
        + 1.45 * is_gold
        + 0.55 * is_silver
        + 0.52 * loyalty
        + 0.20 * value
        + 0.18 * is_app
    )
    is_subscriber = rng.binomial(1, p_sub, size=n).astype(np.int64)
    sub = is_subscriber.astype(np.float64)

    discount_rate = np.clip(
        _sigmoid(
            -1.05
            + 0.90 * sub
            - 0.50 * is_gold
            + 0.28 * is_bronze
            + rng.normal(0.0, 0.55, size=n)
        ),
        0.0,
        1.0,
    )

    # Items bought in the last 30 days: recency gates activity.
    recency_gate = _sigmoid(1.85 - 0.085 * recency_days)
    lam_items = np.clip(
        recency_gate * (0.55 + 0.20 * frequency.astype(np.float64) + 0.55 * sub),
        0.04,
        28.0,
    )
    items_last_month = np.clip(rng.poisson(lam_items), 0, 40).astype(np.int64)

    log_session = (
        2.35
        + 0.26 * digital
        + 0.28 * is_app
        - 0.12 * is_store
        + 0.14 * loyalty
        + rng.normal(0.0, 0.42, size=n)
    )
    session_time = np.clip(np.exp(log_session), 0.35, 180.0)

    lam_cart = np.clip(
        0.85
        + 0.045 * session_time
        + 0.10 * frequency.astype(np.float64)
        + 0.45 * is_app
        + 0.18 * recency_gate,
        0.15,
        32.0,
    )
    cart_adds = np.clip(rng.poisson(lam_cart), 0, 45).astype(np.int64)

    # Zero-inflated returns: structural zero (never-returners) + Poisson.
    pi_zero = _sigmoid(
        0.55
        - 0.30 * log_freq
        - 0.42 * is_gold
        + 0.22 * discount_rate
        - 0.16 * sub
    )
    structural_zero = rng.random(n) < pi_zero
    lam_ret = np.clip(
        0.55 + 0.14 * frequency.astype(np.float64) + 0.55 * discount_rate,
        0.08,
        16.0,
    )
    returns = np.where(
        structural_zero,
        0,
        rng.poisson(lam_ret),
    )
    returns = np.clip(returns, 0, 22).astype(np.int64)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    # y = next-90-day spend (right-skewed). Recency enters negatively.
    # Gold × subscriber is the named interaction a synthesizer must keep.
    eta = (
        4.05
        + 0.40 * log_freq
        + 0.24 * (np.log(monetary) - 5.1)
        - 0.52 * np.log1p(recency_days)
        + 0.16 * is_silver
        + 0.30 * is_gold
        + 0.22 * sub
        + 0.88 * is_gold * sub
        + 0.055 * np.log1p(items_last_month.astype(np.float64))
        + 0.035 * (session_time / 18.0)
        + 0.028 * np.log1p(cart_adds.astype(np.float64))
        - 0.38 * discount_rate
        - 0.045 * returns.astype(np.float64)
        + 0.08 * is_app
        - 0.05 * is_store
        + 0.06 * is_west
        - 0.03 * is_south
        + 0.10 * loyalty
        + 0.08 * value
    )
    sigma = 0.26 * (1.0 + 0.22 * discount_rate)
    y = np.clip(np.exp(eta + sigma * rng.normal(0.0, 1.0, size=n)), 0.40, 22_000.0)

    # y_class = repeat purchaser. Logistic of a structural propensity,
    # sharing some RFM drivers with y but not a cutoff of y.
    repeat_index = (
        -0.20
        - 0.020 * recency_days
        + 0.16 * log_freq
        + 0.95 * sub
        + 0.42 * is_gold
        + 0.58 * (items_last_month > 0).astype(np.float64)
        + 0.055 * cart_adds.astype(np.float64)
        + 0.16 * is_app
        + 0.12 * is_web
        + 0.28 * loyalty
    )
    y_class = rng.binomial(1, _sigmoid(repeat_index), size=n).astype(np.int64)

    recency_out = np.rint(recency_days).astype(np.int64)
    recency_out = np.clip(recency_out, 1, 400)

    frame = pd.DataFrame(
        {
            "recency_days": recency_out,
            "frequency": frequency,
            "monetary": monetary.astype(np.float64),
            "discount_rate": discount_rate.astype(np.float64),
            "channel": channel,
            "tier": tier,
            "items_last_month": items_last_month,
            "is_subscriber": is_subscriber,
            "session_time": session_time.astype(np.float64),
            "cart_adds": cart_adds,
            "returns": returns,
            "region": region,
            "y": y.astype(np.float64),
            "y_class": y_class,
        }
    )
    return frame


def get_ground_truth_description() -> str:
    """Human-readable ground truth for reports / diagnostics."""
    return (
        "Retail RFM DGP: frequency–monetary positively linked; recency "
        "negatively linked to next-90d spend y; gold×subscriber interaction "
        "on y; zero-inflated returns; y_class is a logistic repeat-purchase "
        "draw (not a quantile of y)."
    )


SPEC = DatasetSpec(
    name="retail",
    generate=generate,
    description=get_ground_truth_description(),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=("retail", "rfm", "mixed", "zero-inflated", "interaction"),
)
