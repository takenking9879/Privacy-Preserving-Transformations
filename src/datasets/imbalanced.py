"""Fraud / rare-event DGP for minority-class synthesizer stress tests.

Observed table: mixed-type payment transactions with a continuous fraud
severity ``y`` and a rare binary fraud flag ``y_class`` (prevalence
about 4–8 %).  A small high-risk pocket — international + high velocity
+ low device trust — concentrates the minority class.  ``y`` is a
two-part outcome: almost zero for non-fraud, heavy-tailed for fraud.

A synthesizer MUST recover the rare joint (not just the marginal class
rate).  Oversampling ``y_class`` while breaking ``P(X | y_class=1)``
fails TSTR AUC on the minority class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "txn_amount",
    "txn_hour",
    "merchant_risk",
    "device_trust",
    "country",
    "channel",
    "is_international",
    "n_tx_day",
    "velocity",
    "account_age",
)
TARGET_REG = "y"
TARGET_CLF = "y_class"

COUNTRY_LEVELS: tuple[str, ...] = ("US", "GB", "DE", "BR", "NG")
# Unequal mix (~46 / 20 / 16 / 12 / 6).
_COUNTRY_LOGITS: tuple[float, ...] = (1.15, 0.32, 0.10, -0.18, -0.88)

CHANNEL_LEVELS: tuple[str, ...] = ("card", "web", "mobile")

# Observed pocket definition (fixed thresholds, not sample quantiles).
POCKET_VELOCITY_MIN = 5.50
POCKET_DEVICE_TRUST_MAX = 0.42

# Latent pocket seed rate.  Seeded rows are forced into the observed
# international + high-velocity + low-trust region.
_POCKET_SEED_P = 0.030

_NIGHT_HOURS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 21, 22, 23)
_NIGHT_PROBS: tuple[float, ...] = (
    0.08,
    0.10,
    0.14,
    0.12,
    0.10,
    0.08,
    0.07,
    0.09,
    0.12,
    0.10,
)


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


def in_high_risk_pocket(
    is_international: np.ndarray,
    velocity: np.ndarray,
    device_trust: np.ndarray,
) -> np.ndarray:
    """Return the observed high-risk pocket mask (international × velocity × trust)."""
    return (
        (is_international.astype(np.float64) > 0.5)
        & (velocity >= POCKET_VELOCITY_MIN)
        & (device_trust <= POCKET_DEVICE_TRUST_MAX)
    )


def generate(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` payment transactions from the rare-fraud DGP.

    Parameters
    ----------
    n:
        Number of rows. Must be a positive integer.
    seed:
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS`` plus ``y`` and ``y_class``.  The latent
        pocket seed is not returned; the pocket is recoverable from
        ``is_international``, ``velocity``, and ``device_trust``.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Latents (not exported)
    # ------------------------------------------------------------------
    z_activity = rng.normal(0.0, 1.0, size=n)
    z_risk = 0.28 * z_activity + 0.96 * rng.normal(0.0, 1.0, size=n)
    z_device = rng.normal(0.0, 1.0, size=n)
    z_session = rng.normal(0.0, 1.0, size=n)
    pocket_seed = rng.random(n) < _POCKET_SEED_P
    seed_f = pocket_seed.astype(np.float64)

    # ------------------------------------------------------------------
    # Account age (days, right-skewed).  Older accounts are safer.
    # ------------------------------------------------------------------
    account_age = np.clip(
        np.exp(6.15 + 0.85 * rng.normal(0.0, 1.0, size=n)),
        7.0,
        5_500.0,
    )
    age_c = (np.log(account_age) - 6.15) / 0.85

    # ------------------------------------------------------------------
    # Country (5 unequal levels).  Risk tilts mass toward BR / NG.
    # ------------------------------------------------------------------
    country_logits = np.column_stack(
        [
            np.full(n, _COUNTRY_LOGITS[0]) - 0.10 * z_risk,
            np.full(n, _COUNTRY_LOGITS[1]) - 0.04 * z_risk,
            np.full(n, _COUNTRY_LOGITS[2]),
            np.full(n, _COUNTRY_LOGITS[3]) + 0.18 * z_risk,
            np.full(n, _COUNTRY_LOGITS[4]) + 0.32 * z_risk + 0.25 * seed_f,
        ]
    )
    country_idx = _categorical_from_probs(rng, _softmax_rows(country_logits))
    country = np.array(COUNTRY_LEVELS, dtype=object)[country_idx]
    # Cross-border propensity and residual country risk (NG > BR > DE > GB > US).
    country_intl = np.array([-0.35, -0.05, -0.12, 0.45, 0.85], dtype=np.float64)[
        country_idx
    ]
    country_risk = np.array([-0.18, -0.06, -0.04, 0.22, 0.48], dtype=np.float64)[
        country_idx
    ]

    # ------------------------------------------------------------------
    # Channel (3): card / web / mobile.  Web is slightly riskier.
    # ------------------------------------------------------------------
    chan_logits = np.column_stack(
        [
            0.55 - 0.10 * z_activity,  # card
            0.05 + 0.28 * z_activity + 0.18 * seed_f,  # web
            0.22 + 0.12 * z_activity,  # mobile
        ]
    )
    channel_idx = _categorical_from_probs(rng, _softmax_rows(chan_logits))
    channel = np.array(CHANNEL_LEVELS, dtype=object)[channel_idx]
    is_web = (channel_idx == 1).astype(np.float64)
    is_mobile = (channel_idx == 2).astype(np.float64)

    # ------------------------------------------------------------------
    # is_international.  Pocket seed is forced cross-border.
    # ------------------------------------------------------------------
    intl_lp = -1.85 + country_intl + 0.22 * z_risk - 0.10 * age_c + 0.35 * seed_f
    is_international = rng.binomial(1, _sigmoid(intl_lp), size=n).astype(np.int64)
    is_international = np.where(pocket_seed, 1, is_international).astype(np.int64)
    intl_f = is_international.astype(np.float64)

    # ------------------------------------------------------------------
    # velocity (skewed rate) and device_trust in [0, 1].
    # Session latent makes high velocity and low trust travel together.
    # Pocket seed shifts both into the observed pocket region.
    # ------------------------------------------------------------------
    log_vel = (
        1.18
        + 0.40 * z_activity
        + 0.34 * z_session
        + 0.28 * intl_f
        + 1.20 * seed_f
        + 0.08 * z_risk
        + 0.28 * rng.normal(0.0, 1.0, size=n)
    )
    velocity = np.clip(np.exp(log_vel), 0.05, 140.0)

    trust_lp = (
        0.62
        + 0.55 * z_device
        - 0.48 * z_session
        - 0.58 * intl_f
        + 0.20 * age_c
        - 1.65 * seed_f
        - 0.10 * z_risk
        + 0.32 * rng.normal(0.0, 1.0, size=n)
    )
    device_trust = np.clip(_sigmoid(trust_lp), 0.02, 0.99)

    # ------------------------------------------------------------------
    # n_tx_day (count) and txn_amount (right-skewed).
    # ------------------------------------------------------------------
    lam_tx = np.clip(
        np.exp(
            0.50
            + 0.38 * z_activity
            + 0.16 * np.log1p(velocity)
            + 0.18 * seed_f
        ),
        0.12,
        28.0,
    )
    n_tx_day = np.clip(rng.poisson(lam_tx), 0, 42).astype(np.int64)

    log_amt = (
        3.62
        + 0.20 * z_activity
        + 0.16 * intl_f
        + 0.14 * seed_f
        + 0.88 * rng.normal(0.0, 1.0, size=n)
    )
    txn_amount = np.clip(np.exp(log_amt), 0.50, 80_000.0)

    # ------------------------------------------------------------------
    # txn_hour ∈ {0,…,23}.  Pocket / risk tilt mass into the night.
    # ------------------------------------------------------------------
    p_night = _sigmoid(-1.40 + 1.05 * seed_f + 0.22 * z_risk + 0.18 * intl_f)
    is_night = rng.random(n) < p_night
    day_hour = np.clip(np.rint(rng.normal(13.8, 3.0, size=n)), 7, 20)
    night_hour = rng.choice(
        np.asarray(_NIGHT_HOURS, dtype=np.int64),
        size=n,
        p=np.asarray(_NIGHT_PROBS, dtype=np.float64),
    )
    txn_hour = np.where(is_night, night_hour, day_hour).astype(np.int64)
    night_f = (
        (txn_hour <= 5) | (txn_hour >= 22)
    ).astype(np.float64)

    # ------------------------------------------------------------------
    # merchant_risk ∈ (0, 1).  Jointly higher in the seeded pocket.
    # ------------------------------------------------------------------
    mr_lp = (
        -0.35
        + 0.85 * seed_f
        + 0.32 * z_risk
        + 0.22 * intl_f
        + 0.55 * country_risk
        + 0.18 * night_f
        + 0.40 * rng.normal(0.0, 1.0, size=n)
    )
    merchant_risk = np.clip(_sigmoid(mr_lp), 0.01, 0.99)

    # Observed pocket a synthesizer can read from X (no extra column).
    pocket = in_high_risk_pocket(is_international, velocity, device_trust)
    pocket_f = pocket.astype(np.float64)

    # ------------------------------------------------------------------
    # y_class — rare Bernoulli.  Pocket is the dominant lift.
    # Logistic of a risk index, not a quantile cut on y.
    # ------------------------------------------------------------------
    fraud_lp = (
        -4.05
        + 2.55 * pocket_f
        + 0.95 * merchant_risk
        + 0.32 * intl_f
        + 0.22 * np.log1p(velocity)
        - 0.75 * device_trust
        + 0.38 * night_f
        + 0.10 * (np.log(txn_amount) - 3.62)
        - 0.16 * age_c
        + 0.22 * is_web
        + 0.08 * is_mobile
        + 0.55 * country_risk
        + 0.18 * z_risk
        + 0.06 * np.maximum(n_tx_day.astype(np.float64) - 8.0, 0.0)
    )
    y_class = rng.binomial(1, _sigmoid(fraud_lp), size=n).astype(np.int64)
    fraud_f = y_class.astype(np.float64)

    # ------------------------------------------------------------------
    # y — two-part severity.  Class 0 ≈ 0; class 1 is lognormal + Pareto
    # kicks (heavy right tail), larger still inside the pocket.
    # ------------------------------------------------------------------
    y_neg = np.exp(-6.15 + 0.38 * rng.normal(0.0, 1.0, size=n))
    y_neg = np.clip(y_neg + 2.0e-6 * txn_amount, 1.0e-6, 0.08)

    eta_pos = (
        2.25
        + 0.52 * np.log(txn_amount)
        + 0.72 * pocket_f
        + 0.80 * merchant_risk
        + 0.16 * np.log1p(velocity)
        + 0.12 * night_f
    )
    y_pos = np.exp(eta_pos + 0.82 * rng.normal(0.0, 1.0, size=n))
    pareto_kick = rng.random(n) < 0.08
    y_pos = y_pos * np.where(pareto_kick, rng.pareto(1.35) + 1.0, 1.0)
    y_pos = np.clip(y_pos, 0.40, 4.0e6)

    y = np.where(fraud_f > 0.5, y_pos, y_neg)

    return pd.DataFrame(
        {
            "txn_amount": txn_amount.astype(np.float64),
            "txn_hour": txn_hour,
            "merchant_risk": merchant_risk.astype(np.float64),
            "device_trust": device_trust.astype(np.float64),
            "country": pd.Series(country, dtype="string"),
            "channel": pd.Series(channel, dtype="string"),
            "is_international": is_international,
            "n_tx_day": n_tx_day,
            "velocity": velocity.astype(np.float64),
            "account_age": account_age.astype(np.float64),
            "y": y.astype(np.float64),
            "y_class": y_class,
        }
    )


def get_ground_truth_description() -> str:
    """Human-readable ground truth for reports / diagnostics."""
    return (
        "Fraud / rare-event DGP: y_class prevalence ≈ 4–8 % (logistic of a "
        "risk index, not a y quantile).  Continuous y is two-part — almost "
        "zero for class 0, lognormal + Pareto-kick severity for class 1.  "
        "A small high-risk pocket (is_international=1, velocity ≥ 5.5, "
        "device_trust ≤ 0.42) concentrates the minority class.  Features: "
        "right-skewed txn_amount, txn_hour ∈ {0..23}, merchant_risk, "
        "device_trust, 5 unequal countries, 3 channels, is_international, "
        "n_tx_day (count), velocity, account_age.  Recovering the pocket "
        "joint — not just the class rate — is required for minority AUC."
    )


SPEC = DatasetSpec(
    name="imbalanced",
    generate=generate,
    description=get_ground_truth_description(),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=("imbalanced", "fraud", "rare-event", "heavy-tail", "pocket", "mixed"),
)


def _self_test() -> None:
    """Check class rate, pocket lift, two-part y, and DatasetSpec contract."""
    n = 4_000
    rates: list[float] = []
    pocket_rates: list[float] = []
    pocket_lifts: list[float] = []
    for seed in (0, 1, 2, 3, 4):
        df = generate(n, seed=seed)
        prev = float(df["y_class"].mean())
        rates.append(prev)
        assert 0.030 <= prev <= 0.100, f"seed={seed} y_class rate={prev:.4f} outside 3–10%"

        pocket = in_high_risk_pocket(
            df["is_international"].to_numpy(),
            df["velocity"].to_numpy(dtype=float),
            df["device_trust"].to_numpy(dtype=float),
        )
        pr = float(pocket.mean())
        pocket_rates.append(pr)
        assert 0.008 <= pr <= 0.080, f"seed={seed} pocket rate={pr:.4f}"

        p_in = float(df.loc[pocket, "y_class"].mean()) if pocket.any() else 0.0
        p_out = float(df.loc[~pocket, "y_class"].mean())
        lift = p_in / max(p_out, 1e-9)
        pocket_lifts.append(lift)
        assert p_in > 0.18, f"seed={seed} P(fraud|pocket)={p_in:.4f}"
        assert lift >= 3.0, f"seed={seed} pocket lift={lift:.2f}"

        y0 = df.loc[df["y_class"] == 0, "y"].to_numpy(dtype=float)
        y1 = df.loc[df["y_class"] == 1, "y"].to_numpy(dtype=float)
        assert y0.size > 50 and y1.size > 20
        assert float(np.median(y0)) < 0.02, f"class-0 median y={np.median(y0):.4g}"
        assert float(np.quantile(y0, 0.95)) < 0.08
        assert float(np.median(y1)) > 10.0, f"class-1 median y={np.median(y1):.4g}"
        q50, q99 = np.quantile(y1, [0.50, 0.99])
        assert q99 / max(q50, 1e-9) >= 4.0, f"class-1 tail ratio={q99 / q50:.2f}"

    mean_rate = float(np.mean(rates))
    assert 0.04 <= mean_rate <= 0.08, f"mean y_class rate={mean_rate:.4f} not in 4–8%"

    df0 = generate(800, seed=0)
    prev0 = float(df0["y_class"].mean())
    assert 0.04 <= prev0 <= 0.08, f"n=800 seed=0 y_class rate={prev0:.4f} not in 4–8%"

    assert set(df0["txn_hour"].unique()).issubset(set(range(24)))
    assert 0 <= int(df0["txn_hour"].min()) and int(df0["txn_hour"].max()) <= 23
    assert set(df0["country"].astype(str).unique()).issubset(set(COUNTRY_LEVELS))
    assert df0["country"].nunique() >= 4
    assert set(df0["channel"].astype(str).unique()) == set(CHANNEL_LEVELS)
    assert set(df0["is_international"].unique()).issubset({0, 1})
    amt = df0["txn_amount"].to_numpy(dtype=float)
    assert float(np.mean(amt)) > float(np.median(amt)), "txn_amount should be right-skewed"

    a = generate(120, 7)
    b = generate(120, 7)
    pd.testing.assert_frame_equal(a, b)
    c = generate(120, 8)
    assert not a.equals(c)

    assert SPEC.name == "imbalanced"
    assert SPEC.target_reg == "y"
    assert SPEC.target_clf == "y_class"
    assert SPEC.feature_cols == FEATURE_COLS
    sampled = SPEC.sample(60, seed=3)
    assert len(sampled) == 60
    assert list(sampled.columns) == list(df0.columns)

    print(
        "imbalanced self-test ok  "
        f"y_class n=4000 seeds0-4={rates} mean={mean_rate:.4f}  "
        f"n=800 seed0={prev0:.4f}  "
        f"pocket_rates={['%.4f' % p for p in pocket_rates]}  "
        f"pocket_lifts={['%.2f' % p for p in pocket_lifts]}"
    )


if __name__ == "__main__":
    _self_test()


__all__ = [
    "FEATURE_COLS",
    "TARGET_REG",
    "TARGET_CLF",
    "COUNTRY_LEVELS",
    "CHANNEL_LEVELS",
    "POCKET_VELOCITY_MIN",
    "POCKET_DEVICE_TRUST_MAX",
    "SPEC",
    "generate",
    "get_ground_truth_description",
    "in_high_risk_pocket",
]
