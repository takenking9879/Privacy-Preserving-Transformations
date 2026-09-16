"""Heavy-tailed returns / PnL DGP that breaks Gaussian copulas.

Observed table: market-style features (volatility, momentum, liquidity,
leverage, sector, size, overnight, volume, spread) plus a continuous PnL
target ``y`` and a binary large-loss flag ``y_class``.

``y`` is a Student-t / centered-lognormal mixture (excess kurtosis well
above 3).  Leverage and ``y`` share an inverse-chi² mixer — a
multivariate-t / t-copula construction — so extremes arrive together.
A Gaussian copula can match body ranks and still understate joint tail
mass; methods that keep tail mass (empirical quantiles, sequential trees
that isolate extremes) are the ones that should travel.

``y_class`` is a Bernoulli of a logistic in ``leverage * vol``.  That is
the stochastic analogue of “below the 10% quantile of a latent loss
index”, **not** a cut of observed ``y``.

See :data:`SPEC`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "vol",
    "momentum",
    "liquidity",
    "leverage",
    "sector",
    "size_bucket",
    "overnight",
    "volume",
    "spread",
)

TARGET_REG: str = "y"
TARGET_CLF: str = "y_class"

SECTOR_LEVELS: tuple[str, ...] = (
    "tech",
    "financials",
    "energy",
    "healthcare",
    "industrials",
)
SIZE_LEVELS: tuple[str, ...] = ("small", "mid", "large")

# Unequal mix so a copula cannot treat sector / size as balanced dummies.
_SECTOR_PROBS: tuple[float, ...] = (0.28, 0.22, 0.16, 0.18, 0.16)
_SIZE_PROBS: tuple[float, ...] = (0.40, 0.35, 0.25)

# Multivariate-t mixer (shared by leverage and y).  df just above 4 keeps
# sample fourth moments finite-ish while leaving clear tail dependence.
_TAIL_DF = 4.5
# Negative Gaussian core: high leverage travels with low (loss) y.
_TAIL_RHO = -0.62

# Mixture weight on the Student-t residual (rest is centered lognormal).
# Lognormal mass is high enough that sample excess kurtosis stays > 3
# even when the rare inverse-χ² spike is missing from a given draw.
_MIX_T_P = 0.38
_T_DF = 4.2
_LN_SIGMA = 1.12

# Residual blend: mixture shock (kurtosis) + shared t-factor (tail dep).
_W_MIX = 0.88
_W_TAIL = 0.55

# Pareto shape for volume (alpha < 2 → infinite variance).
_VOLUME_ALPHA = 1.85
_VOLUME_SCALE = 8.5e4

# Logistic of leverage*vol ≈ 10% large-loss rate (latent quantile story).
# Intercept is a bit left of logit(0.10) because leverage×vol is right-skewed.
_YCLASS_INTERCEPT = -2.78
_YCLASS_SLOPE = 1.15
_RISK_CENTER = 0.52
_RISK_SCALE = 0.48


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _excess_kurtosis(x: np.ndarray) -> float:
    """Fisher excess kurtosis (biased / population fourth-moment form)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    centered = x - float(np.mean(x))
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return 0.0
    m4 = float(np.mean(centered**4))
    return m4 / (m2 * m2) - 3.0


def _centered_lognormal(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """Mean-zero lognormal shock (right-skewed, high kurtosis)."""
    raw = rng.lognormal(mean=0.0, sigma=float(sigma), size=n)
    return raw - float(np.exp(0.5 * sigma * sigma))


def generate(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` returns / PnL rows from the heavy-tail DGP.

    Parameters
    ----------
    n:
        Number of rows. Must be a positive integer.
    seed:
        RNG seed (deterministic given ``n``).

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS + (y, y_class)``.  The inverse-chi² mixer
        and mixture indicators are not returned.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Shared inverse-chi² mixer → multivariate-t (t-copula) factor.
    # Large W inflates *both* leverage and |y| — Gaussian copulas miss
    # that joint tail mass.
    # ------------------------------------------------------------------
    chi2 = rng.chisquare(_TAIL_DF, size=n)
    mixer = _TAIL_DF / np.maximum(chi2, 1e-12)
    sqrt_w = np.sqrt(mixer)

    z_lev = rng.normal(0.0, 1.0, size=n)
    z_y = _TAIL_RHO * z_lev + np.sqrt(max(1.0 - _TAIL_RHO**2, 0.0)) * rng.normal(
        0.0, 1.0, size=n
    )
    t_lev = sqrt_w * z_lev
    t_y = sqrt_w * z_y

    # ------------------------------------------------------------------
    # Sector (5) / size_bucket (3) / overnight (0/1)
    # ------------------------------------------------------------------
    sector_idx = rng.choice(5, size=n, p=np.asarray(_SECTOR_PROBS))
    size_idx = rng.choice(3, size=n, p=np.asarray(_SIZE_PROBS))
    sector = np.asarray(SECTOR_LEVELS, dtype=object)[sector_idx]
    size_bucket = np.asarray(SIZE_LEVELS, dtype=object)[size_idx]

    is_small = (size_idx == 0).astype(np.float64)
    is_large = (size_idx == 2).astype(np.float64)
    is_fin = (sector_idx == 1).astype(np.float64)
    is_energy = (sector_idx == 2).astype(np.float64)

    # Overnight more common in energy / financials (gap-risk names).
    overnight_lp = -0.95 + 0.55 * is_energy + 0.35 * is_fin + 0.18 * is_small
    overnight = rng.binomial(1, _sigmoid(overnight_lp), size=n).astype(np.int64)
    overnight_f = overnight.astype(np.float64)

    # ------------------------------------------------------------------
    # Heavy-tailed / skewed market features
    # ------------------------------------------------------------------
    # vol: lognormal, lifted for small caps / energy and by the tail factor.
    sector_vol = np.array([0.04, 0.07, 0.11, -0.03, 0.01], dtype=np.float64)[
        sector_idx
    ]
    log_vol = (
        np.log(0.20)
        + sector_vol
        + 0.20 * is_small
        - 0.14 * is_large
        + 0.06 * overnight_f
        + 0.07 * t_lev
        + rng.normal(0.0, 0.26, size=n)
    )
    vol = np.clip(np.exp(log_vol), 0.03, 3.5)

    # momentum: Student-t recent return (fat-tailed X).
    sector_mom = np.array([0.004, -0.002, 0.001, 0.003, 0.000], dtype=np.float64)[
        sector_idx
    ]
    momentum = sector_mom + 0.065 * rng.standard_t(5.0, size=n)

    # liquidity: higher for large / lower when vol is high (lognormal).
    log_liq = (
        np.log(1.15)
        + 0.42 * is_large
        - 0.28 * is_small
        - 0.55 * np.log(vol / 0.20)
        + rng.normal(0.0, 0.33, size=n)
    )
    liquidity = np.clip(np.exp(log_liq), 0.05, 25.0)

    # leverage: positive, heavy right tail via the shared t factor.
    log_lev = (
        np.log(1.75)
        + 0.38 * t_lev
        + 0.18 * is_small
        - 0.12 * is_large
        + 0.10 * overnight_f
        + 0.08 * is_fin
        + rng.normal(0.0, 0.22, size=n)
    )
    leverage = np.clip(np.exp(log_lev), 0.15, 80.0)

    # volume: Pareto (power-law) scaled by size and liquidity.
    u_vol = rng.uniform(1e-12, 1.0, size=n)
    pareto = u_vol ** (-1.0 / _VOLUME_ALPHA)
    volume = np.clip(
        _VOLUME_SCALE
        * pareto
        * (0.55 + 0.90 * is_large + 0.25 * liquidity / 1.15)
        * (0.85 + 0.6 * vol / 0.20),
        1.0e3,
        5.0e8,
    )

    # spread: inverse to liquidity, wider for high vol / small / overnight.
    log_spread = (
        np.log(8.0e-4)
        - 0.52 * np.log(np.maximum(liquidity, 1e-6))
        + 0.48 * np.log(vol / 0.20)
        + 0.28 * is_small
        + 0.18 * overnight_f
        + rng.normal(0.0, 0.28, size=n)
    )
    spread = np.clip(np.exp(log_spread), 1.0e-5, 0.08)

    # ------------------------------------------------------------------
    # y — PnL: Student-t / centered-lognormal mixture + shared t-factor.
    # Location effects stay modest so residual kurtosis is not diluted.
    # ------------------------------------------------------------------
    sector_alpha = np.array(
        [0.0012, -0.0020, 0.0004, 0.0008, -0.0006], dtype=np.float64
    )[sector_idx]
    mu = (
        0.0015
        + 0.12 * momentum
        + 0.004 * (liquidity - 1.15)
        - 0.0065 * (leverage - 1.75)
        - 0.035 * (vol - 0.20)
        + sector_alpha
        - 0.004 * overnight_f
        - 0.002 * is_small
    )
    sigma = 0.018 * (1.0 + 1.15 * vol + 0.08 * np.maximum(leverage - 1.0, 0.0))
    sigma = np.clip(sigma, 0.008, 0.28)

    is_t = rng.random(n) < _MIX_T_P
    shock_t = rng.standard_t(_T_DF, size=n)
    shock_ln = _centered_lognormal(rng, n, _LN_SIGMA)
    shock = np.where(is_t, shock_t, shock_ln)

    y = mu + sigma * (_W_MIX * shock + _W_TAIL * t_y)

    # ------------------------------------------------------------------
    # y_class — large-loss flag via logistic(leverage * vol).
    # Conceptual 10% quantile of a latent loss index; not a cut of y.
    # ------------------------------------------------------------------
    risk = leverage * vol
    risk_z = (risk - _RISK_CENTER) / _RISK_SCALE
    class_lp = _YCLASS_INTERCEPT + _YCLASS_SLOPE * risk_z
    y_class = rng.binomial(1, _sigmoid(class_lp), size=n).astype(np.int64)

    return pd.DataFrame(
        {
            "vol": vol.astype(np.float64),
            "momentum": momentum.astype(np.float64),
            "liquidity": liquidity.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "sector": pd.Series(sector, dtype="string"),
            "size_bucket": pd.Series(size_bucket, dtype="string"),
            "overnight": overnight,
            "volume": volume.astype(np.float64),
            "spread": spread.astype(np.float64),
            "y": y.astype(np.float64),
            "y_class": y_class,
        }
    )


def get_ground_truth_description() -> str:
    """Return a human-readable summary of the heavy-tail DGP."""
    return (
        "Returns/PnL heavy-tail DGP (Gaussian copula should miss joint tails).  "
        "Features: vol (lognormal); momentum (Student-t); liquidity (lognormal, "
        "inverse to vol); leverage (log-scale load on a multivariate-t factor); "
        f"sector ∈ {SECTOR_LEVELS} with probs {_SECTOR_PROBS}; "
        f"size_bucket ∈ {SIZE_LEVELS} with probs {_SIZE_PROBS}; "
        "overnight ~ Bern(σ(−0.95 + 0.55·I(energy) + 0.35·I(financials) + 0.18·I(small))); "
        f"volume ~ Pareto(α={_VOLUME_ALPHA}) × size/liquidity/vol; "
        "spread lognormal, inverse to liquidity.  "
        "Shared inverse-χ² mixer W = ν/χ²_ν (ν=4.5) builds a t-copula factor "
        "(t_lev, t_y) with Gaussian-core ρ=−0.62, so high leverage and large "
        "losses arrive together in the tails.  "
        "TARGET y = μ(X) + σ(vol, leverage)·(0.88·ε_mix + 0.55·t_y), where "
        "ε_mix is a 38/62 Student-t(4.2) / centered-lognormal(σ=1.12) mixture.  "
        "TARGET y_class ~ Bern(σ(−2.78 + 1.15·z(leverage·vol))) — a logistic "
        "of leverage×vol (latent 10% large-loss flag), not a quantile cut on y.  "
        "A Gaussian copula recovers Spearman in the body and understates "
        "P(leverage extreme, y extreme); excess kurtosis of y exceeds 3."
    )


SPEC = DatasetSpec(
    name="heavytail",
    generate=generate,
    description=get_ground_truth_description(),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=(
        "heavytail",
        "returns",
        "pnl",
        "student-t",
        "lognormal-mixture",
        "tail-dependence",
        "copula-fail",
        "pareto",
        "mixed",
    ),
)


def _self_test(n: int = 2500, seed: int = 0) -> dict[str, float | int | bool | str]:
    """Assert excess kurtosis of ``y`` exceeds 3 and return a summary."""
    df = generate(n, seed=seed)
    y = df["y"].to_numpy(dtype=float)
    kurt = _excess_kurtosis(y)
    if not np.isfinite(kurt) or kurt <= 3.0:
        raise AssertionError(
            f"excess kurtosis of y={kurt:.4f} is not > 3 (n={n}, seed={seed})"
        )

    # Contract / schema.
    if SPEC.name != "heavytail":
        raise AssertionError(f"SPEC.name={SPEC.name!r}")
    missing = [c for c in FEATURE_COLS + (TARGET_REG, TARGET_CLF) if c not in df.columns]
    if missing:
        raise AssertionError(f"missing columns: {missing}")
    n_sector = int(df["sector"].nunique())
    n_size = int(df["size_bucket"].nunique())
    if n_sector != 5:
        raise AssertionError(f"sector levels={n_sector}, expected 5")
    if n_size != 3:
        raise AssertionError(f"size_bucket levels={n_size}, expected 3")

    overnight = df["overnight"].to_numpy()
    if not np.isin(overnight, [0, 1]).all():
        raise AssertionError("overnight is not in {0,1}")

    yc = df["y_class"].to_numpy(dtype=int)
    y_cut = (y <= np.quantile(y, 0.10)).astype(int)
    if np.array_equal(yc, y_cut):
        raise AssertionError("y_class is a 10% quantile cut of observed y")

    prev = float(np.mean(yc))
    lev = df["leverage"].to_numpy(dtype=float)
    lev_hi = lev >= np.quantile(lev, 0.90)
    y_lo = y <= np.quantile(y, 0.10)
    p_y_lo_given_lev_hi = float(np.mean(y_lo[lev_hi])) if np.any(lev_hi) else 0.0

    # Seed stability.
    a = generate(80, 7)
    b = generate(80, 7)
    pd.testing.assert_frame_equal(a, b)

    sampled = SPEC.sample(40, seed=3)
    if len(sampled) != 40 or "y" not in sampled.columns:
        raise AssertionError("DatasetSpec.sample contract failed")

    return {
        "name": SPEC.name,
        "n": int(len(df)),
        "n_cols": int(df.shape[1]),
        "excess_kurtosis_y": kurt,
        "y_mean": float(np.mean(y)),
        "y_q01": float(np.quantile(y, 0.01)),
        "y_q10": float(np.quantile(y, 0.10)),
        "y_q90": float(np.quantile(y, 0.90)),
        "y_q99": float(np.quantile(y, 0.99)),
        "y_class_rate": prev,
        "p_y_lo|lev_hi": p_y_lo_given_lev_hi,
        "n_sector": n_sector,
        "n_size": n_size,
        "overnight_rate": float(np.mean(overnight)),
        "passed": True,
    }


if __name__ == "__main__":
    summary = _self_test()
    print(summary)


__all__ = [
    "FEATURE_COLS",
    "TARGET_REG",
    "TARGET_CLF",
    "SECTOR_LEVELS",
    "SIZE_LEVELS",
    "SPEC",
    "generate",
    "get_ground_truth_description",
]
