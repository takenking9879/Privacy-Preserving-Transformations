"""Clinical mixed-type hospital DGP for synthesizer fidelity tests.

Observed table: vitals / labs / demographics plus a continuous cost-like
target ``y`` and a binary severity flag ``y_class``.  A latent frailty
``S`` (never returned) and an ICU ward jointly shift several labs, so
one-dimensional marginal matching is not enough.

A synthesizer MUST recover: right-skewed BMI/glucose, strong systolic–
diastolic correlation, a smoker × age interaction in ``y``, an ICU
multimodal lab shift, a nonlinear blood-pressure effect in ``y``,
``y_class`` as a logistic of a severity index (not a ``y`` quantile),
and ~5% MAR missingness on cholesterol concentrated in ICU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

FEATURE_COLS: tuple[str, ...] = (
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
    "smoker",
    "sex",
    "ward",
    "comorbidity_count",
    "length_of_stay",
    "lab_flag",
)
TARGET_REG = "y"
TARGET_CLF = "y_class"

WARD_LEVELS: tuple[str, ...] = ("A", "B", "C", "ICU")
SEX_LEVELS: tuple[str, ...] = ("M", "F")


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


def generate(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n`` hospital encounters from the clinical DGP.

    Parameters
    ----------
    n:
        Number of rows. Defaults to 800.
    seed:
        RNG seed. Defaults to 0.

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE_COLS`` plus ``y`` and ``y_class``.  ``cholesterol``
        contains MAR NaNs (~5% overall, driven by ICU).  Latent frailty
        ``S`` is not returned.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Latent frailty (not exported).  Fans out into labs, ward, and both
    # targets so residual dependence remains after conditioning on age.
    # ------------------------------------------------------------------
    frailty = rng.normal(0.0, 1.0, size=n)

    # ------------------------------------------------------------------
    # Demographics
    # ------------------------------------------------------------------
    age = np.clip(rng.normal(58.0, 16.0, size=n), 18.0, 95.0)
    sex_is_m = rng.binomial(1, 0.47, size=n).astype(np.float64)
    sex = np.where(sex_is_m > 0.5, "M", "F")

    # Smoking is more common in men and slightly in younger adults.
    smoker_lp = -1.35 + 0.28 * sex_is_m - 0.012 * (age - 50.0) + 0.10 * frailty
    smoker = rng.binomial(1, _sigmoid(smoker_lp), size=n).astype(np.int64)
    smoker_f = smoker.astype(np.float64)

    # Comorbidity load (integer counts) before ward so ICU can depend on it.
    comorb_lam = np.clip(
        0.55
        + 0.032 * (age - 50.0)
        + 0.42 * frailty
        + 0.38 * smoker_f
        + 0.08 * sex_is_m,
        0.08,
        10.0,
    )
    comorbidity_count = np.clip(rng.poisson(comorb_lam), 0, 12).astype(np.int64)
    comorb_f = comorbidity_count.astype(np.float64)

    # Ward: A/B/C/ICU, unequal.  ICU ~12–16%; frailty / age / comorbidities
    # tilt patients into ICU (multimodal lab cluster below).
    ward_logits = np.column_stack(
        [
            np.full(n, 0.85),
            np.full(n, 0.40),
            np.full(n, 0.00),
            -0.55
            + 0.52 * frailty
            + 0.020 * (age - 55.0)
            + 0.20 * comorb_f
            + 0.22 * smoker_f,
        ]
    )
    ward_idx = _categorical_from_probs(rng, _softmax_rows(ward_logits))
    ward = np.array(WARD_LEVELS, dtype=object)[ward_idx]
    icu = (ward_idx == 3).astype(np.float64)

    # ------------------------------------------------------------------
    # Labs / vitals.  ICU adds a joint location shift → second mode.
    # ------------------------------------------------------------------
    # BMI: log-normal, right-skewed; ICU and frailty lift the body.
    log_bmi = (
        np.log(26.4)
        + 0.055 * frailty
        + 0.16 * icu
        + 0.035 * smoker_f
        + 0.0018 * (age - 50.0)
        + rng.normal(0.0, 0.30, size=n)
    )
    bmi = np.clip(np.exp(log_bmi), 16.0, 58.0)

    # Glucose: log-normal, right-skewed; ICU multiplies by ~e^0.34 ≈ 1.40.
    log_glucose = (
        np.log(95.0)
        + 0.075 * frailty
        + 0.34 * icu
        + 0.11 * smoker_f
        + 0.014 * (bmi - 27.0)
        + rng.normal(0.0, 0.25, size=n)
    )
    glucose = np.clip(np.exp(log_glucose), 55.0, 420.0)

    # Systolic / diastolic: shared Gaussian factor → strong positive corr.
    # ICU shifts both jointly (pulse pressure stays plausible).
    z_bp = rng.normal(0.0, 1.0, size=n)
    sys_noise = 10.5 * z_bp + 6.0 * rng.normal(0.0, 1.0, size=n)
    dia_noise = 6.8 * (0.82 * z_bp + 0.57 * rng.normal(0.0, 1.0, size=n))
    systolic_bp = (
        124.0
        + 0.30 * (age - 55.0)
        + 5.2 * frailty
        + 15.0 * icu
        + 5.5 * smoker_f
        + 0.32 * (bmi - 27.0)
        + sys_noise
    )
    diastolic_bp = (
        76.0
        + 0.11 * (age - 55.0)
        + 3.4 * frailty
        + 9.0 * icu
        + 2.4 * smoker_f
        + 0.14 * (bmi - 27.0)
        + dia_noise
    )
    systolic_bp = np.clip(systolic_bp, 85.0, 230.0)
    diastolic_bp = np.clip(diastolic_bp, 45.0, 135.0)
    diastolic_bp = np.minimum(diastolic_bp, systolic_bp - 18.0)
    diastolic_bp = np.clip(diastolic_bp, 45.0, 135.0)

    cholesterol = (
        188.0
        + 11.0 * frailty
        + 24.0 * icu
        + 9.0 * smoker_f
        + 1.05 * (bmi - 27.0)
        + 0.18 * (age - 55.0)
        + rng.normal(0.0, 30.0, size=n)
    )
    cholesterol = np.clip(cholesterol, 110.0, 380.0)

    # Abnormal-lab flag (binary), more common in ICU and high glucose/BP.
    lab_lp = (
        -2.35
        + 0.016 * (glucose - 100.0)
        + 0.012 * (systolic_bp - 130.0)
        + 1.20 * icu
        + 0.14 * comorb_f
        + 0.18 * frailty
    )
    lab_flag = rng.binomial(1, _sigmoid(lab_lp), size=n).astype(np.int64)
    lab_f = lab_flag.astype(np.float64)

    # Length of stay (days), right-skewed; ICU stretches the stay.
    log_los = (
        1.12
        + 0.92 * icu
        + 0.13 * comorb_f
        + 0.16 * frailty
        + 0.14 * lab_f
        + rng.normal(0.0, 0.48, size=n)
    )
    length_of_stay = np.clip(np.exp(log_los), 0.5, 75.0)

    # ------------------------------------------------------------------
    # y — encounter cost / readmission-risk score (USD-like, right-skewed)
    # Nonlinear BP (quadratic + hockey-stick) and smoker × age interaction.
    # ------------------------------------------------------------------
    sys_c = (systolic_bp - 120.0) / 20.0
    sys_excess = np.maximum(systolic_bp - 145.0, 0.0)
    eta = (
        8.02
        + 0.013 * (age - 55.0)
        + 0.26 * smoker_f
        + 0.015 * smoker_f * (age - 40.0)  # smoker × age
        + 0.20 * (sys_c**2)  # nonlinear BP
        + 0.038 * sys_excess
        + 0.0050 * (glucose - 100.0)
        + 0.016 * (bmi - 27.0)
        + 0.095 * comorb_f
        + 0.46 * icu
        + 0.032 * length_of_stay
        + 0.15 * lab_f
        + 0.11 * frailty
    )
    sigma_y = 0.21 * (1.0 + 0.32 * icu + 0.10 * smoker_f)
    y = np.exp(eta + sigma_y * rng.normal(0.0, 1.0, size=n))
    y = np.clip(y, 400.0, 250_000.0)

    # ------------------------------------------------------------------
    # y_class — Bernoulli of a logistic severity index (NOT a y-quantile).
    # Shares frailty, ICU, smoker×age, BP threshold, and glucose with y.
    # ------------------------------------------------------------------
    severity = (
        -1.82
        + 0.020 * (age - 55.0)
        + 0.58 * smoker_f
        + 0.22 * smoker_f * ((age - 40.0) / 25.0)
        + 0.015 * np.maximum(systolic_bp - 140.0, 0.0)
        + 0.0055 * (glucose - 100.0)
        + 0.19 * comorb_f
        + 1.00 * icu
        + 0.38 * lab_f
        + 0.26 * frailty
        + 0.035 * np.maximum(bmi - 30.0, 0.0)
    )
    y_class = rng.binomial(1, _sigmoid(severity), size=n).astype(np.int64)

    # ------------------------------------------------------------------
    # MAR missingness on cholesterol, concentrated in ICU (~5% overall).
    # P(miss | ward) depends on the fully observed ICU indicator (and a
    # little on lab_flag), so complete-case is a selected subsample.
    # ------------------------------------------------------------------
    p_miss_chol = _sigmoid(-3.95 + 2.95 * icu + 0.20 * lab_f)
    miss_chol = rng.random(n) < p_miss_chol
    cholesterol_obs = cholesterol.copy()
    cholesterol_obs[miss_chol] = np.nan

    return pd.DataFrame(
        {
            "age": age.astype(np.float64),
            "bmi": bmi.astype(np.float64),
            "systolic_bp": systolic_bp.astype(np.float64),
            "diastolic_bp": diastolic_bp.astype(np.float64),
            "glucose": glucose.astype(np.float64),
            "cholesterol": cholesterol_obs.astype(np.float64),
            "smoker": smoker,
            "sex": pd.Series(sex, dtype="string"),
            "ward": pd.Series(ward, dtype="string"),
            "comorbidity_count": comorbidity_count,
            "length_of_stay": length_of_stay.astype(np.float64),
            "lab_flag": lab_flag,
            "y": y.astype(np.float64),
            "y_class": y_class,
        }
    )


SPEC = DatasetSpec(
    name="healthcare",
    generate=generate,
    description=(
        "Clinical mixed table (n default 800): right-skewed BMI/glucose; "
        "strongly correlated systolic/diastolic BP; ICU jointly shifts labs "
        "(multimodal); smoker×age interaction and nonlinear BP in continuous "
        "cost y; y_class is a logistic of a severity index (not a y quantile); "
        "~5% MAR missing cholesterol driven by ICU."
    ),
    feature_cols=FEATURE_COLS,
    target_reg=TARGET_REG,
    target_clf=TARGET_CLF,
    tags=("healthcare", "clinical", "mixed", "interaction", "multimodal", "mar"),
)


if __name__ == "__main__":
    df = generate(200, seed=0)
    y = df["y"].to_numpy(dtype=float)
    glu = df["glucose"].to_numpy(dtype=float)
    corr = float(np.corrcoef(y, glu)[0, 1])
    balance = float(df["y_class"].mean())
    print(f"shape={df.shape}")
    print(f"corr(y, glucose)={corr:.4f}")
    print(f"class balance (P(y_class=1))={balance:.4f}")
