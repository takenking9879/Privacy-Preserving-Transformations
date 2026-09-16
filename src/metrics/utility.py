"""Model-utility evaluation of a synthetic table against the original.

Protocol (holdout is always the *original* test split when the name
says ``TR*`` / ``*TR``):

* **TRTR** — train ``M`` on real train, evaluate on real holdout.
* **TSTR** — train ``M'`` on synth train, evaluate on the **same** real holdout.
* **TRTS** — train on real train, evaluate on synth holdout.
* **TSTS** — train on synth train, evaluate on synth holdout.

A frozen copy of each real-trained model is then applied to the full
feature matrices ``X`` and ``X'``.  If the rows look paired (positional
feature equality on ≥ 90% of rows), prediction agreement is Spearman /
Pearson of ``M(X)`` vs ``M(X')``.  Otherwise only distributional
summaries are used (mean, std, KS on equal-sized samples).  Synthetic
tables are **not** assumed paired with the original.

``utility_score`` is in ``[0, 1]`` (1 = interchangeable with real for
the frozen model families).  ``tstr_gap_r2 = TRTR R² − TSTR R²``; values
near 0 mean high utility.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-12
RF_N_ESTIMATORS = 80
RF_MAX_DEPTH = 8
FORCED_CATEGORICAL = ("region", "segment")
PAIRED_ROW_MATCH = 0.90

UTILITY_SCORE_FORMULA = (
    "utility_score = clip(w-mean of available terms, 0, 1): "
    "0.50 * mean_m (1 - clip(|R2_TRTR - R2_TSTR|, 0, 1))  [regression TSTR]; "
    "0.20 * mean_m (1 - clip(|AUC_TRTR - AUC_TSTR|, 0, 1))  [clf TSTR, "
    "fallback to accuracy then f1]; "
    "0.30 * agreement_score. "
    "agreement_score = 0.55 * clip(corr, 0, 1) + 0.45 * (1 - KS) when paired "
    "(corr = mean of clip(spearman,0,1) and clip(pearson,0,1)); "
    "otherwise 0.70 * (1 - KS) + 0.30 * moment_score, where "
    "moment_score = 1 - 0.5 * |μr-μs| / (|μr|+|μs|+σr+σs+ε) "
    "- 0.5 * |σr-σs| / (σr+σs+ε). "
    "Per-task agreement is averaged over the fitted frozen models. "
    "Missing families are dropped and the remaining weights renormalized."
)

INTERPRETATION = {
    "tstr_gap_r2": (
        "TRTR R² minus TSTR R² (mean across regression models). "
        "0 means the synthetic table trains as well as the original. "
        "Negative ⇒ TSTR beat TRTR (unusual; extra synth n or leakage). "
        "Good: |gap| ≤ 0.05 (strong), ≤ 0.08 (typical pass bar). "
        "Poor: |gap| ≥ 0.20, or TSTR R² < 0 while TRTR R² > 0.2."
    ),
    "utility_score": (
        "Scalar in [0, 1].  1 = synth is interchangeable with real for "
        "these model families.  ≥ 0.85 strong substitute; 0.70–0.85 usable; "
        "< 0.55 failed / negative-control-like (independence shuffle, "
        "random features)."
    ),
    "pred_agreement": (
        "Frozen real-trained model applied to X and X'.  Paired rows: "
        "Spearman/Pearson of scores near 1 is good; label_agreement near 1 "
        "for classifiers.  Unpaired (the usual synthesizer case): compare "
        "score mean/std and KS.  KS ≤ 0.10 strong, ≤ 0.20 typical pass, "
        "> 0.35 distributionally distinct.  Do not read positional "
        "correlation when paired=False — rows are not aligned."
    ),
    "trtr": "Ceiling: real train → real holdout.  Reference for every gap.",
    "tstr": (
        "Product gate: synth train → same real holdout.  Should track TRTR. "
        "Closer TSTR is to TRTR, higher the modelling utility of X'."
    ),
    "trts": (
        "Real train → synth holdout.  Asks whether X' is in-distribution "
        "for a model that already works on real data."
    ),
    "tsts": (
        "Synth train → synth holdout.  Inflated TSTS + weak TSTR means "
        "the synthesizer is self-consistent but not faithful to P(Y|X)."
    ),
    "regression_metrics": "r2 (higher better), rmse / mae (lower better).",
    "classification_metrics": (
        "accuracy, roc_auc, f1 (higher better).  roc_auc is binary on P(y=1) "
        "or macro-OVR if more than two classes.  f1 is binary (pos_label=1 "
        "when present) or macro."
    ),
}

METRIC_KEYS = [
    "regression",
    "classification",
    "tstr_gap_r2",
    "pred_agreement",
    "utility_score",
    "utility_score_formula",
    "meta",
]


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mean_finite(values: list[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return None
    return float(np.mean(xs))


def _feature_columns(
    real: pd.DataFrame, target_reg: str, target_clf: str
) -> list[str]:
    exclude = {target_reg, target_clf}
    return [c for c in real.columns if c not in exclude]


def _column_roles(
    frame: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in feature_cols:
        if col in FORCED_CATEGORICAL:
            categorical.append(col)
            continue
        series = frame[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            numeric.append(col)
        elif pd.api.types.is_bool_dtype(series):
            numeric.append(col)
        elif isinstance(series.dtype, pd.CategoricalDtype):
            categorical.append(col)
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        ):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def _align_feature_frame(
    frame: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    data: dict[str, Any] = {}
    n = len(frame)
    for col in feature_cols:
        if col in frame.columns:
            series = frame[col]
        else:
            series = pd.Series(np.full(n, np.nan), index=frame.index)
        if col in cat_cols:
            as_obj = series.astype(object)
            data[col] = as_obj.where(series.notna(), other=np.nan)
        else:
            data[col] = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(data, index=frame.index)


def _looks_paired(
    real: pd.DataFrame, synth: pd.DataFrame, feature_cols: list[str]
) -> bool:
    if len(real) != len(synth) or len(real) == 0 or not feature_cols:
        return False
    real_f = real.loc[:, [c for c in feature_cols if c in real.columns]]
    synth_f = synth.loc[:, [c for c in feature_cols if c in synth.columns]]
    shared = [c for c in real_f.columns if c in synth_f.columns]
    if not shared:
        return False
    left = real_f.loc[:, shared].reset_index(drop=True)
    right = synth_f.loc[:, shared].reset_index(drop=True)
    equal = left.eq(right) | (left.isna() & right.isna())
    return bool(equal.all(axis=1).mean() >= PAIRED_ROW_MATCH)


def _make_preprocessor(
    num_cols: list[str], cat_cols: list[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No feature columns available to build a preprocessor")
    return ColumnTransformer(transformers, remainder="drop")


def _make_pipeline(
    estimator: Any, num_cols: list[str], cat_cols: list[str]
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _make_preprocessor(num_cols, cat_cols)),
            ("model", estimator),
        ]
    )


def _reg_estimators(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "linear": LinearRegression,
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=int(seed),
        ),
    }


def _clf_estimators(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "logistic": lambda: LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=int(seed),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=int(seed),
        ),
    }


def _xy(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    mask = frame[target].notna()
    features = _align_feature_frame(frame.loc[mask], feature_cols, cat_cols)
    labels = frame.loc[mask, target]
    return features, labels


def _safe_split(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify_col: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    if n < 2:
        raise ValueError("Need at least 2 rows to form a train/holdout split")
    n_test = int(np.ceil(n * float(test_size)))
    n_test = min(max(n_test, 1), n - 1)
    actual_size = n_test / n
    stratify = None
    if stratify_col is not None and stratify_col in frame.columns:
        labels = frame[stratify_col]
        if labels.notna().all() and int(labels.nunique()) >= 2:
            counts = labels.value_counts()
            if counts.min() >= 2 and (counts * actual_size).min() >= 1:
                stratify = labels
    try:
        train, test = train_test_split(
            frame,
            test_size=actual_size,
            random_state=int(seed),
            stratify=stratify,
        )
    except ValueError:
        train, test = train_test_split(
            frame,
            test_size=actual_size,
            random_state=int(seed),
            stratify=None,
        )
    return train, test


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return {"r2": None, "rmse": None, "mae": None}
    return {
        "r2": _f(r2_score(y_true, y_pred)),
        "rmse": _f(root_mean_squared_error(y_true, y_pred)),
        "mae": _f(mean_absolute_error(y_true, y_pred)),
    }


def _positive_index(classes: np.ndarray) -> int:
    classes = np.asarray(classes)
    for marker in (1, 1.0, True, "1"):
        hits = np.flatnonzero(classes == marker)
        if hits.size:
            return int(hits[0])
    return int(len(classes) - 1)


def _f1(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> Optional[float]:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred), np.asarray(classes)]))
    if labels.size <= 2:
        pos = 1 if 1 in set(labels.tolist()) else labels[-1]
        return _f(f1_score(y_true, y_pred, average="binary", pos_label=pos, zero_division=0))
    return _f(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _roc_auc(
    y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> Optional[float]:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return _f(roc_auc_score(y_true, proba))
    if proba.shape[1] == 2:
        return _f(roc_auc_score(y_true, proba[:, _positive_index(classes)]))
    try:
        return _f(
            roc_auc_score(
                y_true,
                proba,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except ValueError:
        return None


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    classes: np.ndarray,
) -> dict[str, Optional[float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"accuracy": None, "roc_auc": None, "f1": None}
    return {
        "accuracy": _f(accuracy_score(y_true, y_pred)),
        "roc_auc": _roc_auc(y_true, y_proba, classes) if y_proba is not None else None,
        "f1": _f1(y_true, y_pred, classes),
    }


def _score_reg(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Optional[float]]:
    if len(X) == 0:
        return {"r2": None, "rmse": None, "mae": None}
    pred = pipe.predict(X)
    return _regression_metrics(y.to_numpy(), pred)


def _score_clf(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Optional[float]]:
    if len(X) == 0:
        return {"accuracy": None, "roc_auc": None, "f1": None}
    pred = pipe.predict(X)
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
        except Exception:
            proba = None
    classes = np.asarray(pipe.named_steps["model"].classes_)
    return _classification_metrics(y.to_numpy(), pred, proba, classes)


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    if n < 3:
        return None, None
    a = a[:n]
    b = b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.size < 3:
        return None, None
    if np.std(a) <= EPS or np.std(b) <= EPS:
        if np.allclose(a, b, atol=1e-8, rtol=0.0):
            return 1.0, 1.0
        return None, None
    try:
        sp = spearmanr(a, b).statistic
    except Exception:
        sp = None
    try:
        pr = pearsonr(a, b).statistic
    except Exception:
        pr = None
    return _f(sp), _f(pr)


def _equal_size_pair(
    a: np.ndarray, b: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n = int(min(a.size, b.size))
    if n == 0:
        return a, b
    rng = np.random.default_rng(int(seed))
    if a.size > n:
        a = a[rng.choice(a.size, size=n, replace=False)]
    if b.size > n:
        b = b[rng.choice(b.size, size=n, replace=False)]
    return a, b


def _ks(a: np.ndarray, b: np.ndarray, seed: int) -> tuple[Optional[float], Optional[float], int]:
    left, right = _equal_size_pair(a, b, seed)
    if left.size == 0 or right.size == 0:
        return None, None, 0
    if left.size < 2 or right.size < 2:
        identical = left.size == right.size and np.allclose(left, right, atol=1e-12)
        return (0.0 if identical else 1.0), None, int(min(left.size, right.size))
    try:
        stat, pvalue = ks_2samp(left, right, method="auto")
    except Exception:
        return None, None, int(min(left.size, right.size))
    return _f(stat), _f(pvalue), int(min(left.size, right.size))


def _positive_scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipe.predict_proba(X)
    classes = np.asarray(pipe.named_steps["model"].classes_)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba.astype(float)
    return proba[:, _positive_index(classes)].astype(float)


def _agreement_block(
    scores_real: np.ndarray,
    scores_synth: np.ndarray,
    paired: bool,
    seed: int,
    labels_real: Optional[np.ndarray] = None,
    labels_synth: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    scores_real = np.asarray(scores_real, dtype=float).ravel()
    scores_synth = np.asarray(scores_synth, dtype=float).ravel()
    spearman = pearson = label_agreement = None
    if paired and scores_real.size == scores_synth.size:
        spearman, pearson = _corr(scores_real, scores_synth)
        if labels_real is not None and labels_synth is not None:
            lr = np.asarray(labels_real).ravel()
            ls = np.asarray(labels_synth).ravel()
            m = min(lr.size, ls.size)
            if m:
                label_agreement = _f(np.mean(lr[:m] == ls[:m]))
    ks_stat, ks_p, n_ks = _ks(scores_real, scores_synth, seed)
    return {
        "paired": bool(paired and scores_real.size == scores_synth.size),
        "spearman": spearman if paired else None,
        "pearson": pearson if paired else None,
        "label_agreement": label_agreement if paired else None,
        "pred_mean_real": _f(np.mean(scores_real)) if scores_real.size else None,
        "pred_mean_synth": _f(np.mean(scores_synth)) if scores_synth.size else None,
        "pred_std_real": _f(np.std(scores_real, ddof=1)) if scores_real.size > 1 else None,
        "pred_std_synth": _f(np.std(scores_synth, ddof=1)) if scores_synth.size > 1 else None,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
        "n_real": int(scores_real.size),
        "n_synth": int(scores_synth.size),
        "n_ks": int(n_ks),
    }


def _agreement_score(block: dict[str, Any]) -> Optional[float]:
    ks = block.get("ks_stat")
    ks_term = _clip01(1.0 - ks) if ks is not None else None
    mu_r, mu_s = block.get("pred_mean_real"), block.get("pred_mean_synth")
    sd_r, sd_s = block.get("pred_std_real"), block.get("pred_std_synth")
    moment = None
    if None not in (mu_r, mu_s, sd_r, sd_s):
        mean_pen = abs(mu_r - mu_s) / (abs(mu_r) + abs(mu_s) + abs(sd_r) + abs(sd_s) + EPS)
        std_pen = abs(sd_r - sd_s) / (abs(sd_r) + abs(sd_s) + EPS)
        moment = _clip01(1.0 - 0.5 * mean_pen - 0.5 * std_pen)
    if block.get("paired"):
        corr_bits = []
        for key in ("spearman", "pearson"):
            val = block.get(key)
            if val is not None:
                corr_bits.append(_clip01(val))
        if block.get("label_agreement") is not None:
            corr_bits.append(_clip01(float(block["label_agreement"])))
        corr = float(np.mean(corr_bits)) if corr_bits else None
        parts = [(0.55, corr), (0.45, ks_term)]
    else:
        parts = [(0.70, ks_term), (0.30, moment)]
    return _weighted_mean(parts)


def _weighted_mean(parts: list[tuple[float, Optional[float]]]) -> Optional[float]:
    usable = [(w, v) for w, v in parts if v is not None and np.isfinite(v)]
    if not usable:
        return None
    total_w = sum(w for w, _ in usable)
    if total_w <= 0:
        return None
    return float(sum((w / total_w) * v for w, v in usable))


def _gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def _gap_score(gap: Optional[float]) -> Optional[float]:
    if gap is None:
        return None
    return _clip01(1.0 - abs(gap))


def _empty_reg() -> dict[str, Optional[float]]:
    return {"r2": None, "rmse": None, "mae": None}


def _empty_clf() -> dict[str, Optional[float]]:
    return {"accuracy": None, "roc_auc": None, "f1": None}


def _run_task(
    *,
    task: str,
    real_tr: pd.DataFrame,
    real_te: pd.DataFrame,
    synth_tr: pd.DataFrame,
    synth_te: pd.DataFrame,
    real_all: pd.DataFrame,
    synth_all: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    estimators: dict[str, Callable[[], Any]],
    paired: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_model: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    score_fn = _score_reg if task == "reg" else _score_clf

    for name, factory in estimators.items():
        Xr_tr, yr_tr = _xy(real_tr, feature_cols, cat_cols, target)
        Xr_te, yr_te = _xy(real_te, feature_cols, cat_cols, target)
        Xs_tr, ys_tr = _xy(synth_tr, feature_cols, cat_cols, target)
        Xs_te, ys_te = _xy(synth_te, feature_cols, cat_cols, target)
        empty = _empty_reg() if task == "reg" else _empty_clf()

        if len(Xr_tr) < 2 or len(Xs_tr) < 2:
            block = {
                "trtr": dict(empty),
                "tstr": dict(empty),
                "trts": dict(empty),
                "tsts": dict(empty),
            }
            if task == "reg":
                block["tstr_gap_r2"] = None
            else:
                block["tstr_gap_accuracy"] = None
                block["tstr_gap_roc_auc"] = None
                block["tstr_gap_f1"] = None
            per_model[name] = block
            continue

        real_pipe = _make_pipeline(factory(), num_cols, cat_cols)
        real_pipe.fit(Xr_tr, yr_tr)
        synth_pipe = _make_pipeline(factory(), num_cols, cat_cols)
        synth_pipe.fit(Xs_tr, ys_tr)

        trtr = score_fn(real_pipe, Xr_te, yr_te)
        tstr = score_fn(synth_pipe, Xr_te, yr_te)
        trts = score_fn(real_pipe, Xs_te, ys_te)
        tsts = score_fn(synth_pipe, Xs_te, ys_te)

        block = {"trtr": trtr, "tstr": tstr, "trts": trts, "tsts": tsts}
        if task == "reg":
            block["tstr_gap_r2"] = _gap(trtr.get("r2"), tstr.get("r2"))
        else:
            block["tstr_gap_accuracy"] = _gap(trtr.get("accuracy"), tstr.get("accuracy"))
            block["tstr_gap_roc_auc"] = _gap(trtr.get("roc_auc"), tstr.get("roc_auc"))
            block["tstr_gap_f1"] = _gap(trtr.get("f1"), tstr.get("f1"))
        per_model[name] = block

        X_real_all, _ = _xy(real_all, feature_cols, cat_cols, target)
        X_synth_all, _ = _xy(synth_all, feature_cols, cat_cols, target)
        if task == "reg":
            pred_r = real_pipe.predict(X_real_all)
            pred_s = real_pipe.predict(X_synth_all)
            agreement[name] = _agreement_block(pred_r, pred_s, paired, seed)
        else:
            scores_r = _positive_scores(real_pipe, X_real_all)
            scores_s = _positive_scores(real_pipe, X_synth_all)
            labels_r = real_pipe.predict(X_real_all)
            labels_s = real_pipe.predict(X_synth_all)
            agreement[name] = _agreement_block(
                scores_r,
                scores_s,
                paired,
                seed,
                labels_real=labels_r,
                labels_synth=labels_s,
            )

    return per_model, agreement


def _summarize_reg(per_model: dict[str, Any]) -> dict[str, Any]:
    protocols = ("trtr", "tstr", "trts", "tsts")
    summary: dict[str, Any] = {p: _empty_reg() for p in protocols}
    for proto in protocols:
        for metric in ("r2", "rmse", "mae"):
            summary[proto][metric] = _mean_finite(
                [
                    (per_model[m].get(proto) or {}).get(metric)
                    for m in per_model
                ]
            )
    summary["tstr_gap_r2"] = _mean_finite(
        [per_model[m].get("tstr_gap_r2") for m in per_model]
    )
    return summary


def _summarize_clf(per_model: dict[str, Any]) -> dict[str, Any]:
    protocols = ("trtr", "tstr", "trts", "tsts")
    summary: dict[str, Any] = {p: _empty_clf() for p in protocols}
    for proto in protocols:
        for metric in ("accuracy", "roc_auc", "f1"):
            summary[proto][metric] = _mean_finite(
                [
                    (per_model[m].get(proto) or {}).get(metric)
                    for m in per_model
                ]
            )
    for gap_key in ("tstr_gap_accuracy", "tstr_gap_roc_auc", "tstr_gap_f1"):
        summary[gap_key] = _mean_finite(
            [per_model[m].get(gap_key) for m in per_model]
        )
    return summary


def _clf_tstr_term(summary: dict[str, Any]) -> Optional[float]:
    for gap_key in ("tstr_gap_roc_auc", "tstr_gap_accuracy", "tstr_gap_f1"):
        score = _gap_score(summary.get(gap_key))
        if score is not None:
            return score
    return None


def _utility_score(
    reg_summary: Optional[dict[str, Any]],
    clf_summary: Optional[dict[str, Any]],
    pred_agreement: dict[str, Any],
) -> Optional[float]:
    agree_scores: list[Optional[float]] = []
    for family in ("regression", "classification"):
        block = pred_agreement.get(family) or {}
        for name, inner in block.items():
            if name == "paired" or not isinstance(inner, dict):
                continue
            agree_scores.append(_agreement_score(inner))
    agree = _mean_finite(agree_scores)

    parts: list[tuple[float, Optional[float]]] = []
    if reg_summary is not None:
        parts.append((0.50, _gap_score(reg_summary.get("tstr_gap_r2"))))
    if clf_summary is not None:
        parts.append((0.20, _clf_tstr_term(clf_summary)))
    parts.append((0.30, agree))
    score = _weighted_mean(parts)
    if score is None:
        return None
    return _clip01(score)


def evaluate_model_utility(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_reg: str = "y",
    target_clf: str = "y_class",
    test_size: float = 0.3,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare modelling utility of ``synth`` against ``real``.

    Parameters
    ----------
    real, synth:
        Tables with the same schema.  Feature columns are every column
        except ``target_reg`` and ``target_clf``.  ``region`` / ``segment``
        (if present) are one-hot encoded; remaining numeric columns are
        median-imputed and standardized.  Other object/string columns are
        treated as categorical.  Missing features in ``synth`` are filled
        with NaN and imputed.
    target_reg, target_clf:
        Column names of the continuous and discrete targets.
    test_size, seed:
        Shared ``train_test_split`` arguments.  Classification is
        stratified on ``target_clf`` when that is feasible.  Both tables
        are split independently with the same seed so that ``synth is
        real.copy()`` yields a vanishing TSTR gap.

    Returns
    -------
    dict
        Nested metrics keyed by ``METRIC_KEYS``.  See ``INTERPRETATION``.
    """
    if not isinstance(real, pd.DataFrame) or not isinstance(synth, pd.DataFrame):
        raise TypeError("real and synth must be pandas DataFrames")
    if real.empty or synth.empty:
        raise ValueError("real and synth must be non-empty")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be in (0, 1)")

    real_df = real.copy().reset_index(drop=True)
    synth_df = synth.copy().reset_index(drop=True)

    feature_cols = _feature_columns(real_df, target_reg, target_clf)
    if not feature_cols:
        raise ValueError("No feature columns remain after dropping the targets")

    present_for_roles = [c for c in feature_cols if c in real_df.columns]
    num_cols, cat_cols = _column_roles(real_df.loc[:, present_for_roles], present_for_roles)
    paired = _looks_paired(real_df, synth_df, feature_cols)

    stratify_col = target_clf if target_clf in real_df.columns else None
    real_tr, real_te = _safe_split(real_df, test_size, seed, stratify_col)
    synth_strat = target_clf if target_clf in synth_df.columns else None
    synth_tr, synth_te = _safe_split(synth_df, test_size, seed, synth_strat)

    have_reg = target_reg in real_df.columns and target_reg in synth_df.columns
    have_clf = target_clf in real_df.columns and target_clf in synth_df.columns

    regression: dict[str, Any] = {}
    classification: dict[str, Any] = {}
    pred_agreement: dict[str, Any] = {"paired": bool(paired)}

    if have_reg:
        per_model, agree = _run_task(
            task="reg",
            real_tr=real_tr,
            real_te=real_te,
            synth_tr=synth_tr,
            synth_te=synth_te,
            real_all=real_df,
            synth_all=synth_df,
            target=target_reg,
            feature_cols=feature_cols,
            num_cols=num_cols,
            cat_cols=cat_cols,
            estimators=_reg_estimators(seed),
            paired=paired,
            seed=seed,
        )
        summary = _summarize_reg(per_model)
        regression = {**per_model, "mean": summary, "tstr_gap_r2": summary["tstr_gap_r2"]}
        pred_agreement["regression"] = agree

    if have_clf:
        per_model, agree = _run_task(
            task="clf",
            real_tr=real_tr,
            real_te=real_te,
            synth_tr=synth_tr,
            synth_te=synth_te,
            real_all=real_df,
            synth_all=synth_df,
            target=target_clf,
            feature_cols=feature_cols,
            num_cols=num_cols,
            cat_cols=cat_cols,
            estimators=_clf_estimators(seed),
            paired=paired,
            seed=seed,
        )
        summary = _summarize_clf(per_model)
        classification = {**per_model, "mean": summary}
        pred_agreement["classification"] = agree

    tstr_gap_r2 = regression.get("tstr_gap_r2") if regression else None
    utility = _utility_score(
        regression.get("mean") if regression else None,
        classification.get("mean") if classification else None,
        pred_agreement,
    )

    return {
        "regression": regression,
        "classification": classification,
        "tstr_gap_r2": tstr_gap_r2,
        "pred_agreement": pred_agreement,
        "utility_score": utility,
        "utility_score_formula": UTILITY_SCORE_FORMULA,
        "meta": {
            "n_real": int(len(real_df)),
            "n_synth": int(len(synth_df)),
            "n_real_train": int(len(real_tr)),
            "n_real_test": int(len(real_te)),
            "n_synth_train": int(len(synth_tr)),
            "n_synth_test": int(len(synth_te)),
            "test_size": float(test_size),
            "seed": int(seed),
            "target_reg": target_reg,
            "target_clf": target_clf,
            "feature_cols": feature_cols,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "paired_rows": bool(paired),
            "protocol": {
                "trtr": "train real train → test real holdout",
                "tstr": "train synth train → test the same real holdout",
                "trts": "train real train → test synth holdout",
                "tsts": "train synth train → test synth holdout",
            },
        },
    }


def _make_toy(
    n: int = 420,
    seed: int = 0,
    *,
    randomize_features: bool = False,
    permute_seed: int = 99,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    age = rng.normal(42.0, 10.0, n)
    income = rng.normal(55.0, 12.0, n)
    credit = rng.uniform(0.08, 0.92, n)
    region = rng.choice(
        np.array(["Northeast", "Midwest", "South", "West"], dtype=object), n
    )
    segment = rng.choice(
        np.array(["Mass", "Affluent", "Private"], dtype=object), n
    )
    noise = rng.normal(0.0, 2.5, n)
    y = (
        1.15 * age
        - 0.35 * income
        + 22.0 * credit
        + 7.0 * (region == "West").astype(float)
        + 4.5 * (segment == "Private").astype(float)
        + noise
    )
    y_class = (y > np.median(y)).astype(np.int64)
    frame = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit": credit,
            "region": pd.Series(region, dtype="string"),
            "segment": pd.Series(segment, dtype="string"),
            "y": y,
            "y_class": y_class,
        }
    )
    miss = rng.random(n) < 0.06
    frame.loc[miss, "income"] = np.nan
    if randomize_features:
        prng = np.random.default_rng(int(permute_seed))
        for col in ("age", "income", "credit", "region", "segment"):
            frame[col] = prng.permutation(frame[col].to_numpy())
    return frame


def _self_test() -> dict[str, Any]:
    real = _make_toy(n=420, seed=0)
    identity = evaluate_model_utility(real, real.copy(), test_size=0.3, seed=0)
    random_feat = _make_toy(n=420, seed=0, randomize_features=True, permute_seed=99)
    shuffled = evaluate_model_utility(real, random_feat, test_size=0.3, seed=0)

    id_gap = identity["tstr_gap_r2"]
    sh_gap = shuffled["tstr_gap_r2"]
    id_util = identity["utility_score"]
    sh_util = shuffled["utility_score"]
    id_pair = identity["pred_agreement"]["paired"]
    sh_pair = shuffled["pred_agreement"]["paired"]

    id_rf_r2 = identity["regression"]["random_forest"]["trtr"]["r2"]
    sh_tstr_r2 = shuffled["regression"]["random_forest"]["tstr"]["r2"]

    return {
        "identity_tstr_gap_r2": id_gap,
        "random_tstr_gap_r2": sh_gap,
        "identity_utility_score": id_util,
        "random_utility_score": sh_util,
        "identity_paired": id_pair,
        "random_paired": sh_pair,
        "identity_rf_trtr_r2": id_rf_r2,
        "random_rf_tstr_r2": sh_tstr_r2,
        "identity_tiny_gap": id_gap is not None and abs(id_gap) < 1e-8,
        "random_large_gap": (
            sh_gap is not None
            and id_gap is not None
            and sh_gap > 0.25
            and sh_gap > abs(id_gap) + 0.20
        ),
        "identity_high_utility": id_util is not None and id_util >= 0.90,
        "random_lower_utility": (
            sh_util is not None and id_util is not None and sh_util < id_util - 0.20
        ),
        "identity_trtr_equals_tstr_r2": (
            identity["regression"]["linear"]["trtr"]["r2"]
            == identity["regression"]["linear"]["tstr"]["r2"]
        ),
    }


__all__ = [
    "evaluate_model_utility",
    "INTERPRETATION",
    "METRIC_KEYS",
    "UTILITY_SCORE_FORMULA",
]


if __name__ == "__main__":
    report = _self_test()
    print("METRIC_KEYS:", METRIC_KEYS)
    print("INTERPRETATION:")
    for key, text in INTERPRETATION.items():
        print(f"  {key}: {text}")
    print("UTILITY_SCORE_FORMULA:", UTILITY_SCORE_FORMULA)
    print("SELF-TEST:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    if not report["identity_tiny_gap"]:
        raise SystemExit("identical synth==real did not yield a tiny tstr_gap_r2")
    if not report["random_large_gap"]:
        raise SystemExit("random-feature synth did not yield a large tstr_gap_r2")
    if not report["identity_high_utility"]:
        raise SystemExit("identity utility_score was not high")
    if not report["random_lower_utility"]:
        raise SystemExit("random-feature utility_score was not clearly lower")
    print("self-test OK")
