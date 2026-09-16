"""Evaluate synthetic-data statistical fidelity and TSTR model utility.

Run from the repo root::

    python -m src.evaluate
    SYNTH_N=400 SYNTH_QUICK=1 python -m src.evaluate
    python -m src.evaluate --n 800 --seed 7 --out artifacts/evaluation.json

Environment
-----------
``SYNTH_N``
    Original DGP size (default ``1500``).
``SYNTH_QUICK``
    If ``1`` / ``true``, cap ``n`` at 200 and skip extra diagnostics.
``SYNTH_SEED``
    Master seed (default ``42``).
``SYNTH_OUT``
    JSON destination (default ``/workspace/artifacts/evaluation.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

DEFAULT_N = 1500
QUICK_N_CAP = 200
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.30
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "evaluation.json"
DEFAULT_DATA = ROOT / "data"
SCHEMA_VERSION = "1.0"

# PLAN.md gates (informational; recorded, not used to abort).
GATE_FIDELITY = 0.70
GATE_CORR_FROB = 0.30
GATE_TSTR_R2_GAP = 0.08
GATE_TSTR_AUC_GAP = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if pd.isna(obj):
        return None
    return obj


def _is_categorical(s: pd.Series, max_levels: int = 12) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and int(s.nunique(dropna=True)) <= max_levels:
        return True
    return False


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


def _load_dgp() -> tuple[Callable[[int, int], pd.DataFrame], list[str], list[str], list[str], str]:
    try:
        from src.dgp import FEATURE_COLS, TARGET_CLF, TARGET_REG, generate_original

        return (
            generate_original,
            _as_list(FEATURE_COLS),
            _as_list(TARGET_REG),
            _as_list(TARGET_CLF),
            "src.dgp",
        )
    except ImportError as exc:
        warnings.warn(
            f"src.dgp is not importable ({exc}); using an in-module smoke DGP.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fallback_generate, ["x0", "x1", "x2", "x3", "x4"], ["y"], ["y_class"], "fallback_mock"


def _fallback_generate(n: int, seed: int) -> pd.DataFrame:
    """Minimal correlated DGP so ``python -m src.evaluate`` still runs."""
    rng = np.random.default_rng(int(seed))
    z = rng.normal(size=n)
    x0 = z + 0.25 * rng.normal(size=n)
    x1 = 0.65 * z + 0.75 * rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = (x0 * x1 + 0.25 * rng.normal(size=n) > 0).astype(np.int64)
    x4 = rng.choice(["a", "b", "c"], size=n, p=[0.50, 0.30, 0.20])
    y = 1.1 * x0 + 0.7 * x1 + 1.4 * x0 * x1 + 0.35 * x2 + rng.normal(0.0, 0.45, n)
    y_class = (y + 0.6 * x3 + rng.normal(0.0, 0.35, n) > np.median(y)).astype(np.int64)
    return pd.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": pd.Series(x4, dtype="string"),
            "y": y,
            "y_class": y_class,
        }
    )


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------


def _corr_frobenius_relative(real: pd.DataFrame, synth: pd.DataFrame, cols: list[str]) -> Optional[float]:
    use = [c for c in cols if c in real.columns and c in synth.columns]
    if len(use) < 2:
        return None
    cr = real[use].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    cs = synth[use].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    cr = cr.fillna(0.0)
    cs = cs.fillna(0.0)
    keep = [c for c in use if c in cr.columns and c in cs.columns]
    if len(keep) < 2:
        return None
    a = cr.loc[keep, keep].to_numpy(dtype=float)
    b = cs.loc[keep, keep].to_numpy(dtype=float)
    denom = float(np.linalg.norm(a, ord="fro"))
    if denom <= 1e-12:
        return None
    return float(np.linalg.norm(a - b, ord="fro") / denom)


def _fidelity_fallback(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    """Self-contained fidelity if ``src.metrics.statistical`` is absent."""
    from scipy.stats import ks_2samp, wasserstein_distance

    common = [c for c in real.columns if c in synth.columns]
    ks_vals: list[float] = []
    tvd_vals: list[float] = []
    wass_vals: list[float] = []
    per_column: dict[str, Any] = {}
    numeric_cols: list[str] = []
    cat_cols: list[str] = []

    for c in common:
        rs, ss = real[c], synth[c]
        if _is_categorical(rs) or not pd.api.types.is_numeric_dtype(rs):
            cat_cols.append(c)
            a = rs.dropna().astype(str)
            b = ss.dropna().astype(str)
            if a.empty or b.empty:
                tvd = 1.0
            else:
                pa = a.value_counts(normalize=True)
                pb = b.value_counts(normalize=True)
                idx = pa.index.union(pb.index)
                tvd = 0.5 * float(
                    np.abs(pa.reindex(idx, fill_value=0.0) - pb.reindex(idx, fill_value=0.0)).sum()
                )
            tvd_vals.append(tvd)
            per_column[c] = {"type": "categorical", "tvd": tvd}
        else:
            numeric_cols.append(c)
            a = pd.to_numeric(rs, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(ss, errors="coerce").to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if a.size == 0 or b.size == 0:
                ks, w = 1.0, 1.0
            else:
                ks = float(ks_2samp(a, b, method="auto").statistic)
                scale = float(np.std(a))
                w_raw = float(wasserstein_distance(a, b))
                w = w_raw / scale if scale > 1e-12 else (0.0 if w_raw <= 1e-12 else w_raw)
            ks_vals.append(ks)
            wass_vals.append(w)
            per_column[c] = {"type": "numeric", "ks_statistic": ks, "wasserstein_1_normalized": w}

    def _mean(xs: list[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    ks_mean = _mean(ks_vals)
    tvd_mean = _mean(tvd_vals)
    wass_mean = _mean(wass_vals)
    rel = _corr_frobenius_relative(real, synth, numeric_cols)
    sims = []
    if ks_mean is not None:
        sims.append(_clip01(1.0 - ks_mean))
    if tvd_mean is not None:
        sims.append(_clip01(1.0 - tvd_mean))
    if wass_mean is not None:
        sims.append(float(1.0 / (1.0 + max(wass_mean, 0.0))))
    if rel is not None:
        sims.append(_clip01(1.0 - rel))
    score = float(np.mean(sims)) if sims else 0.0
    return {
        "n_rows_real": int(len(real)),
        "n_rows_synth": int(len(synth)),
        "n_cols_compared": int(len(common)),
        "columns_compared": common,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "per_column": per_column,
        "ks_mean": ks_mean,
        "tvd_mean": tvd_mean,
        "wasserstein_normalized_mean": wass_mean,
        "spearman_corr_frobenius_relative": rel,
        "fidelity_score": _clip01(score),
        "fidelity_score_formula": (
            "fallback: mean of {1-KS, 1-TVD, 1/(1+W1/std), 1-corr_frobenius_rel}"
        ),
        "source": "src.evaluate._fidelity_fallback",
    }


def compute_fidelity(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_reg: Optional[str],
) -> dict[str, Any]:
    source = "src.evaluate._fidelity_fallback"
    details: dict[str, Any]
    try:
        from src.metrics.statistical import compare_xy_relationship, compute_statistical_fidelity

        details = compute_statistical_fidelity(real, synth)
        source = "src.metrics.statistical"
        if target_reg and target_reg in real.columns and target_reg in synth.columns:
            try:
                details["xy_relationship"] = compare_xy_relationship(
                    real, synth, target=target_reg
                )
            except Exception as exc:  # noqa: BLE001
                details["xy_relationship_error"] = str(exc)
    except ImportError:
        details = _fidelity_fallback(real, synth)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"src.metrics.statistical failed ({exc}); using fallback fidelity.",
            RuntimeWarning,
            stacklevel=2,
        )
        details = _fidelity_fallback(real, synth)

    num_cols = details.get("numeric_columns") or []
    rel = details.get("spearman_corr_frobenius_relative")
    if rel is None:
        rel = _corr_frobenius_relative(real, synth, list(num_cols))
    details["spearman_corr_frobenius_relative"] = rel
    details["source"] = source
    if "fidelity_score" not in details or details["fidelity_score"] is None:
        details["fidelity_score"] = 0.0
    return details


# ---------------------------------------------------------------------------
# Utility (TSTR / TRTR / TRTS)
# ---------------------------------------------------------------------------


def _split_kinds(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in cols:
        if c not in df.columns:
            continue
        if _is_categorical(df[c]) or pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(
            df[c]
        ):
            cat.append(c)
        else:
            num.append(c)
    return num, cat


def _preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat,
            )
        )
    if not transformers:
        # sklearn requires at least one transformer
        transformers.append(("passthrough", "passthrough", []))
    return ColumnTransformer(transformers, remainder="drop")


def _reg_pipeline(num: list[str], cat: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(num, cat)),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _clf_pipeline(num: list[str], cat: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(num, cat)),
            (
                "model",
                LogisticRegression(max_iter=400, solver="lbfgs"),
            ),
        ]
    )


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if y_true.size < 2:
        return None
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:  # noqa: BLE001
        return None


def _regression_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Optional[float]]:
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yv)
    if mask.sum() < 2:
        return {"r2": None, "mae": None, "rmse": None}
    pred = np.asarray(model.predict(X.loc[mask]), dtype=float)
    yt = yv[mask]
    return {
        "r2": _safe_r2(yt, pred),
        "mae": float(mean_absolute_error(yt, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yt, pred))),
    }


def _classification_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Optional[float]]:
    yv = y.to_numpy()
    mask = pd.notna(y)
    if int(mask.sum()) < 2:
        return {"auc": None, "acc": None, "f1": None, "brier": None}
    Xv = X.loc[mask]
    yt = y.loc[mask]
    pred = model.predict(Xv)
    acc = float(accuracy_score(yt, pred))
    try:
        f1 = float(f1_score(yt, pred, average="binary" if len(np.unique(yt)) == 2 else "macro"))
    except Exception:  # noqa: BLE001
        f1 = None
    auc: Optional[float] = None
    brier: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(Xv)
            classes = list(getattr(model, "classes_", []))
            if proba.ndim == 2 and proba.shape[1] == 2:
                pos = proba[:, 1]
                auc = float(roc_auc_score(yt, pos))
                # Brier needs 0/1 labels
                y_bin = pd.to_numeric(yt, errors="coerce").to_numpy(dtype=float)
                if np.isfinite(y_bin).all() and set(np.unique(y_bin)).issubset({0.0, 1.0}):
                    brier = float(brier_score_loss(y_bin, pos))
            elif proba.ndim == 2 and proba.shape[1] > 2:
                auc = float(roc_auc_score(yt, proba, multi_class="ovr", average="macro"))
        except Exception:  # noqa: BLE001
            auc = None
    return {"auc": auc, "acc": acc, "f1": f1, "brier": brier}


def _retention(baseline: Optional[float], observed: Optional[float], chance: float = 0.0) -> Optional[float]:
    if baseline is None or observed is None:
        return None
    denom = float(baseline) - chance
    if abs(denom) < 1e-8:
        return 1.0 if abs(float(observed) - float(baseline)) < 1e-6 else 0.0
    return _clip01((float(observed) - chance) / denom)


def _error_retention(baseline: Optional[float], observed: Optional[float]) -> Optional[float]:
    """For MAE/RMSE: smaller is better; 1.0 when synth matches or beats real."""
    if baseline is None or observed is None:
        return None
    if observed <= 1e-12:
        return 1.0
    return _clip01(float(baseline) / float(observed))


def compute_utility(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    feature_cols: list[str],
    target_reg: Optional[str],
    target_clf: Optional[str],
    *,
    real_full: Optional[pd.DataFrame] = None,
    seed: int = 0,
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict[str, Any]:
    """TSTR / TRTR utility. Prefers ``src.metrics.utility`` when importable."""
    try:
        from src.metrics.utility import evaluate_model_utility

        real = real_full if real_full is not None else pd.concat([train, test], ignore_index=True)
        ext = evaluate_model_utility(
            real,
            synth,
            target_reg=target_reg or "y",
            target_clf=target_clf or "y_class",
            test_size=float(test_size),
            seed=int(seed),
        )
        ext["source"] = "src.metrics.utility"
        if ext.get("utility_score") is None:
            ext["utility_score"] = 0.0
        return ext
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"src.metrics.utility failed ({exc}); using fallback TSTR.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _utility_fallback(train, test, synth, feature_cols, target_reg, target_clf)


def _utility_fallback(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    feature_cols: list[str],
    target_reg: Optional[str],
    target_clf: Optional[str],
) -> dict[str, Any]:
    feats = [c for c in feature_cols if c in train.columns]
    num, cat = _split_kinds(train, feats)
    out: dict[str, Any] = {"features": feats, "numeric_features": num, "categorical_features": cat}
    retentions: list[float] = []

    if target_reg and target_reg in train.columns and target_reg in test.columns:
        block: dict[str, Any] = {}
        try:
            ytr = train[target_reg]
            mask = pd.to_numeric(ytr, errors="coerce").notna()
            model = _reg_pipeline(num, cat)
            model.fit(train.loc[mask, feats], pd.to_numeric(ytr.loc[mask], errors="coerce"))
            trtr = _regression_scores(model, test[feats], test[target_reg])
            trts = (
                _regression_scores(model, synth[feats], synth[target_reg])
                if target_reg in synth.columns
                else {"r2": None, "mae": None, "rmse": None}
            )
            model_s = _reg_pipeline(num, cat)
            ysyn = synth[target_reg] if target_reg in synth.columns else None
            if ysyn is not None:
                smask = pd.to_numeric(ysyn, errors="coerce").notna()
                model_s.fit(
                    synth.loc[smask, feats],
                    pd.to_numeric(ysyn.loc[smask], errors="coerce"),
                )
                tstr = _regression_scores(model_s, test[feats], test[target_reg])
            else:
                tstr = {"r2": None, "mae": None, "rmse": None}
            block = {
                "trtr": trtr,
                "tstr": tstr,
                "trts": trts,
                "r2_retention": _retention(trtr.get("r2"), tstr.get("r2"), chance=0.0),
                "mae_retention": _error_retention(trtr.get("mae"), tstr.get("mae")),
            }
            for key in ("r2_retention", "mae_retention"):
                if block[key] is not None:
                    retentions.append(float(block[key]))
        except Exception as exc:  # noqa: BLE001
            block = {"error": str(exc)}
        out["regression"] = block

    if target_clf and target_clf in train.columns and target_clf in test.columns:
        block = {}
        try:
            ytr = train[target_clf]
            mask = ytr.notna()
            # Need at least two classes in the training slice.
            if int(ytr.loc[mask].nunique()) < 2:
                block = {"error": "training classification target has <2 classes"}
            else:
                model = _clf_pipeline(num, cat)
                model.fit(train.loc[mask, feats], ytr.loc[mask])
                trtr = _classification_scores(model, test[feats], test[target_clf])
                trts = (
                    _classification_scores(model, synth[feats], synth[target_clf])
                    if target_clf in synth.columns
                    else {"auc": None, "acc": None, "f1": None, "brier": None}
                )
                model_s = _clf_pipeline(num, cat)
                ysyn = synth[target_clf] if target_clf in synth.columns else None
                if ysyn is not None and int(ysyn.dropna().nunique()) >= 2:
                    smask = ysyn.notna()
                    model_s.fit(synth.loc[smask, feats], ysyn.loc[smask])
                    tstr = _classification_scores(model_s, test[feats], test[target_clf])
                else:
                    tstr = {"auc": None, "acc": None, "f1": None, "brier": None}
                block = {
                    "trtr": trtr,
                    "tstr": tstr,
                    "trts": trts,
                    "auc_retention": _retention(trtr.get("auc"), tstr.get("auc"), chance=0.5),
                    "acc_retention": _retention(trtr.get("acc"), tstr.get("acc"), chance=0.0),
                    "f1_retention": _retention(trtr.get("f1"), tstr.get("f1"), chance=0.0),
                }
                for key in ("auc_retention", "acc_retention"):
                    if block[key] is not None:
                        retentions.append(float(block[key]))
        except Exception as exc:  # noqa: BLE001
            block = {"error": str(exc)}
        out["classification"] = block

    out["utility_score"] = float(np.mean(retentions)) if retentions else 0.0
    out["utility_score_formula"] = (
        "mean of available retentions: "
        "R2 (chance=0), MAE (trtr/tstr), AUC (chance=0.5), accuracy (chance=0)"
    )
    reg = out.get("regression") or {}
    trtr_r2 = (reg.get("trtr") or {}).get("r2") if isinstance(reg, dict) else None
    tstr_r2 = (reg.get("tstr") or {}).get("r2") if isinstance(reg, dict) else None
    if trtr_r2 is not None and tstr_r2 is not None:
        out["tstr_gap_r2"] = float(trtr_r2) - float(tstr_r2)
    out["source"] = "src.evaluate._utility_fallback"
    return out


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _drop_targets(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    drop = [c for c in target_cols if c in df.columns]
    return df.drop(columns=drop) if drop else df


def identity_synth(
    df: pd.DataFrame,
    n: int,
    seed: int,
    include_targets: bool = True,
    target_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Positive control: bootstrap copy of the training table."""
    work = df if include_targets else _drop_targets(df, target_cols or [])
    rng = np.random.default_rng(int(seed))
    n = int(n)
    if n == len(work):
        return work.reset_index(drop=True).copy()
    replace = n > len(work)
    idx = rng.choice(len(work), size=n, replace=replace)
    return work.iloc[idx].reset_index(drop=True)


def negative_control_synth(
    df: pd.DataFrame,
    n: int,
    seed: int,
    include_targets: bool = True,
    target_cols: Optional[list[str]] = None,
    noise_scale: float = 0.15,
) -> pd.DataFrame:
    """Destroy relationships: independent column shuffle + Gaussian noise.

    Marginals stay roughly intact (shuffle is exact; noise only on continuous
    columns), so univariate metrics can remain high while joints / TSTR collapse.
    That is the point: dependence and utility gates must fire.
    """
    work = df if include_targets else _drop_targets(df, target_cols or [])
    rng = np.random.default_rng(int(seed))
    n = int(n)
    idx = rng.choice(len(work), size=n, replace=(n > len(work)))
    out = work.iloc[idx].reset_index(drop=True).copy()

    for col in out.columns:
        out[col] = rng.permutation(out[col].to_numpy())

    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        nunique = int(s.nunique(dropna=True))
        if nunique <= 2:
            continue
        # Leave low-cardinality integer counts as a pure shuffle.
        if pd.api.types.is_integer_dtype(s) and nunique <= 16:
            continue
        vals = pd.to_numeric(s, errors="coerce")
        std = float(vals.std())
        if not np.isfinite(std) or std <= 1e-12:
            continue
        noise = rng.normal(0.0, noise_scale * std, size=len(out))
        noisy = vals.to_numpy(dtype=float) + noise
        na = vals.isna().to_numpy()
        noisy[na] = np.nan
        out[col] = noisy
    return out


def _align_schema(synth: pd.DataFrame, template: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in template.columns if c not in synth.columns]
    extra = [c for c in synth.columns if c not in template.columns]
    if missing:
        warnings.warn(
            f"Synthesizer '{name}' missing columns {missing}; filling with NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        for c in missing:
            synth[c] = np.nan
    if extra:
        synth = synth.drop(columns=extra)
    ordered = synth.loc[:, list(template.columns)].copy()
    return ordered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _nested_r2(block: Any, protocol: str) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    proto = block.get(protocol)
    if isinstance(proto, dict) and proto.get("r2") is not None:
        return float(proto["r2"])
    mean = block.get("mean")
    if isinstance(mean, dict):
        inner = mean.get(protocol)
        if isinstance(inner, dict) and inner.get("r2") is not None:
            return float(inner["r2"])
    return None


def _nested_auc(block: Any, protocol: str) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    proto = block.get(protocol)
    if isinstance(proto, dict):
        for key in ("auc", "roc_auc"):
            if proto.get(key) is not None:
                return float(proto[key])
    mean = block.get("mean")
    if isinstance(mean, dict):
        inner = mean.get(protocol)
        if isinstance(inner, dict):
            for key in ("auc", "roc_auc"):
                if inner.get(key) is not None:
                    return float(inner[key])
    return None


def _gates(fid: dict[str, Any], util: dict[str, Any]) -> dict[str, Any]:
    score = fid.get("fidelity_score")
    rel = fid.get("spearman_corr_frobenius_relative")
    raw_f = fid.get("spearman_corr_frobenius")
    # Prefer relative; fall back to raw/n.
    if rel is None and raw_f is not None:
        n = max(len(fid.get("numeric_columns") or []), 1)
        rel = float(raw_f) / float(n)

    reg = util.get("regression") or {}
    clf = util.get("classification") or {}
    r2_gap = util.get("tstr_gap_r2")
    if r2_gap is None and isinstance(reg, dict):
        r2_gap = reg.get("tstr_gap_r2")
        if r2_gap is None and isinstance(reg.get("mean"), dict):
            r2_gap = reg["mean"].get("tstr_gap_r2")
        if r2_gap is None:
            trtr_r2 = _nested_r2(reg, "trtr")
            tstr_r2 = _nested_r2(reg, "tstr")
            if trtr_r2 is not None and tstr_r2 is not None:
                r2_gap = float(trtr_r2) - float(tstr_r2)
    if r2_gap is not None:
        r2_gap = float(r2_gap)

    trtr_auc = _nested_auc(clf, "trtr")
    tstr_auc = _nested_auc(clf, "tstr")
    auc_gap = (
        float(trtr_auc) - float(tstr_auc)
        if trtr_auc is not None and tstr_auc is not None
        else None
    )

    checks = {
        "fidelity_score_ge_0.70": (score is not None and float(score) >= GATE_FIDELITY),
        "corr_frobenius_rel_le_0.30": (rel is None or float(rel) <= GATE_CORR_FROB),
        "tstr_r2_gap_le_0.08": (r2_gap is None or r2_gap <= GATE_TSTR_R2_GAP),
        "tstr_auc_gap_le_0.05": (auc_gap is None or auc_gap <= GATE_TSTR_AUC_GAP),
    }
    return {
        **checks,
        "r2_gap": r2_gap,
        "auc_gap": auc_gap,
        "corr_frobenius_relative": rel,
        "passes": bool(all(checks.values())),
    }


def _flatten_utility_keys(util: dict[str, Any]) -> dict[str, Any]:
    """Lift report.py-friendly TSTR keys onto the result row."""
    flat: dict[str, Any] = {}
    if util.get("tstr_gap_r2") is not None:
        flat["tstr_gap_r2"] = util["tstr_gap_r2"]
    reg = util.get("regression") or {}
    if isinstance(reg, dict):
        if "tstr_gap_r2" in reg and flat.get("tstr_gap_r2") is None:
            flat["tstr_gap_r2"] = reg.get("tstr_gap_r2")
        mean = reg.get("mean") if isinstance(reg.get("mean"), dict) else {}
        if flat.get("tstr_gap_r2") is None and mean.get("tstr_gap_r2") is not None:
            flat["tstr_gap_r2"] = mean.get("tstr_gap_r2")
        for model_name, dest_tstr, dest_trtr in (
            ("linear", "tstr_r2_linear", "trtr_r2_linear"),
            ("ridge", "tstr_r2_linear", "trtr_r2_linear"),
            ("random_forest", "tstr_r2_rf", "trtr_r2_rf"),
            ("rf", "tstr_r2_rf", "trtr_r2_rf"),
        ):
            block = reg.get(model_name)
            if not isinstance(block, dict):
                continue
            tstr = block.get("tstr") if isinstance(block.get("tstr"), dict) else {}
            trtr = block.get("trtr") if isinstance(block.get("trtr"), dict) else {}
            if tstr.get("r2") is not None:
                flat[dest_tstr] = tstr["r2"]
            if trtr.get("r2") is not None:
                flat[dest_trtr] = trtr["r2"]
        if "tstr_r2_linear" not in flat:
            tstr = _nested_r2(reg, "tstr")
            trtr = _nested_r2(reg, "trtr")
            if tstr is not None:
                flat["tstr_r2_linear"] = tstr
            if trtr is not None:
                flat["trtr_r2_linear"] = trtr
    return flat


def _evaluate_one(
    name: str,
    role: str,
    fn: Callable[..., pd.DataFrame],
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_reg: Optional[str],
    target_clf: Optional[str],
    seed: int,
    *,
    real_full: Optional[pd.DataFrame] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    split_seed: int = 0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result: dict[str, Any] = {
        "name": name,
        "role": role,
        "status": "ok",
        "error": None,
        "n_synthetic": None,
        "elapsed_sec": None,
        "is_negative_control": role == "negative_control",
    }
    try:
        synth = fn(train, n=len(train), seed=seed, include_targets=True)
        if not isinstance(synth, pd.DataFrame):
            raise TypeError(f"{name} returned {type(synth).__name__}, not DataFrame")
        synth = _align_schema(synth, train, name)
        result["n_synthetic"] = int(len(synth))
        data_dir = DEFAULT_DATA
        data_dir.mkdir(parents=True, exist_ok=True)
        synth_path = data_dir / f"synthetic_{name}.csv"
        synth.to_csv(synth_path, index=False)
        result["synth_csv"] = str(synth_path)
        fid = compute_fidelity(train, synth, target_reg=target_reg)
        util = compute_utility(
            train,
            test,
            synth,
            feature_cols,
            target_reg,
            target_clf,
            real_full=real_full,
            seed=split_seed,
            test_size=test_size,
        )
        result["fidelity"] = fid
        result["utility"] = util
        result["fidelity_score"] = float(fid.get("fidelity_score") or 0.0)
        result["utility_score"] = float(util.get("utility_score") or 0.0)
        result["combined_score"] = result["fidelity_score"] + result["utility_score"]
        result["gates"] = _gates(fid, util)
        result.update(_flatten_utility_keys(util))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["fidelity_score"] = 0.0
        result["utility_score"] = 0.0
        result["combined_score"] = 0.0
        warnings.warn(f"Synthesizer '{name}' failed: {exc}", RuntimeWarning, stacklevel=2)
    result["elapsed_sec"] = float(time.perf_counter() - t0)
    return result


def run_evaluation(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    quick: bool = False,
    out_path: Optional[Path] = None,
) -> dict[str, Any]:
    generate_original, feature_cols, target_reg_cols, target_clf_cols, dgp_source = _load_dgp()
    target_cols = target_reg_cols + target_clf_cols
    target_reg = target_reg_cols[0] if target_reg_cols else None
    target_clf = target_clf_cols[0] if target_clf_cols else None

    print(f"[evaluate] DGP={dgp_source}  n={n}  seed={seed}  quick={quick}")
    df = generate_original(int(n), int(seed))
    if not isinstance(df, pd.DataFrame):
        raise TypeError("generate_original must return a pandas DataFrame")

    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in target_cols]

    stratify = None
    if target_clf and target_clf in df.columns:
        y = df[target_clf]
        try:
            if y.notna().all() and y.nunique() >= 2 and type_of_target(y) in {
                "binary",
                "multiclass",
            }:
                stratify = y
        except Exception:  # noqa: BLE001
            stratify = None

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=int(seed),
        stratify=stratify,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"[evaluate] split  train={len(train)}  test={len(test)}")
    DEFAULT_DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_DATA / "original.csv", index=False)
    train.to_csv(DEFAULT_DATA / "original_train.csv", index=False)
    test.to_csv(DEFAULT_DATA / "original_test.csv", index=False)

    try:
        from src.synthesizers.registry import SYNTHESIZERS
    except ImportError as exc:
        warnings.warn(f"Could not import synthesizer registry: {exc}", RuntimeWarning)
        SYNTHESIZERS = {}

    jobs: list[tuple[str, str, Callable[..., pd.DataFrame], int]] = [
        (
            "identity",
            "positive_control",
            lambda d, n, seed, include_targets=True: identity_synth(
                d, n, seed, include_targets, target_cols
            ),
            seed + 1,
        )
    ]
    for name, fn in SYNTHESIZERS.items():
        jobs.append((name, "method", fn, seed + 10 + (sum(map(ord, name)) % 50)))
    jobs.append(
        (
            "negative_control",
            "negative_control",
            lambda d, n, seed, include_targets=True: negative_control_synth(
                d, n, seed, include_targets, target_cols
            ),
            seed + 99,
        )
    )

    results: list[dict[str, Any]] = []
    for name, role, fn, job_seed in jobs:
        print(f"[evaluate] running {name} ({role}) …", flush=True)
        rec = _evaluate_one(
            name,
            role,
            fn,
            train,
            test,
            feature_cols,
            target_reg,
            target_clf,
            job_seed,
            real_full=df,
            test_size=test_size,
            split_seed=int(seed),
        )
        status = rec["status"]
        combo = rec.get("combined_score")
        print(
            f"           status={status}  fidelity={rec.get('fidelity_score')}  "
            f"utility={rec.get('utility_score')}  combined={combo}  "
            f"elapsed={rec.get('elapsed_sec'):.2f}s"
        )
        results.append(rec)

    ranked = sorted(
        results,
        key=lambda r: (
            -(r.get("combined_score") or 0.0),
            -(r.get("utility_score") or 0.0),
            r["name"],
        ),
    )
    leaderboard = []
    for i, r in enumerate(ranked, start=1):
        row = {
            "rank": i,
            "name": r["name"],
            "role": r["role"],
            "status": r["status"],
            "fidelity_score": r.get("fidelity_score"),
            "utility_score": r.get("utility_score"),
            "combined_score": r.get("combined_score"),
            "elapsed_sec": r.get("elapsed_sec"),
            "is_negative_control": bool(r.get("is_negative_control")),
            "tstr_gap_r2": r.get("tstr_gap_r2"),
            "tstr_r2_linear": r.get("tstr_r2_linear"),
            "trtr_r2_linear": r.get("trtr_r2_linear"),
            "tstr_r2_rf": r.get("tstr_r2_rf"),
            "trtr_r2_rf": r.get("trtr_r2_rf"),
        }
        leaderboard.append(row)

    neg = next((r for r in results if r["name"] == "negative_control"), None)
    methods_ok = [r for r in results if r["role"] == "method" and r["status"] == "ok"]
    identity = next((r for r in results if r["name"] == "identity"), None)
    fires = {
        "expected": (
            "negative_control should sit below identity / real synthesizers on "
            "combined_score and fail a dependence or TSTR gate"
        ),
        "negative_control_last": bool(
            leaderboard and leaderboard[-1]["name"] == "negative_control"
        ),
        "failed_a_gate": bool(neg and not (neg.get("gates") or {}).get("passes", True)),
        "below_identity": bool(
            neg
            and identity
            and (neg.get("combined_score") or 0) < (identity.get("combined_score") or 0) - 0.05
        ),
        "below_best_method": bool(
            neg
            and methods_ok
            and (neg.get("combined_score") or 0)
            < max(r.get("combined_score") or 0 for r in methods_ok) - 0.05
        )
        if methods_ok
        else None,
    }
    fires["metrics_fire"] = bool(
        fires["failed_a_gate"] and (fires["below_identity"] or fires["negative_control_last"])
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "n": int(n),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_size": float(test_size),
            "seed": int(seed),
            "quick": bool(quick),
            "dgp_source": dgp_source,
            "feature_cols": feature_cols,
            "target_reg": target_reg,
            "target_clf": target_clf,
            "target_reg_cols": target_reg_cols,
            "target_clf_cols": target_clf_cols,
            "registered_synthesizers": list(SYNTHESIZERS.keys()),
            "output_path": str(out_path) if out_path is not None else None,
        },
        "results": results,
        "leaderboard": leaderboard,
        "negative_control_fires": fires,
        "winner": next(
            (
                r["name"]
                for r in leaderboard
                if r.get("role") == "method" and r.get("status") == "ok"
            ),
            next(
                (r["name"] for r in leaderboard if r.get("status") == "ok"),
                None,
            ),
        ),
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
        payload["meta"]["output_path"] = str(out_path)
        print(f"[evaluate] wrote {out_path}")

    _print_leaderboard(leaderboard, fires)
    return payload


def _print_leaderboard(leaderboard: list[dict[str, Any]], fires: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("SYNTHETIC DATA LEADERBOARD  (sorted by utility_score + fidelity_score)")
    print("=" * 78)
    header = f"{'rank':>4}  {'name':<22} {'role':<18} {'fid':>6} {'util':>6} {'sum':>6}  status"
    print(header)
    print("-" * 78)
    for row in leaderboard:
        fid = row.get("fidelity_score")
        util = row.get("utility_score")
        combo = row.get("combined_score")
        print(
            f"{row['rank']:>4}  {row['name']:<22} {row['role']:<18} "
            f"{_fmt(fid):>6} {_fmt(util):>6} {_fmt(combo):>6}  {row['status']}"
        )
    print("=" * 78)
    print(
        f"negative control fires: metrics_fire={fires.get('metrics_fire')}  "
        f"last={fires.get('negative_control_last')}  "
        f"failed_gate={fires.get('failed_a_gate')}"
    )


def _fmt(x: Any) -> str:
    if x is None:
        return "  n/a"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)[:6]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate synthetic-data fidelity and TSTR utility."
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=f"Original DGP size (default: $SYNTH_N or {DEFAULT_N})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Master seed (default: $SYNTH_SEED or {DEFAULT_SEED})",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke run (also honoured via SYNTH_QUICK=1).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"JSON output path (default: $SYNTH_OUT or {DEFAULT_OUT})",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Hold-out fraction (default {DEFAULT_TEST_SIZE}).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    quick = bool(args.quick or _env_flag("SYNTH_QUICK"))
    n = args.n if args.n is not None else _env_int("SYNTH_N", DEFAULT_N)
    if quick:
        n = min(int(n), QUICK_N_CAP)
    seed = args.seed if args.seed is not None else _env_int("SYNTH_SEED", DEFAULT_SEED)
    out = args.out or os.environ.get("SYNTH_OUT") or str(DEFAULT_OUT)
    return run_evaluation(
        n=int(n),
        seed=int(seed),
        test_size=float(args.test_size),
        quick=quick,
        out_path=Path(out),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
