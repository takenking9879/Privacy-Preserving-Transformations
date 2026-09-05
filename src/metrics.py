"""Utility, reconstruction, and structure metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or np.allclose(y_true, y_true[0]):
        return 1.0 if np.allclose(y_pred, y_true) else 0.0
    return float(r2_score(y_true, y_pred))


def per_feature_r2(X_true: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    d = X_true.shape[1]
    out = np.empty(d, dtype=np.float64)
    for j in range(d):
        out[j] = safe_r2(X_true[:, j], X_hat[:, j])
    return out


def reconstruction_report(X_true: np.ndarray, X_hat: np.ndarray) -> dict:
    X_true = np.asarray(X_true, dtype=np.float64)
    X_hat = np.asarray(X_hat, dtype=np.float64)
    if X_hat.shape != X_true.shape:
        # Pad / crop so attacks on reduced representations still score.
        n = min(X_true.shape[0], X_hat.shape[0])
        d = min(X_true.shape[1], X_hat.shape[1])
        aligned = np.zeros_like(X_true[:n])
        aligned[:, :d] = X_hat[:n, :d]
        X_hat = aligned
        X_true = X_true[:n]
    r2j = per_feature_r2(X_true, X_hat)
    order = np.argsort(r2j)[::-1]
    return {
        "global_r2": float(r2_score(X_true, X_hat, multioutput="uniform_average"))
        if X_true.size
        else float("nan"),
        "mean_feature_r2": float(np.mean(r2j)) if r2j.size else float("nan"),
        "median_feature_r2": float(np.median(r2j)) if r2j.size else float("nan"),
        "max_feature_r2": float(np.max(r2j)) if r2j.size else float("nan"),
        "mae": float(mean_absolute_error(X_true, X_hat)) if X_true.size else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(X_true, X_hat))) if X_true.size else float("nan"),
        "per_feature_r2": r2j.tolist(),
        "top5": [{"index": int(i), "r2": float(r2j[i])} for i in order[: min(5, len(order))]],
    }


def retention(transformed: float, raw: float) -> float:
    if raw is None or not np.isfinite(raw) or abs(raw) < 1e-12:
        return float("nan")
    return float(transformed / raw)


def pairwise_distance_corr(X: np.ndarray, Z: np.ndarray, max_n: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    n = min(len(X), max_n)
    if n < 8:
        return float("nan")
    idx = rng.choice(len(X), size=n, replace=False)
    dx = _upper_distances(X[idx])
    dz = _upper_distances(Z[idx])
    if dx.std() < 1e-12 or dz.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(dx, dz)[0, 1])


def rank_preservation(X: np.ndarray, Z: np.ndarray, max_cols: int = 8) -> float:
    """Mean |Spearman| between each X column and its best-matching Z column."""
    d = min(X.shape[1], Z.shape[1], max_cols)
    if d == 0:
        return float("nan")
    scores = []
    for j in range(d):
        best = 0.0
        for k in range(min(Z.shape[1], 24)):
            rho, _ = spearmanr(X[:, j], Z[:, k])
            if np.isfinite(rho):
                best = max(best, abs(float(rho)))
        scores.append(best)
    return float(np.mean(scores))


def corr_frobenius_gap(X: np.ndarray, Z: np.ndarray) -> float:
    d = min(X.shape[1], Z.shape[1])
    if d < 2:
        return float("nan")
    cx = np.nan_to_num(np.corrcoef(X[:, :d], rowvar=False))
    cz = np.nan_to_num(np.corrcoef(Z[:, :d], rowvar=False))
    return float(np.linalg.norm(cx - cz) / d)


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if y_proba is not None and len(np.unique(y_true)) == 2:
        proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def _upper_distances(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    g = A @ A.T
    nrm = np.diag(g)
    d2 = nrm[:, None] + nrm[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(A.shape[0], k=1)
    return np.sqrt(d2[iu])
