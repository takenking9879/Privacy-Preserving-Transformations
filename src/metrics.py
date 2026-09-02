"""Utility and reconstruction metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def retention(metric_t: float, metric_raw: float) -> float:
    if metric_raw == 0:
        return float("nan")
    return float(metric_t / metric_raw)


def per_feature_r2(X_true: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    d = X_true.shape[1]
    out = np.empty(d, dtype=np.float64)
    for j in range(d):
        y = X_true[:, j]
        p = X_hat[:, j]
        if np.allclose(y, y[0]):
            out[j] = 1.0 if np.allclose(p, y[0]) else 0.0
        else:
            out[j] = float(r2_score(y, p))
    return out


def reconstruction_report(X_true: np.ndarray, X_hat: np.ndarray) -> dict:
    r2j = per_feature_r2(X_true, X_hat)
    order = np.argsort(r2j)[::-1]
    top5 = [{"index": int(i), "r2": float(r2j[i])} for i in order[:5]]
    return {
        "global_r2": float(r2_score(X_true, X_hat, multioutput="uniform_average")),
        "mean_feature_r2": float(np.mean(r2j)),
        "median_feature_r2": float(np.median(r2j)),
        "max_feature_r2": float(np.max(r2j)),
        "mae": float(mean_absolute_error(X_true, X_hat)),
        "per_feature_r2": r2j.tolist(),
        "top5": top5,
    }


def pairwise_distance_corr(X: np.ndarray, Z: np.ndarray, max_n: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    n = min(len(X), max_n)
    if n < 8:
        return float("nan")
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = X[idx]
    Zs = Z[idx]
    dx = _upper_distances(Xs)
    dz = _upper_distances(Zs)
    if dx.std() < 1e-12 or dz.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(dx, dz)[0, 1])


def _upper_distances(A: np.ndarray) -> np.ndarray:
    g = A @ A.T
    nrm = np.diag(g)
    d2 = nrm[:, None] + nrm[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(A.shape[0], k=1)
    return np.sqrt(d2[iu])
