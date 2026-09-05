"""Phase-1 structural diagnostics: what a transform preserves vs destroys."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, spearmanr, skew

from src.metrics import corr_frobenius_gap, pairwise_distance_corr, rank_preservation


def column_moments(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    feats = np.column_stack(
        [
            A.mean(axis=0),
            A.std(axis=0),
            skew(A, axis=0, bias=False, nan_policy="omit"),
            kurtosis(A, axis=0, bias=False, nan_policy="omit"),
            A.min(axis=0),
            A.max(axis=0),
            (np.abs(A) < 1e-8).mean(axis=0),
        ]
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def analyze_properties(X: np.ndarray, Z: np.ndarray, seed: int) -> dict:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    xm = column_moments(X)
    zm = column_moments(Z)
    xc = np.atleast_2d(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    zc = np.atleast_2d(np.nan_to_num(np.corrcoef(Z, rowvar=False)))
    xeigs = np.sort(np.abs(np.linalg.eigvalsh(np.cov(X, rowvar=False) + 1e-8 * np.eye(X.shape[1]))))[::-1]
    zeigs = np.sort(np.abs(np.linalg.eigvalsh(np.cov(Z, rowvar=False) + 1e-8 * np.eye(Z.shape[1]))))[::-1]
    return {
        "x_mean_abs_skew": float(np.mean(np.abs(xm[:, 2]))),
        "z_mean_abs_skew": float(np.mean(np.abs(zm[:, 2]))),
        "x_mean_kurtosis": float(np.mean(xm[:, 3])),
        "z_mean_kurtosis": float(np.mean(zm[:, 3])),
        "x_sparsity": float(np.mean(xm[:, 6])),
        "z_sparsity": float(np.mean(zm[:, 6])),
        "x_offdiag_corr": float(np.mean(np.abs(xc - np.diag(np.diag(xc))))),
        "z_offdiag_corr": float(np.mean(np.abs(zc - np.diag(np.diag(zc))))),
        "corr_frobenius_gap": corr_frobenius_gap(X, Z),
        "rank_match": rank_preservation(X, Z),
        "distance_corr": pairwise_distance_corr(X, Z, 400, seed),
        "x_eig_top5_frac": float(xeigs[:5].sum() / (xeigs.sum() + 1e-12)),
        "z_eig_top5_frac": float(zeigs[:5].sum() / (zeigs.sum() + 1e-12)),
        "dim_x": int(X.shape[1]),
        "dim_z": int(Z.shape[1]),
        "same_dimension": bool(X.shape[1] == Z.shape[1]),
    }


def pairwise_spearman_leak(X: np.ndarray, Z: np.ndarray) -> float:
    """Max |Spearman| between any raw column and any transformed column."""
    best = 0.0
    for j in range(min(X.shape[1], 16)):
        for k in range(min(Z.shape[1], 24)):
            rho, _ = spearmanr(X[:, j], Z[:, k])
            if np.isfinite(rho):
                best = max(best, abs(float(rho)))
    return float(best)
