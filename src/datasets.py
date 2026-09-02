"""Dataset loaders: one real tabular regression set plus two synthetic suites."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml

from src.config import CACHE_DIR


@dataclass
class DatasetBundle:
    name: str
    kind: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    n: int
    d: int
    description: str


def _as_float(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def load_real_tabular(quick: bool = False) -> DatasetBundle:
    """Type A: real-world tabular regression.

    Preference order:
    1. OpenML superconduct (n≈21263, d=81)
    2. OpenML ailerons (n≈13750, d=40)
    3. California housing expanded with real derived census interactions
       only as a last resort, documented as such.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for name, version in (("superconduct", 1), ("ailerons", 1)):
        try:
            ds = fetch_openml(
                name=name,
                version=version,
                as_frame=False,
                parser="auto",
                data_home=str(CACHE_DIR),
            )
            X, y = _as_float(ds.data, ds.target)
            feature_names = [str(c) for c in ds.feature_names]
            if X.shape[1] < 8:
                raise ValueError(f"{name} has too few features: {X.shape[1]}")
            if quick:
                rng = np.random.RandomState(0)
                idx = rng.choice(len(X), size=min(4000, len(X)), replace=False)
                X, y = X[idx], y[idx]
            return DatasetBundle(
                name=f"openml_{name}",
                kind="real",
                X=X,
                y=y,
                feature_names=feature_names,
                n=int(X.shape[0]),
                d=int(X.shape[1]),
                description=(
                    f"OpenML '{name}' regression, n={X.shape[0]}, d={X.shape[1]}. "
                    "Real tabular features; target is the dataset default."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")

    housing = fetch_california_housing()
    X, y = _as_float(housing.data, housing.target)
    # Expand with documented derived features so d >= 20 while remaining
    # a function of the original census variables (still real measurements).
    medinc, houseage, rooms, bedrooms, pop, occup, lat, lon = X.T
    derived = np.column_stack(
        [
            rooms / np.clip(occup, 1e-6, None),
            bedrooms / np.clip(rooms, 1e-6, None),
            pop / np.clip(occup, 1e-6, None),
            medinc * occup,
            medinc / np.clip(houseage, 1e-6, None),
            np.log1p(pop),
            np.log1p(np.abs(rooms)),
            np.sin(np.deg2rad(lat)),
            np.cos(np.deg2rad(lat)),
            np.sin(np.deg2rad(lon)),
            np.cos(np.deg2rad(lon)),
            lat * lon,
            medinc**2,
            occup**2,
            houseage * medinc,
        ]
    )
    X_full = np.column_stack([X, derived])
    names = list(housing.feature_names) + [
        "rooms_per_occup",
        "bedroom_ratio",
        "households",
        "inc_x_occup",
        "inc_over_age",
        "log_pop",
        "log_rooms",
        "sin_lat",
        "cos_lat",
        "sin_lon",
        "cos_lon",
        "lat_x_lon",
        "medinc_sq",
        "occup_sq",
        "age_x_inc",
    ]
    if quick:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X_full), size=min(4000, len(X_full)), replace=False)
        X_full, y = X_full[idx], y[idx]
    return DatasetBundle(
        name="california_housing_expanded",
        kind="real",
        X=X_full,
        y=y,
        feature_names=names,
        n=int(X_full.shape[0]),
        d=int(X_full.shape[1]),
        description=(
            "California housing (8 census measurements) plus 15 derived real-valued "
            f"transforms of those measurements. n={X_full.shape[0]}, d={X_full.shape[1]}. "
            "Fallback used because OpenML downloads failed: " + " | ".join(errors)
        ),
    )


def _heterogeneous_observe(h_col: np.ndarray, kind: int, rng: np.random.RandomState) -> np.ndarray:
    z = h_col
    if kind == 0:
        return z + rng.normal(0.0, 0.15, size=z.shape)
    if kind == 1:
        return np.exp(0.55 * z) + rng.normal(0.0, 0.05, size=z.shape)
    if kind == 2:
        return 1.0 / (1.0 + np.exp(-1.6 * z)) + rng.normal(0.0, 0.02, size=z.shape)
    if kind == 3:
        return np.sinh(0.6 * z) + rng.standard_t(5, size=z.shape) * 0.1
    if kind == 4:
        return np.clip(z, -1.5, 1.5) + rng.uniform(-0.05, 0.05, size=z.shape)
    if kind == 5:
        return z**2 + rng.normal(0.0, 0.1, size=z.shape)
    if kind == 6:
        return np.abs(z) + rng.exponential(0.08, size=z.shape)
    return np.tan(np.clip(0.4 * z, -1.2, 1.2)) + rng.normal(0.0, 0.08, size=z.shape)


def make_banking_synthetic(
    n: int,
    d: int,
    q: int = 12,
    seed: int = 0,
    name: str = "banking_synthetic",
) -> DatasetBundle:
    """Type B/C: correlated latents, heterogeneous marginals, nonlinear target.

    y depends on only a subset of latents, so X contains surplus information.
    """
    rng = np.random.RandomState(seed)
    corr = 0.55
    sigma = corr * np.ones((q, q)) + (1.0 - corr) * np.eye(q)
    L = np.linalg.cholesky(sigma)
    H = rng.normal(size=(n, q)) @ L.T

    X = np.zeros((n, d))
    names: list[str] = []
    families = [
        "income",
        "balance",
        "utilization",
        "inquiries",
        "tenure",
        "limit",
        "spend",
        "delinq",
    ]
    for j in range(d):
        latent = H[:, j % q]
        mix = 0.25 * H[:, (j * 3) % q]
        kind = j % 8
        X[:, j] = _heterogeneous_observe(latent + mix, kind, rng)
        names.append(f"{families[j % len(families)]}_{j:03d}")

    noise = rng.normal(0.0, 0.35, size=n)
    h = lambda i: H[:, i % q]
    y = (
        1.4 * h(0)
        + 0.9 * h(1)
        + 0.8 * h(2) * h(3)
        + 1.1 * np.sin(1.3 * h(4))
        + 0.6 * (h(5) ** 2)
        + noise
    )
    return DatasetBundle(
        name=name,
        kind="synthetic",
        X=X,
        y=y,
        feature_names=names,
        n=n,
        d=d,
        description=(
            f"Synthetic banking-like tabular data n={n}, d={d}, q={q} latents. "
            "Heterogeneous observation maps; y is a nonlinear function of 6 latents."
        ),
    )


def load_all_datasets(quick: bool = False) -> list[DatasetBundle]:
    real = load_real_tabular(quick=quick)
    if quick:
        b = make_banking_synthetic(n=3500, d=40, q=8, seed=1, name="banking_d40")
        c = make_banking_synthetic(n=3500, d=80, q=8, seed=2, name="banking_d80")
    else:
        b = make_banking_synthetic(n=18000, d=80, q=12, seed=1, name="banking_d80")
        c = make_banking_synthetic(n=18000, d=200, q=12, seed=2, name="banking_d200")
    return [real, b, c]


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(round(0.15 * n))
    n_aux = int(round(0.15 * n))
    test_idx = np.sort(perm[:n_test])
    aux_idx = np.sort(perm[n_test : n_test + n_aux])
    train_idx = np.sort(perm[n_test + n_aux :])
    return {"train": train_idx, "test": test_idx, "aux": aux_idx}
