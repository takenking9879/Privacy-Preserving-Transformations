"""Attack suite against privacy-preserving representations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis, skew, ortho_group
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import DISTANCE_SUBSAMPLE, KNOWN_PAIR_GRID, NEURAL_ATTACK_SUBSAMPLE
from src.metrics import pairwise_distance_corr, reconstruction_report
from src.transforms import PrivacyTransform

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


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
        ]
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def statistical_inspection(X: np.ndarray, Z: np.ndarray) -> dict[str, Any]:
    xm = column_moments(X)
    zm = column_moments(Z)
    xc = np.corrcoef(X, rowvar=False)
    zc = np.corrcoef(Z, rowvar=False)
    xc = np.atleast_2d(np.nan_to_num(xc))
    zc = np.atleast_2d(np.nan_to_num(zc))
    xeigs = np.sort(np.abs(np.linalg.eigvalsh(np.cov(X, rowvar=False))))[::-1]
    zeigs = np.sort(np.abs(np.linalg.eigvalsh(np.cov(Z, rowvar=False))))[::-1]
    return {
        "x_mean_abs_skew": float(np.mean(np.abs(xm[:, 2]))),
        "z_mean_abs_skew": float(np.mean(np.abs(zm[:, 2]))),
        "x_mean_kurtosis": float(np.mean(xm[:, 3])),
        "z_mean_kurtosis": float(np.mean(zm[:, 3])),
        "x_offdiag_corr_mean_abs": float(np.mean(np.abs(xc - np.diag(np.diag(xc))))),
        "z_offdiag_corr_mean_abs": float(np.mean(np.abs(zc - np.diag(np.diag(zc))))),
        "x_eig_top5_frac": float(xeigs[:5].sum() / (xeigs.sum() + 1e-12)),
        "z_eig_top5_frac": float(zeigs[:5].sum() / (zeigs.sum() + 1e-12)),
        "x_range_median": float(np.median(xm[:, 5] - xm[:, 4])),
        "z_range_median": float(np.median(zm[:, 5] - zm[:, 4])),
    }


def _signature_match_accuracy(src: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    if src.shape[1] != tgt.shape[1]:
        d = min(src.shape[1], tgt.shape[1])
        src = src[:, :d]
        tgt = tgt[:, :d]
    s = column_moments(src)
    t = column_moments(tgt)
    s = (s - s.mean(axis=0)) / (s.std(axis=0) + 1e-8)
    t = (t - t.mean(axis=0)) / (t.std(axis=0) + 1e-8)
    d2 = ((s[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
    ri, ci = linear_sum_assignment(d2)
    acc = float(np.mean(ri == ci))
    return {
        "assignment_accuracy": acc,
        "random_baseline": 1.0 / src.shape[1],
        "accuracy_over_random": acc * src.shape[1],
    }


def semantic_matching(X: np.ndarray, Z: np.ndarray, X_aux: np.ndarray) -> dict[str, Any]:
    unpaired = _signature_match_accuracy(X_aux, Z)
    d = min(X.shape[1], Z.shape[1])
    Xs, Zs = X[:, :d], Z[:, :d]
    C = np.abs(np.corrcoef(np.hstack([Xs, Zs]), rowvar=False)[:d, d:])
    C = np.nan_to_num(C)
    ri, ci = linear_sum_assignment(-C)
    paired_acc = float(np.mean(ri == ci))
    matched_corr = float(C[ri, ci].mean()) if len(ri) else 0.0
    return {
        "unpaired_moment_match": unpaired,
        "paired_corr_assignment_accuracy": paired_acc,
        "paired_matched_mean_abs_corr": matched_corr,
        "max_abs_corr": float(C.max()) if C.size else 0.0,
        "random_baseline": 1.0 / d if d else float("nan"),
    }


def _fit_predict_multi(model, Z_tr, X_tr, Z_te) -> np.ndarray:
    model.fit(Z_tr, X_tr)
    return model.predict(Z_te)


def known_pair_attacks(
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_test: np.ndarray,
    Z_test: np.ndarray,
    X_aux: np.ndarray,
    transform: PrivacyTransform,
    seed: int,
    heavy: bool,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    n_train = len(X_train)
    d = X_train.shape[1]
    grid = [n for n in KNOWN_PAIR_GRID if n < 0.5 * n_train]
    if not grid:
        grid = [max(1, n_train // 10)]

    curves: list[dict] = []
    for n in grid:
        idx = rng.choice(n_train, size=n, replace=False)
        Zk, Xk = Z_train[idx], X_train[idx]
        row: dict[str, Any] = {"n": int(n)}

        ridge = Ridge(alpha=1.0)
        pred = _fit_predict_multi(ridge, Zk, Xk, Z_test)
        row["ridge"] = reconstruction_report(X_test, pred)

        try:
            pinv = np.linalg.pinv(Zk)
            W = pinv @ Xk
            pred = Z_test @ W
            row["pinv"] = reconstruction_report(X_test, pred)
        except Exception as exc:  # noqa: BLE001
            row["pinv"] = {"error": str(exc)}

            row["informed_procrustes"] = None
            if n in {10, 50, 100, 500, 1000} or n == grid[-1]:
                row["informed_procrustes"] = _informed_procrustes(
                    Xk, Zk, X_test, Z_test, X_aux, transform
                )

        if heavy and n in {100, 500} and n >= 20:
            try:
                row["mlp"] = _torch_mlp_attack(Zk, Xk, Z_test, X_test, seed, hidden=(64, 64), epochs=25)
            except Exception as exc:  # noqa: BLE001
                row["mlp"] = {"error": str(exc)}

        curves.append(row)

    worst = []
    for row in curves:
        r = row.get("ridge", {})
        if isinstance(r, dict) and "global_r2" in r:
            worst.append((row["n"], r["global_r2"], r["max_feature_r2"]))
    n_for_r2_05 = None
    for n, g, _ in worst:
        if g >= 0.5:
            n_for_r2_05 = n
            break
    return {
        "curve": curves,
        "n_for_ridge_global_r2_0.5": n_for_r2_05,
        "known_pair_robustness": _robustness_label(worst, d),
    }


def _robustness_label(points: list[tuple[int, float, float]], d: int) -> str:
    if not points:
        return "unknown"
    large = [g for n, g, _ in points if n >= max(d, 50)]
    if large and max(large) >= 0.9:
        return "collapses_with_enough_pairs"
    if large and max(large) >= 0.5:
        return "partial_recovery_with_enough_pairs"
    if points and max(g for _, g, _ in points) >= 0.5:
        return "partial_recovery"
    return "resistant_in_tested_budget"


def _informed_procrustes(
    Xk: np.ndarray,
    Zk: np.ndarray,
    X_test: np.ndarray,
    Z_test: np.ndarray,
    X_aux: np.ndarray,
    transform: PrivacyTransform,
) -> dict:
    """Algorithm-aware attack: fit G,W on auxiliary X, recover R from pairs."""
    try:
        from sklearn.preprocessing import QuantileTransformer

        spec = transform.spec
        if spec.bottleneck or spec.name == "raw":
            ridge = Ridge(alpha=1.0)
            pred = _fit_predict_multi(ridge, Zk, Xk, Z_test)
            return reconstruction_report(X_test, pred)

        n_q = min(1000, len(X_aux))
        g = QuantileTransformer(
            n_quantiles=max(10, n_q),
            output_distribution="normal",
            subsample=len(X_aux),
            random_state=0,
            copy=True,
        )
        V_aux = g.fit_transform(X_aux)
        mu = V_aux.mean(axis=0)
        cov = np.cov(V_aux - mu, rowvar=False) + 1e-6 * np.eye(V_aux.shape[1])
        u, s, _ = np.linalg.svd(cov, full_matrices=False)
        W = (u * (1.0 / np.sqrt(s))) @ u.T
        W_inv = (u * np.sqrt(s)) @ u.T

        Vk = (g.transform(Xk) - mu) @ W
        kdim = min(Vk.shape[1], Zk.shape[1])
        Vk_u, _, Zk_u = np.linalg.svd(Vk[:, :kdim].T @ Zk[:, :kdim], full_matrices=False)
        R_hat = Vk_u @ Zk_u
        V_hat = Z_test[:, :kdim] @ R_hat.T
        if V_hat.shape[1] < W_inv.shape[0]:
            pad = np.zeros((len(V_hat), W_inv.shape[0]))
            pad[:, : V_hat.shape[1]] = V_hat
            V_hat = pad
        Xg = V_hat[:, : W_inv.shape[0]] @ W_inv + mu
        X_hat = g.inverse_transform(Xg)
        return reconstruction_report(X_test, X_hat)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def auxiliary_unpaired_attack(
    X_aux: np.ndarray,
    Z_train: np.ndarray,
    X_test: np.ndarray,
    Z_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """No row pairs. Attacker knows the algorithm class but not the secret key."""
    rng = np.random.RandomState(seed)
    out: dict[str, Any] = {}

    # Random orthogonal inverse in Z-space cannot recover instance values.
    k = Z_test.shape[1]
    d = X_test.shape[1]
    R_rand = ortho_group.rvs(k, random_state=rng) if k > 1 else np.ones((1, 1))
    mapped = Z_test @ R_rand
    if mapped.shape[1] < d:
        pad = rng.normal(size=(len(mapped), d))
        pad[:, : mapped.shape[1]] = mapped
        mapped = pad
    elif mapped.shape[1] > d:
        mapped = mapped[:, :d]
    # Scale columns to aux moments — marginal matching without pairing.
    aux_mean, aux_std = X_aux.mean(axis=0), X_aux.std(axis=0) + 1e-8
    mapped = (mapped - mapped.mean(axis=0)) / (mapped.std(axis=0) + 1e-8)
    mapped = mapped * aux_std[: mapped.shape[1]] + aux_mean[: mapped.shape[1]]
    out["random_rotation_moment_match"] = reconstruction_report(X_test, mapped)

    # ICA on Z, match independent components to aux columns by moments.
    try:
        import warnings

        ica = FastICA(
            n_components=min(k, 12),
            random_state=seed,
            max_iter=400,
            tol=1e-3,
            whiten="unit-variance",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_sub = min(len(Z_train), 4000)
            ica.fit(Z_train[:n_sub])
            S = ica.transform(Z_test)
        d_m = min(S.shape[1], X_test.shape[1])
        cost = np.zeros((d_m, d_m))
        sm = column_moments(S[:, :d_m])
        xm = column_moments(X_aux[:, :d_m])
        sm = (sm - sm.mean(0)) / (sm.std(0) + 1e-8)
        xm = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
        cost = ((sm[:, None, :] - xm[None, :, :]) ** 2).sum(axis=2)
        ri, ci = linear_sum_assignment(cost)
        X_hat = np.zeros_like(X_test[:, :d_m])
        for a, b in zip(ri, ci):
            col = S[:, a]
            X_hat[:, b] = (col - col.mean()) / (col.std() + 1e-8) * X_aux[:, b].std() + X_aux[:, b].mean()
        full = np.zeros_like(X_test)
        full[:, :d_m] = X_hat
        out["ica_moment_align"] = reconstruction_report(X_test, full)
    except Exception as exc:  # noqa: BLE001
        out["ica_moment_align"] = {"error": str(exc)}

    # CCA between Z and aux is not row-aligned; report as inapplicable and
    # instead try CCA on randomly paired equal-length prefixes as a naive baseline.
    try:
        n = min(len(X_aux), len(Z_train), 2000)
        cca = CCA(n_components=min(8, k, d, n - 2))
        cca.fit(Z_train[:n], X_aux[:n])
        pred = cca.predict(Z_test)
        if pred.shape[1] < d:
            pad = np.zeros((len(pred), d))
            pad[:, : pred.shape[1]] = pred
            pred = pad
        out["naive_cca_shuffled_aux"] = reconstruction_report(X_test, pred[:, :d])
    except Exception as exc:  # noqa: BLE001
        out["naive_cca_shuffled_aux"] = {"error": str(exc)}

    return out


def _torch_mlp_attack(
    Z_tr, X_tr, Z_te, X_te, seed: int, hidden: tuple[int, ...] = (128, 64), epochs: int = 30
) -> dict:
    if torch is None or nn is None:
        return {"error": "torch unavailable"}
    scaler_z = StandardScaler()
    scaler_x = StandardScaler()
    Zs = scaler_z.fit_transform(Z_tr)
    Xs = scaler_x.fit_transform(X_tr)
    Zte = scaler_z.transform(Z_te)
    layers: list[nn.Module] = []
    d = Zs.shape[1]
    for h in hidden:
        layers.extend([nn.Linear(d, h), nn.ReLU()])
        d = h
    layers.append(nn.Linear(d, Xs.shape[1]))
    torch.manual_seed(seed)
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.tensor(Zs, dtype=torch.float32)
    yt = torch.tensor(Xs, dtype=torch.float32)
    n = len(xt)
    batch = min(256, n)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            sl = perm[i : i + batch]
            pred = model(xt[sl])
            loss = torch.mean((pred - yt[sl]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Zte, dtype=torch.float32)).numpy()
    pred = scaler_x.inverse_transform(pred)
    return reconstruction_report(X_te, pred)


def neural_reconstruction_attacks(
    Z_train: np.ndarray,
    X_train: np.ndarray,
    Z_test: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = min(len(Z_train), NEURAL_ATTACK_SUBSAMPLE)
    idx = rng.choice(len(Z_train), size=n, replace=False)
    Ztr, Xtr = Z_train[idx], X_train[idx]
    results: dict[str, Any] = {}
    try:
        results["mlp_small"] = _torch_mlp_attack(Ztr, Xtr, Z_test, X_test, seed, hidden=(64,), epochs=25)
    except Exception as exc:  # noqa: BLE001
        results["mlp_small"] = {"error": str(exc)}
    try:
        results["mlp_medium"] = _torch_mlp_attack(
            Ztr, Xtr, Z_test, X_test, seed, hidden=(128, 128), epochs=30
        )
    except Exception as exc:  # noqa: BLE001
        results["mlp_medium"] = {"error": str(exc)}
    scaler_z = StandardScaler()
    scaler_x = StandardScaler()
    Zs = scaler_z.fit_transform(Ztr)
    Xs = scaler_x.fit_transform(Xtr)
    Zte = scaler_z.transform(Z_test)
    results["mlp_residual"] = _residual_attack(Zs, Xs, Zte, scaler_x, X_test, seed)
    return results


def _residual_attack(Zs, Xs, Zte, scaler_x, X_test, seed: int) -> dict:
    if torch is None or nn is None:
        return {"error": "torch unavailable"}

    class ResBlock(nn.Module):
        def __init__(self, h: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(h, h)
            self.fc2 = nn.Linear(h, h)

        def forward(self, x):
            return x + torch.relu(self.fc2(torch.relu(self.fc1(x))))

    class ResidualMLP(nn.Module):
        def __init__(self, d_in: int, d_out: int, h: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, h),
                nn.ReLU(),
                ResBlock(h),
                ResBlock(h),
                ResBlock(h),
                nn.Linear(h, d_out),
            )

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(seed)
    model = ResidualMLP(Zs.shape[1], Xs.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.tensor(Zs, dtype=torch.float32)
    yt = torch.tensor(Xs, dtype=torch.float32)
    model.train()
    n = len(xt)
    batch = 256
    for _ in range(25):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            sl = perm[i : i + batch]
            pred = model(xt[sl])
            loss = torch.mean((pred - yt[sl]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Zte, dtype=torch.float32)).numpy()
    pred = scaler_x.inverse_transform(pred)
    return reconstruction_report(X_test, pred)


def run_attacks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_aux: np.ndarray,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    Z_aux: np.ndarray,
    transform: PrivacyTransform,
    seed: int,
    heavy: bool,
) -> dict[str, Any]:
    del y_train, y_test, Z_aux
    stats = statistical_inspection(X_train, Z_train)
    semantic = semantic_matching(X_train, Z_train, X_aux)
    dist = pairwise_distance_corr(X_test, Z_test, DISTANCE_SUBSAMPLE, seed)
    known = known_pair_attacks(
        X_train, Z_train, X_test, Z_test, X_aux, transform, seed, heavy=heavy
    )
    aux = auxiliary_unpaired_attack(X_aux, Z_train, X_test, Z_test, seed)

    # Representative leakage from the largest ridge known-pair budget tested.
    last_ridge = None
    for row in reversed(known["curve"]):
        if isinstance(row.get("ridge"), dict) and "global_r2" in row["ridge"]:
            last_ridge = row["ridge"]
            break

    out: dict[str, Any] = {
        "statistical_inspection": stats,
        "semantic_matching": semantic,
        "distance_preservation": dist,
        "known_pair": known,
        "auxiliary_unpaired": aux,
        "summary_recon_r2_knownpair_ridge": None if last_ridge is None else last_ridge["global_r2"],
        "summary_worst_attr_knownpair_ridge": None if last_ridge is None else last_ridge["max_feature_r2"],
        "known_pair_robustness": known["known_pair_robustness"],
    }
    if heavy:
        neural = neural_reconstruction_attacks(Z_train, X_train, Z_test, X_test, seed)
        out["neural"] = neural
        best = None
        for v in neural.values():
            if isinstance(v, dict) and "global_r2" in v:
                if best is None or v["global_r2"] > best["global_r2"]:
                    best = v
        out["neural_best"] = best
    return out
