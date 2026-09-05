"""Attacker A (known-pair) and Attacker B (black-box semantic/statistical)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.metrics import reconstruction_report
from src.properties import column_moments, pairwise_spearman_leak


KNOWN_PAIR_GRID = [1, 2, 5, 10, 25, 50, 100, 200]


def _fit_predict_multi(model, Z_tr, X_tr, Z_te) -> np.ndarray:
    model.fit(Z_tr, X_tr)
    return np.asarray(model.predict(Z_te), dtype=np.float64)


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


def _rank_attack(Zk: np.ndarray, Xk: np.ndarray, Z_test: np.ndarray, X_test: np.ndarray) -> dict:
    """Recover each X column from the Z column with highest Spearman on pairs."""
    d = Xk.shape[1]
    X_hat = np.zeros_like(X_test)
    used = []
    for j in range(d):
        best_k, best_abs = 0, -1.0
        for k in range(Zk.shape[1]):
            rho, _ = spearmanr(Xk[:, j], Zk[:, k])
            rho = 0.0 if not np.isfinite(rho) else abs(float(rho))
            if rho > best_abs:
                best_abs, best_k = rho, k
        used.append((j, best_k, best_abs))
        # Monotone map via ranks on the matched column.
        order = np.argsort(Zk[:, best_k])
        x_sorted = np.sort(Xk[:, j])
        # Map test ranks into the known-pair value range.
        ranks = np.argsort(np.argsort(Z_test[:, best_k]))
        idx = np.clip((ranks * (len(x_sorted) - 1) / max(len(ranks) - 1, 1)).astype(int), 0, len(x_sorted) - 1)
        X_hat[:, j] = x_sorted[idx]
    report = reconstruction_report(X_test, X_hat)
    report["matches"] = used[:8]
    return report


def _frequency_cat_attack(
    Zk: np.ndarray,
    Xk: np.ndarray,
    Z_test: np.ndarray,
    X_test: np.ndarray,
    cat_idx: list[int],
) -> dict:
    """Match categorical columns by histogram of a 1-d projection of Z."""
    if not cat_idx:
        return {"n_cats": 0, "mean_accuracy": float("nan")}
    accs = []
    z_proj = Zk.mean(axis=1)
    z_te = Z_test.mean(axis=1)
    for j in cat_idx:
        # Bin Z and assign the majority X label in each bin.
        try:
            qs = np.quantile(z_proj, np.linspace(0, 1, 9))
            qs[0] -= 1e-6
            qs[-1] += 1e-6
            bins = np.digitize(z_proj, qs)
            bins_te = np.digitize(z_te, qs)
            mapping: dict[int, float] = {}
            for b in np.unique(bins):
                vals, counts = np.unique(Xk[bins == b, j], return_counts=True)
                mapping[int(b)] = float(vals[np.argmax(counts)])
            pred = np.array([mapping.get(int(b), np.median(Xk[:, j])) for b in bins_te])
            accs.append(float(np.mean(np.rint(pred) == np.rint(X_test[:, j]))))
        except Exception:
            accs.append(0.0)
    return {"n_cats": len(cat_idx), "mean_accuracy": float(np.mean(accs) if accs else np.nan)}


def attacker_a(
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_test: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sensitive_mask: np.ndarray | None,
    cat_idx: list[int],
    seed: int,
    heavy: bool,
) -> dict[str, Any]:
    """Known-pair / known-plaintext attacker. Actively searches for inverses."""
    rng = np.random.RandomState(seed)
    n_train = len(X_train)
    d = X_train.shape[1]
    grid = [n for n in KNOWN_PAIR_GRID if n < 0.45 * n_train]
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
            W = np.linalg.pinv(Zk) @ Xk
            row["pinv"] = reconstruction_report(X_test, Z_test @ W)
        except Exception as exc:
            row["pinv"] = {"error": str(exc)}
        row["rank_order"] = _rank_attack(Zk, Xk, Z_test, X_test)
        row["frequency_cat"] = _frequency_cat_attack(Zk, Xk, Z_test, X_test, cat_idx)
        if n >= 25:
            # Nonlinear inversion on a subset of columns (speed).
            try:
                X_hat = np.zeros_like(X_test)
                take = min(6, Xk.shape[1])
                for j in range(take):
                    m = HistGradientBoostingRegressor(
                        max_iter=40, max_depth=4, random_state=seed
                    )
                    m.fit(Zk, Xk[:, j])
                    X_hat[:, j] = m.predict(Z_test)
                row["hgb_partial"] = reconstruction_report(X_test[:, :take], X_hat[:, :take])
            except Exception as exc:
                row["hgb_partial"] = {"error": str(exc)}
        if heavy and n in {50, 100, 200} and n >= 20:
            try:
                from src.learned import train_bottleneck

                # Reuse a small MLP as inversion model: Z -> X via supervised bottleneck
                # trained the other way around.
                inv = _mlp_invert(Zk, Xk, Z_test, X_test, seed)
                row["mlp"] = inv
            except Exception as exc:
                row["mlp"] = {"error": str(exc)}
        # Utility of reconstructed X: how well does a linear model on X_hat predict y?
        try:
            lin = Ridge(alpha=1.0)
            lin.fit(pred, y_test)  # cheating upper bound using test y — report separately
        except Exception:
            pass
        try:
            # Honest: train ridge on reconstructed train-like mapping applied to Z_test vs y_test
            # using a model trained on true X_train[:,] would leak. Instead score X_hat vs y
            # as "can reconstructed features still predict y?"
            from sklearn.linear_model import Ridge as R2

            probe = R2(alpha=1.0)
            probe.fit(X_train, y_train)
            # Compare probe(X_test) vs probe(X_hat_ridge)
            p_true = probe.predict(X_test)
            p_hat = probe.predict(pred)
            row["probe_pred_corr"] = float(np.corrcoef(p_true, p_hat)[0, 1])
        except Exception:
            row["probe_pred_corr"] = float("nan")
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
    last = next((row["ridge"] for row in reversed(curves) if "global_r2" in row.get("ridge", {})), {})
    sens_r2 = float("nan")
    if sensitive_mask is not None and last.get("per_feature_r2"):
        r2s = np.asarray(last["per_feature_r2"], dtype=float)
        mask = sensitive_mask[: len(r2s)]
        if mask.any():
            sens_r2 = float(np.max(r2s[mask]))
    return {
        "curve": curves,
        "n_for_ridge_global_r2_0.5": n_for_r2_05,
        "known_pair_robustness": _robustness_label(worst, d),
        "summary_recon_r2": last.get("global_r2"),
        "summary_worst_attr": last.get("max_feature_r2"),
        "summary_sensitive_attr": sens_r2,
        "spearman_leak": pairwise_spearman_leak(X_test, Z_test),
    }


def _mlp_invert(Zk, Xk, Z_test, X_test, seed: int) -> dict:
    import torch
    from torch import nn

    scaler_z = StandardScaler()
    scaler_x = StandardScaler()
    Zs = scaler_z.fit_transform(Zk)
    Xs = scaler_x.fit_transform(Xk)
    Zte = scaler_z.transform(Z_test)
    layers: list[nn.Module] = [nn.Linear(Zs.shape[1], 64), nn.ReLU(), nn.Linear(64, Xs.shape[1])]
    torch.manual_seed(seed)
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.tensor(Zs, dtype=torch.float32)
    yt = torch.tensor(Xs, dtype=torch.float32)
    n = len(xt)
    batch = min(128, n)
    model.train()
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


# ---------------------------------------------------------------------------
# Attacker B: only Z plus a short context string.
# ---------------------------------------------------------------------------

SEMANTIC_LIBRARY = {
    "income": {"skew_min": 0.6, "positive": True, "role": "income"},
    "age": {"range_lo": 15, "range_hi": 100, "integerish": True, "role": "age"},
    "credit_score": {"range_lo": 250, "range_hi": 900, "role": "credit_score"},
    "utilization": {"range_lo": 0, "range_hi": 2, "role": "credit_utilization"},
    "gender": {"card": 2, "role": "gender"},
    "zip": {"card_min": 50, "role": "zip"},
    "identifier": {"unique_frac_min": 0.9, "role": "row_id"},
    "amount": {"skew_min": 0.4, "positive": True, "role": "account_balance"},
}


def _type_guess(col: np.ndarray) -> str:
    nuniq = len(np.unique(np.round(col, 6)))
    if nuniq <= 2:
        return "binary"
    if nuniq <= 12:
        return "categorical"
    # integer-ish?
    if np.mean(np.abs(col - np.round(col)) < 1e-6) > 0.95 and nuniq < 0.3 * len(col):
        return "ordinal_or_count"
    return "continuous"


def _semantic_guess(col: np.ndarray, context: str) -> list[dict]:
    guesses = []
    nuniq = len(np.unique(np.round(col, 8)))
    unique_frac = nuniq / max(len(col), 1)
    sk = float(0 if np.std(col) < 1e-12 else ((col - col.mean()) ** 3).mean() / (col.std() ** 3 + 1e-12))
    lo, hi = float(np.min(col)), float(np.max(col))
    integerish = float(np.mean(np.abs(col - np.round(col)) < 1e-6))
    if unique_frac > 0.92:
        guesses.append({"label": "identifier", "score": unique_frac})
    if nuniq <= 2:
        guesses.append({"label": "gender_or_binary_flag", "score": 0.6})
    if 15 <= lo and hi <= 100 and integerish > 0.8:
        guesses.append({"label": "age", "score": 0.7})
    if 250 <= lo and hi <= 900 and nuniq > 20:
        guesses.append({"label": "credit_score", "score": 0.55})
    if 0 <= lo and hi <= 2 and nuniq > 10:
        guesses.append({"label": "rate_or_utilization", "score": 0.5})
    if sk > 0.8 and lo >= 0:
        guesses.append({"label": "money_or_income", "score": min(0.85, 0.4 + 0.2 * sk)})
    if "bank" in context.lower() and sk > 0.5 and lo >= 0:
        guesses.append({"label": "bank_amount", "score": 0.45})
    if not guesses:
        guesses.append({"label": "unknown_continuous" if nuniq > 12 else "unknown_categorical", "score": 0.2})
    guesses.sort(key=lambda g: -g["score"])
    return guesses[:3]


def attacker_b(
    Z: np.ndarray,
    context: str,
    true_roles: list[str],
    true_kinds: list[str],
    sensitive_mask: np.ndarray,
    X_aux_public: np.ndarray | None,
    train_mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Black-box attacker: Z + a realistic context sentence. No keys, no names."""
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape
    type_preds = [_type_guess(Z[:, j]) for j in range(d)]
    guesses = [_semantic_guess(Z[:, j], context) for j in range(d)]
    cards = [int(len(np.unique(np.round(Z[:, j], 6)))) for j in range(d)]
    ents = []
    for j in range(d):
        vals, counts = np.unique(np.round(Z[:, j], 5), return_counts=True)
        p = counts / counts.sum()
        ents.append(float(entropy(p)))
    # Correlation / PCA structure
    C = np.nan_to_num(np.corrcoef(Z, rowvar=False))
    pca = PCA(n_components=min(6, d, n - 1), random_state=seed)
    pca.fit(StandardScaler().fit_transform(Z))
    # Clustering / re-identification risk: singleton-ish clusters
    k = min(20, max(4, n // 80))
    km = KMeans(n_clusters=k, n_init=5, random_state=seed)
    labels = km.fit_predict(StandardScaler().fit_transform(Z))
    _, counts = np.unique(labels, return_counts=True)
    # Membership inference: nearest-neighbor self-distance vs holdout
    mi = _membership_inference(Z, train_mask, seed)
    # Linkage to auxiliary public-like numeric table if provided
    linkage = {"enabled": False}
    if X_aux_public is not None and X_aux_public.size:
        linkage = _linkage_attack(Z, X_aux_public, seed)
    # Semantic identification accuracy vs true roles when dimensions match
    sem_acc = _semantic_accuracy(guesses, true_roles)
    # Can we tell which columns were originally categorical?
    kind_acc = _kind_accuracy(type_preds, true_kinds)
    # Sensitive-column ranking: high-entropy unique or money-like
    sens_scores = []
    for j in range(d):
        g0 = guesses[j][0]
        score = g0["score"] if g0["label"] in {"identifier", "money_or_income", "age", "gender_or_binary_flag", "bank_amount"} else 0.1
        if cards[j] <= 2 or cards[j] > 0.9 * n:
            score += 0.2
        sens_scores.append(score)
    sens_rank = np.argsort(sens_scores)[::-1]
    n_sens = int(sensitive_mask.sum()) if sensitive_mask is not None and len(sensitive_mask) == d else 0
    if n_sens and len(sensitive_mask) == d:
        top = set(sens_rank[:n_sens].tolist())
        true = set(np.where(sensitive_mask)[0].tolist())
        sens_hit = len(top & true) / max(n_sens, 1)
    else:
        sens_hit = float("nan")
    return {
        "n_rows": n,
        "n_cols": d,
        "context_used": context,
        "type_preds": type_preds,
        "cardinality": cards,
        "entropy": ents,
        "semantic_guesses": guesses,
        "semantic_top1_accuracy": sem_acc,
        "kind_accuracy_if_aligned": kind_acc,
        "pca_var_top3": float(pca.explained_variance_ratio_[:3].sum()),
        "mean_abs_corr": float(np.mean(np.abs(C - np.eye(d)))),
        "cluster_min_size": int(counts.min()),
        "cluster_singleton_risk": float(np.mean(counts == 1)),
        "membership_inference_auc": mi,
        "linkage": linkage,
        "sensitive_column_hit_rate": sens_hit,
        "max_abs_corr": float(np.max(np.abs(C - np.eye(d)))) if d > 1 else 0.0,
    }


def _semantic_accuracy(guesses: list[list[dict]], true_roles: list[str]) -> float:
    if not true_roles or len(true_roles) != len(guesses):
        return float("nan")
    hits = 0
    for role, gs in zip(true_roles, guesses):
        labels = " ".join(g["label"] for g in gs)
        role_l = role.lower()
        if role_l and any(tok in labels for tok in role_l.split("_") if len(tok) > 2):
            hits += 1
        elif role_l in {"income", "account_balance", "spend"} and "money" in labels:
            hits += 1
        elif role_l == "gender" and "gender" in labels:
            hits += 1
        elif role_l in {"row_id", "email"} and "identifier" in labels:
            hits += 1
    return float(hits / len(true_roles))


def _kind_accuracy(type_preds: list[str], true_kinds: list[str]) -> float:
    if not true_kinds or len(true_kinds) != len(type_preds):
        return float("nan")
    hits = 0
    for pred, kind in zip(type_preds, true_kinds):
        if kind == "numeric" and pred in {"continuous", "ordinal_or_count"}:
            hits += 1
        elif kind == "categorical" and pred in {"binary", "categorical"}:
            hits += 1
        elif kind == "string" and pred in {"categorical", "ordinal_or_count", "continuous"}:
            # hashed strings may look continuous
            hits += 0
    return float(hits / len(true_kinds))


def _membership_inference(Z: np.ndarray, train_mask: np.ndarray, seed: int) -> float:
    """Distance-to-nearest-other-row: train rows of a deterministic map can be
    slightly more clustered. This is a weak but honest black-box MI attempt.
    """
    if train_mask is None or train_mask.sum() < 20 or (~train_mask).sum() < 20:
        return float("nan")
    rng = np.random.RandomState(seed)
    take = min(800, len(Z))
    idx = rng.choice(len(Z), size=take, replace=False)
    Zs = StandardScaler().fit_transform(Z[idx])
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(Zs)
    dist = nn.kneighbors(Zs)[0][:, 1]
    y = train_mask[idx].astype(int)
    # Higher uniqueness (larger NN distance) as a membership score — try both orientations.
    try:
        auc1 = roc_auc_score(y, dist)
        auc2 = roc_auc_score(y, -dist)
        return float(max(auc1, auc2))
    except ValueError:
        return float("nan")


def _linkage_attack(Z: np.ndarray, X_pub: np.ndarray, seed: int) -> dict:
    """Try to match rows of Z to a public-like table via PCA-space NN.

    If the transform preserves neighborhoods, linkage succeeds even without pairs.
    """
    rng = np.random.RandomState(seed)
    n = min(len(Z), len(X_pub), 600)
    z_idx = rng.choice(len(Z), size=n, replace=False)
    # Assume a corresponding public slice exists (worst-case overlapping cohort).
    x_idx = z_idx if len(X_pub) == len(Z) else rng.choice(len(X_pub), size=n, replace=False)
    k = min(6, Z.shape[1], X_pub.shape[1], n - 2)
    Zs = StandardScaler().fit_transform(Z[z_idx])
    Xs = StandardScaler().fit_transform(X_pub[x_idx])
    zp = PCA(n_components=k, random_state=seed).fit_transform(Zs)
    xp = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(xp)
    pred = nn.kneighbors(zp, return_distance=False)[:, 0]
    # Success if we recover the same index in this aligned subsample.
    success = float(np.mean(pred == np.arange(n))) if len(X_pub) == len(Z) else float("nan")
    return {
        "enabled": True,
        "nn_self_match_rate": success,
        "random_baseline": 1.0 / n,
    }
