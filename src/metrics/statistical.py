"""Statistical fidelity metrics for real vs. synthetic tabular data.

``compute_statistical_fidelity`` compares aligned common columns only
(columns present in just one frame are ignored). Constant / empty /
all-NaN columns are handled without raising.

Fidelity score
--------------
Each available family is mapped to a similarity in ``[0, 1]``, then
averaged (equal weight). Families with no applicable columns are omitted.

    s_ks       = 1 - mean_i KS_i
    s_wass     = 1 / (1 + mean_i W1_i / scale_i)
                 scale_i = std(real_i) if std(real_i) > 0 else 1
    s_mean     = 1 / (1 + mean_i |μs − μr| / (|μr| + ε))
    s_std      = 1 / (1 + mean_i |σs − σr| / (|σr| + ε))
    s_tvd      = 1 - mean_j TVD_j
    s_pearson  = 1 / (1 + ||P_r − P_s||_F / n)
    s_spearman = 1 / (1 + ||S_r − S_s||_F / n)
    s_mi       = 1 / (1 + mean_k |MI_r − MI_s|_k)
    s_miss     = 1 - mean_c |miss_r − miss_s|_c
                 (included only when any NaNs exist in either frame)

    fidelity_score = 0.40 * mean(marginal s_*) + 0.60 * mean(dependence s_*)

ε = 1e-12.  Pearson / Spearman use numeric-typed common columns (bool
included as 0/1); ``n`` is the number of those columns used in the
matrix.  Mutual-information drift uses ``sklearn``
``mutual_info_regression`` on a deterministic sample of the highest-|Pearson|
numeric pairs (up to ``MAX_MI_PAIRS``).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression

EPS = 1e-12
MAX_MI_PAIRS = 5
MI_SAMPLE_SIZE = 2000
MI_RANDOM_STATE = 0

MARGINAL_SIM_KEYS: tuple[str, ...] = (
    "ks_sim",
    "wass_sim",
    "mean_sim",
    "std_sim",
    "tvd_sim",
    "miss_sim",
)
DEPENDENCE_SIM_KEYS: tuple[str, ...] = (
    "pearson_sim",
    "spearman_sim",
    "mi_sim",
)
MARGINAL_WEIGHT = 0.30
DEPENDENCE_WEIGHT = 0.30
XY_WEIGHT = 0.40

FIDELITY_SCORE_FORMULA = (
    "fidelity_score = 0.30 * mean(marginal sims) + 0.30 * mean(X-structure sims) "
    "+ 0.40 * xy_sim when target y is present; otherwise 0.40/0.60 marginal/dep. "
    "Marginal: s_ks=1-mean(KS); s_wass=1/(1+mean(W1/std_real)); "
    "s_mean=1/(1+mean(mean_rel_err)); s_std=1/(1+mean(std_rel_err)); "
    "s_tvd=1-mean(TVD); s_miss=1-mean(|miss_r-miss_s|) if NaNs exist. "
    "X-structure: s_pearson=exp(-||P_r-P_s||_F/sqrt(n)); same for Spearman; "
    "s_mi=1/(1+mean(|MI_r-MI_s|)). "
    "xy_sim = 0.5*(1-mean|corr(X,y)_r-corr(X,y)_s|) + 0.5*clip(OLS coef cosine,0,1). "
    "Missing families drop out and remaining weights renormalize. Clipped to [0, 1]."
)


def _align_frames(
    real: pd.DataFrame, synth: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    common = [c for c in real.columns if c in synth.columns]
    # Preserve real column order; drop duplicate names if any.
    seen: set[str] = set()
    ordered: list[str] = []
    for c in common:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return real.loc[:, ordered], synth.loc[:, ordered], ordered


def _finite_values(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _is_cat_or_binary(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return int(s.dropna().nunique()) <= 2
    return True


def _classify_columns(
    real: pd.DataFrame, cols: list[str]
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for c in cols:
        s = real[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            numeric.append(c)
            continue
        if _is_cat_or_binary(s):
            categorical.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def _numeric_typed_columns(real: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns usable in Pearson/Spearman/MI (bool + numeric + datetime)."""
    out: list[str] = []
    for c in cols:
        s = real[c]
        if (
            pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s)
        ):
            out.append(c)
    return out


def _as_float_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    pieces: dict[str, np.ndarray] = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            pieces[c] = s.view("int64").to_numpy(dtype=float)
        elif pd.api.types.is_bool_dtype(s):
            pieces[c] = s.astype(float).to_numpy()
        else:
            pieces[c] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(pieces, index=df.index)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return None
    return float(np.mean(xs))


def _rel_err(est: float, ref: float) -> float:
    return float(abs(est - ref) / (abs(ref) + EPS))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0
    return float(ks_2samp(a, b, method="auto").statistic)


def _wasserstein_normalized(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0
    w1 = float(wasserstein_distance(a, b))
    scale = float(np.std(a))
    if scale <= EPS:
        # Constant real column: W1 is |Δmean| for two constants; scale by 1.
        return 0.0 if w1 <= EPS else float(w1)
    return w1 / scale


def _tvd(real_s: pd.Series, synth_s: pd.Series) -> float:
    a = real_s.dropna()
    b = synth_s.dropna()
    if a.empty and b.empty:
        return 0.0
    if a.empty or b.empty:
        return 1.0
    pa = a.astype(str).value_counts(normalize=True)
    pb = b.astype(str).value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    pa = pa.reindex(idx, fill_value=0.0)
    pb = pb.reindex(idx, fill_value=0.0)
    return 0.5 * float(np.abs(pa.to_numpy() - pb.to_numpy()).sum())


def _corr_frobenius(
    real_num: pd.DataFrame,
    synth_num: pd.DataFrame,
    method: str,
) -> Optional[float]:
    if real_num.shape[1] < 2:
        return None
    real_std = real_num.std(ddof=0, numeric_only=True)
    keep = [c for c in real_num.columns if float(real_std.get(c, 0.0) or 0.0) > EPS]
    if len(keep) < 2:
        return None

    if method == "pearson":
        cr = real_num[keep].corr(method="pearson")
        cs = synth_num[keep].corr(method="pearson")
    elif method == "spearman":
        # pandas spearman can be slow / nan-heavy; use scipy on complete cases
        # per pair via DataFrame.corr which already does pairwise complete.
        cr = real_num[keep].corr(method="spearman")
        cs = synth_num[keep].corr(method="spearman")
    else:
        raise ValueError(method)

    cr = cr.fillna(0.0)
    cs = cs.fillna(0.0)
    # Align in case a column vanished (all-nan in synth).
    cols = [c for c in keep if c in cr.columns and c in cs.columns]
    if len(cols) < 2:
        return None
    diff = cr.loc[cols, cols].to_numpy() - cs.loc[cols, cols].to_numpy()
    return float(np.linalg.norm(diff, ord="fro"))


def _top_mi_pairs(real_num: pd.DataFrame, max_pairs: int) -> list[tuple[str, str]]:
    cols = list(real_num.columns)
    if len(cols) < 2:
        return []
    corr = real_num.corr(method="pearson").abs()
    scored: list[tuple[float, str, str]] = []
    for i, ci in enumerate(cols):
        for cj in cols[i + 1 :]:
            val = corr.loc[ci, cj]
            if np.isfinite(val):
                scored.append((float(val), ci, cj))
            else:
                scored.append((0.0, ci, cj))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(a, b) for _, a, b in scored[:max_pairs]]


def _pairwise_mi_drift(
    real_num: pd.DataFrame,
    synth_num: pd.DataFrame,
) -> list[dict[str, Any]]:
    pairs = _top_mi_pairs(real_num, MAX_MI_PAIRS)
    results: list[dict[str, Any]] = []
    if not pairs:
        return results

    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= MI_SAMPLE_SIZE:
            return df
        return df.sample(n=MI_SAMPLE_SIZE, random_state=MI_RANDOM_STATE)

    real_s = _sample(real_num)
    synth_s = _sample(synth_num)

    for a, b in pairs:
        def _mi(df: pd.DataFrame, x: str, y: str) -> Optional[float]:
            sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 4:
                return None
            xv = sub[[x]].to_numpy(dtype=float)
            yv = sub[y].to_numpy(dtype=float)
            if np.std(xv) <= EPS or np.std(yv) <= EPS:
                return 0.0
            try:
                est = mutual_info_regression(
                    xv,
                    yv,
                    discrete_features=False,
                    random_state=MI_RANDOM_STATE,
                )
                return float(est[0])
            except ValueError:
                return None

        mi_r = _mi(real_s, a, b)
        mi_s = _mi(synth_s, a, b)
        delta = (
            abs(mi_r - mi_s)
            if mi_r is not None and mi_s is not None
            else None
        )
        results.append(
            {
                "pair": [a, b],
                "mi_real": mi_r,
                "mi_synth": mi_s,
                "abs_delta": delta,
            }
        )
    return results


def _sim_one_minus(mean_stat: Optional[float]) -> Optional[float]:
    if mean_stat is None:
        return None
    return float(np.clip(1.0 - mean_stat, 0.0, 1.0))


def _sim_inv(mean_stat: Optional[float]) -> Optional[float]:
    if mean_stat is None:
        return None
    return float(1.0 / (1.0 + max(mean_stat, 0.0)))


def compute_statistical_fidelity(
    real: pd.DataFrame, synth: pd.DataFrame
) -> dict[str, Any]:
    """Compare marginals, dependence, and missingness of ``synth`` vs ``real``.

    Returns a dict whose keys are listed in ``METRIC_KEYS`` / the
    ``fidelity_score_formula`` field.  ``fidelity_score`` is in ``[0, 1]``
    (1 = indistinguishable on the measured statistics).
    """
    if not isinstance(real, pd.DataFrame) or not isinstance(synth, pd.DataFrame):
        raise TypeError("real and synth must be pandas DataFrames")

    real_a, synth_a, cols = _align_frames(real, synth)
    numeric_cols, cat_cols = _classify_columns(real_a, cols)
    dep_cols = _numeric_typed_columns(real_a, cols)

    per_column: dict[str, dict[str, Any]] = {}
    ks_vals: list[float] = []
    wass_vals: list[float] = []
    mean_rel_vals: list[float] = []
    std_rel_vals: list[float] = []
    tvd_vals: list[float] = []

    for c in numeric_cols:
        a = _finite_values(real_a[c])
        b = _finite_values(synth_a[c])
        ks = _ks_statistic(a, b)
        wass = _wasserstein_normalized(a, b)
        if a.size == 0 and b.size == 0:
            mean_re = 0.0
            std_re = 0.0
        elif a.size == 0 or b.size == 0:
            mean_re = 1.0
            std_re = 1.0
        else:
            mean_re = _rel_err(float(np.mean(b)), float(np.mean(a)))
            std_re = _rel_err(float(np.std(b)), float(np.std(a)))
        ks_vals.append(ks)
        wass_vals.append(wass)
        mean_rel_vals.append(mean_re)
        std_rel_vals.append(std_re)
        per_column[c] = {
            "type": "numeric",
            "ks_statistic": ks,
            "wasserstein_1_normalized": wass,
            "mean_relative_error": mean_re,
            "std_relative_error": std_re,
            "tvd": None,
        }

    for c in cat_cols:
        tvd = _tvd(real_a[c], synth_a[c])
        tvd_vals.append(tvd)
        per_column[c] = {
            "type": "categorical",
            "ks_statistic": None,
            "wasserstein_1_normalized": None,
            "mean_relative_error": None,
            "std_relative_error": None,
            "tvd": tvd,
        }

    real_num = _as_float_frame(real_a, dep_cols) if dep_cols else pd.DataFrame()
    synth_num = _as_float_frame(synth_a, dep_cols) if dep_cols else pd.DataFrame()

    pearson_f = _corr_frobenius(real_num, synth_num, "pearson") if dep_cols else None
    spearman_f = _corr_frobenius(real_num, synth_num, "spearman") if dep_cols else None
    n_dep = int(real_num.shape[1]) if dep_cols else 0

    mi_pairs = _pairwise_mi_drift(real_num, synth_num) if n_dep >= 2 else []
    mi_mean = _safe_mean(p["abs_delta"] for p in mi_pairs)

    any_nan = bool(real_a.isna().any().any() or synth_a.isna().any().any())
    missingness: Optional[dict[str, dict[str, float]]] = None
    missingness_mae: Optional[float] = None
    if any_nan:
        missingness = {}
        deltas: list[float] = []
        n_r = max(len(real_a), 1)
        n_s = max(len(synth_a), 1)
        for c in cols:
            mr = float(real_a[c].isna().mean()) if n_r else 0.0
            ms = float(synth_a[c].isna().mean()) if n_s else 0.0
            d = abs(mr - ms)
            missingness[c] = {"real": mr, "synth": ms, "abs_delta": d}
            deltas.append(d)
        missingness_mae = _safe_mean(deltas)

    ks_mean = _safe_mean(ks_vals)
    wass_mean = _safe_mean(wass_vals)
    mean_re_mean = _safe_mean(mean_rel_vals)
    std_re_mean = _safe_mean(std_rel_vals)
    tvd_mean = _safe_mean(tvd_vals)

    pearson_scaled = (
        pearson_f / n_dep if pearson_f is not None and n_dep > 0 else None
    )
    spearman_scaled = (
        spearman_f / n_dep if spearman_f is not None and n_dep > 0 else None
    )

    def _corr_sim(scaled: Optional[float], n_cols: int) -> Optional[float]:
        if scaled is None or n_cols <= 0:
            return None
        # harsher than 1/(1+F/n): independence shuffle of weak-corr tables
        # used to look "fine" under the old transform.
        raw = float(scaled) * float(n_cols)  # recover Frobenius
        return float(np.clip(np.exp(-raw / max(np.sqrt(n_cols), 1.0)), 0.0, 1.0))

    xy_payload: Optional[dict[str, Any]] = None
    xy_sim: Optional[float] = None
    if "y" in cols:
        xy_payload = compare_xy_relationship(real_a, synth_a, target="y")
        parts: list[float] = []
        delta = xy_payload.get("corr_abs_delta_mean")
        if delta is not None and np.isfinite(delta):
            parts.append(float(np.clip(1.0 - float(delta), 0.0, 1.0)))
        cos = xy_payload.get("coef_cosine_similarity")
        if cos is not None and np.isfinite(cos):
            parts.append(float(np.clip(cos, 0.0, 1.0)))
        if parts:
            xy_sim = float(np.mean(parts))

    component_scores: dict[str, Optional[float]] = {
        "ks_sim": _sim_one_minus(ks_mean),
        "wass_sim": _sim_inv(wass_mean),
        "mean_sim": _sim_inv(mean_re_mean),
        "std_sim": _sim_inv(std_re_mean),
        "tvd_sim": _sim_one_minus(tvd_mean),
        "pearson_sim": _corr_sim(pearson_scaled, n_dep),
        "spearman_sim": _corr_sim(spearman_scaled, n_dep),
        "mi_sim": _sim_inv(mi_mean),
        "miss_sim": _sim_one_minus(missingness_mae) if any_nan else None,
        "xy_sim": xy_sim,
    }

    def _finite_mean(keys: tuple[str, ...]) -> Optional[float]:
        vals = [
            float(component_scores[k])
            for k in keys
            if component_scores.get(k) is not None and np.isfinite(component_scores[k])
        ]
        if not vals:
            return None
        return float(np.mean(vals))

    marginal_score = _finite_mean(MARGINAL_SIM_KEYS)
    dependence_score = _finite_mean(DEPENDENCE_SIM_KEYS)
    component_scores["marginal_score"] = marginal_score
    component_scores["dependence_score"] = dependence_score

    weighted: list[tuple[float, float]] = []
    if marginal_score is not None:
        weighted.append((MARGINAL_WEIGHT, marginal_score))
    if dependence_score is not None:
        weighted.append((DEPENDENCE_WEIGHT, dependence_score))
    if xy_sim is not None:
        weighted.append((XY_WEIGHT, xy_sim))
    if not cols or not weighted:
        fidelity = 0.0
    else:
        wsum = float(sum(w for w, _ in weighted))
        fidelity = float(
            np.clip(sum(w * s for w, s in weighted) / max(wsum, EPS), 0.0, 1.0)
        )

    return {
        "n_rows_real": int(len(real_a)),
        "n_rows_synth": int(len(synth_a)),
        "n_cols_compared": int(len(cols)),
        "columns_compared": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "per_column": per_column,
        "ks_mean": ks_mean,
        "wasserstein_normalized_mean": wass_mean,
        "mean_relative_error_mean": mean_re_mean,
        "std_relative_error_mean": std_re_mean,
        "tvd_mean": tvd_mean,
        "pearson_corr_frobenius": pearson_f,
        "spearman_corr_frobenius": spearman_f,
        "mutual_info_pairs": mi_pairs,
        "mutual_info_drift_mean": mi_mean,
        "missingness": missingness,
        "missingness_mae": missingness_mae,
        "component_scores": component_scores,
        "xy_relationship": xy_payload,
        "fidelity_score": fidelity,
        "fidelity_score_formula": FIDELITY_SCORE_FORMULA,
    }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= EPS:
        return 1.0 if ss_res <= EPS else 0.0
    return float(1.0 - ss_res / ss_tot)


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= EPS and nv <= EPS:
        return 1.0
    if nu <= EPS or nv <= EPS:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _pearson_xy(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else None


def compare_xy_relationship(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target: str = "y",
) -> dict[str, Any]:
    """Compare feature–target dependence and a linear ``y ~ X`` fit.

    Reports per-feature Pearson correlation with ``target`` (real, synth,
    abs delta) plus:

    * ``r2_real`` — in-sample R² of OLS fit on real data
    * ``r2_synth_with_real_coefs`` — R² of those same coefficients on synth
    * ``r2_synth_refit`` — in-sample R² of a separate OLS fit on synth
    * ``coef_cosine_similarity`` — cosine similarity of the two OLS
      coefficient vectors (intercept included)

    Only numeric/bool features present in both frames (excluding
    ``target``) are used.  Constant columns are kept for OLS but skipped
    for correlation.  Does not raise on rank-deficient or constant ``y``.
    """
    real_a, synth_a, cols = _align_frames(real, synth)
    empty = {
        "target": target,
        "feature_corr": {},
        "corr_abs_delta_mean": None,
        "r2_real": None,
        "r2_synth_with_real_coefs": None,
        "r2_synth_refit": None,
        "r2_delta": None,
        "coef_cosine_similarity": None,
        "n_features": 0,
        "features": [],
    }
    if target not in cols:
        empty["error"] = f"target {target!r} not in common columns"
        return empty

    feat_cols = [
        c
        for c in cols
        if c != target
        and (
            pd.api.types.is_numeric_dtype(real_a[c])
            or pd.api.types.is_bool_dtype(real_a[c])
        )
    ]

    def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        use = feat_cols + [target]
        frame = _as_float_frame(df, use).replace([np.inf, -np.inf], np.nan).dropna()
        return frame[feat_cols], frame[target]

    Xr, yr = _xy(real_a)
    Xs, ys = _xy(synth_a)

    feature_corr: dict[str, dict[str, Optional[float]]] = {}
    deltas: list[float] = []
    for c in feat_cols:
        cr = _pearson_xy(Xr[c].to_numpy() if c in Xr else np.array([]), yr.to_numpy() if len(yr) else np.array([]))
        cs = _pearson_xy(Xs[c].to_numpy() if c in Xs else np.array([]), ys.to_numpy() if len(ys) else np.array([]))
        delta = abs(cr - cs) if cr is not None and cs is not None else None
        if delta is not None:
            deltas.append(delta)
        feature_corr[c] = {
            "corr_real": cr,
            "corr_synth": cs,
            "abs_delta": delta,
        }

    r2_real: Optional[float] = None
    r2_transfer: Optional[float] = None
    r2_synth: Optional[float] = None
    r2_delta: Optional[float] = None
    cos_sim: Optional[float] = None

    can_real = len(Xr) >= 2 and Xr.shape[1] >= 1
    can_synth = len(Xs) >= 2 and Xs.shape[1] >= 1

    coef_r = None
    if can_real:
        try:
            model_r = LinearRegression()
            model_r.fit(Xr.to_numpy(dtype=float), yr.to_numpy(dtype=float))
            pred_r = model_r.predict(Xr.to_numpy(dtype=float))
            r2_real = _r2(yr.to_numpy(dtype=float), pred_r)
            coef_r = np.concatenate(
                [[float(model_r.intercept_)], model_r.coef_.astype(float).ravel()]
            )
            if can_synth and list(Xs.columns) == list(Xr.columns):
                pred_s = model_r.predict(Xs.to_numpy(dtype=float))
                r2_transfer = _r2(ys.to_numpy(dtype=float), pred_s)
        except (ValueError, np.linalg.LinAlgError):
            pass

    if can_synth:
        try:
            model_s = LinearRegression()
            model_s.fit(Xs.to_numpy(dtype=float), ys.to_numpy(dtype=float))
            pred_ss = model_s.predict(Xs.to_numpy(dtype=float))
            r2_synth = _r2(ys.to_numpy(dtype=float), pred_ss)
            coef_s = np.concatenate(
                [[float(model_s.intercept_)], model_s.coef_.astype(float).ravel()]
            )
            if coef_r is not None and coef_r.shape == coef_s.shape:
                cos_sim = _cosine(coef_r, coef_s)
        except (ValueError, np.linalg.LinAlgError):
            pass

    if r2_real is not None and r2_transfer is not None:
        r2_delta = float(r2_real - r2_transfer)

    return {
        "target": target,
        "feature_corr": feature_corr,
        "corr_abs_delta_mean": _safe_mean(deltas),
        "r2_real": r2_real,
        "r2_synth_with_real_coefs": r2_transfer,
        "r2_synth_refit": r2_synth,
        "r2_delta": r2_delta,
        "coef_cosine_similarity": cos_sim,
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
    }


METRIC_KEYS = [
    "n_rows_real",
    "n_rows_synth",
    "n_cols_compared",
    "columns_compared",
    "numeric_columns",
    "categorical_columns",
    "per_column",
    "ks_mean",
    "wasserstein_normalized_mean",
    "mean_relative_error_mean",
    "std_relative_error_mean",
    "tvd_mean",
    "pearson_corr_frobenius",
    "spearman_corr_frobenius",
    "mutual_info_pairs",
    "mutual_info_drift_mean",
    "missingness",
    "missingness_mae",
    "component_scores",
    "fidelity_score",
    "fidelity_score_formula",
]

COMPARE_XY_KEYS = [
    "target",
    "feature_corr",
    "corr_abs_delta_mean",
    "r2_real",
    "r2_synth_with_real_coefs",
    "r2_synth_refit",
    "r2_delta",
    "coef_cosine_similarity",
    "n_features",
    "features",
]


def _self_test() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(5.0, 2.0, n)
    x3 = 0.8 * x1 + rng.normal(0.0, 0.4, n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    binary = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    y = 1.5 * x1 - 0.4 * x2 + 0.3 * x3 + rng.normal(0.0, 0.25, n)
    extra_only_real = rng.normal(size=n)
    real = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "cat": cat,
            "bin": binary,
            "y": y,
            "const": np.ones(n),
            "only_real": extra_only_real,
        }
    )
    # Inject a few NaNs so missingness is exercised on the identity path.
    real.loc[0:4, "x2"] = np.nan

    identity = compute_statistical_fidelity(real, real.copy())
    xy_id = compare_xy_relationship(real, real.copy(), target="y")

    shuffled = real.copy()
    for c in shuffled.columns:
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    # Drop the real-only column analogue; add a synth-only column.
    shuffled["only_synth"] = rng.normal(size=n)
    shuffled_metrics = compute_statistical_fidelity(real, shuffled)
    xy_shuf = compare_xy_relationship(real, shuffled, target="y")

    indep = pd.DataFrame(
        {
            "x1": rng.normal(8.0, 3.0, n),
            "x2": rng.uniform(-10.0, 10.0, n),
            "x3": rng.exponential(4.0, n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.1, 0.1, 0.8]),
            "bin": rng.choice([0, 1], size=n, p=[0.2, 0.8]),
            "y": rng.normal(50.0, 10.0, n),
            "const": np.full(n, 7.0),
        }
    )
    indep.loc[10:19, "x2"] = np.nan
    indep_metrics = compute_statistical_fidelity(real, indep)
    xy_indep = compare_xy_relationship(real, indep, target="y")

    # Constant-column / empty-overlap guards (must not raise).
    compute_statistical_fidelity(
        pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]}),
        pd.DataFrame({"a": [1, 1, 1], "z": [0, 0, 0]}),
    )
    compare_xy_relationship(
        pd.DataFrame({"x": [0.0, 0.0, 0.0], "y": [1.0, 1.0, 1.0]}),
        pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [3.0, 3.0, 3.0]}),
        target="y",
    )

    return {
        "identity_fidelity_score": identity["fidelity_score"],
        "shuffled_fidelity_score": shuffled_metrics["fidelity_score"],
        "independent_fidelity_score": indep_metrics["fidelity_score"],
        "identity_ks_mean": identity["ks_mean"],
        "identity_tvd_mean": identity["tvd_mean"],
        "identity_pearson_frobenius": identity["pearson_corr_frobenius"],
        "identity_mi_drift_mean": identity["mutual_info_drift_mean"],
        "identity_missingness_mae": identity["missingness_mae"],
        "shuffled_pearson_frobenius": shuffled_metrics["pearson_corr_frobenius"],
        "independent_ks_mean": indep_metrics["ks_mean"],
        "xy_identity_r2_real": xy_id["r2_real"],
        "xy_identity_r2_transfer": xy_id["r2_synth_with_real_coefs"],
        "xy_identity_coef_cos": xy_id["coef_cosine_similarity"],
        "xy_shuffled_corr_abs_delta_mean": xy_shuf["corr_abs_delta_mean"],
        "xy_indep_coef_cos": xy_indep["coef_cosine_similarity"],
        "identity_better_than_shuffled": (
            identity["fidelity_score"] > shuffled_metrics["fidelity_score"]
        ),
        "shuffled_better_than_independent": (
            shuffled_metrics["fidelity_score"] > indep_metrics["fidelity_score"]
        ),
        "identity_near_one": identity["fidelity_score"] >= 0.99,
    }


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    report = _self_test()
    print("METRIC_KEYS:", METRIC_KEYS)
    print("COMPARE_XY_KEYS:", COMPARE_XY_KEYS)
    print("FIDELITY_SCORE_FORMULA:", FIDELITY_SCORE_FORMULA)
    print("SELF-TEST:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if not report["identity_near_one"]:
        raise SystemExit("identity fidelity_score not ~1")
    if not report["identity_better_than_shuffled"]:
        raise SystemExit("shuffled score was not lower than identity")
    if not report["shuffled_better_than_independent"]:
        raise SystemExit("independent score was not lower than shuffled")
    print("self-test OK")
