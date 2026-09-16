"""Conditional Gaussian-mixture synthesizer for multimodal tabular data.

Stratifies by ``(region, segment)`` when those columns exist, otherwise by a
KMeans partition of numeric features.  Each stratum gets a Bayesian (or
classical) Gaussian mixture on numeric columns; leftover categoricals are
drawn from a cluster-conditional multinomial (log-reg or empirical).  Tiny
strata fall back to a parent (region-pooled) or global mixture.

This beats a single copula when the DGP is a mixture or when dependence
changes across segments.  It fails when strata are tiny, components are
badly non-Gaussian, or categorical–numeric coupling is finer than a
cluster label.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler

TARGET_ALIASES = frozenset(
    {
        "y",
        "target",
        "label",
        "labels",
        "outcome",
        "response",
        "class",
        "cls",
        "dependent",
    }
)
REGION_ALIASES = frozenset(
    {
        "region",
        "region_id",
        "geo",
        "geography",
        "zone",
        "zona",
        "state",
        "country",
        "market",
    }
)
SEGMENT_ALIASES = frozenset(
    {
        "segment",
        "segment_id",
        "seg",
        "customer_segment",
        "cust_segment",
        "tier",
        "persona",
        "cohort",
    }
)

_MIN_ROWS_GMM = 8
_MIN_ROWS_LOGREG = 24
_LAPLACE = 0.5


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _is_target_name(name: str) -> bool:
    key = _norm(name)
    return key in TARGET_ALIASES or key.startswith("target_")


def _match_alias(columns: Iterable[str], aliases: frozenset[str]) -> str | None:
    by_norm = {_norm(c): c for c in columns}
    for alias in aliases:
        if alias in by_norm:
            return by_norm[alias]
    return None


def _is_bool_dtype(dtype: Any) -> bool:
    return pd.api.types.is_bool_dtype(dtype)


def _is_numeric_dtype(dtype: Any) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not _is_bool_dtype(dtype)


def _unique_non_null(s: pd.Series) -> np.ndarray:
    return pd.unique(s.dropna())


def _empirical(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, counts = np.unique(values, return_counts=True)
    probs = counts.astype(float) + _LAPLACE
    probs /= probs.sum()
    return vals, probs


def _safe_scaler(X: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(X)
    scale = np.asarray(scaler.scale_, dtype=float)
    scale[scale < 1e-12] = 1.0
    scaler.scale_ = scale
    Xs = (X - scaler.mean_) / scaler.scale_
    return scaler, Xs


def _component_cov(gmm: Any, k: int) -> np.ndarray:
    ct = gmm.covariance_type
    if ct == "full":
        cov = np.array(gmm.covariances_[k], dtype=float, copy=True)
    elif ct == "tied":
        cov = np.array(gmm.covariances_, dtype=float, copy=True)
    elif ct == "diag":
        cov = np.diag(np.asarray(gmm.covariances_[k], dtype=float))
    elif ct == "spherical":
        d = gmm.means_.shape[1]
        cov = np.eye(d) * float(gmm.covariances_[k])
    else:
        raise ValueError(f"unknown covariance_type {ct!r}")
    d = cov.shape[0]
    jitter = 1e-5
    for _ in range(6):
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            cov = cov + np.eye(d) * jitter
            jitter *= 10.0
    return cov + np.eye(d) * 1e-3


def _sample_gmm(
    gmm: Any, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(gmm.weights_, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    comps = rng.choice(len(weights), size=n, p=weights)
    out = np.empty((n, gmm.means_.shape[1]), dtype=float)
    for k in np.unique(comps):
        mask = comps == k
        nk = int(mask.sum())
        mean = np.asarray(gmm.means_[k], dtype=float)
        cov = _component_cov(gmm, int(k))
        out[mask] = rng.multivariate_normal(mean, cov, size=nk)
    return out, comps


def _fit_gmm(
    X: np.ndarray, n_components: int, random_state: int
) -> Any | None:
    n, d = X.shape
    if n < 2 or d < 1:
        return None
    k = max(1, min(int(n_components), n // 4, n))
    if n < _MIN_ROWS_GMM:
        k = 1
    cov = "full" if n > (d * 4 + 8) and k > 1 else "diag"
    if n <= d + 2:
        cov = "diag"
    common = dict(
        n_components=k,
        covariance_type=cov,
        reg_covar=1e-3,
        max_iter=250,
        random_state=random_state,
        tol=1e-3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1.0 / max(k, 1),
                **common,
            )
            model.fit(X)
            return model
        except Exception:
            pass
        try:
            model = GaussianMixture(n_init=1, **common)
            model.fit(X)
            return model
        except Exception:
            if cov != "diag":
                try:
                    common["covariance_type"] = "diag"
                    model = GaussianMixture(n_init=1, **common)
                    model.fit(X)
                    return model
                except Exception:
                    return None
            return None


def _one_hot_clusters(cluster_ids: np.ndarray, n_comp: int) -> np.ndarray:
    oh = np.zeros((len(cluster_ids), n_comp), dtype=float)
    valid = (cluster_ids >= 0) & (cluster_ids < n_comp)
    if valid.any():
        oh[np.flatnonzero(valid), cluster_ids[valid]] = 1.0
    return oh


def _fit_cat_given_cluster(
    cluster_ids: np.ndarray, y: np.ndarray, n_comp: int
) -> dict[str, Any]:
    y = np.asarray(y)
    mask = pd.notna(y)
    if mask.sum() == 0:
        return {"kind": "empty"}
    cluster_ids = np.asarray(cluster_ids)[mask]
    y = y[mask]
    classes = np.unique(y)
    if len(classes) == 1:
        return {"kind": "constant", "value": classes[0]}

    emp: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(n_comp):
        yk = y[cluster_ids == k]
        if len(yk) == 0:
            continue
        emp[k] = _empirical(yk)
    global_emp = _empirical(y)

    if (
        len(y) >= _MIN_ROWS_LOGREG
        and len(classes) >= 2
        and len(np.unique(cluster_ids)) >= 2
    ):
        try:
            Xoh = _one_hot_clusters(cluster_ids, n_comp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(
                    solver="lbfgs",
                    max_iter=300,
                    C=1.0,
                )
                clf.fit(Xoh, y)
            return {
                "kind": "logreg",
                "clf": clf,
                "n_comp": n_comp,
                "classes": np.asarray(clf.classes_),
                "emp": emp,
                "global": global_emp,
            }
        except Exception:
            pass
    return {"kind": "empirical_cluster", "emp": emp, "global": global_emp}


def _sample_cat(
    spec: dict[str, Any],
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(cluster_ids)
    kind = spec.get("kind")
    if kind == "empty":
        return np.array([None] * n, dtype=object)
    if kind == "constant":
        return np.full(n, spec["value"], dtype=object)

    out = np.empty(n, dtype=object)
    if kind == "logreg":
        Xoh = _one_hot_clusters(cluster_ids, int(spec["n_comp"]))
        try:
            proba = spec["clf"].predict_proba(Xoh)
            classes = spec["classes"]
            for i in range(n):
                p = proba[i]
                p = np.clip(p, 0.0, None)
                z = p.sum()
                if z <= 0:
                    vals, probs = spec["global"]
                    out[i] = rng.choice(vals, p=probs)
                else:
                    out[i] = rng.choice(classes, p=p / z)
            return out
        except Exception:
            pass

    emp: dict[int, tuple[np.ndarray, np.ndarray]] = spec.get("emp", {})
    global_emp = spec.get("global")
    for i, k in enumerate(cluster_ids):
        pair = emp.get(int(k))
        if pair is None:
            pair = global_emp
        if pair is None:
            out[i] = None
        else:
            vals, probs = pair
            out[i] = rng.choice(vals, p=probs)
    return out


def _restore_dtype(series: pd.Series, dtype: Any) -> pd.Series:
    if _is_bool_dtype(dtype):
        vals = series.to_numpy()
        as_bool = []
        for v in vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                as_bool.append(False)
            else:
                as_bool.append(bool(int(round(float(v)))))
        return pd.Series(as_bool, index=series.index, dtype="bool")
    if pd.api.types.is_integer_dtype(dtype):
        nums = pd.to_numeric(series, errors="coerce")
        nums = nums.round()
        if nums.isna().any():
            fill = nums.median()
            if pd.isna(fill):
                fill = 0
            nums = nums.fillna(fill)
        try:
            return nums.astype(dtype)
        except (TypeError, ValueError):
            return nums.astype("int64")
    if pd.api.types.is_float_dtype(dtype):
        nums = pd.to_numeric(series, errors="coerce")
        return nums.astype(float)
    if isinstance(dtype, pd.CategoricalDtype) or str(dtype) == "category":
        return series.astype("category")
    try:
        return series.astype(dtype)
    except (TypeError, ValueError):
        return series


@dataclass
class _MixtureFit:
    gmm: Any | None
    scaler: StandardScaler | None
    numeric_cols: list[str]
    cat_specs: dict[str, dict[str, Any]]
    cat_global: dict[str, tuple[np.ndarray, np.ndarray]]
    binary_idx: list[int]
    n_rows: int = 0
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    integer_cols: set[str] = field(default_factory=set)
    unit_interval_cols: set[str] = field(default_factory=set)


class ConditionalMixtureSynthesizer:
    """Stratified Bayesian/Gaussian mixture synthesizer.

    Parameters are supplied to :meth:`fit` (not ``__init__``) so a single
    instance can be refit on different frames.
    """

    def __init__(self) -> None:
        self.include_targets_: bool = True
        self.n_components_: int = 6
        self.columns_: list[str] = []
        self.dtypes_: dict[str, Any] = {}
        self.target_cols_: list[str] = []
        self.region_col_: str | None = None
        self.segment_col_: str | None = None
        self.strata_kind_: str = "global"
        self.numeric_cols_: list[str] = []
        self.binary_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.stratum_cat_cols_: list[str] = []
        self.free_cat_cols_: list[str] = []
        self.stratum_keys_: list[Hashable] = []
        self.stratum_probs_: np.ndarray = np.array([1.0])
        self.stratum_labels_: dict[Hashable, dict[str, Any]] = {}
        self.stratum_model_key_: dict[Hashable, Hashable] = {}
        self.models_: dict[Hashable, _MixtureFit] = {}
        self.kmeans_: KMeans | None = None
        self.kmeans_scaler_: StandardScaler | None = None
        self._fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        include_targets: bool = True,
        n_components: int = 6,
    ) -> ConditionalMixtureSynthesizer:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if df.empty:
            raise ValueError("Cannot fit ConditionalMixtureSynthesizer on an empty frame.")

        self.include_targets_ = bool(include_targets)
        self.n_components_ = max(1, int(n_components))
        self.columns_ = list(df.columns)
        self.dtypes_ = {c: df[c].dtype for c in self.columns_}
        self.target_cols_ = [c for c in self.columns_ if _is_target_name(c)]

        work = df.copy()
        if not include_targets and self.target_cols_:
            work = work.drop(columns=self.target_cols_)

        self._classify_columns(work)
        strata = self._assign_strata(work)
        self._fit_hierarchy(work, strata)
        self._fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        n = int(n)
        if n <= 0:
            cols = self._output_columns()
            return pd.DataFrame({c: [] for c in cols})

        rng = np.random.default_rng(seed)
        keys = self.stratum_keys_
        probs = self.stratum_probs_
        chosen = rng.choice(len(keys), size=n, p=probs)

        frames: list[pd.DataFrame] = []
        for i, key in enumerate(keys):
            count = int((chosen == i).sum())
            if count == 0:
                continue
            frames.append(self._sample_stratum(key, count, rng))
        out = pd.concat(frames, ignore_index=True)
        if len(out) > n:
            out = out.iloc[:n].reset_index(drop=True)
        elif len(out) < n:
            extra = self._sample_stratum(keys[int(chosen[0])], n - len(out), rng)
            out = pd.concat([out, extra], ignore_index=True)

        out = out[self._output_columns()]
        for c in out.columns:
            if c in self.dtypes_:
                out[c] = _restore_dtype(out[c], self.dtypes_[c])
        return out.reset_index(drop=True)

    def _output_columns(self) -> list[str]:
        if self.include_targets_ or not self.target_cols_:
            return list(self.columns_)
        return [c for c in self.columns_ if c not in self.target_cols_]

    def _classify_columns(self, work: pd.DataFrame) -> None:
        numeric: list[str] = []
        binary: list[str] = []
        cats: list[str] = []
        for c in work.columns:
            s = work[c]
            if _is_bool_dtype(s.dtype):
                binary.append(c)
                continue
            if _is_numeric_dtype(s.dtype):
                vals = _unique_non_null(s)
                if len(vals) <= 2 and set(_as_float_set(vals)).issubset({0.0, 1.0}):
                    binary.append(c)
                else:
                    numeric.append(c)
                continue
            cats.append(c)

        self.region_col_ = _match_alias(work.columns, REGION_ALIASES)
        self.segment_col_ = _match_alias(work.columns, SEGMENT_ALIASES)
        # Region/segment must stay categorical strata even if encoded as ints.
        for col in (self.region_col_, self.segment_col_):
            if col is None:
                continue
            if col in numeric:
                numeric.remove(col)
                cats.append(col)
            elif col in binary:
                binary.remove(col)
                cats.append(col)

        self.numeric_cols_ = numeric
        self.binary_cols_ = binary
        self.cat_cols_ = cats
        stratum_cats = [c for c in (self.region_col_, self.segment_col_) if c is not None]
        self.stratum_cat_cols_ = stratum_cats
        self.free_cat_cols_ = [c for c in cats if c not in stratum_cats]

    def _assign_strata(self, work: pd.DataFrame) -> pd.Series:
        if self.region_col_ is not None and self.segment_col_ is not None:
            self.strata_kind_ = "region_segment"
            keys = list(
                zip(
                    work[self.region_col_].astype("string").fillna("<na>"),
                    work[self.segment_col_].astype("string").fillna("<na>"),
                )
            )
            return pd.Series(keys, index=work.index, dtype=object)

        if self.region_col_ is not None:
            self.strata_kind_ = "region"
            keys = work[self.region_col_].astype("string").fillna("<na>").tolist()
            return pd.Series(keys, index=work.index, dtype=object)

        if self.segment_col_ is not None:
            self.strata_kind_ = "segment"
            keys = work[self.segment_col_].astype("string").fillna("<na>").tolist()
            return pd.Series(keys, index=work.index, dtype=object)

        feat_cols = self.numeric_cols_ + self.binary_cols_
        if len(feat_cols) >= 1 and len(work) >= 16:
            X = _numeric_matrix(work, feat_cols)
            n_clusters = int(np.clip(len(work) // 40, 2, 8))
            n_clusters = min(n_clusters, max(2, len(work) // 8))
            scaler, Xs = _safe_scaler(X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=0,
                    max_iter=200,
                )
                labels = km.fit_predict(Xs)
            self.kmeans_ = km
            self.kmeans_scaler_ = scaler
            self.strata_kind_ = "kmeans"
            return pd.Series([("kmeans", int(k)) for k in labels], index=work.index)

        self.strata_kind_ = "global"
        return pd.Series([("global",)] * len(work), index=work.index)

    def _fit_hierarchy(self, work: pd.DataFrame, strata: pd.Series) -> None:
        counts = strata.value_counts()
        self.stratum_keys_ = list(counts.index)
        raw = counts.to_numpy(dtype=float)
        self.stratum_probs_ = raw / raw.sum()

        self.stratum_labels_ = {}
        for key in self.stratum_keys_:
            labels: dict[str, Any] = {}
            if self.strata_kind_ == "region_segment":
                labels[self.region_col_] = key[0]  # type: ignore[index]
                labels[self.segment_col_] = key[1]  # type: ignore[index]
            elif self.strata_kind_ == "region" and self.region_col_:
                labels[self.region_col_] = key
            elif self.strata_kind_ == "segment" and self.segment_col_:
                labels[self.segment_col_] = key
            self.stratum_labels_[key] = labels

        gmm_cols = self.numeric_cols_ + self.binary_cols_
        min_size = max(2 * self.n_components_, _MIN_ROWS_GMM)

        global_fit = self._fit_block(work, gmm_cols, random_state=0)
        self.models_[("global",)] = global_fit

        parents: dict[Hashable, pd.DataFrame] = {}
        if self.strata_kind_ == "region_segment" and self.region_col_ is not None:
            for region, grp in work.groupby(self.region_col_, dropna=False):
                rkey = ("region", str(region) if pd.notna(region) else "<na>")
                parents[rkey] = grp
                self.models_[rkey] = self._fit_block(
                    grp, gmm_cols, random_state=1 + (hash(rkey) % 10_000)
                )

        self.stratum_model_key_ = {}
        for key in self.stratum_keys_:
            mask = strata == key
            block = work.loc[mask]
            n_rows = len(block)
            parent_key = self._parent_key(key)
            if n_rows >= min_size and len(block) >= 3:
                fit = self._fit_block(
                    block, gmm_cols, random_state=2 + (hash(key) % 10_000)
                )
                if fit.gmm is not None or not gmm_cols:
                    self.models_[("stratum", key)] = fit
                    self.stratum_model_key_[key] = ("stratum", key)
                    continue
            if (
                parent_key is not None
                and parent_key in self.models_
                and self.models_[parent_key].n_rows >= min_size
            ):
                self.stratum_model_key_[key] = parent_key
            else:
                self.stratum_model_key_[key] = ("global",)

            # Still record a local categorical/empirical snapshot for free cats
            # when the numeric model is borrowed.
            local = self._fit_block(block, gmm_cols=[], random_state=0)
            self.models_[("local_cat", key)] = local

    def _parent_key(self, key: Hashable) -> Hashable | None:
        if self.strata_kind_ == "region_segment" and isinstance(key, tuple) and len(key) == 2:
            return ("region", str(key[0]))
        return None

    def _fit_block(
        self,
        block: pd.DataFrame,
        gmm_cols: list[str],
        random_state: int,
    ) -> _MixtureFit:
        numeric_present = [c for c in gmm_cols if c in block.columns]
        binary_idx = [i for i, c in enumerate(numeric_present) if c in self.binary_cols_]
        integer_cols = {
            c
            for c in numeric_present
            if c in self.dtypes_ and pd.api.types.is_integer_dtype(self.dtypes_[c])
        }
        unit_interval_cols: set[str] = set()
        bounds: dict[str, tuple[float, float]] = {}
        for c in numeric_present:
            s = pd.to_numeric(block[c], errors="coerce")
            lo = float(s.min()) if s.notna().any() else 0.0
            hi = float(s.max()) if s.notna().any() else 1.0
            if hi < lo:
                hi = lo
            bounds[c] = (lo, hi)
            if lo >= 0.0 and hi <= 1.0:
                unit_interval_cols.add(c)

        gmm = None
        scaler = None
        cluster_ids = np.zeros(len(block), dtype=int)
        n_comp = 1
        if numeric_present and len(block) >= 2:
            X = _numeric_matrix(block, numeric_present)
            scaler, Xs = _safe_scaler(X)
            gmm = _fit_gmm(Xs, self.n_components_, random_state)
            if gmm is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        cluster_ids = np.asarray(gmm.predict(Xs), dtype=int)
                    except Exception:
                        cluster_ids = np.zeros(len(block), dtype=int)
                n_comp = int(getattr(gmm, "n_components", 1))

        cat_specs: dict[str, dict[str, Any]] = {}
        cat_global: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for c in self.free_cat_cols_:
            if c not in block.columns:
                continue
            y = block[c].astype(object).to_numpy()
            valid = pd.notna(block[c]).to_numpy()
            if valid.any():
                cat_global[c] = _empirical(y[valid])
            cat_specs[c] = _fit_cat_given_cluster(cluster_ids, y, n_comp)

        return _MixtureFit(
            gmm=gmm,
            scaler=scaler,
            numeric_cols=numeric_present,
            cat_specs=cat_specs,
            cat_global=cat_global,
            binary_idx=binary_idx,
            n_rows=len(block),
            bounds=bounds,
            integer_cols=integer_cols,
            unit_interval_cols=unit_interval_cols,
        )

    def _sample_stratum(
        self, key: Hashable, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        model_key = self.stratum_model_key_.get(key, ("global",))
        fit = self.models_.get(model_key) or self.models_[("global",)]
        local_cats = self.models_.get(("local_cat", key))

        data: dict[str, Any] = {}
        cluster_ids = np.zeros(n, dtype=int)

        if fit.gmm is not None and fit.numeric_cols and fit.scaler is not None:
            Xs, cluster_ids = _sample_gmm(fit.gmm, n, rng)
            X = Xs * fit.scaler.scale_ + fit.scaler.mean_
            means_orig = (
                np.asarray(fit.gmm.means_) * fit.scaler.scale_ + fit.scaler.mean_
            )
            for j, c in enumerate(fit.numeric_cols):
                col = X[:, j].astype(float)
                lo, hi = fit.bounds.get(c, (col.min(), col.max()))
                span = hi - lo
                if c in fit.unit_interval_cols or c in self.binary_cols_:
                    lo_clip, hi_clip = 0.0, 1.0
                else:
                    pad = 0.25 * span if span > 0 else 1.0
                    lo_clip, hi_clip = lo - pad, hi + pad
                col = np.clip(col, lo_clip, hi_clip)
                if j in fit.binary_idx:
                    p = np.clip(means_orig[cluster_ids, j], 0.0, 1.0)
                    col = rng.binomial(1, p).astype(float)
                elif c in fit.integer_cols:
                    col = np.rint(col)
                    col = np.clip(col, np.floor(lo), np.ceil(hi))
                data[c] = col
        else:
            for c in self.numeric_cols_ + self.binary_cols_:
                data[c] = np.zeros(n, dtype=float)
            if self.binary_cols_:
                # Degenerate numeric model: independent Bernoulli from global mean.
                src = fit if fit.n_rows else self.models_[("global",)]
                for c in self.binary_cols_:
                    lo, hi = src.bounds.get(c, (0.0, 1.0))
                    p = float(np.clip(0.5 * (lo + hi), 0.0, 1.0))
                    data[c] = rng.binomial(1, p, size=n).astype(float)

        labels = self.stratum_labels_.get(key, {})
        for c, value in labels.items():
            data[c] = np.full(n, value, dtype=object)

        cat_fit = local_cats if local_cats is not None and local_cats.cat_specs else fit
        for c in self.free_cat_cols_:
            spec = cat_fit.cat_specs.get(c) or fit.cat_specs.get(c)
            if spec is None:
                emp = cat_fit.cat_global.get(c) or fit.cat_global.get(c)
                if emp is None:
                    data[c] = np.array([None] * n, dtype=object)
                else:
                    vals, probs = emp
                    data[c] = rng.choice(vals, size=n, p=probs)
            else:
                data[c] = _sample_cat(spec, cluster_ids, rng)

        cols = self._output_columns()
        frame = pd.DataFrame({c: data[c] for c in cols if c in data})
        for c in cols:
            if c not in frame.columns:
                frame[c] = np.nan
        return frame[cols]


def _as_float_set(vals: np.ndarray) -> set[float]:
    out: set[float] = set()
    for v in vals:
        try:
            if pd.isna(v):
                continue
            out.add(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.empty((len(df), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        s = pd.to_numeric(df[c], errors="coerce")
        med = s.median()
        if pd.isna(med):
            med = 0.0
        X[:, j] = s.fillna(float(med)).to_numpy(dtype=float)
    return X


def synthesize_mixture(
    df: pd.DataFrame,
    n: int | None = None,
    seed: int = 0,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Fit a :class:`ConditionalMixtureSynthesizer` and draw ``n`` rows.

    ``n`` defaults to ``len(df)``.
    """
    if n is None:
        n = len(df)
    return (
        ConditionalMixtureSynthesizer()
        .fit(df, include_targets=include_targets)
        .sample(int(n), seed=seed)
    )


def _self_test(n_synth: int = 200) -> pd.DataFrame:
    """Generate a multimodal, segment-specific table and synthesize it."""
    rng = np.random.default_rng(7)
    n = 500
    region = rng.choice(["north", "south", "west"], size=n, p=[0.45, 0.35, 0.20])
    segment = rng.choice(["growth", "value"], size=n, p=[0.55, 0.45])
    channel = rng.choice(["web", "store", "partner"], size=n, p=[0.5, 0.3, 0.2])

    x1 = np.empty(n)
    x2 = np.empty(n)
    y = np.empty(n)
    flag = np.empty(n, dtype=int)
    for i in range(n):
        if segment[i] == "growth":
            # Mode A: positive dependence, higher mean.
            mean = np.array([2.5, 1.8])
            cov = np.array([[0.6, 0.45], [0.45, 0.7]])
            slope = 1.4
            intercept = 0.5
            p_flag = 0.75
        else:
            # Mode B: negative dependence, lower mean.
            mean = np.array([-1.2, 0.4])
            cov = np.array([[0.5, -0.35], [-0.35, 0.55]])
            slope = -0.8
            intercept = 2.0
            p_flag = 0.20
        if region[i] == "west":
            mean = mean + np.array([1.5, -0.8])
        xy = rng.multivariate_normal(mean, cov)
        x1[i], x2[i] = xy
        y[i] = intercept + slope * x1[i] + 0.3 * x2[i] + rng.normal(0.0, 0.25)
        flag[i] = int(rng.random() < p_flag)

    df = pd.DataFrame(
        {
            "region": region,
            "segment": segment,
            "channel": channel,
            "x1": x1,
            "x2": x2,
            "flag": flag,
            "y": y,
        }
    )
    syn = synthesize_mixture(df, n=n_synth, seed=0, include_targets=True)
    if len(syn) != n_synth:
        raise AssertionError(f"expected {n_synth} rows, got {len(syn)}")
    if list(syn.columns) != list(df.columns):
        raise AssertionError(f"column mismatch: {list(syn.columns)}")
    if syn["region"].nunique() < 2 or syn["segment"].nunique() < 2:
        raise AssertionError("strata collapsed in sample")
    if syn["channel"].nunique() < 2:
        raise AssertionError("free categorical collapsed")
    if set(syn["flag"].unique()) - {0, 1}:
        raise AssertionError("binary column not 0/1")

    # Segment-specific relationship should be visible (sign of corr(x1, x2)).
    real_g = np.corrcoef(df.loc[df.segment == "growth", ["x1", "x2"]].to_numpy().T)[0, 1]
    syn_g = np.corrcoef(syn.loc[syn.segment == "growth", ["x1", "x2"]].to_numpy().T)[0, 1]
    if np.isnan(syn_g) or real_g * syn_g < 0:
        raise AssertionError(
            f"growth-segment correlation sign lost (real={real_g:.3f}, syn={syn_g:.3f})"
        )

    syn_x = synthesize_mixture(df, n=64, seed=1, include_targets=False)
    if "y" in syn_x.columns:
        raise AssertionError("include_targets=False should drop y")

    # Clustering path (no region/segment).
    numeric_only = df[["x1", "x2", "flag", "y"]].copy()
    syn_k = synthesize_mixture(numeric_only, n=80, seed=2)
    if len(syn_k) != 80:
        raise AssertionError("kmeans-stratum path failed")

    # Tiny-stratum fallback: one rare (region, segment) cell.
    tiny = df.copy()
    rare = (tiny["region"] == "west") & (tiny["segment"] == "value")
    tiny = pd.concat([tiny.loc[~rare], tiny.loc[rare].head(3)], ignore_index=True)
    syn_t = synthesize_mixture(tiny, n=40, seed=3)
    if len(syn_t) != 40:
        raise AssertionError("tiny-stratum fallback failed")

    return syn


if __name__ == "__main__":
    sample = _self_test(200)
    print(
        f"self-test ok: {len(sample)} rows, columns={list(sample.columns)}, "
        f"dtypes={sample.dtypes.to_dict()}"
    )
