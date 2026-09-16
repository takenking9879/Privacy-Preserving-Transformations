"""Gaussian-copula tabular synthesizer (numpy / pandas / scipy only).

API
---
- ``GaussianCopulaSynthesizer.fit(df, exclude_targets=False)``
- ``GaussianCopulaSynthesizer.sample(n, seed=None)``
- ``synthesize_copula(df, n=None, seed=0, exclude_targets=False)``

When ``exclude_targets=False`` (default) every column — including ``y`` and
``y_class`` if present — enters the copula so the joint ``(X, y)`` is
preserved.  When ``exclude_targets=True`` those target columns are ignored
during fitting and are omitted from samples.

Limitations
-----------
A Gaussian copula recovers 1-D marginals plus *elliptical / rank-linear*
dependence.  It misses:

* **Multimodality in the joint** (e.g. a rare distressed cluster that
  shifts several columns together).  Unimodal KDE marginals also smear
  marginal modes; the empirical-quantile fallback keeps 1-D atoms/modes
  but not their joint occurrence.
* **Nonlinear / tail / asymmetric dependence** (interactions, hockey-stick
  thresholds, heteroscedasticity, copulas with tail dependence).
* **Discrete atoms** (ties in counts/binaries): average ranks are a
  continuous approximation; the latent Gaussian cannot represent
  ``P(X = c) > 0`` exactly, though the inverse-quantile step can
  reintroduce atoms marginally.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde, norm

TARGET_COLUMNS: frozenset[str] = frozenset({"y", "y_class"})

_KNOWN_CATEGORICAL: frozenset[str] = frozenset({"region", "segment"})
_KNOWN_BINARY: frozenset[str] = frozenset({"is_premium"})
_KNOWN_COUNT: frozenset[str] = frozenset(
    {"num_products", "complaint_count", "tenure_months"}
)
_KNOWN_CONTINUOUS: frozenset[str] = frozenset(
    {"age", "income", "risk_score", "usage", "engagement", "credit_util"}
)

_BINARY_VALUE_SETS = (
    {0, 1},
    {0.0, 1.0},
    {False, True},
    {"0", "1"},
    {"false", "true"},
    {"no", "yes"},
    {"n", "y"},
)

_U_LO = 1e-6
_U_HI = 1.0 - 1e-6
_KDE_GRID = 512
_MIN_EIG = 1e-8


def _is_binary_values(series: pd.Series) -> bool:
    vals = pd.unique(series.dropna())
    if len(vals) == 0 or len(vals) > 2:
        return False
    if len(vals) == 1:
        key = vals[0]
        if isinstance(key, str):
            key = key.lower()
        return key in {0, 1, 0.0, 1.0, False, True, "0", "1", "false", "true"}
    normalized: set[Any] = set()
    for v in vals:
        if isinstance(v, str):
            normalized.add(v.strip().lower())
        elif isinstance(v, (bool, np.bool_)):
            normalized.add(bool(v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            if float(v) in (0.0, 1.0):
                normalized.add(float(v))
            else:
                return False
        else:
            return False
    return normalized in _BINARY_VALUE_SETS or normalized <= {0.0, 1.0, False, True}


def _infer_kind(name: str, series: pd.Series) -> str:
    """Classify a column as continuous / categorical / binary / count."""
    if name in _KNOWN_BINARY:
        return "binary"
    if name in _KNOWN_CATEGORICAL:
        return "categorical"
    if name in _KNOWN_COUNT:
        return "count"
    if name in _KNOWN_CONTINUOUS:
        return "continuous"

    if name == "y_class":
        if _is_binary_values(series):
            return "binary"
        return "categorical"
    if name == "y":
        if pd.api.types.is_float_dtype(series):
            return "continuous"
        if _is_binary_values(series):
            return "binary"
        if pd.api.types.is_numeric_dtype(series):
            nunq = int(series.nunique(dropna=True))
            if nunq <= 12 and pd.api.types.is_integer_dtype(series):
                return "categorical"
            return "continuous"
        return "categorical"

    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if isinstance(series.dtype, pd.CategoricalDtype):
        if _is_binary_values(series):
            return "binary"
        return "categorical"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        if _is_binary_values(series):
            return "binary"
        return "categorical"

    nunq = int(series.nunique(dropna=True))
    if nunq <= 2 and _is_binary_values(series):
        return "binary"
    if pd.api.types.is_integer_dtype(series):
        observed = series.dropna()
        if len(observed) and float(observed.min()) >= 0:
            return "count"
        return "continuous"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "categorical"


def _normal_scores(values: np.ndarray) -> np.ndarray:
    """Van der Waerden scores; NaNs stay NaN; ties use average ranks."""
    s = pd.Series(values)
    z = np.full(len(s), np.nan, dtype=float)
    mask = s.notna().to_numpy()
    m = int(mask.sum())
    if m == 0:
        return z
    if m == 1:
        z[mask] = 0.0
        return z
    ranks = s[mask].rank(method="average").to_numpy(dtype=float)
    u = np.clip((ranks - 0.5) / m, _U_LO, _U_HI)
    z[mask] = norm.ppf(u)
    return z


def _shrink_corr(corr: np.ndarray) -> np.ndarray:
    """Symmetrize, NaN-fill, identity-shrink if near-singular, PSD-project."""
    r = np.asarray(corr, dtype=float)
    if r.size == 0:
        return r
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("correlation matrix must be square")
    d = r.shape[0]
    if d == 1:
        return np.array([[1.0]])

    r = 0.5 * (r + r.T)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(r, 1.0)

    eigvals = np.linalg.eigvalsh(r)
    min_ev = float(eigvals.min())
    max_ev = float(eigvals.max())
    cond = max_ev / max(abs(min_ev), 1e-16)

    lam = 0.0
    if min_ev < _MIN_EIG or cond > 1e8:
        if min_ev >= _MIN_EIG:
            lam = 0.05
        else:
            lam = (_MIN_EIG - min_ev) / max(1.0 - min_ev, 1e-12)
            lam = float(np.clip(lam, 0.05, 0.95))
        r = (1.0 - lam) * r + lam * np.eye(d)
        np.fill_diagonal(r, 1.0)

    evals, evecs = np.linalg.eigh(r)
    evals = np.maximum(evals, _MIN_EIG)
    r = (evecs * evals) @ evecs.T
    std = np.sqrt(np.clip(np.diag(r), 1e-12, None))
    r = r / np.outer(std, std)
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 1.0)
    return r


def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    jitter = 1e-10
    d = corr.shape[0]
    eye = np.eye(d)
    for _ in range(10):
        try:
            return np.linalg.cholesky(corr + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    evals, evecs = np.linalg.eigh(corr)
    evals = np.maximum(evals, _MIN_EIG)
    psd = (evecs * evals) @ evecs.T
    return np.linalg.cholesky(psd + 1e-8 * eye)


def _kde_cdf_grid(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8 or np.unique(x).size < 8:
        return None
    if float(np.std(x)) <= 1e-15:
        return None
    try:
        kde = gaussian_kde(x, bw_method="scott")
        bw = float(np.sqrt(np.squeeze(kde.covariance)))
    except Exception:
        return None
    lo = float(x.min()) - 4.0 * bw
    hi = float(x.max()) + 4.0 * bw
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, _KDE_GRID)
    try:
        pdf = np.clip(kde(grid), 0.0, None)
    except Exception:
        return None
    if not np.isfinite(pdf).all() or float(pdf.sum()) <= 0.0:
        return None
    cdf = cumulative_trapezoid(pdf, grid, initial=0.0)
    total = float(cdf[-1])
    if not np.isfinite(total) or total <= 0.0:
        return None
    cdf = np.clip(cdf / total, 0.0, 1.0)
    # strictly increasing for interpolation
    cdf = np.maximum.accumulate(cdf)
    span = float(cdf[-1] - cdf[0])
    if span <= 0.0:
        return None
    cdf = (cdf - cdf[0]) / span
    cdf[-1] = 1.0
    return grid, cdf


def _choose_continuous_method(values: np.ndarray) -> str:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n_unique = int(np.unique(x).size)
    if x.size < 20 or n_unique < 12:
        return "quantile"
    try:
        # Fisher–Pearson skewness; heavy skew → empirical quantile
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if sd <= 1e-15:
            return "quantile"
        g1 = float(np.mean(((x - mu) / sd) ** 3))
        if abs(g1) > 1.5:
            return "quantile"
    except Exception:
        return "quantile"
    return "kde"


def _ppf_quantile(values: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if x.size == 0:
        return np.full(u.shape, np.nan)
    if x.size == 1:
        return np.full(u.shape, x[0])
    return np.quantile(x, u, method="linear")


def _ppf_kde(grid_x: np.ndarray, grid_cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), _U_LO, _U_HI)
    return np.interp(u, grid_cdf, grid_x)


def _ppf_discrete(levels: np.ndarray, cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0 - 1e-12)
    idx = np.searchsorted(cdf, u, side="left")
    idx = np.clip(idx, 0, len(levels) - 1)
    return levels[idx]


def _unique_preserve(series: pd.Series) -> np.ndarray:
    return pd.unique(series.dropna())


class GaussianCopulaSynthesizer:
    """Fit a Gaussian copula on mixed-type tabular data and sample from it."""

    def __init__(self) -> None:
        self.fitted_: bool = False
        self.exclude_targets_: bool = False
        self.columns_: list[str] = []
        self.dtypes_: dict[str, Any] = {}
        self.specs_: dict[str, dict[str, Any]] = {}
        self.copula_columns_: list[str] = []
        self.corr_: np.ndarray | None = None
        self._chol_: np.ndarray | None = None

    def fit(
        self, df: pd.DataFrame, exclude_targets: bool = False
    ) -> GaussianCopulaSynthesizer:
        """Estimate marginals and a regularized Gaussian-copula correlation.

        Parameters
        ----------
        df:
            Mixed-type training table.
        exclude_targets:
            If True, drop ``y`` / ``y_class`` from the copula (and from
            later samples).  If False (default), include every column so
            the joint ``(X, y)`` is preserved.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.shape[1] == 0:
            raise ValueError("df must contain at least one column")

        self.exclude_targets_ = bool(exclude_targets)
        if self.exclude_targets_:
            cols = [c for c in df.columns if c not in TARGET_COLUMNS]
        else:
            cols = list(df.columns)
        if not cols:
            raise ValueError("no columns left to fit after excluding targets")

        work = df.loc[:, cols]
        self.columns_ = list(cols)
        self.dtypes_ = {c: work[c].dtype for c in cols}
        self.specs_ = {}

        score_cols: list[str] = []
        score_mat: list[np.ndarray] = []

        for col in cols:
            spec = self._fit_column(col, work[col])
            self.specs_[col] = spec
            if spec["constant"]:
                continue
            scores = spec["scores"]
            if np.isfinite(scores).sum() >= 2:
                score_cols.append(col)
                score_mat.append(scores)

        self.copula_columns_ = score_cols
        if score_cols:
            z_df = pd.DataFrame({c: s for c, s in zip(score_cols, score_mat)})
            raw = z_df.corr(method="pearson").to_numpy(dtype=float)
            self.corr_ = _shrink_corr(raw)
            self._chol_ = _safe_cholesky(self.corr_)
        else:
            self.corr_ = np.zeros((0, 0))
            self._chol_ = np.zeros((0, 0))

        # drop bulky per-row scores from stored specs
        for spec in self.specs_.values():
            spec.pop("scores", None)

        self.fitted_ = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        """Draw ``n`` rows from the fitted copula, restoring dtypes/levels."""
        if not self.fitted_:
            raise RuntimeError("call fit() before sample()")
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")

        rng = np.random.default_rng(seed)
        d = len(self.copula_columns_)
        u_map: dict[str, np.ndarray] = {}
        if n == 0:
            uniforms = np.zeros((0, d))
        elif d == 0:
            uniforms = np.zeros((n, 0))
        else:
            z = rng.standard_normal((n, d)) @ self._chol_.T
            uniforms = np.clip(norm.cdf(z), _U_LO, _U_HI)
        for j, col in enumerate(self.copula_columns_):
            u_map[col] = uniforms[:, j] if d else np.full(n, 0.5)

        data: dict[str, pd.Series] = {}
        for col in self.columns_:
            spec = self.specs_[col]
            u = u_map.get(col)
            if u is None:
                u = rng.random(n)
            values = self._invert_column(spec, u)
            values = self._apply_domain(spec, values)
            series = self._restore_series(spec, values, n)
            series = self._apply_missing(spec, series, rng)
            data[col] = series

        out = pd.DataFrame({c: data[c] for c in self.columns_})
        return out

    # ------------------------------------------------------------------
    # fitting helpers
    # ------------------------------------------------------------------

    def _fit_column(self, name: str, series: pd.Series) -> dict[str, Any]:
        kind = _infer_kind(str(name), series)
        observed = series.dropna()
        n_total = int(len(series))
        n_obs = int(len(observed))
        missing_rate = float((n_total - n_obs) / n_total) if n_total else 0.0

        spec: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "dtype": series.dtype,
            "missing_rate": missing_rate,
            "constant": False,
            "constant_value": None,
            "method": "quantile",
            "values": None,
            "grid_x": None,
            "grid_cdf": None,
            "levels": None,
            "level_cdf": None,
            "p": None,
            "zero": None,
            "one": None,
            "lo": None,
            "hi": None,
            "categories": None,
            "ordered": False,
            "scores": np.full(n_total, np.nan, dtype=float),
        }

        if isinstance(series.dtype, pd.CategoricalDtype):
            spec["categories"] = series.dtype.categories
            spec["ordered"] = bool(series.dtype.ordered)

        if n_obs == 0:
            spec["constant"] = True
            spec["constant_value"] = np.nan
            return spec

        n_unique = int(observed.nunique())
        if n_unique <= 1:
            spec["constant"] = True
            spec["constant_value"] = observed.iloc[0]
            spec["lo"] = self._numeric_bound(observed, "min")
            spec["hi"] = self._numeric_bound(observed, "max")
            if kind in {"categorical", "binary"}:
                levels = _unique_preserve(observed)
                spec["levels"] = levels
                spec["level_cdf"] = np.ones(len(levels), dtype=float)
                if kind == "binary":
                    spec["zero"], spec["one"], spec["p"] = self._binary_params(
                        observed
                    )
            else:
                vals = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
                spec["values"] = vals
            spec["scores"] = _normal_scores(self._rank_payload(kind, series, spec))
            return spec

        if kind == "binary":
            zero, one, p = self._binary_params(observed)
            spec["method"] = "bernoulli"
            spec["zero"] = zero
            spec["one"] = one
            spec["p"] = p
            spec["levels"] = np.array([zero, one], dtype=object)
            encoded = self._encode_binary(series, zero, one)
            spec["scores"] = _normal_scores(encoded)
            return spec

        if kind == "categorical":
            counts = observed.value_counts(dropna=True)
            levels = counts.index.to_numpy()
            probs = counts.to_numpy(dtype=float)
            probs = probs / probs.sum()
            spec["method"] = "empirical"
            spec["levels"] = levels
            spec["level_cdf"] = np.cumsum(probs)
            spec["level_cdf"][-1] = 1.0
            codes = self._map_to_codes(series, levels)
            spec["scores"] = _normal_scores(codes)
            return spec

        # continuous or count (count: fit as continuous, invert then round)
        numeric = pd.to_numeric(observed, errors="coerce")
        vals = numeric.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            spec["constant"] = True
            spec["constant_value"] = np.nan
            return spec
        spec["values"] = vals
        spec["lo"] = float(np.min(vals))
        spec["hi"] = float(np.max(vals))
        method = _choose_continuous_method(vals)
        if method == "kde":
            grid = _kde_cdf_grid(vals)
            if grid is None:
                method = "quantile"
            else:
                spec["grid_x"], spec["grid_cdf"] = grid
        spec["method"] = method
        full_numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        spec["scores"] = _normal_scores(full_numeric)
        return spec

    @staticmethod
    def _numeric_bound(observed: pd.Series, which: str) -> float | None:
        numeric = pd.to_numeric(observed, errors="coerce")
        numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
        if numeric.empty:
            return None
        return float(numeric.min() if which == "min" else numeric.max())

    @staticmethod
    def _binary_params(observed: pd.Series) -> tuple[Any, Any, float]:
        levels = list(_unique_preserve(observed))
        if len(levels) == 1:
            only = levels[0]
            if _truthy_level(only):
                return _complement_level(only), only, 1.0
            return only, _complement_level(only), 0.0
        a, b = levels[0], levels[1]
        # Prefer (0, 1) / (False, True) ordering when recognizable.
        if _truthy_level(a) and not _truthy_level(b):
            zero, one = b, a
        elif _truthy_level(b) and not _truthy_level(a):
            zero, one = a, b
        else:
            try:
                zero, one = sorted([a, b], key=lambda v: (str(type(v)), str(v)))
            except Exception:
                zero, one = a, b
        p = float((observed == one).mean())
        p = float(np.clip(p, 0.0, 1.0))
        return zero, one, p

    @staticmethod
    def _encode_binary(series: pd.Series, zero: Any, one: Any) -> np.ndarray:
        out = np.full(len(series), np.nan, dtype=float)
        mask = series.notna().to_numpy()
        vals = series.to_numpy()
        eq_one = np.zeros(len(series), dtype=bool)
        for i, flag in enumerate(mask):
            if flag:
                eq_one[i] = vals[i] == one
        out[mask] = np.where(eq_one[mask], 1.0, 0.0)
        return out

    @staticmethod
    def _map_to_codes(series: pd.Series, levels: np.ndarray) -> np.ndarray:
        index = {v: i for i, v in enumerate(levels)}
        out = np.full(len(series), np.nan, dtype=float)
        for i, val in enumerate(series.to_numpy()):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if pd.isna(val):
                continue
            out[i] = index.get(val, np.nan)
        return out

    @staticmethod
    def _rank_payload(kind: str, series: pd.Series, spec: dict[str, Any]) -> np.ndarray:
        if kind == "binary" and spec.get("one") is not None:
            return GaussianCopulaSynthesizer._encode_binary(
                series, spec["zero"], spec["one"]
            )
        if kind == "categorical" and spec.get("levels") is not None:
            return GaussianCopulaSynthesizer._map_to_codes(series, spec["levels"])
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # inverse transform + restoration
    # ------------------------------------------------------------------

    def _invert_column(self, spec: dict[str, Any], u: np.ndarray) -> np.ndarray:
        if spec["constant"]:
            return np.full(len(u), spec["constant_value"], dtype=object)

        kind = spec["kind"]
        if kind == "binary":
            p = float(spec["p"])
            drawn = np.where(u < p, spec["one"], spec["zero"])
            return drawn

        if kind == "categorical":
            return _ppf_discrete(spec["levels"], spec["level_cdf"], u)

        if spec["method"] == "kde" and spec["grid_x"] is not None:
            return _ppf_kde(spec["grid_x"], spec["grid_cdf"], u)
        return _ppf_quantile(spec["values"], u)

    def _apply_domain(self, spec: dict[str, Any], values: np.ndarray) -> np.ndarray:
        name = spec["name"]
        kind = spec["kind"]
        if kind in {"binary", "categorical"}:
            return values
        arr = np.array(values, dtype=float, copy=True)
        if name == "credit_util":
            arr = np.clip(arr, 0.0, 1.0)
        elif name == "age":
            lo = spec["lo"] if spec["lo"] is not None else np.nanmin(arr)
            hi = spec["hi"] if spec["hi"] is not None else np.nanmax(arr)
            arr = np.clip(arr, lo, hi)
        if kind == "count":
            arr = np.rint(arr)
            arr = np.clip(arr, 0.0, None)
        return arr

    def _restore_series(
        self, spec: dict[str, Any], values: np.ndarray, n: int
    ) -> pd.Series:
        dtype = spec["dtype"]
        kind = spec["kind"]
        name = spec["name"]

        if spec["constant"] and (
            spec["constant_value"] is None
            or (isinstance(spec["constant_value"], float) and np.isnan(spec["constant_value"]))
        ):
            return pd.Series(np.full(n, np.nan), name=name, dtype=dtype)

        if kind == "categorical":
            if isinstance(dtype, pd.CategoricalDtype):
                cats = spec["categories"] if spec["categories"] is not None else dtype.categories
                cat = pd.Categorical(
                    values, categories=cats, ordered=bool(spec["ordered"])
                )
                return pd.Series(cat, name=name, dtype=dtype)
            series = pd.Series(values, name=name)
            try:
                return series.astype(dtype)
            except (TypeError, ValueError):
                return series

        if kind == "binary":
            series = pd.Series(list(values), name=name)
            try:
                return series.astype(dtype)
            except (TypeError, ValueError):
                return series

        # continuous / count
        arr = np.asarray(values, dtype=float)
        if kind == "count" or pd.api.types.is_integer_dtype(dtype):
            rounded = np.rint(arr)
            rounded = np.where(np.isfinite(rounded), rounded, np.nan)
            if pd.api.types.is_integer_dtype(dtype) and not _is_nullable_dtype(dtype):
                # non-nullable integer: NaNs should not appear (counts have none)
                filled = np.where(np.isfinite(rounded), rounded, 0.0)
                series = pd.Series(filled, name=name)
            else:
                series = pd.Series(rounded, name=name)
            try:
                return series.astype(dtype)
            except (TypeError, ValueError):
                return series.astype("Int64")

        series = pd.Series(arr, name=name)
        try:
            return series.astype(dtype)
        except (TypeError, ValueError):
            return series

    @staticmethod
    def _apply_missing(
        spec: dict[str, Any], series: pd.Series, rng: np.random.Generator
    ) -> pd.Series:
        p = float(spec.get("missing_rate") or 0.0)
        if p <= 0.0 or len(series) == 0:
            return series
        mask = rng.random(len(series)) < p
        if not mask.any():
            return series
        out = series.copy()
        try:
            out.loc[mask] = np.nan
        except (TypeError, ValueError):
            # non-nullable integer/bool — leave as-is
            return series
        return out


def _is_nullable_dtype(dtype: Any) -> bool:
    try:
        return bool(pd.api.types.is_extension_array_dtype(dtype))
    except Exception:
        return False


def _truthy_level(val: Any) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y"}
    try:
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        return float(val) == 1.0
    except (TypeError, ValueError):
        return False


def _complement_level(val: Any) -> Any:
    if isinstance(val, (bool, np.bool_)):
        return type(val)(not bool(val))
    if isinstance(val, str):
        mapping = {
            "1": "0",
            "0": "1",
            "true": "false",
            "false": "true",
            "yes": "no",
            "no": "yes",
            "y": "n",
            "n": "y",
        }
        key = val.strip().lower()
        if key in mapping:
            flipped = mapping[key]
            return flipped if val.islower() else flipped.capitalize() if val.istitle() else flipped
        return val
    try:
        return type(val)(0 if float(val) == 1.0 else 1)
    except (TypeError, ValueError):
        return 0


def synthesize_copula(
    df: pd.DataFrame,
    n: int | None = None,
    seed: int = 0,
    exclude_targets: bool = False,
) -> pd.DataFrame:
    """Fit a Gaussian copula on ``df`` and draw ``n`` synthetic rows.

    ``n`` defaults to ``len(df)``.  See :class:`GaussianCopulaSynthesizer`
    for ``exclude_targets`` semantics.
    """
    if n is None:
        n = int(len(df))
    syn = GaussianCopulaSynthesizer()
    syn.fit(df, exclude_targets=exclude_targets)
    return syn.sample(int(n), seed=seed)


def _self_test() -> None:
    rng = np.random.default_rng(42)
    n = 100
    region = rng.choice(["Northeast", "Midwest", "South", "West"], n)
    segment = rng.choice(["Mass", "Affluent", "Private"], n)
    df = pd.DataFrame(
        {
            "age": rng.normal(42.0, 12.0, n).clip(18.0, 79.0),
            "income": rng.lognormal(10.4, 0.4, n),
            "tenure_months": rng.integers(0, 240, n, dtype=np.int64),
            "region": pd.Series(region, dtype="string"),
            "segment": pd.Categorical(
                segment, categories=["Mass", "Affluent", "Private"]
            ),
            "risk_score": rng.normal(50.0, 10.0, n).clip(5.0, 99.0),
            "num_products": rng.integers(0, 8, n, dtype=np.int64),
            "is_premium": rng.integers(0, 2, n, dtype=np.int64),
            "usage": rng.lognormal(2.5, 0.5, n),
            "engagement": rng.normal(60.0, 10.0, n).clip(0.0, 100.0),
            "complaint_count": rng.poisson(1.2, n).astype(np.int64),
            "credit_util": rng.beta(2.0, 3.0, n),
            "y": rng.lognormal(3.5, 0.5, n),
            "y_class": rng.integers(0, 2, n, dtype=np.int64),
        }
    )
    miss_idx = rng.choice(n, size=6, replace=False)
    df.loc[miss_idx[:3], "income"] = np.nan
    df.loc[miss_idx[3:], "engagement"] = np.nan

    syn = GaussianCopulaSynthesizer().fit(df, exclude_targets=False)
    out = syn.sample(n, seed=1)
    cols_match = list(out.columns) == list(df.columns)
    dtype_pairs = [(c, df[c].dtype, out[c].dtype) for c in df.columns]
    dtypes_match = all(a == b for _, a, b in dtype_pairs)
    cats_match = list(out["segment"].cat.categories) == list(df["segment"].cat.categories)

    syn_x = GaussianCopulaSynthesizer().fit(df, exclude_targets=True)
    out_x = syn_x.sample(40, seed=2)
    targets_excluded = (
        "y" not in syn_x.columns_
        and "y_class" not in syn_x.columns_
        and "y" not in syn_x.copula_columns_
        and "y_class" not in syn_x.copula_columns_
        and "y" not in out_x.columns
        and "y_class" not in out_x.columns
    )
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    features_match = list(out_x.columns) == feature_cols
    feature_dtypes = all(out_x[c].dtype == df[c].dtype for c in feature_cols)

    wrapped = synthesize_copula(df, n=30, seed=0, exclude_targets=False)
    wrap_ok = list(wrapped.columns) == list(df.columns)

    age_ok = bool(out["age"].min() >= df["age"].min() - 1e-9 and out["age"].max() <= df["age"].max() + 1e-9)
    util_ok = bool(out["credit_util"].min() >= -1e-12 and out["credit_util"].max() <= 1.0 + 1e-12)
    counts_ok = bool(
        (out["complaint_count"].to_numpy() >= 0).all()
        and (out["num_products"].to_numpy() >= 0).all()
        and (out["tenure_months"].to_numpy() >= 0).all()
    )

    passed = all(
        [
            cols_match,
            dtypes_match,
            cats_match,
            targets_excluded,
            features_match,
            feature_dtypes,
            wrap_ok,
            age_ok,
            util_ok,
            counts_ok,
        ]
    )
    print(
        "self-test:",
        f"columns_match={cols_match}",
        f"dtypes_match={dtypes_match}",
        f"category_levels_match={cats_match}",
        f"exclude_targets_ok={targets_excluded and features_match and feature_dtypes}",
        f"domain_ok={age_ok and util_ok and counts_ok}",
        f"passed={passed}",
    )
    if not dtypes_match:
        print("dtype mismatches:", [(c, a, b) for c, a, b in dtype_pairs if a != b])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    _self_test()
