"""Synthpop-like sequential CART / conditional synthesizer.

Visit-order heuristic
---------------------
1. Optionally drop target columns ``y`` and ``y_class`` when
   ``include_targets=False``.
2. Split remaining columns into *categorical* vs *numeric*:
   - categorical: bool / object / string / pandas Categorical, any column
     named ``region``, ``segment``, ``is_premium``, or ``y_class``, and
     integer columns whose cardinality is at most ``min(15, 10% of n)``
     (so ``num_products``, ``complaint_count``, binary flags stay discrete);
   - numeric: everything else (``age``, ``income``, ``tenure_months``,
     ``risk_score``, ``usage``, ``engagement``, ``credit_util``, ``y``, …).
3. Visit order:
   a. categoricals by *increasing* cardinality — simplest discrete margins
      first, so later models condition on cheap, well-estimated factors;
   b. numerics by *decreasing* mean |Spearman| correlation with the other
      numeric columns — highly associated variables appear earlier so the
      main dependence spine is available to later conditionals;
   c. if included, ``y`` then ``y_class`` are always appended last so the
      targets condition on the full synthesized feature vector (and
      ``y_class`` may depend on ``y``).

Generation
----------
The first visited column is drawn from its empirical distribution.  Each
later column is a CART conditional: a constrained
``DecisionTreeClassifier`` / ``DecisionTreeRegressor`` is fit on the
*real* previous columns to predict the *real* current column.  At sample
time the already-synthesized previous columns are dropped down that tree
and a value is drawn from the leaf — class probabilities for categoricals,
prediction + an empirical leaf residual for numerics.  Mean-only
prediction is never used; residual / probability sampling is what stops
the leaves from collapsing to piecewise-constant over-smoothing.

Trees use ``DecisionTree*`` rather than ``HistGradientBoosting*`` so each
row maps to an explicit leaf with a stored residual bag.  Depth and leaf
size are capped (default ``max_depth=10``, ``min_samples_leaf=20``) as a
privacy-adjacent regularizer: no leaf is allowed to isolate a handful of
individuals, while interactions up to that depth are still represented.

A Gaussian copula only encodes pairwise rank correlations under a
Gaussian dependence structure.  Sequential CART captures non-monotonic,
heteroscedastic, and higher-order interactions (e.g. income high only
when ``is_premium`` and ``region`` take a particular combination) because
each split is a local conditional, and mixed types are native.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

TARGET_COLS: tuple[str, ...] = ("y", "y_class")
FORCE_CATEGORICAL: frozenset[str] = frozenset(
    {"region", "segment", "is_premium", "y_class"}
)
FORCE_NUMERIC: frozenset[str] = frozenset(
    {
        "age",
        "income",
        "tenure_months",
        "risk_score",
        "usage",
        "engagement",
        "credit_util",
        "y",
    }
)


def _is_bool_dtype(dtype: Any) -> bool:
    return dtype == bool or isinstance(dtype, pd.BooleanDtype)


def _is_categorical_series(name: str, s: pd.Series, n: int) -> bool:
    if name in FORCE_NUMERIC:
        return False
    if name in FORCE_CATEGORICAL:
        return True
    if _is_bool_dtype(s.dtype):
        return True
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        nunique = int(s.nunique(dropna=True))
        return nunique <= min(15, max(2, int(0.10 * n)))
    return False


def _spearman_connectedness(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Mean |Spearman| of each numeric column with the other numerics."""
    scores = {c: 0.0 for c in cols}
    if len(cols) < 2:
        return scores
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman").abs()
    for c in cols:
        if c not in corr.columns:
            continue
        others = [o for o in cols if o != c and o in corr.columns]
        if not others:
            continue
        vals = corr.loc[c, others].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            scores[c] = float(np.mean(finite))
    return scores


def _visit_order(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    features = [c for c in df.columns if c not in TARGET_COLS]
    cats = [c for c in features if kinds.get(c) == "cat"]
    nums = [c for c in features if kinds.get(c) == "num"]
    cats.sort(key=lambda c: (int(df[c].nunique(dropna=True)), c))
    conn = _spearman_connectedness(df, nums)
    nums.sort(key=lambda c: (-conn[c], c))
    ordered = cats + nums
    for t in TARGET_COLS:
        if t in kinds:
            ordered.append(t)
    return ordered


@dataclass
class _FirstStep:
    column: str
    values: np.ndarray


@dataclass
class _CatStep:
    column: str
    predictors: list[str]
    tree: DecisionTreeClassifier
    classes: np.ndarray


@dataclass
class _NumStep:
    column: str
    predictors: list[str]
    tree: DecisionTreeRegressor
    leaf_residuals: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class _ConstStep:
    column: str
    value: Any
    kind: str


_Step = _FirstStep | _CatStep | _NumStep | _ConstStep


class CARTSequentialSynthesizer:
    """Sequential CART synthesizer (synthpop-style conditionals)."""

    def __init__(self) -> None:
        self.include_targets_: bool | None = None
        self.max_depth_: int | None = None
        self.min_samples_leaf_: int | None = None
        self.column_order_: list[str] = []
        self.output_columns_: list[str] = []
        self.kinds_: dict[str, str] = {}
        self.dtypes_: dict[str, Any] = {}
        self.cat_categories_: dict[str, pd.Index] = {}
        self.cat_fill_: dict[str, Any] = {}
        self.num_fill_: dict[str, float] = {}
        self.steps_: list[_Step] = []
        self.n_train_: int = 0

    def fit(
        self,
        df: pd.DataFrame,
        include_targets: bool = True,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
    ) -> CARTSequentialSynthesizer:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df must contain at least one row")

        work = df.copy()
        if not include_targets:
            drop = [c for c in TARGET_COLS if c in work.columns]
            work = work.drop(columns=drop)

        n = len(work)
        kinds = {
            c: "cat" if _is_categorical_series(c, work[c], n) else "num"
            for c in work.columns
        }
        order = _visit_order(work, kinds)

        self.include_targets_ = bool(include_targets)
        self.max_depth_ = int(max_depth)
        self.min_samples_leaf_ = int(min_samples_leaf)
        self.column_order_ = list(order)
        self.output_columns_ = [c for c in work.columns]
        self.kinds_ = kinds
        self.dtypes_ = {c: work[c].dtype for c in work.columns}
        self.cat_categories_ = {}
        self.cat_fill_ = {}
        self.num_fill_ = {}
        self.steps_ = []
        self.n_train_ = n

        for col, kind in kinds.items():
            s = work[col]
            if kind == "cat":
                cats = pd.Index(pd.unique(s.dropna()))
                self.cat_categories_[col] = cats
                mode = s.mode(dropna=True)
                if len(mode):
                    self.cat_fill_[col] = mode.iloc[0]
                elif len(cats):
                    self.cat_fill_[col] = cats[0]
                else:
                    self.cat_fill_[col] = None
            else:
                numeric = pd.to_numeric(s, errors="coerce")
                med = numeric.median()
                self.num_fill_[col] = float(med) if pd.notna(med) else 0.0

        leaf = max(1, int(min_samples_leaf))
        depth = max(1, int(max_depth))
        split = max(2, 2 * leaf)
        rng_tree = 0

        for i, col in enumerate(order):
            y = work[col]
            observed = y.notna()
            if not observed.any():
                fill = (
                    self.cat_fill_.get(col)
                    if kinds[col] == "cat"
                    else self.num_fill_.get(col, 0.0)
                )
                self.steps_.append(_ConstStep(column=col, value=fill, kind=kinds[col]))
                continue

            y_obs = y.loc[observed]
            n_unique = int(y_obs.nunique(dropna=True))
            if n_unique <= 1:
                self.steps_.append(
                    _ConstStep(column=col, value=y_obs.iloc[0], kind=kinds[col])
                )
                continue

            if i == 0:
                self.steps_.append(
                    _FirstStep(column=col, values=y_obs.to_numpy())
                )
                continue

            predictors = order[:i]
            X = self._encode_block(work.loc[observed], predictors)
            y_arr = y_obs.to_numpy()

            if kinds[col] == "cat":
                tree = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=min(leaf, max(1, int(observed.sum()))),
                    min_samples_split=min(split, max(2, int(observed.sum()))),
                    random_state=rng_tree,
                )
                tree.fit(X, y_arr)
                self.steps_.append(
                    _CatStep(
                        column=col,
                        predictors=predictors,
                        tree=tree,
                        classes=np.asarray(tree.classes_),
                    )
                )
            else:
                y_num = pd.to_numeric(y_obs, errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(y_num)
                if finite.sum() <= 1:
                    val = float(y_num[finite][0]) if finite.any() else 0.0
                    self.steps_.append(_ConstStep(column=col, value=val, kind="num"))
                    continue
                tree = DecisionTreeRegressor(
                    max_depth=depth,
                    min_samples_leaf=min(leaf, max(1, int(finite.sum()))),
                    min_samples_split=min(split, max(2, int(finite.sum()))),
                    random_state=rng_tree,
                )
                X_fit = X[finite]
                y_fit = y_num[finite]
                tree.fit(X_fit, y_fit)
                pred = tree.predict(X_fit)
                leaves = tree.apply(X_fit)
                residuals = y_fit - pred
                bag: dict[int, np.ndarray] = {}
                for leaf_id in np.unique(leaves):
                    bag[int(leaf_id)] = residuals[leaves == leaf_id]
                self.steps_.append(
                    _NumStep(
                        column=col,
                        predictors=predictors,
                        tree=tree,
                        leaf_residuals=bag,
                    )
                )
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.steps_ and not self.output_columns_:
            return pd.DataFrame(index=range(int(n)))
        if not self.column_order_:
            raise RuntimeError("CARTSequentialSynthesizer.fit() must be called first")
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return pd.DataFrame(
                {c: pd.Series(dtype=self.dtypes_[c]) for c in self.output_columns_}
            )
        rng = np.random.default_rng(seed)
        syn: dict[str, np.ndarray] = {}
        global_num_resid = np.array([0.0])

        for step in self.steps_:
            if isinstance(step, _ConstStep):
                syn[step.column] = np.full(n, step.value, dtype=object)
                continue
            if isinstance(step, _FirstStep):
                idx = rng.integers(0, len(step.values), size=n)
                syn[step.column] = step.values[idx]
                continue

            prev = pd.DataFrame({c: syn[c] for c in step.predictors})
            X = self._encode_block(prev, step.predictors)

            if isinstance(step, _CatStep):
                proba = step.tree.predict_proba(X)
                syn[step.column] = _sample_from_proba(rng, step.classes, proba)
            else:
                pred = step.tree.predict(X)
                leaves = step.tree.apply(X)
                out = np.empty(n, dtype=float)
                # Vectorize by unique leaf so we still draw leaf-local residuals.
                for leaf_id in np.unique(leaves):
                    mask = leaves == leaf_id
                    resid = step.leaf_residuals.get(int(leaf_id))
                    if resid is None or len(resid) == 0:
                        resid = global_num_resid
                    draws = rng.choice(resid, size=int(mask.sum()), replace=True)
                    out[mask] = pred[mask] + draws
                syn[step.column] = out

        data: dict[str, Any] = {}
        for col in self.output_columns_:
            data[col] = self._cast_column(col, syn[col])
        return pd.DataFrame(data, index=range(n))

    def _encode_block(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.empty((len(df), 0), dtype=float)
        mats: list[np.ndarray] = []
        for col in cols:
            if self.kinds_[col] == "cat":
                mats.append(self._encode_categorical(df[col], col))
            else:
                numeric = pd.to_numeric(pd.Series(df[col].to_numpy()), errors="coerce")
                filled = numeric.fillna(self.num_fill_[col]).to_numpy(dtype=float)
                mats.append(filled)
        return np.column_stack(mats)

    def _encode_categorical(self, series: pd.Series, col: str) -> np.ndarray:
        cats = self.cat_categories_[col]
        fill = self.cat_fill_[col]
        fill_code = _category_code(fill, cats) if fill is not None else -1
        values = series.to_numpy()
        if _is_bool_like(cats) or _is_bool_dtype(self.dtypes_.get(col)):
            norm_cats = [bool(c) for c in cats]
            norm_vals: list[Any] = []
            for v in values:
                if _is_na(v):
                    norm_vals.append(None)
                else:
                    norm_vals.append(bool(v))
            cat = pd.Categorical(norm_vals, categories=norm_cats)
        else:
            cat = pd.Categorical(values, categories=list(cats))
        codes = cat.codes.astype(float)
        if fill_code >= 0:
            codes = np.where(codes < 0, float(fill_code), codes)
        return codes

    def _cast_column(self, col: str, values: np.ndarray) -> Any:
        dtype = self.dtypes_[col]
        if self.kinds_[col] == "cat":
            cats = self.cat_categories_.get(col)
            if _is_bool_dtype(dtype):
                return np.asarray([bool(v) if not _is_na(v) else False for v in values])
            if isinstance(dtype, pd.CategoricalDtype):
                return pd.Categorical(values, categories=dtype.categories)
            if cats is not None and pd.api.types.is_integer_dtype(dtype):
                arr = np.array([int(v) for v in values], dtype=np.int64)
                try:
                    return arr.astype(dtype, copy=False)
                except (TypeError, ValueError):
                    return arr
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                return np.asarray(values, dtype=object)
            return values
        arr = np.asarray(values, dtype=float)
        if pd.api.types.is_integer_dtype(dtype):
            rounded = np.rint(arr)
            try:
                return rounded.astype(dtype, copy=False)
            except (TypeError, ValueError):
                return rounded.astype(np.int64, copy=False)
        return arr


def _is_na(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (ValueError, TypeError):
        return False


def _is_bool_like(cats: pd.Index) -> bool:
    if len(cats) == 0:
        return False
    return all(isinstance(v, (bool, np.bool_)) for v in cats)


def _category_code(value: Any, cats: pd.Index) -> int:
    for i, c in enumerate(cats):
        if c == value:
            return i
    return -1


def _sample_from_proba(
    rng: np.random.Generator, classes: np.ndarray, proba: np.ndarray
) -> np.ndarray:
    """Draw one class per row from a probability matrix (no argmax)."""
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 0.0, None)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    proba = proba / row_sum
    cdf = np.cumsum(proba, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(proba.shape[0])
    idx = (u[:, None] <= cdf).argmax(axis=1)
    return classes[idx]


def synthesize_cart(
    df: pd.DataFrame,
    n: int | None = None,
    seed: int = 0,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Fit a sequential CART synthesizer and draw ``n`` rows (default ``len(df)``)."""
    if n is None:
        n = len(df)
    synth = CARTSequentialSynthesizer()
    synth.fit(df, include_targets=include_targets)
    return synth.sample(int(n), seed=seed)
