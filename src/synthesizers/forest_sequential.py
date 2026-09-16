"""Synthpop-style sequential RandomForest / ExtraTrees synthesizer.

Visit columns in a schema-agnostic order, draw the first from its empirical
distribution, then fit a forest conditional of each later column on the
already-visited ones.  At sample time a *random tree* is chosen per row and
a value is drawn from that tree's leaf (class probabilities, or leaf mean
plus an empirical residual).  Mean-only / argmax prediction is never used.

Type inference is dtype-driven only — no dataset-specific column names:

* object / category / string → categorical
* bool, 0-1, and other low-cardinality integers → categorical / binary
* everything else → numeric

``y`` and ``y_class`` (when present and ``include_targets=True``) are visited
last so targets condition on the full feature vector.  If those names are
absent the last numeric column is visited last instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

TARGET_COLS: tuple[str, ...] = ("y", "y_class")

# Integer cardinality at or below this (and ≤ 10% of n) stays discrete.
_LOW_CARD_CAP = 15


def _is_bool_dtype(dtype: Any) -> bool:
    return pd.api.types.is_bool_dtype(dtype)


def _is_na(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (ValueError, TypeError):
        return False


def _unique_dropna(s: pd.Series) -> np.ndarray:
    return pd.unique(s.dropna())


def _is_zero_one(s: pd.Series) -> bool:
    """True when every non-null value is in {0, 1} (int, float, or bool)."""
    vals = _unique_dropna(s)
    if vals.size == 0 or vals.size > 2:
        return False
    for v in vals:
        if isinstance(v, (bool, np.bool_)):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if fv not in (0.0, 1.0) or not np.isfinite(fv):
            return False
    return True


def _is_categorical_series(s: pd.Series, n: int) -> bool:
    """Infer categorical / binary vs numeric from dtype and cardinality only."""
    dtype = s.dtype
    if _is_bool_dtype(dtype):
        return True
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return True
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(
        dtype
    ):
        return False
    if not pd.api.types.is_numeric_dtype(dtype):
        return True
    if _is_zero_one(s):
        return True
    if pd.api.types.is_integer_dtype(dtype):
        nunique = int(s.nunique(dropna=True))
        return nunique <= min(_LOW_CARD_CAP, max(2, int(0.10 * n)))
    return False


def _visit_order(
    columns: list[str], kinds: dict[str, str], include_targets: bool
) -> list[str]:
    """Given order, then targets last — or the last numeric if no target names."""
    cols = list(columns)
    present_targets = [c for c in TARGET_COLS if c in cols]
    if not include_targets:
        cols = [c for c in cols if c not in TARGET_COLS]
        present_targets = []
    if present_targets:
        rest = [c for c in cols if c not in TARGET_COLS]
        return rest + present_targets
    nums = [c for c in cols if kinds.get(c) == "num"]
    if nums:
        last_num = nums[-1]
        return [c for c in cols if c != last_num] + [last_num]
    return cols


def _pick_ensemble(
    kind: str, n_rows: int, n_unique: int
) -> type:
    """Choose ExtraTrees vs RandomForest per column (CPU-cheap heuristic)."""
    # ExtraTrees: random splits, diverse leaves, faster — prefer for numerics
    # and simple binaries.  RandomForest: bootstrap class frequencies, better
    # when a categorical has several levels or n is small.
    if kind == "num":
        if n_rows < 80:
            return RandomForestRegressor
        return ExtraTreesRegressor
    if n_unique <= 2 and n_rows >= 80:
        return ExtraTreesClassifier
    return RandomForestClassifier


def _forest_params(
    n_rows: int,
    n_features: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> dict[str, Any]:
    leaf = max(1, min(int(min_samples_leaf), max(1, n_rows)))
    split = max(2, min(2 * leaf, max(2, n_rows)))
    depth = max(1, int(max_depth))
    n_est = max(1, int(n_estimators))
    max_features: Any = "sqrt" if n_features >= 4 else None
    return {
        "n_estimators": n_est,
        "max_depth": depth,
        "min_samples_leaf": leaf,
        "min_samples_split": split,
        "n_jobs": 1,
        "random_state": int(random_state),
        "max_features": max_features,
    }


@dataclass
class _FirstStep:
    column: str
    values: np.ndarray


@dataclass
class _CatStep:
    column: str
    predictors: list[str]
    model: Any
    classes: np.ndarray
    family: str


@dataclass
class _NumStep:
    column: str
    predictors: list[str]
    model: Any
    leaf_residuals: list[dict[int, np.ndarray]] = field(default_factory=list)
    family: str = "extra"


@dataclass
class _ConstStep:
    column: str
    value: Any
    kind: str


_Step = _FirstStep | _CatStep | _NumStep | _ConstStep


def _leaf_residual_bags(estimators: list[Any], X: np.ndarray, y: np.ndarray) -> list[dict[int, np.ndarray]]:
    bags: list[dict[int, np.ndarray]] = []
    for tree in estimators:
        pred = tree.predict(X)
        leaves = tree.apply(X)
        resid = y - pred
        bag: dict[int, np.ndarray] = {}
        for leaf_id in np.unique(leaves):
            bag[int(leaf_id)] = resid[leaves == leaf_id]
        bags.append(bag)
    return bags


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


def _sample_random_tree_proba(
    rng: np.random.Generator,
    X: np.ndarray,
    estimators: list[Any],
    classes: np.ndarray,
    n: int,
) -> np.ndarray:
    n_trees = len(estimators)
    tree_ids = rng.integers(0, n_trees, size=n)
    out = np.empty(n, dtype=object)
    for t in range(n_trees):
        mask = tree_ids == t
        if not np.any(mask):
            continue
        tree = estimators[t]
        proba = tree.predict_proba(X[mask])
        tree_classes = np.asarray(getattr(tree, "classes_", classes))
        drawn = _sample_from_proba(rng, tree_classes, proba)
        out[mask] = drawn
    return out


def _sample_random_tree_residuals(
    rng: np.random.Generator,
    X: np.ndarray,
    estimators: list[Any],
    leaf_bags: list[dict[int, np.ndarray]],
    n: int,
) -> np.ndarray:
    n_trees = len(estimators)
    tree_ids = rng.integers(0, n_trees, size=n)
    out = np.empty(n, dtype=float)
    fallback = np.array([0.0])
    for t in range(n_trees):
        mask = tree_ids == t
        if not np.any(mask):
            continue
        tree = estimators[t]
        Xt = X[mask]
        pred = tree.predict(Xt)
        leaves = tree.apply(Xt)
        bags = leaf_bags[t] if t < len(leaf_bags) else {}
        draws = np.empty(int(mask.sum()), dtype=float)
        for leaf_id in np.unique(leaves):
            lm = leaves == leaf_id
            resid = bags.get(int(leaf_id))
            if resid is None or len(resid) == 0:
                # Pool residual bags of this leaf id across trees, else 0.
                pooled = [leaf_bags[j].get(int(leaf_id), fallback) for j in range(n_trees)]
                resid = np.concatenate(pooled) if pooled else fallback
                if resid.size == 0:
                    resid = fallback
            draws[lm] = rng.choice(resid, size=int(lm.sum()), replace=True)
        out[mask] = pred + draws
    return out


class ForestSequentialSynthesizer:
    """Sequential forest synthesizer (synthpop-style conditionals)."""

    def __init__(self) -> None:
        self.include_targets_: bool | None = None
        self.n_estimators_: int | None = None
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
        n_estimators: int = 25,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
    ) -> ForestSequentialSynthesizer:
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
            c: "cat" if _is_categorical_series(work[c], n) else "num"
            for c in work.columns
        }
        order = _visit_order(list(work.columns), kinds, include_targets=True)

        self.include_targets_ = bool(include_targets)
        self.n_estimators_ = int(n_estimators)
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
                cats = pd.Index(_unique_dropna(s))
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

        rng_tree = 0
        n_est = max(1, int(n_estimators))
        depth = max(1, int(max_depth))
        leaf = max(1, int(min_samples_leaf))

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
                self.steps_.append(_FirstStep(column=col, values=y_obs.to_numpy()))
                continue

            predictors = order[:i]
            X = self._encode_block(work.loc[observed], predictors)
            params = _forest_params(
                n_rows=int(observed.sum()),
                n_features=X.shape[1],
                n_estimators=n_est,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=rng_tree,
            )
            rng_tree += 1
            cls = _pick_ensemble(kinds[col], int(observed.sum()), n_unique)
            family = "extra" if cls in (ExtraTreesClassifier, ExtraTreesRegressor) else "rf"

            if kinds[col] == "cat":
                y_arr = y_obs.to_numpy()
                model = cls(**params)
                model.fit(X, y_arr)
                self.steps_.append(
                    _CatStep(
                        column=col,
                        predictors=predictors,
                        model=model,
                        classes=np.asarray(model.classes_),
                        family=family,
                    )
                )
            else:
                y_num = pd.to_numeric(y_obs, errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(y_num)
                if finite.sum() <= 1:
                    val = float(y_num[finite][0]) if finite.any() else 0.0
                    self.steps_.append(_ConstStep(column=col, value=val, kind="num"))
                    continue
                if finite.sum() < int(observed.sum()):
                    params = _forest_params(
                        n_rows=int(finite.sum()),
                        n_features=X.shape[1],
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        random_state=rng_tree - 1,
                    )
                X_fit = X[finite]
                y_fit = y_num[finite]
                model = cls(**params)
                model.fit(X_fit, y_fit)
                bags = _leaf_residual_bags(list(model.estimators_), X_fit, y_fit)
                self.steps_.append(
                    _NumStep(
                        column=col,
                        predictors=predictors,
                        model=model,
                        leaf_residuals=bags,
                        family=family,
                    )
                )
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.steps_ and not self.output_columns_:
            return pd.DataFrame(index=range(int(n)))
        if not self.column_order_:
            raise RuntimeError("ForestSequentialSynthesizer.fit() must be called first")
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return pd.DataFrame(
                {c: pd.Series(dtype=self.dtypes_[c]) for c in self.output_columns_}
            )
        rng = np.random.default_rng(seed)
        syn: dict[str, np.ndarray] = {}

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
                syn[step.column] = _sample_random_tree_proba(
                    rng, X, list(step.model.estimators_), step.classes, n
                )
            else:
                syn[step.column] = _sample_random_tree_residuals(
                    rng,
                    X,
                    list(step.model.estimators_),
                    step.leaf_residuals,
                    n,
                )

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


def _is_bool_like(cats: pd.Index) -> bool:
    if len(cats) == 0:
        return False
    return all(isinstance(v, (bool, np.bool_)) for v in cats)


def _category_code(value: Any, cats: pd.Index) -> int:
    for i, c in enumerate(cats):
        if c == value:
            return i
    return -1


def synthesize_forest(
    df: pd.DataFrame,
    n: int | None = None,
    seed: int = 0,
    include_targets: bool = True,
    n_estimators: int = 25,
    max_depth: int = 8,
    min_samples_leaf: int = 20,
) -> pd.DataFrame:
    """Fit a sequential forest synthesizer and draw ``n`` rows (default ``len(df)``)."""
    if n is None:
        n = len(df)
    synth = ForestSequentialSynthesizer()
    synth.fit(
        df,
        include_targets=include_targets,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    return synth.sample(int(n), seed=seed)
