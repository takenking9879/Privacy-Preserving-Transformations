"""Hybrid synthesizer: copula for X + residual-sampling models for Y.

This is the method most aligned with the product goal.

1. Fit a Gaussian copula on the *features* so Law(X') ≈ Law(X)
   (marginals + rank-linear dependence).
2. Fit supervised models of P(Y | X) and P(Y_class | X) on the real table.
3. Draw X' from the copula, then draw Y' from the fitted conditionals
   *with residual / probability sampling* so we do not collapse to the
   conditional mean.

TSTR then sees a table whose X-structure and X→Y map were estimated
separately — the usual failure mode of a joint Gaussian (missed
interactions in Y) and of a small-n CART (broken X joints) is reduced.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.synthesizers.copula import GaussianCopulaSynthesizer

TARGET_COLS: tuple[str, ...] = ("y", "y_class")
FORCED_CATEGORICAL: frozenset[str] = frozenset({"region", "segment"})


def _is_categorical(name: str, s: pd.Series) -> bool:
    if name in FORCED_CATEGORICAL:
        return True
    if pd.api.types.is_bool_dtype(s):
        return False
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    return False


class HybridSynthesizer:
    """Copula features + random-forest conditionals for the targets."""

    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 8,
        min_samples_leaf: int = 15,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.include_targets: bool = True
        self.feature_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.copula: Optional[GaussianCopulaSynthesizer] = None
        self.pre: Optional[ColumnTransformer] = None
        self.reg: Optional[Pipeline] = None
        self.clf: Optional[Pipeline] = None
        self.residuals_: Optional[np.ndarray] = None
        self.y_log_: bool = False
        self.columns_: list[str] = []
        self.dtypes_: dict[str, Any] = {}
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, include_targets: bool = True) -> "HybridSynthesizer":
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.include_targets = bool(include_targets)
        self.columns_ = list(df.columns)
        self.dtypes_ = {c: df[c].dtype for c in df.columns}
        self.feature_cols = [c for c in df.columns if c not in TARGET_COLS]
        if not self.feature_cols:
            raise ValueError("HybridSynthesizer needs at least one feature column")

        self.cat_cols = [c for c in self.feature_cols if _is_categorical(c, df[c])]
        self.num_cols = [c for c in self.feature_cols if c not in self.cat_cols]

        self.copula = GaussianCopulaSynthesizer()
        self.copula.fit(df.loc[:, self.feature_cols], exclude_targets=True)

        X = df.loc[:, self.feature_cols]
        self.pre = self._make_preprocessor()

        if self.include_targets and "y" in df.columns:
            y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.sum() >= 20:
                y_obs = y[mask]
                self.y_log_ = bool(np.nanmin(y_obs) > 0.0)
                y_fit = np.log(y_obs) if self.y_log_ else y_obs
                self.reg = Pipeline(
                    [
                        ("pre", self._make_preprocessor()),
                        (
                            "rf",
                            RandomForestRegressor(
                                n_estimators=self.n_estimators,
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                random_state=0,
                                n_jobs=1,
                            ),
                        ),
                    ]
                )
                self.reg.fit(X.loc[mask], y_fit)
                pred = np.asarray(self.reg.predict(X.loc[mask]), dtype=float)
                resid = y_fit - pred
                # Keep a non-degenerate residual bag even if the forest is sharp.
                if float(np.std(resid)) < 1e-8:
                    resid = resid + np.random.default_rng(0).normal(0.0, 1e-3, size=resid.shape)
                self.residuals_ = resid

        if self.include_targets and "y_class" in df.columns:
            yc = pd.to_numeric(df["y_class"], errors="coerce")
            mask = yc.notna().to_numpy()
            if mask.sum() >= 20 and int(yc.nunique(dropna=True)) >= 2:
                self.clf = Pipeline(
                    [
                        ("pre", self._make_preprocessor()),
                        (
                            "rf",
                            RandomForestClassifier(
                                n_estimators=self.n_estimators,
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                random_state=0,
                                n_jobs=1,
                            ),
                        ),
                    ]
                )
                self.clf.fit(X.loc[mask], yc.loc[mask].astype(int))

        self._fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self._fitted or self.copula is None:
            raise RuntimeError("HybridSynthesizer.fit() must be called first")
        rng = np.random.default_rng(None if seed is None else int(seed))
        copula_seed = None if seed is None else int(seed) + 17
        out = self.copula.sample(int(n), seed=copula_seed)
        # Restore any feature columns the copula dropped.
        for c in self.feature_cols:
            if c not in out.columns:
                out[c] = np.nan
        out = out.loc[:, [c for c in self.feature_cols if c in out.columns]].copy()

        if self.include_targets and self.reg is not None and self.residuals_ is not None:
            pred = np.asarray(self.reg.predict(out), dtype=float)
            resid = rng.choice(self.residuals_, size=len(out), replace=True)
            y = pred + resid
            if self.y_log_:
                y = np.exp(np.clip(y, -20.0, 20.0))
            out["y"] = y
        elif self.include_targets and "y" in self.columns_:
            out["y"] = np.nan

        if self.include_targets and self.clf is not None:
            proba = np.asarray(self.clf.predict_proba(out), dtype=float)
            classes = np.asarray(self.clf.named_steps["rf"].classes_)
            draws = np.empty(len(out), dtype=classes.dtype)
            for i in range(len(out)):
                p = proba[i]
                p = p / p.sum() if p.sum() > 0 else np.full(len(classes), 1.0 / len(classes))
                draws[i] = rng.choice(classes, p=p)
            out["y_class"] = draws.astype(np.int64)
        elif self.include_targets and "y_class" in self.columns_:
            out["y_class"] = 0

        # Column order and dtypes from the training frame.
        ordered = []
        for c in self.columns_:
            if c in out.columns:
                ordered.append(c)
            elif self.include_targets or c not in TARGET_COLS:
                out[c] = np.nan
                ordered.append(c)
        out = out.loc[:, ordered]
        for c, dtype in self.dtypes_.items():
            if c not in out.columns:
                continue
            try:
                if c == "y_class":
                    out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.int64)
                elif str(dtype).startswith("int") and c != "y":
                    out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
                    if not out[c].isna().any():
                        out[c] = out[c].astype(np.int64)
                else:
                    out[c] = out[c].astype(dtype, errors="ignore")
            except (TypeError, ValueError):
                pass
        return out.reset_index(drop=True)

    def _make_preprocessor(self) -> ColumnTransformer:
        transformers = []
        if self.num_cols:
            transformers.append(
                (
                    "num",
                    SimpleImputer(strategy="median"),
                    self.num_cols,
                )
            )
        if self.cat_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            (
                                "oh",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    self.cat_cols,
                )
            )
        return ColumnTransformer(transformers, remainder="drop")


def synthesize_hybrid(
    df: pd.DataFrame,
    n: int | None = None,
    seed: int = 0,
    include_targets: bool = True,
) -> pd.DataFrame:
    if n is None:
        n = int(len(df))
    return HybridSynthesizer().fit(df, include_targets=include_targets).sample(int(n), seed=seed)
