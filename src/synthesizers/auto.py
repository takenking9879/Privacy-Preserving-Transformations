"""Portfolio / bake-off synthesizer (the candidate GENERAL method).

``AutoSynthesizer`` tries the registered families (copula, cart, mixture,
hybrid, plus forest / knn when present) on a hold-out of the *real*
table, picks a winner, then re-fits that winner on the full frame.

Winner rule
-----------
Each surviving candidate is scored

    combined = fidelity_score + utility_score

where ``fidelity_score`` is ``compute_statistical_fidelity(train, synth)``
and ``utility_score`` is a cheap linear TSTR on ``y`` / ``y_class`` when
those columns exist:

    utility = mean of clip(1 - max(0, gap), 0, 1)

for LinearRegression R² gap (TRTR − TSTR) and LogisticRegression
accuracy / ROC-AUC gap.  The winner is the highest ``combined``, with
ties broken by **lowest** ``tstr_gap_r2`` then highest fidelity.

``evaluate_model_utility`` is not used in the bake-off: it fits random
forests / HGB and is too slow for a multi-candidate sweep.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.metrics.statistical import compute_statistical_fidelity
from src.synthesizers.registry import SYNTHESIZERS, adapt, try_register

BAKEOFF_MAX_ROWS = 400
HOLDOUT_FRAC = 0.25
TARGET_REG = "y"
TARGET_CLF = "y_class"
TARGET_NAMES = frozenset({TARGET_REG, TARGET_CLF})
FORCED_CATEGORICAL = frozenset({"region", "segment"})
PREFERRED = ("copula", "cart", "mixture", "hybrid", "forest", "knn")
_SKIP_NAMES = frozenset({"auto", "identity", "negative", "negative_control"})

SELECTION_RULE = (
    "combined = fidelity_score + utility_score; "
    "utility_score is a cheap linear TSTR "
    "(clip(1 - max(0, TRTR-TSTR gap), 0, 1) on y / y_class). "
    "Winner = highest combined, then lowest tstr_gap_r2, then highest fidelity. "
    "Winner is re-fit on the full table."
)

_OPTIONAL_FAMILIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "forest",
        (
            "src.synthesizers.forest",
            "src.synthesizers.random_forest",
            "src.synthesizers.rf",
        ),
        (
            "ForestSynthesizer",
            "RandomForestSynthesizer",
            "RFSynthesizer",
        ),
        (
            "synthesize_forest",
            "synthesize_random_forest",
            "synthesize_rf",
        ),
    ),
    (
        "knn",
        (
            "src.synthesizers.knn",
            "src.synthesizers.nearest_neighbor",
            "src.synthesizers.nearest_neighbors",
        ),
        (
            "KNNSynthesizer",
            "KnnSynthesizer",
            "NearestNeighborSynthesizer",
        ),
        (
            "synthesize_knn",
            "synthesize_nearest_neighbor",
        ),
    ),
)


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name != "self"
    }
    return {k: v for k, v in kwargs.items() if k in accepted}


def _unwrap_registered(
    fn: Callable[..., Any],
) -> tuple[Optional[type], Optional[Callable[..., Any]]]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None, fn if callable(fn) else None
    cls = params["_cls"].default if "_cls" in params else None
    inner = params["_fn"].default if "_fn" in params else None
    if isinstance(cls, type):
        return cls, None
    if callable(inner) and not isinstance(inner, type):
        return None, inner
    if isinstance(inner, type):
        return inner, None
    if callable(fn):
        return None, fn
    return None, None


def _construct(cls: type, seed: int) -> Any:
    last: Optional[BaseException] = None
    for kw in ({"seed": seed}, {"random_state": seed}, {}):
        try:
            return cls(**_filter_kwargs(cls.__init__, kw))
        except TypeError as exc:
            last = exc
    if last is not None:
        try:
            return cls()
        except TypeError:
            raise last
    return cls()


def _call_fit(inst: Any, df: pd.DataFrame, include_targets: bool, seed: int) -> Any:
    fit = getattr(inst, "fit", None)
    if not callable(fit):
        raise TypeError(f"{type(inst).__name__} has no fit()")
    attempts = (
        {
            "include_targets": include_targets,
            "exclude_targets": not include_targets,
            "seed": seed,
            "random_state": seed,
        },
        {"include_targets": include_targets, "seed": seed},
        {"exclude_targets": not include_targets},
        {"include_targets": include_targets},
        {"seed": seed},
        {},
    )
    last: Optional[BaseException] = None
    for kw in attempts:
        try:
            out = fit(df, **_filter_kwargs(fit, kw))
            return out if out is not None and hasattr(out, "sample") else inst
        except TypeError as exc:
            last = exc
    if last is not None:
        raise last
    return inst


def _align_schema(synth: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    out = synth.copy()
    for col in template.columns:
        if col not in out.columns:
            out[col] = np.nan
    extra = [c for c in out.columns if c not in template.columns]
    if extra:
        out = out.drop(columns=extra)
    return out.loc[:, list(template.columns)].reset_index(drop=True)


def _is_categorical(name: str, series: pd.Series) -> bool:
    if name in FORCED_CATEGORICAL:
        return True
    if pd.api.types.is_bool_dtype(series):
        return False
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return True
    return False


def _column_roles(
    frame: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in feature_cols:
        series = frame[col] if col in frame.columns else pd.Series(dtype=float)
        if _is_categorical(col, series):
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def _feature_frame(
    frame: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    data: dict[str, Any] = {}
    n = len(frame)
    index = frame.index
    for col in feature_cols:
        if col in frame.columns:
            series = frame[col]
        else:
            series = pd.Series(np.full(n, np.nan), index=index)
        if col in cat_cols:
            as_obj = series.astype(object)
            data[col] = as_obj.where(series.notna(), other=np.nan)
        else:
            data[col] = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(data, index=index)


def _make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
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
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("no feature columns for TSTR")
    return ColumnTransformer(transformers, remainder="drop")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _gap_to_utility(gap: Optional[float]) -> Optional[float]:
    if gap is None or not np.isfinite(gap):
        return None
    return _clip01(1.0 - max(0.0, float(gap)))


def _regression_r2(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    pre: ColumnTransformer,
) -> Optional[float]:
    mask_tr = y_tr.notna()
    mask_te = y_te.notna()
    if int(mask_tr.sum()) < 3 or int(mask_te.sum()) < 2:
        return None
    try:
        pipe = Pipeline([("pre", pre), ("m", LinearRegression())])
        pipe.fit(X_tr.loc[mask_tr], y_tr.loc[mask_tr])
        pred = pipe.predict(X_te.loc[mask_te])
        score = float(r2_score(y_te.loc[mask_te], pred))
    except Exception:  # noqa: BLE001 — bake-off must not crash
        return None
    return score if np.isfinite(score) else None


def _classification_score(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    pre: ColumnTransformer,
) -> Optional[float]:
    mask_tr = y_tr.notna()
    mask_te = y_te.notna()
    y_tr = y_tr.loc[mask_tr]
    y_te = y_te.loc[mask_te]
    X_tr = X_tr.loc[mask_tr]
    X_te = X_te.loc[mask_te]
    if len(y_tr) < 3 or len(y_te) < 2 or y_tr.nunique(dropna=True) < 2:
        return None
    try:
        pipe = Pipeline(
            [
                ("pre", pre),
                (
                    "m",
                    LogisticRegression(
                        max_iter=250,
                        solver="lbfgs",
                        random_state=0,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        if y_te.nunique(dropna=True) == 2:
            try:
                proba = pipe.predict_proba(X_te)
                classes = list(pipe.named_steps["m"].classes_)
                pos: Any
                if 1 in classes:
                    pos = 1
                elif True in classes:
                    pos = True
                elif "1" in classes:
                    pos = "1"
                else:
                    pos = classes[-1]
                col = classes.index(pos)
                auc = float(roc_auc_score(y_te, proba[:, col]))
                if np.isfinite(auc):
                    return auc
            except Exception:  # noqa: BLE001
                pass
        acc = float(accuracy_score(y_te, pred))
        return acc if np.isfinite(acc) else None
    except Exception:  # noqa: BLE001
        return None


def _cheap_linear_tstr(
    real_train: pd.DataFrame,
    real_valid: pd.DataFrame,
    synth: pd.DataFrame,
) -> dict[str, Any]:
    """Train-on-synth / test-on-real-holdout linear TSTR vs TRTR ceiling."""
    feature_cols = [c for c in real_train.columns if c not in TARGET_NAMES]
    has_y = TARGET_REG in real_train.columns and TARGET_REG in synth.columns
    has_yc = TARGET_CLF in real_train.columns and TARGET_CLF in synth.columns
    empty = {
        "utility_score": 0.0,
        "tstr_gap_r2": None,
        "tstr_gap_clf": None,
        "trtr_r2": None,
        "tstr_r2": None,
    }
    if not feature_cols or (not has_y and not has_yc):
        return empty
    if len(real_train) < 8 or len(real_valid) < 3 or len(synth) < 8:
        return empty

    num_cols, cat_cols = _column_roles(real_train, feature_cols)
    try:
        pre = _make_preprocessor(num_cols, cat_cols)
    except ValueError:
        return empty

    X_real_tr = _feature_frame(real_train, feature_cols, cat_cols)
    X_real_te = _feature_frame(real_valid, feature_cols, cat_cols)
    X_syn = _feature_frame(synth, feature_cols, cat_cols)

    gap_r2: Optional[float] = None
    trtr_r2: Optional[float] = None
    tstr_r2: Optional[float] = None
    if has_y:
        y_tr = pd.to_numeric(real_train[TARGET_REG], errors="coerce")
        y_te = pd.to_numeric(real_valid[TARGET_REG], errors="coerce")
        y_syn = pd.to_numeric(synth[TARGET_REG], errors="coerce")
        trtr_r2 = _regression_r2(X_real_tr, y_tr, X_real_te, y_te, pre)
        tstr_r2 = _regression_r2(X_syn, y_syn, X_real_te, y_te, pre)
        if trtr_r2 is not None and tstr_r2 is not None:
            gap_r2 = float(trtr_r2 - tstr_r2)

    gap_clf: Optional[float] = None
    if has_yc:
        y_tr = real_train[TARGET_CLF]
        y_te = real_valid[TARGET_CLF]
        y_syn = synth[TARGET_CLF]
        trtr_c = _classification_score(X_real_tr, y_tr, X_real_te, y_te, pre)
        tstr_c = _classification_score(X_syn, y_syn, X_real_te, y_te, pre)
        if trtr_c is not None and tstr_c is not None:
            gap_clf = float(trtr_c - tstr_c)

    parts = [_gap_to_utility(g) for g in (gap_r2, gap_clf)]
    finite = [p for p in parts if p is not None]
    utility = float(np.mean(finite)) if finite else 0.0
    return {
        "utility_score": utility,
        "tstr_gap_r2": gap_r2,
        "tstr_gap_clf": gap_clf,
        "trtr_r2": trtr_r2,
        "tstr_r2": tstr_r2,
    }


@dataclass
class _Candidate:
    name: str
    synthesize: Callable[..., pd.DataFrame]
    cls: Optional[type] = None


class _FnAdapter:
    """Re-fit a registry callable on every ``sample`` (function-only winner)."""

    def __init__(
        self,
        fn: Callable[..., pd.DataFrame],
        df: pd.DataFrame,
        include_targets: bool,
    ) -> None:
        self.fn = fn
        self.df = df
        self.include_targets = include_targets

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        s = 0 if seed is None else int(seed)
        return self.fn(
            self.df, n=int(n), seed=s, include_targets=self.include_targets
        )


class _BootstrapSynthesizer:
    """Last-resort row bootstrap so ``sample`` still works if all candidates die."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, include_targets: bool = True, seed: int = 0) -> "_BootstrapSynthesizer":
        self._df = df.reset_index(drop=True).copy()
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("call fit() before sample()")
        rng = np.random.default_rng(None if seed is None else int(seed))
        idx = rng.integers(0, len(self._df), size=int(n))
        return self._df.iloc[idx].reset_index(drop=True)


def _discover_optional(
    name: str,
    modules: tuple[str, ...],
    class_names: tuple[str, ...],
    fn_names: tuple[str, ...],
) -> Optional[_Candidate]:
    for mod_path in modules:
        try:
            module = importlib.import_module(mod_path)
        except Exception:  # noqa: BLE001
            continue
        picked: Any = None
        for attr in fn_names:
            obj = getattr(module, attr, None)
            if callable(obj) and not isinstance(obj, type):
                picked = obj
                break
        if picked is None:
            for attr in class_names:
                obj = getattr(module, attr, None)
                if isinstance(obj, type):
                    picked = obj
                    break
        if picked is None:
            for attr in dir(module):
                if attr.startswith("_"):
                    continue
                obj = getattr(module, attr, None)
                if (
                    isinstance(obj, type)
                    and attr.endswith("Synthesizer")
                    and attr not in {"BaseSynthesizer", "Synthesizer", "AutoSynthesizer"}
                ):
                    picked = obj
                    break
                if callable(obj) and not isinstance(obj, type) and attr.startswith("synthesize"):
                    picked = obj
                    break
        if picked is None:
            continue
        try:
            fn = adapt(picked, name)
        except Exception:  # noqa: BLE001
            continue
        cls = picked if isinstance(picked, type) else None
        return _Candidate(name, fn, cls)
    return None


def _collect_candidates() -> dict[str, _Candidate]:
    """Registered families first, then duck-typed forest / knn extras."""
    try:
        from src.synthesizers import registry as _reg

        for name in ("forest", "knn"):
            if name in getattr(_reg, "_SPECS", {}) and name not in SYNTHESIZERS:
                try:
                    try_register(name)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass

    out: dict[str, _Candidate] = {}
    for name, fn in list(SYNTHESIZERS.items()):
        if name in _SKIP_NAMES or not callable(fn):
            continue
        cls, _inner = _unwrap_registered(fn)
        out[name] = _Candidate(name, fn, cls)

    for name, modules, class_names, fn_names in _OPTIONAL_FAMILIES:
        if name in out:
            continue
        extra = _discover_optional(name, modules, class_names, fn_names)
        if extra is not None:
            out[name] = extra
    return out


def _ordered_names(candidates: dict[str, _Candidate]) -> list[str]:
    preferred = [n for n in PREFERRED if n in candidates]
    rest = sorted(n for n in candidates if n not in PREFERRED)
    return preferred + rest


def _subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)


def _holdout_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    if TARGET_CLF in df.columns:
        yc = df[TARGET_CLF]
        counts = yc.value_counts(dropna=True)
        if yc.notna().all() and len(counts) >= 2 and int(counts.min()) >= 2:
            stratify = yc
    test_size = HOLDOUT_FRAC
    if len(df) < 8:
        # Tiny tables: keep at least a couple of validation rows.
        test_size = max(1 / max(len(df), 1), min(HOLDOUT_FRAC, 0.4))
    train, valid = train_test_split(
        df,
        test_size=test_size,
        random_state=int(seed),
        stratify=stratify,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def _rank_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    gap = rec.get("tstr_gap_r2")
    gap_key = float(gap) if gap is not None and np.isfinite(gap) else 1.0e9
    return (
        -float(rec.get("combined_score") or 0.0),
        gap_key,
        -float(rec.get("fidelity_score") or 0.0),
        str(rec.get("name") or ""),
    )


def _refit_winner(
    cand: _Candidate, df: pd.DataFrame, include_targets: bool, seed: int
) -> Any:
    if cand.cls is not None:
        try:
            inst = _construct(cand.cls, seed)
            fitted = _call_fit(inst, df, include_targets, seed)
            if hasattr(fitted, "sample") and callable(fitted.sample):
                return fitted
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Re-fit of winner class {cand.cls!r} failed ({exc}); "
                "falling back to the registry callable.",
                RuntimeWarning,
                stacklevel=2,
            )
    return _FnAdapter(cand.synthesize, df, include_targets)


class AutoSynthesizer:
    """Hold-out bake-off over registered synthesizers; ``sample`` uses the winner."""

    def __init__(self) -> None:
        self.include_targets: bool = True
        self.columns_: list[str] = []
        self.candidate_scores: dict[str, dict[str, Any]] = {}
        self._winner_name: str = ""
        self._winner: Any = None
        self._fitted: bool = False

    @property
    def winner_name(self) -> str:
        return self._winner_name

    def fit(
        self,
        df: pd.DataFrame,
        include_targets: bool = True,
        seed: int = 0,
    ) -> "AutoSynthesizer":
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df must contain at least one row")

        full = df.reset_index(drop=True).copy()
        self.include_targets = bool(include_targets)
        self.columns_ = list(full.columns)
        self.candidate_scores = {}
        self._winner = None
        self._winner_name = ""
        self._fitted = False

        train_full, valid = _holdout_split(full, int(seed))
        bakeoff_train = _subsample(train_full, BAKEOFF_MAX_ROWS, int(seed) + 1)
        n_synth = int(len(bakeoff_train))

        candidates = _collect_candidates()
        records: list[dict[str, Any]] = []
        for i, name in enumerate(_ordered_names(candidates)):
            cand = candidates[name]
            rec: dict[str, Any] = {
                "name": name,
                "status": "ok",
                "error": None,
                "fidelity_score": 0.0,
                "utility_score": 0.0,
                "combined_score": 0.0,
                "tstr_gap_r2": None,
            }
            try:
                synth = cand.synthesize(
                    bakeoff_train,
                    n_synth,
                    int(seed) + 17 + i,
                    include_targets=True,
                )
                if not isinstance(synth, pd.DataFrame):
                    raise TypeError(
                        f"{name} returned {type(synth).__name__}, not DataFrame"
                    )
                if synth.empty:
                    raise ValueError(f"{name} returned an empty table")
                synth = _align_schema(synth, bakeoff_train)
                fid = compute_statistical_fidelity(bakeoff_train, synth)
                fidelity = float(fid.get("fidelity_score") or 0.0)
                tstr = _cheap_linear_tstr(bakeoff_train, valid, synth)
                utility = float(tstr.get("utility_score") or 0.0)
                rec.update(
                    {
                        "fidelity_score": fidelity,
                        "utility_score": utility,
                        "combined_score": fidelity + utility,
                        "tstr_gap_r2": tstr.get("tstr_gap_r2"),
                        "tstr_gap_clf": tstr.get("tstr_gap_clf"),
                        "trtr_r2": tstr.get("trtr_r2"),
                        "tstr_r2": tstr.get("tstr_r2"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — skip crashed candidates
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    f"AutoSynthesizer skipped candidate '{name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.candidate_scores[name] = rec
            if rec["status"] == "ok":
                records.append(rec)

        if records:
            winner_rec = sorted(records, key=_rank_key)[0]
            winner = candidates[str(winner_rec["name"])]
            self._winner_name = winner.name
            self._winner = _refit_winner(
                winner, full, self.include_targets, int(seed)
            )
        else:
            warnings.warn(
                "AutoSynthesizer: every candidate failed; using row bootstrap.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._winner_name = "bootstrap"
            self._winner = _BootstrapSynthesizer().fit(full)

        self._fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self._fitted or self._winner is None:
            raise RuntimeError("call fit() before sample()")
        n = int(n)
        meth = getattr(self._winner, "sample", None)
        if not callable(meth):
            raise RuntimeError("winner has no sample() method")
        try:
            out = meth(n, seed=seed)
        except TypeError:
            try:
                out = meth(n, random_state=seed)
            except TypeError:
                out = meth(n)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"winner.sample returned {type(out).__name__}, not DataFrame"
            )
        aligned = _align_schema(out, pd.DataFrame(columns=self.columns_))
        if len(aligned) > n:
            aligned = aligned.iloc[:n].reset_index(drop=True)
        elif len(aligned) < n and len(aligned) > 0:
            aligned = aligned.iloc[np.arange(n) % len(aligned)].reset_index(drop=True)
        return aligned


def synthesize_auto(
    df: pd.DataFrame,
    n: int | None = None,
    seed: int = 0,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Fit :class:`AutoSynthesizer` on ``df`` and draw ``n`` rows (default ``len(df)``)."""
    if n is None:
        n = int(len(df))
    return (
        AutoSynthesizer()
        .fit(df, include_targets=include_targets, seed=int(seed))
        .sample(int(n), seed=int(seed))
    )


def _self_test() -> dict[str, Any]:
    try:
        from src.dgp import generate_original

        df = generate_original(n=160, seed=7)
        source = "generate_original"
    except Exception:  # noqa: BLE001
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "age": rng.normal(42.0, 10.0, 120),
                "region": rng.choice(["N", "S"], 120),
                "y": rng.normal(size=120),
                "y_class": rng.integers(0, 2, 120),
            }
        )
        source = "fallback"

    auto = AutoSynthesizer()
    auto.fit(df, include_targets=True, seed=0)
    out = auto.sample(48, seed=1)
    no_tgt_cols = [c for c in df.columns if c not in TARGET_NAMES]
    auto_x = AutoSynthesizer().fit(df[no_tgt_cols], include_targets=True, seed=1)
    out_x = auto_x.sample(24, seed=2)
    helper = synthesize_auto(df.iloc[:80].reset_index(drop=True), n=20, seed=2)

    ok = (
        isinstance(out, pd.DataFrame)
        and len(out) == 48
        and list(out.columns) == list(df.columns)
        and auto.winner_name != ""
        and len(out_x) == 24
        and list(out_x.columns) == no_tgt_cols
        and len(helper) == 20
    )
    return {
        "ok": ok,
        "source": source,
        "winner": auto.winner_name,
        "selection_rule": SELECTION_RULE,
        "scores": auto.candidate_scores,
        "n_out": int(len(out)),
        "columns": list(out.columns),
        "winner_no_targets": auto_x.winner_name,
    }


__all__ = [
    "AutoSynthesizer",
    "SELECTION_RULE",
    "synthesize_auto",
]


if __name__ == "__main__":
    result = _self_test()
    print("winner:", result["winner"])
    print("rule:", result["selection_rule"])
    for name, rec in result["scores"].items():
        print(
            f"  {name}: status={rec.get('status')} "
            f"fid={rec.get('fidelity_score')} util={rec.get('utility_score')} "
            f"sum={rec.get('combined_score')} gap={rec.get('tstr_gap_r2')} "
            f"err={rec.get('error')}"
        )
    if not result["ok"]:
        raise SystemExit("auto synthesizer self-test failed")
    print("self-test ok")
