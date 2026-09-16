"""Row-mix ensemble of the available tabular synthesizers.

``EnsembleSynthesizer`` fits every family that can be imported and that
succeeds on the training table (copula, cart, mixture, hybrid, and forest
when that module is present).  ``sample`` draws a *preferred* mix of rows

    40% cart, 30% hybrid, 20% mixture, 10% copula

plus a 15% share for forest when it fitted, then **renormalizes** those
weights among the members that actually fitted and concatenates the
blocks.  Rows are shuffled so the output is not method-blocked.

The mixer is schema-agnostic: any mixed-type frame works.  Conventional
target names ``y`` / ``y_class`` are dropped from the output only when
``include_targets=False``.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

TARGET_COLS: tuple[str, ...] = ("y", "y_class")

# Preferred mixing weights.  Forest is optional; its share is used only
# when that family imported and fitted.  Surviving weights are renormalized.
PREFERRED_SHARES: dict[str, float] = {
    "cart": 0.40,
    "hybrid": 0.30,
    "mixture": 0.20,
    "copula": 0.10,
    "forest": 0.15,
}

_FAMILY_SPECS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "cart",
        (
            "src.synthesizers.cart_sequential",
            "src.synthesizers.sequential_cart",
            "src.synthesizers.cart",
        ),
        (
            "CARTSequentialSynthesizer",
            "SequentialCARTSynthesizer",
            "CARTSynthesizer",
            "CartSynthesizer",
        ),
        ("synthesize_cart", "synthesize_sequential_cart"),
    ),
    (
        "hybrid",
        (
            "src.synthesizers.hybrid",
            "src.synthesizers.copula_conditional",
        ),
        ("HybridSynthesizer", "CopulaConditionalSynthesizer"),
        ("synthesize_hybrid",),
    ),
    (
        "mixture",
        (
            "src.synthesizers.conditional_mixture",
            "src.synthesizers.mixture",
            "src.synthesizers.gmm",
        ),
        (
            "ConditionalMixtureSynthesizer",
            "MixtureSynthesizer",
            "GaussianMixtureSynthesizer",
        ),
        ("synthesize_mixture", "synthesize_conditional_mixture"),
    ),
    (
        "copula",
        (
            "src.synthesizers.copula",
            "src.synthesizers.gaussian_copula",
            "src.synthesizers.copulas",
        ),
        (
            "GaussianCopulaSynthesizer",
            "CopulaSynthesizer",
            "GaussianCopula",
        ),
        ("synthesize_copula", "synthesize_gaussian_copula"),
    ),
    (
        "forest",
        (
            "src.synthesizers.forest_sequential",
            "src.synthesizers.forest",
            "src.synthesizers.random_forest",
            "src.synthesizers.rf",
        ),
        (
            "ForestSequentialSynthesizer",
            "ForestSynthesizer",
            "RandomForestSynthesizer",
            "RFSynthesizer",
        ),
        ("synthesize_forest", "synthesize_random_forest", "synthesize_rf"),
    ),
)

_SEED_OFFSETS: dict[str, int] = {
    "cart": 11,
    "hybrid": 23,
    "mixture": 37,
    "copula": 53,
    "forest": 71,
    "bootstrap": 97,
}


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


def _output_columns(columns: list[str], include_targets: bool) -> list[str]:
    if include_targets:
        return list(columns)
    return [c for c in columns if c not in TARGET_COLS]


def _is_target_col(name: str) -> bool:
    return name in TARGET_COLS


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
        {"exclude_targets": not include_targets, "seed": seed},
        {"include_targets": include_targets},
        {"exclude_targets": not include_targets},
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


def _call_sample(obj: Any, n: int, seed: int | None) -> pd.DataFrame:
    meth = getattr(obj, "sample", None)
    if not callable(meth):
        raise TypeError(f"{type(obj).__name__} has no sample()")
    attempts = (
        lambda: meth(n, seed=seed),
        lambda: meth(n, random_state=seed),
        lambda: meth(int(n), seed=seed),
        lambda: meth(n),
        lambda: meth(n_samples=n, seed=seed),
    )
    last: Optional[BaseException] = None
    for attempt in attempts:
        try:
            out = attempt()
        except TypeError as exc:
            last = exc
            continue
        if isinstance(out, pd.DataFrame):
            return out
        last = TypeError(f"sample returned {type(out).__name__}, not DataFrame")
    if last is not None:
        raise last
    raise TypeError("sample() produced no DataFrame")


def _pick_from_module(
    module: Any,
    class_names: tuple[str, ...],
    fn_names: tuple[str, ...],
) -> Optional[Any]:
    for attr in class_names:
        obj = getattr(module, attr, None)
        if isinstance(obj, type) and hasattr(obj, "fit") and hasattr(obj, "sample"):
            return obj
    for attr in fn_names:
        obj = getattr(module, attr, None)
        if callable(obj) and not isinstance(obj, type):
            return obj
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        obj = getattr(module, attr, None)
        if (
            isinstance(obj, type)
            and attr.endswith("Synthesizer")
            and attr
            not in {
                "BaseSynthesizer",
                "Synthesizer",
                "AutoSynthesizer",
                "EnsembleSynthesizer",
            }
            and hasattr(obj, "fit")
            and hasattr(obj, "sample")
        ):
            return obj
    return None


def _discover_families() -> dict[str, Any]:
    """Return ``{name: class_or_fn}`` for every importable family."""
    found: dict[str, Any] = {}
    for name, modules, class_names, fn_names in _FAMILY_SPECS:
        for mod_path in modules:
            try:
                module = importlib.import_module(mod_path)
            except Exception:  # noqa: BLE001 — family is optional
                continue
            picked = _pick_from_module(module, class_names, fn_names)
            if picked is None:
                continue
            found[name] = picked
            break
    return found


class _FnMember:
    """Adapter so a ``synthesize_*`` helper can sit next to fit/sample classes."""

    def __init__(
        self,
        fn: Callable[..., Any],
        df: pd.DataFrame,
        include_targets: bool,
    ) -> None:
        self.fn = fn
        self.df = df
        self.include_targets = include_targets

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        s = 0 if seed is None else int(seed)
        kw_full = {
            "n": int(n),
            "seed": s,
            "n_samples": int(n),
            "random_state": s,
            "include_targets": self.include_targets,
            "exclude_targets": not self.include_targets,
        }
        attempts = (
            lambda: self.fn(
                self.df, int(n), s, include_targets=self.include_targets
            ),
            lambda: self.fn(
                self.df, int(n), s, exclude_targets=not self.include_targets
            ),
            lambda: self.fn(self.df, **_filter_kwargs(self.fn, kw_full)),
            lambda: self.fn(self.df, int(n), s),
            lambda: self.fn(self.df, int(n)),
        )
        last: Optional[BaseException] = None
        for attempt in attempts:
            try:
                out = attempt()
            except TypeError as exc:
                last = exc
                continue
            if isinstance(out, pd.DataFrame):
                return out
            last = TypeError(f"helper returned {type(out).__name__}")
        if last is not None:
            raise last
        raise TypeError("synthesize helper produced no DataFrame")


class _BootstrapMember:
    """Row resampling fallback when every parametric family fails."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True).copy()

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        n = int(n)
        if n <= 0 or self._df.empty:
            return self._df.iloc[0:0].copy()
        rng = np.random.default_rng(None if seed is None else int(seed))
        idx = rng.integers(0, len(self._df), size=n)
        return self._df.iloc[idx].reset_index(drop=True)


def _allocate_counts(n: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation so integer row counts sum to ``n``."""
    names = [k for k, w in shares.items() if w > 0]
    if not names:
        return {}
    weights = np.asarray([float(shares[k]) for k in names], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return {k: 0 for k in names}
    weights = weights / total
    raw = weights * int(n)
    counts = np.floor(raw).astype(int)
    leftover = int(n) - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for i in range(max(0, leftover)):
        counts[int(order[i])] += 1
    return {names[i]: int(counts[i]) for i in range(len(names))}


def _restore_dtypes(frame: pd.DataFrame, dtypes: dict[str, Any]) -> pd.DataFrame:
    out = frame
    for col, dtype in dtypes.items():
        if col not in out.columns:
            continue
        try:
            if isinstance(dtype, pd.CategoricalDtype):
                out[col] = pd.Categorical(
                    out[col], categories=list(dtype.categories), ordered=bool(dtype.ordered)
                )
            elif pd.api.types.is_bool_dtype(dtype):
                out[col] = out[col].astype(dtype, errors="ignore")
            elif pd.api.types.is_integer_dtype(dtype):
                num = pd.to_numeric(out[col], errors="coerce")
                if num.isna().any() or pd.api.types.is_extension_array_dtype(dtype):
                    out[col] = num.round().astype("Int64")
                else:
                    out[col] = num.round().astype(dtype)
            else:
                out[col] = out[col].astype(dtype, errors="ignore")
        except (TypeError, ValueError):
            pass
    return out


def _align_frame(
    frame: pd.DataFrame,
    columns: list[str],
    train: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = frame.copy()
    n = len(out)
    for col in columns:
        if col in out.columns:
            continue
        if col in train.columns and len(train) and n:
            idx = rng.integers(0, len(train), size=n)
            out[col] = train.iloc[idx][col].to_numpy()
        else:
            out[col] = np.nan
    extra = [c for c in out.columns if c not in columns]
    if extra:
        out = out.drop(columns=extra)
    if columns:
        out = out.loc[:, columns]
    return out.reset_index(drop=True)


def _pad_or_trim(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    n = int(n)
    if n <= 0:
        return frame.iloc[0:0].copy()
    if len(frame) == n:
        return frame.reset_index(drop=True)
    if len(frame) == 0:
        return frame
    if len(frame) > n:
        return frame.iloc[:n].reset_index(drop=True)
    extra = rng.integers(0, len(frame), size=n - len(frame))
    return pd.concat([frame, frame.iloc[extra]], ignore_index=True)


class EnsembleSynthesizer:
    """Fit several synthesizers; sample a renormalized concatenation of each."""

    def __init__(self) -> None:
        self.include_targets_: bool = True
        self.columns_: list[str] = []
        self.dtypes_: dict[str, Any] = {}
        self.fitted_names_: list[str] = []
        self.shares_: dict[str, float] = {}
        self.members_: list[tuple[str, Any]] = []
        self._train: Optional[pd.DataFrame] = None
        self._bootstrap: Optional[_BootstrapMember] = None
        self._fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        include_targets: bool = True,
        seed: int = 0,
    ) -> "EnsembleSynthesizer":
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df must contain at least one row")

        include_targets = bool(include_targets)
        seed = int(seed)
        work = df.reset_index(drop=True).copy()
        out_cols = _output_columns(list(work.columns), include_targets)
        if not include_targets:
            dropped = [c for c in TARGET_COLS if c in work.columns]
            if dropped:
                work = work.drop(columns=dropped)
        if not out_cols:
            raise ValueError("no columns left to synthesize after dropping targets")

        work = work.loc[:, out_cols]
        self.include_targets_ = include_targets
        self.columns_ = list(out_cols)
        self.dtypes_ = {c: work[c].dtype for c in out_cols}
        self.fitted_names_ = []
        self.shares_ = {}
        self.members_ = []
        self._train = work
        self._bootstrap = _BootstrapMember(work)
        self._fitted = False

        families = _discover_families()
        for i, (name, share) in enumerate(PREFERRED_SHARES.items()):
            picked = families.get(name)
            if picked is None:
                continue
            member_seed = seed + _SEED_OFFSETS.get(name, 13 + i)
            try:
                if isinstance(picked, type):
                    inst = _construct(picked, member_seed)
                    fitted = _call_fit(inst, work, include_targets, member_seed)
                    if not hasattr(fitted, "sample"):
                        raise TypeError(f"{name} fitted object has no sample()")
                    member: Any = fitted
                else:
                    member = _FnMember(picked, work, include_targets)
                    # Probe the helper so a broken function is skipped at fit time.
                    probe_n = min(8, max(1, len(work)))
                    probe = member.sample(probe_n, seed=member_seed)
                    if not isinstance(probe, pd.DataFrame) or probe.empty:
                        raise TypeError(f"{name} helper returned an unusable frame")
            except Exception as exc:  # noqa: BLE001 — skip a broken family
                warnings.warn(
                    f"EnsembleSynthesizer skipped '{name}': {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            self.members_.append((name, member))
            self.fitted_names_.append(name)
            self.shares_[name] = float(share)

        if self.shares_:
            total = float(sum(self.shares_.values()))
            if total > 0:
                self.shares_ = {k: v / total for k, v in self.shares_.items()}
        else:
            warnings.warn(
                "EnsembleSynthesizer: every family failed; using row bootstrap.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.members_.append(("bootstrap", self._bootstrap))
            self.fitted_names_.append("bootstrap")
            self.shares_ = {"bootstrap": 1.0}

        self._fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self._fitted or self._train is None:
            raise RuntimeError("EnsembleSynthesizer.fit() must be called first")
        n = int(n)
        columns = list(self.columns_)
        if n <= 0:
            return pd.DataFrame(
                {c: pd.Series(dtype=self.dtypes_.get(c, object)) for c in columns}
            )

        rng = np.random.default_rng(None if seed is None else int(seed))
        counts = _allocate_counts(n, self.shares_)
        train = self._train
        frames: list[pd.DataFrame] = []
        leftover = 0

        for name, member in self.members_:
            k = int(counts.get(name, 0)) + leftover
            leftover = 0
            if k <= 0:
                continue
            child_seed = (
                None if seed is None else int(seed) + _SEED_OFFSETS.get(name, 3)
            )
            try:
                part = _call_sample(member, k, child_seed)
                if not isinstance(part, pd.DataFrame):
                    raise TypeError(f"{name}.sample returned {type(part).__name__}")
                part = _align_frame(part, columns, train, rng)
                part = _pad_or_trim(part, k, rng)
            except Exception as exc:  # noqa: BLE001 — keep the mix alive
                warnings.warn(
                    f"EnsembleSynthesizer member '{name}' failed at sample "
                    f"({type(exc).__name__}: {exc}); filling with bootstrap rows.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if self._bootstrap is None:
                    leftover += k
                    continue
                part = _align_frame(
                    self._bootstrap.sample(k, seed=child_seed), columns, train, rng
                )
                part = _pad_or_trim(part, k, rng)
            if len(part) < k:
                leftover += k - len(part)
            frames.append(part)

        if leftover > 0 and self._bootstrap is not None:
            extra = _align_frame(
                self._bootstrap.sample(leftover, seed=seed), columns, train, rng
            )
            frames.append(_pad_or_trim(extra, leftover, rng))

        if not frames:
            if self._bootstrap is None:
                raise RuntimeError("ensemble has no members to sample from")
            out = _align_frame(
                self._bootstrap.sample(n, seed=seed), columns, train, rng
            )
        else:
            out = pd.concat(frames, ignore_index=True, sort=False)
            out = _align_frame(out, columns, train, rng)
            out = _pad_or_trim(out, n, rng)

        if len(out) > 1:
            out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
        out = _restore_dtypes(out, self.dtypes_)
        return out.reset_index(drop=True)


def synthesize_ensemble(
    df: pd.DataFrame,
    n: int | None = None,
    seed: int = 0,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Fit :class:`EnsembleSynthesizer` on ``df`` and draw ``n`` rows.

    ``n`` defaults to ``len(df)``.
    """
    if n is None:
        n = int(len(df))
    return (
        EnsembleSynthesizer()
        .fit(df, include_targets=include_targets, seed=int(seed))
        .sample(int(n), seed=int(seed))
    )


# Alias used by the multi-dataset runner / older plan name.
MixtureCartEnsemble = EnsembleSynthesizer


def _self_test() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    n = 80
    region = rng.choice(["north", "south", "west"], size=n, p=[0.45, 0.35, 0.20])
    segment = rng.choice(["growth", "value"], size=n)
    df = pd.DataFrame(
        {
            "age": rng.normal(42.0, 10.0, n).clip(18.0, 80.0),
            "income": rng.lognormal(10.2, 0.35, n),
            "region": pd.Series(region, dtype="string"),
            "segment": pd.Categorical(segment, categories=["growth", "value"]),
            "flag": rng.integers(0, 2, n, dtype=np.int64),
            "y": rng.normal(3.0, 1.0, n) + 0.4 * (segment == "growth"),
            "y_class": (rng.random(n) < 0.4).astype(np.int64),
        }
    )

    ens = EnsembleSynthesizer()
    ens.fit(df, include_targets=True, seed=0)
    out = ens.sample(50, seed=1)

    if not ens.fitted_names_:
        raise AssertionError("no ensemble members fitted")
    if abs(sum(ens.shares_.values()) - 1.0) > 1e-9:
        raise AssertionError(f"shares do not sum to 1: {ens.shares_}")
    if len(out) != 50:
        raise AssertionError(f"expected 50 rows, got {len(out)}")
    if list(out.columns) != list(df.columns):
        raise AssertionError(f"column mismatch: {list(out.columns)} vs {list(df.columns)}")

    no_tgt = EnsembleSynthesizer().fit(df, include_targets=False, seed=1)
    out_x = no_tgt.sample(24, seed=2)
    want_x = [c for c in df.columns if c not in TARGET_COLS]
    if list(out_x.columns) != want_x:
        raise AssertionError(f"include_targets=False columns {list(out_x.columns)}")
    if any(_is_target_col(c) for c in out_x.columns):
        raise AssertionError("targets leaked when include_targets=False")
    if len(out_x) != 24:
        raise AssertionError("include_targets=False row count")

    # Schema-agnostic table: no conventional target names.
    other = pd.DataFrame(
        {
            "height": rng.normal(170.0, 8.0, 60),
            "city": rng.choice(["a", "b", "c"], 60),
            "score": rng.normal(size=60),
        }
    )
    ens_o = EnsembleSynthesizer().fit(other, include_targets=True, seed=2)
    out_o = ens_o.sample(20, seed=3)
    if list(out_o.columns) != list(other.columns) or len(out_o) != 20:
        raise AssertionError("schema-agnostic path failed")

    helper = synthesize_ensemble(df.iloc[:50].reset_index(drop=True), n=16, seed=4)
    if len(helper) != 16 or list(helper.columns) != list(df.columns):
        raise AssertionError("synthesize_ensemble helper failed")

    empty = ens.sample(0, seed=5)
    if len(empty) != 0 or list(empty.columns) != list(df.columns):
        raise AssertionError("n=0 sample failed")

    return {
        "ok": True,
        "fitted": list(ens.fitted_names_),
        "shares": dict(ens.shares_),
        "n_out": int(len(out)),
        "columns": list(out.columns),
        "fitted_no_targets": list(no_tgt.fitted_names_),
        "fitted_other": list(ens_o.fitted_names_),
    }


__all__ = [
    "EnsembleSynthesizer",
    "MixtureCartEnsemble",
    "PREFERRED_SHARES",
    "synthesize_ensemble",
]


if __name__ == "__main__":
    result = _self_test()
    print("fitted:", result["fitted"])
    print("shares:", {k: round(v, 4) for k, v in result["shares"].items()})
    print("n_out:", result["n_out"], "columns:", result["columns"])
    print("fitted_no_targets:", result["fitted_no_targets"])
    print("fitted_other:", result["fitted_other"])
    if not result["ok"]:
        raise SystemExit("ensemble synthesizer self-test failed")
    print("self-test ok")
