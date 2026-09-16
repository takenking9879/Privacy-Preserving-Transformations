"""Schema-agnostic general synthesizer smoke tests.

``AutoSynthesizer`` or ``ForestSequentialSynthesizer`` or ``synthesize_auto``
must work on two different schemas (credit-like when ``src.dgp`` exists,
plus a tiny generic ``foo/bar/grp/y/y_class`` frame with no region/segment).
Output has the same columns and the same n.

Skips if none of those APIs import. Fast. pytest + unittest main fallback.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def _skip(reason: str) -> None:
    if pytest is not None:
        pytest.skip(reason)
    raise unittest.SkipTest(reason)


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except ImportError:
        return None


def _load_general_api():
    """Return (kind, obj) for the first available general synthesizer API."""
    forest_mod = _try_import("src.synthesizers.forest_sequential")
    if forest_mod is not None:
        cls = getattr(forest_mod, "ForestSequentialSynthesizer", None)
        if inspect.isclass(cls):
            return "class", cls
        fn = getattr(forest_mod, "synthesize_forest", None)
        if callable(fn):
            return "fn", fn

    auto_mod = _try_import("src.synthesizers.auto")
    if auto_mod is not None:
        cls = getattr(auto_mod, "AutoSynthesizer", None)
        if inspect.isclass(cls):
            return "class", cls
        fn = getattr(auto_mod, "synthesize_auto", None)
        if callable(fn):
            return "fn", fn

    return None, None


def _credit_like(n: int, seed: int) -> pd.DataFrame:
    dgp = _try_import("src.dgp")
    if dgp is not None:
        fn = getattr(dgp, "generate_original", None) or getattr(dgp, "generate", None)
        if callable(fn):
            try:
                return fn(n=n, seed=seed)
            except TypeError:
                return fn(n, seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(42.0, 10.0, n),
            "income": rng.lognormal(10.0, 0.4, n),
            "region": rng.choice(["Northeast", "South", "West"], n),
            "segment": rng.choice(["Mass", "Affluent"], n),
            "y": rng.normal(size=n),
            "y_class": rng.integers(0, 2, n),
        }
    )


def _generic_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "foo": rng.normal(0.0, 1.0, n),
            "bar": rng.uniform(-1.0, 1.0, n),
            "grp": rng.choice(["A", "B", "C"], n),
            "y": rng.normal(size=n),
            "y_class": rng.integers(0, 2, n),
        }
    )


def _fit_class(cls, df: pd.DataFrame, seed: int):
    try:
        inst = cls()
    except TypeError:
        try:
            inst = cls(seed=seed)
        except TypeError:
            inst = cls(random_state=seed)
    fit = getattr(inst, "fit", None)
    if not callable(fit):
        raise AssertionError(f"{cls!r} has no fit()")
    attempts = (
        dict(include_targets=True, n_estimators=8, max_depth=4, min_samples_leaf=8, seed=seed),
        dict(include_targets=True, n_estimators=8, max_depth=4, min_samples_leaf=8),
        dict(include_targets=True, seed=seed),
        dict(include_targets=True),
        dict(seed=seed),
        {},
    )
    last_err: Exception | None = None
    for kw in attempts:
        try:
            out = fit(df, **kw)
            return out if out is not None and hasattr(out, "sample") else inst
        except TypeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return inst


def _sample(inst, n: int, seed: int) -> pd.DataFrame:
    meth = getattr(inst, "sample", None) or getattr(inst, "generate", None)
    if not callable(meth):
        raise AssertionError(f"{type(inst).__name__} has no sample/generate")
    try:
        return meth(n, seed=seed)
    except TypeError:
        try:
            return meth(n, random_state=seed)
        except TypeError:
            try:
                return meth(n)
            except TypeError:
                return meth(n_samples=n, seed=seed)


def _call_fn(fn, df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    try:
        return fn(df, n=n, seed=seed)
    except TypeError:
        try:
            return fn(df, n, seed)
        except TypeError:
            try:
                return fn(df, n)
            except TypeError:
                return fn(df, n=n, seed=seed, include_targets=True)


def _synthesize(kind: str, obj, df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if kind == "class":
        return _sample(_fit_class(obj, df, seed), n, seed)
    if kind == "fn":
        return _call_fn(obj, df, n, seed)
    raise AssertionError("no general synthesizer API loaded")


def _assert_same_schema_and_n(label: str, real: pd.DataFrame, synth: pd.DataFrame, n: int) -> None:
    assert isinstance(synth, pd.DataFrame), f"{label} returned {type(synth).__name__}"
    assert len(synth) == n, f"{label} n={len(synth)}, expected {n}"
    got = set(synth.columns)
    want = set(real.columns)
    assert got == want, f"{label} columns {sorted(got)} != {sorted(want)}"


def test_general_synth_two_schemas_same_columns_and_n():
    """Credit-like + generic foo/bar/grp frames keep columns and n; no region/segment required."""
    kind, obj = _load_general_api()
    if obj is None:
        _skip(
            "AutoSynthesizer / ForestSequentialSynthesizer / synthesize_auto "
            "are not importable"
        )

    n = 80
    schemas = (
        ("credit-like", _credit_like(n, seed=7)),
        ("generic", _generic_frame(n, seed=3)),
    )
    generic = schemas[1][1]
    assert "region" not in generic.columns and "segment" not in generic.columns

    for label, real in schemas:
        assert list(real.columns)
        synth = _synthesize(kind, obj, real, n=n, seed=0)
        _assert_same_schema_and_n(label, real, synth, n)

    # Explicit schema-agnostic check: generic frame never had region/segment
    # and the synthesizer still returned that schema (not a credit template).
    generic_out = _synthesize(kind, obj, generic, n=n, seed=1)
    _assert_same_schema_and_n("generic-rerun", generic, generic_out, n)
    assert "region" not in generic_out.columns
    assert "segment" not in generic_out.columns
    for col in ("foo", "bar", "grp", "y", "y_class"):
        assert col in generic_out.columns, f"generic output missing {col}"


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and inspect.isfunction(_fn):
            suite.addTest(unittest.FunctionTestCase(_fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
