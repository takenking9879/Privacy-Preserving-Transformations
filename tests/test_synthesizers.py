"""Each available synthesizer returns the same columns and n without crashing.

Discovers ``*Synthesizer`` classes (fit/sample) and ``synthesize*`` helpers
under ``src.synthesizers``. Skips if none can be imported. Fast: n ≈ 80.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


_SYNTH_MODULES = (
    "src.synthesizers.cart_sequential",
    "src.synthesizers.sequential_cart",
    "src.synthesizers.gaussian_copula",
    "src.synthesizers.copula",
    "src.synthesizers.conditional_mixture",
    "src.synthesizers.mixture",
    "src.synthesizers.negative",
    "src.synthesizers.smote_baseline",
    "src.synthesizers.ensemble",
    "src.synthesizers.hybrid",
    "src.synthesizers.base",
)
_REGISTRY_MODULES = ("src.synthesizers.registry", "src.synthesizers")


def _skip(reason: str) -> None:
    if pytest is not None:
        pytest.skip(reason)
    raise unittest.SkipTest(reason)


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except ImportError:
        return None


def _tiny_real(n: int = 80, seed: int = 7) -> pd.DataFrame:
    dgp = _try_import("src.dgp")
    if dgp is not None:
        fn = getattr(dgp, "generate_original", None) or getattr(dgp, "generate", None)
        if callable(fn):
            return fn(n=n, seed=seed)
    _skip("src.dgp is not available and no fallback table was requested")


def _family_key(name: str) -> str:
    lowered = name.lower()
    for token in ("copula", "cart", "mixture", "hybrid", "smote", "shuffle", "jitter", "ensemble"):
        if token in lowered:
            return token
    return name


def _collect_from_modules(mod_names) -> dict[str, object]:
    found: dict[str, object] = {}
    for mod_name in mod_names:
        mod = _try_import(mod_name)
        if mod is None:
            continue
        for attr in ("AVAILABLE_SYNTHESIZERS", "SYNTHESIZERS"):
            registry = getattr(mod, attr, None)
            if isinstance(registry, dict):
                for key, obj in registry.items():
                    found[str(key)] = obj
            elif isinstance(registry, (list, tuple)):
                for obj in registry:
                    found[getattr(obj, "__name__", str(obj))] = obj
        for name, obj in vars(mod).items():
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) and name.endswith("Synthesizer"):
                if name in {"BaseSynthesizer", "Synthesizer"}:
                    continue
                found[f"{mod_name}.{name}"] = obj
            elif inspect.isfunction(obj) and name.startswith("synthesize"):
                found[f"{mod_name}.{name}"] = obj
    return found


def _discover_synthesizers():
    """One object per synthesizer family; prefer native classes over adapters."""
    native = _collect_from_modules(_SYNTH_MODULES)
    if native:
        chosen: dict[str, object] = {}
        for name, obj in native.items():
            # Prefer classes (fit/sample) over synthesize_* helpers.
            key = _family_key(name)
            prev = chosen.get(key)
            if prev is None or (inspect.isclass(obj) and not inspect.isclass(prev)):
                chosen[key] = obj
        return chosen
    return _collect_from_modules(_REGISTRY_MODULES)


def _run_synthesizer(obj, real: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if inspect.isclass(obj):
        try:
            inst = obj()
        except TypeError:
            _skip(f"{obj!r} is not instantiable (abstract base?)")
        if hasattr(inst, "fit") and hasattr(inst, "sample"):
            inst.fit(real)
            return inst.sample(n, seed=seed)
        if hasattr(inst, "fit") and hasattr(inst, "generate"):
            inst.fit(real)
            return inst.generate(n)
        raise AssertionError(f"{obj!r} has no fit/sample interface")
    if callable(obj):
        try:
            return obj(real, n=n, seed=seed)
        except TypeError:
            try:
                return obj(real, n)
            except TypeError:
                return obj(real)
    raise AssertionError(f"do not know how to run synthesizer {obj!r}")


def test_available_synthesizers_same_schema_and_n():
    """Every discovered synthesizer emits n rows and the input columns."""
    if pytest is not None:
        pytest.importorskip("src.synthesizers")
    else:
        if _try_import("src.synthesizers") is None and not any(
            _try_import(name) for name in _SYNTH_MODULES[1:]
        ):
            raise unittest.SkipTest("src.synthesizers is not available")

    available = _discover_synthesizers()
    if not available:
        _skip("no synthesizer classes or synthesize_* helpers found")

    n = 80
    real = _tiny_real(n=n, seed=7)
    expected_cols = list(real.columns)
    failures = []
    ran = 0
    for name, obj in sorted(available.items(), key=lambda kv: kv[0]):
        try:
            synth = _run_synthesizer(obj, real, n=n, seed=0)
        except unittest.SkipTest:
            continue
        except Exception as exc:  # noqa: BLE001 — contract is "no crash"
            failures.append(f"{name} crashed: {type(exc).__name__}: {exc}")
            continue
        ran += 1
        if not isinstance(synth, pd.DataFrame):
            failures.append(f"{name} returned {type(synth).__name__}, not DataFrame")
            continue
        if len(synth) != n:
            failures.append(f"{name} returned n={len(synth)}, expected {n}")
        got_cols = set(synth.columns)
        want_cols = set(expected_cols)
        if got_cols != want_cols:
            failures.append(
                f"{name} columns {sorted(got_cols)} != {sorted(want_cols)}"
            )
    assert ran >= 1, "no synthesizer actually produced a table"
    assert not failures, "synthesizer contract failures:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and inspect.isfunction(_fn):
            suite.addTest(unittest.FunctionTestCase(_fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
