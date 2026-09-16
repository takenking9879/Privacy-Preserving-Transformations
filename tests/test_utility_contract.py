"""Utility contract: evaluate_model_utility(real, real) has a tiny TSTR R² gap.

Skips if ``evaluate_model_utility`` is not importable. Fast: n ≈ 120.
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


def _load_evaluate_model_utility():
    for mod_name in (
        "src.metrics.utility",
        "src.metrics",
        "src.metrics.tstr",
        "src.utility",
        "src.evaluate",
        "src.metrics.statistical",
    ):
        mod = _try_import(mod_name)
        if mod is None:
            continue
        fn = getattr(mod, "evaluate_model_utility", None)
        if callable(fn):
            return fn
    _skip("evaluate_model_utility is not available")


def _lookup(result, *keys):
    if isinstance(result, dict):
        for key in keys:
            if key in result and result[key] is not None:
                return result[key]
        for value in result.values():
            if isinstance(value, dict):
                found = _lookup(value, *keys)
                if found is not None:
                    return found
        return None
    for key in keys:
        if hasattr(result, key):
            val = getattr(result, key)
            if val is not None:
                return val
    return None


def _real_frame(n: int = 120, seed: int = 7) -> pd.DataFrame:
    dgp = _try_import("src.dgp")
    if dgp is not None:
        fn = getattr(dgp, "generate_original", None) or getattr(dgp, "generate", None)
        if callable(fn):
            return fn(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    risk = rng.normal(50.0, 10.0, size=n)
    y = 0.7 * risk + rng.normal(0.0, 5.0, size=n)
    return pd.DataFrame(
        {
            "risk_score": risk,
            "x1": rng.normal(size=n),
            "y": y,
            "y_class": (y > np.median(y)).astype(np.int64),
        }
    )


def test_evaluate_model_utility_real_vs_real_small_tstr_gap_r2():
    """Identity utility: |tstr_gap_r2| < 0.15 when synth is a copy of real."""
    fn = _load_evaluate_model_utility()
    real = _real_frame(120, seed=7)
    result = fn(real, real.copy())
    gap = _lookup(result, "tstr_gap_r2")
    if gap is None:
        raise AssertionError(
            "evaluate_model_utility(real, real) must expose tstr_gap_r2; "
            f"got {result!r}"
        )
    gap = float(gap)
    assert abs(gap) < 0.15, (
        f"evaluate_model_utility(real, real) tstr_gap_r2={gap:.4f}, "
        "expected abs(tstr_gap_r2) < 0.15"
    )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and inspect.isfunction(_fn):
            suite.addTest(unittest.FunctionTestCase(_fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
