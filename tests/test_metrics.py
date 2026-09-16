"""Fidelity / utility metric contracts.

Identical frames must score high; a full independence shuffle of ``y`` and
the features must score lower; TSTR gap is small when synth == real.

Skips if the metrics package is missing. Fast: n ≈ 120.
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


def _load_fidelity_fn():
    candidates = (
        "src.metrics.statistical",
        "src.metrics",
        "src.metrics.fidelity",
    )
    names = (
        "compute_statistical_fidelity",
        "evaluate_fidelity",
        "statistical_fidelity",
        "fidelity_score",
        "compute_fidelity",
    )
    for mod_name in candidates:
        mod = _try_import(mod_name)
        if mod is None:
            continue
        for fn_name in names:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
    _skip("statistical fidelity function is not available")


def _load_utility_fn():
    candidates = (
        "src.metrics.utility",
        "src.metrics",
        "src.metrics.tstr",
        "src.utility",
        "src.evaluate",
        "src.metrics.statistical",
    )
    for mod_name in candidates:
        mod = _try_import(mod_name)
        if mod is None:
            continue
        fn = getattr(mod, "evaluate_model_utility", None)
        if callable(fn):
            return fn
    _skip("evaluate_model_utility is not available")


def _fidelity_score(result) -> float:
    if isinstance(result, (int, float, np.floating)):
        return float(result)
    if isinstance(result, dict):
        for key in ("fidelity_score", "fidelity", "score", "mean_1m_ks_or_tv"):
            if key in result and result[key] is not None:
                return float(result[key])
    for key in ("fidelity_score", "fidelity", "score"):
        if hasattr(result, key):
            val = getattr(result, key)
            if val is not None:
                return float(val)
    raise AssertionError(f"could not read a fidelity score from {result!r}")


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


def _real_frame(n: int = 120, seed: int = 1) -> pd.DataFrame:
    dgp = _try_import("src.dgp")
    if dgp is not None:
        fn = getattr(dgp, "generate_original", None) or getattr(dgp, "generate", None)
        if callable(fn):
            return fn(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    risk = 50.0 + 10.0 * x1 + rng.normal(0.0, 3.0, size=n)
    y = 0.8 * risk + rng.normal(0.0, 4.0, size=n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": rng.normal(size=n),
            "risk_score": risk,
            "y": y,
            "y_class": (y > np.median(y)).astype(np.int64),
        }
    )


def _independence_shuffle(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Destroy joints: independently permute every column, including y."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for col in out.columns:
        values = out[col].to_numpy(copy=True)
        rng.shuffle(values)
        out[col] = values
    return out


def test_identical_frames_high_fidelity():
    """A table compared to itself must report near-perfect fidelity."""
    fn = _load_fidelity_fn()
    real = _real_frame(120, seed=2)
    score = _fidelity_score(fn(real, real.copy()))
    assert score >= 0.90, f"identical-frame fidelity_score={score:.4f} < 0.90"


def test_shuffled_y_and_independent_features_lower_fidelity():
    """Independence-shuffled y and features must score below the identity."""
    fn = _load_fidelity_fn()
    real = _real_frame(120, seed=2)
    ident = _fidelity_score(fn(real, real.copy()))
    broken = _independence_shuffle(real, seed=99)
    lower = _fidelity_score(fn(real, broken))
    assert lower < ident, (
        f"shuffled/independent fidelity ({lower:.4f}) was not lower "
        f"than identical ({ident:.4f})"
    )
    assert lower < 0.80, (
        f"independence shuffle still looks high-fidelity ({lower:.4f}); "
        "dependence must dominate the score"
    )


def test_tstr_gap_small_when_synth_equals_real():
    """Train-on-synth/test-on-real gap is small when the two tables match."""
    fn = _load_utility_fn()
    real = _real_frame(120, seed=4)
    result = fn(real, real.copy())
    gap = _lookup(result, "tstr_gap", "tstr_gap_r2", "gap_r2", "r2_gap")
    if gap is None:
        raise AssertionError(
            f"evaluate_model_utility did not return tstr_gap/tstr_gap_r2: {result!r}"
        )
    gap = float(gap)
    assert abs(gap) < 0.15, f"tstr_gap={gap:.4f} not small when synth==real"


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and inspect.isfunction(_fn):
            suite.addTest(unittest.FunctionTestCase(_fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
