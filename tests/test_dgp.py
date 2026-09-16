"""DGP contract: shape, schema, varying targets, and y–risk_score dependence.

Runnable with pytest (preferred) or ``python tests/test_dgp.py`` via unittest.
Skips if ``src.dgp`` is missing.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

if pytest is not None:
    dgp = pytest.importorskip("src.dgp")
else:  # pragma: no cover
    try:
        import src.dgp as dgp
    except ImportError:
        dgp = None


_FALLBACK_FEATURES = (
    "age",
    "income",
    "tenure_months",
    "region",
    "segment",
    "risk_score",
    "num_products",
    "is_premium",
    "usage",
    "engagement",
    "complaint_count",
    "credit_util",
)
_CLASS_ALIASES = ("y_class", "class", "cls")


def _skip_if_missing() -> None:
    if dgp is None:
        raise unittest.SkipTest("src.dgp is not available")


def _generate(n: int, seed: int = 7):
    _skip_if_missing()
    fn = getattr(dgp, "generate_original", None) or getattr(dgp, "generate", None)
    if fn is None:
        raise unittest.SkipTest("src.dgp has no generate_original/generate")
    return fn(n=n, seed=seed)


def _required_feature_cols() -> list[str]:
    cols = getattr(dgp, "FEATURE_COLS", None)
    if cols:
        return list(cols)
    return list(_FALLBACK_FEATURES)


def _class_col(frame) -> str:
    for name in _CLASS_ALIASES:
        if name in frame.columns:
            return name
    raise AssertionError(
        f"binary class column not found; expected one of {_CLASS_ALIASES}, "
        f"got {list(frame.columns)}"
    )


def test_dgp_shape():
    """Requested n is honoured and the table has features + both targets."""
    n = 120
    frame = _generate(n, seed=3)
    assert frame.shape[0] == n
    expected_min = len(_required_feature_cols()) + 2  # + y + class
    assert frame.shape[1] >= expected_min
    assert frame.shape[1] == len(frame.columns)


def test_dgp_required_columns():
    """Schema columns from FEATURE_COLS plus y, risk_score, and a 0/1 class."""
    frame = _generate(100, seed=7)
    required = set(_required_feature_cols())
    required.add("y")
    required.add("risk_score")
    missing = required - set(frame.columns)
    assert not missing, f"DGP missing required columns: {sorted(missing)}"
    _class_col(frame)


def test_dgp_y_varies():
    """Continuous target is non-degenerate."""
    frame = _generate(100, seed=11)
    assert "y" in frame.columns
    y = np.asarray(frame["y"], dtype=float)
    finite = y[np.isfinite(y)]
    assert finite.size == y.size, "y must be fully observed"
    assert np.unique(finite).size > 1
    assert float(np.std(finite)) > 0.0


def test_dgp_class_is_binary_01():
    """Classification target is {0, 1} and both labels appear."""
    frame = _generate(120, seed=11)
    col = _class_col(frame)
    values = np.asarray(frame[col])
    unique = set(int(v) for v in np.unique(values) if v == v)
    assert unique.issubset({0, 1}), f"{col} values must be 0/1, got {unique}"
    assert unique == {0, 1}, f"{col} must contain both classes, got {unique}"


def test_dgp_risk_score_correlates_with_y():
    """On n=400 the DGP keeps a usable |corr(y, risk_score)|."""
    frame = _generate(400, seed=7)
    assert "risk_score" in frame.columns and "y" in frame.columns
    y = np.asarray(frame["y"], dtype=float)
    risk = np.asarray(frame["risk_score"], dtype=float)
    mask = np.isfinite(y) & np.isfinite(risk)
    corr = float(np.corrcoef(y[mask], risk[mask])[0, 1])
    assert np.isfinite(corr)
    assert abs(corr) > 0.15, f"|corr(y, risk_score)|={corr:.4f} <= 0.15"


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and inspect.isfunction(_fn):
            suite.addTest(unittest.FunctionTestCase(_fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
