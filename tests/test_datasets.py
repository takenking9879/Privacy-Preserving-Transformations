"""Every discovered DatasetSpec generates a usable table (n=80).

Checks: unique names; ``y`` present and non-degenerate; ``y_class`` is
{0, 1} when present. Fast. Skips if the datasets package is missing.

Runnable with pytest or ``python tests/test_datasets.py`` via unittest.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
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


def _skip(reason: str) -> None:
    if pytest is not None:
        pytest.skip(reason)
    raise unittest.SkipTest(reason)


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except ImportError:
        return None


def _dataset_spec_cls():
    for mod_name in ("src.datasets.base", "src.datasets"):
        mod = _try_import(mod_name)
        if mod is None:
            continue
        cls = getattr(mod, "DatasetSpec", None)
        if inspect.isclass(cls):
            return cls
    return None


def _generate_from_spec(spec, n: int, seed: int):
    gen = getattr(spec, "generate", None)
    if callable(gen):
        try:
            return gen(n, seed)
        except TypeError:
            try:
                return gen(n=n, seed=seed)
            except TypeError:
                return gen(n)
    sample = getattr(spec, "sample", None)
    if callable(sample):
        try:
            return sample(n=n, seed=seed)
        except TypeError:
            return sample(n)
    raise AssertionError(f"{getattr(spec, 'name', spec)!r} has no generate/sample")


def _add_spec(bucket: list, seen: set, spec, spec_cls) -> None:
    if spec is None or not isinstance(spec, spec_cls):
        return
    name = getattr(spec, "name", None)
    key = name if name is not None else id(spec)
    if key in seen:
        return
    seen.add(key)
    bucket.append(spec)


def _discover_specs() -> list:
    spec_cls = _dataset_spec_cls()
    if spec_cls is None:
        _skip("src.datasets.DatasetSpec is not available")

    specs: list = []
    seen: set = set()

    for mod_name in ("src.datasets.registry", "src.datasets"):
        mod = _try_import(mod_name)
        if mod is None:
            continue
        available = getattr(mod, "available", None)
        getter = getattr(mod, "get", None)
        if callable(available):
            try:
                items = list(available())
            except Exception:  # noqa: BLE001 — discovery must not crash the suite
                items = []
            for item in items:
                if isinstance(item, spec_cls):
                    _add_spec(specs, seen, item, spec_cls)
                elif isinstance(item, str) and callable(getter):
                    try:
                        _add_spec(specs, seen, getter(item), spec_cls)
                    except Exception:  # noqa: BLE001
                        continue
        mapping = getattr(mod, "DATASETS", None)
        if isinstance(mapping, dict):
            for value in mapping.values():
                _add_spec(specs, seen, value, spec_cls)
            for key in mapping:
                if callable(getter):
                    try:
                        _add_spec(specs, seen, getter(key), spec_cls)
                    except Exception:  # noqa: BLE001
                        continue

    pkg = _try_import("src.datasets")
    if pkg is not None and getattr(pkg, "__path__", None) is not None:
        for _finder, name, _ispkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            if name.rsplit(".", 1)[-1].startswith("_"):
                continue
            try:
                sub = importlib.import_module(name)
            except Exception:  # noqa: BLE001
                continue
            _add_spec(specs, seen, getattr(sub, "SPEC", None), spec_cls)

    dgp = _try_import("src.dgp")
    if dgp is not None:
        _add_spec(specs, seen, getattr(dgp, "SPEC", None), spec_cls)

    return specs


def test_dataset_spec_names_unique():
    """Discovered DatasetSpec.name values are unique and non-empty."""
    specs = _discover_specs()
    if not specs:
        _skip("no DatasetSpec objects discovered")
    names = [str(spec.name) for spec in specs]
    assert all(name.strip() for name in names), f"blank DatasetSpec names: {names}"
    dups = sorted({n for n in names if names.count(n) > 1})
    assert len(names) == len(set(names)), f"duplicate DatasetSpec names: {dups}"


def test_every_dataset_spec_generate_n80():
    """Each spec: generate(n=80) has y, y varies, y_class is 0/1 if present."""
    specs = _discover_specs()
    if not specs:
        _skip("no DatasetSpec objects discovered")

    n = 80
    failures = []
    for spec in specs:
        name = getattr(spec, "name", "<unnamed>")
        try:
            frame = _generate_from_spec(spec, n=n, seed=0)
        except unittest.SkipTest:
            raise
        except Exception as exc:  # noqa: BLE001 — contract is "generate works"
            failures.append(f"{name} generate crashed: {type(exc).__name__}: {exc}")
            continue
        if frame is None or not hasattr(frame, "columns"):
            failures.append(f"{name} generate returned {type(frame).__name__}")
            continue
        if len(frame) != n:
            failures.append(f"{name} n={len(frame)}, expected {n}")
        if "y" not in frame.columns:
            failures.append(f"{name} missing column 'y'; got {list(frame.columns)}")
            continue
        y = np.asarray(frame["y"], dtype=float)
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            failures.append(f"{name} y is all-non-finite")
            continue
        if np.unique(finite).size <= 1 and float(np.nanstd(finite)) <= 0.0:
            failures.append(f"{name} y does not vary")
        if "y_class" in frame.columns:
            raw = np.asarray(frame["y_class"])
            values = set()
            for v in np.unique(raw):
                if v is None:
                    continue
                try:
                    if v != v:  # NaN
                        continue
                except Exception:  # noqa: BLE001
                    pass
                try:
                    values.add(int(v))
                except (TypeError, ValueError):
                    values.add(v)
            if not values.issubset({0, 1}):
                failures.append(f"{name} y_class values must be 0/1, got {values}")

    assert not failures, "DatasetSpec contract failures:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and inspect.isfunction(_fn):
            suite.addTest(unittest.FunctionTestCase(_fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
