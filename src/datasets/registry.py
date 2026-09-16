"""Discover ``DatasetSpec`` objects from ``src.datasets.*`` modules."""

from __future__ import annotations

import importlib
import warnings
from typing import Optional

from src.datasets.base import DatasetSpec

DATASETS: dict[str, DatasetSpec] = {}

_MODULE_CANDIDATES: tuple[str, ...] = (
    "src.datasets.credit",
    "src.datasets.healthcare",
    "src.datasets.retail",
    "src.datasets.insurance",
    "src.datasets.interactions",
    "src.datasets.multimodal",
    "src.datasets.imbalanced",
    "src.datasets.heavytail",
    "src.datasets.panel",
    "src.datasets.sparse_linear",
    "src.dgp",
)


def _register(spec: DatasetSpec) -> None:
    DATASETS[spec.name] = spec


def _try_module(path: str) -> None:
    try:
        mod = importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001 — discovery must not crash
        warnings.warn(f"dataset module {path} skipped: {exc}", RuntimeWarning)
        return
    spec = getattr(mod, "SPEC", None)
    if isinstance(spec, DatasetSpec):
        _register(spec)
        return
    gen = getattr(mod, "generate_original", None) or getattr(mod, "generate", None)
    if callable(gen) and path.endswith(".dgp"):
        feats = tuple(getattr(mod, "FEATURE_COLS", ()) or ())
        _register(
            DatasetSpec(
                name="credit",
                generate=lambda n, seed, _g=gen: _g(n, seed),
                description=str(
                    getattr(mod, "get_ground_truth_description", lambda: "credit DGP")()
                ),
                feature_cols=feats,
                tags=("credit", "mixed", "legacy"),
            )
        )


def _discover() -> None:
    DATASETS.clear()
    for path in _MODULE_CANDIDATES:
        _try_module(path)


def available() -> list[str]:
    if not DATASETS:
        _discover()
    return list(DATASETS.keys())


def get(name: str) -> Optional[DatasetSpec]:
    if not DATASETS:
        _discover()
    return DATASETS.get(name)


_discover()
