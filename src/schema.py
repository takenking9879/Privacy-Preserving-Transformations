"""Shared types for heterogeneous tables and transformed representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

TaskKind = Literal["regression", "classification"]
ColumnKind = Literal["numeric", "categorical", "string"]


@dataclass
class ColumnSpec:
    name: str
    kind: ColumnKind
    sensitive: bool = False
    semantic_role: str = ""
    # True for row-identifiers that must not be used as predictive features.
    identifier: bool = False


@dataclass
class RawTable:
    """Owner-side table. Column names and semantics stay on this side."""

    name: str
    frame: pd.DataFrame
    y: np.ndarray
    columns: list[ColumnSpec]
    task: TaskKind
    context: str
    description: str

    @property
    def n(self) -> int:
        return int(len(self.frame))

    @property
    def feature_names(self) -> list[str]:
        return [c.name for c in self.columns if not c.identifier]

    def predictive_frame(self) -> pd.DataFrame:
        keep = [c.name for c in self.columns if not c.identifier]
        return self.frame.loc[:, keep].copy()

    def column_map(self) -> dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}


@dataclass
class TargetMap:
    """Invertible map  y -> y_tilde  so predictions can be returned to y-space."""

    task: TaskKind
    scale: float = 1.0
    shift: float = 0.0
    class_forward: dict[int, int] = field(default_factory=dict)
    class_inverse: dict[int, int] = field(default_factory=dict)

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if self.task == "regression":
            return self.scale * y.astype(np.float64) + self.shift
        mapped = np.empty_like(y, dtype=np.int64)
        for i, v in enumerate(y.astype(int)):
            mapped[i] = self.class_forward[int(v)]
        return mapped

    def inverse(self, y_tilde: np.ndarray) -> np.ndarray:
        y_tilde = np.asarray(y_tilde)
        if self.task == "regression":
            return (y_tilde.astype(np.float64) - self.shift) / self.scale
        mapped = np.empty_like(y_tilde, dtype=np.int64)
        for i, v in enumerate(np.rint(y_tilde).astype(int)):
            mapped[i] = self.class_inverse.get(int(v), int(v))
        return mapped


@dataclass
class TransformedView:
    """What the training party is allowed to see."""

    Z: np.ndarray
    y: np.ndarray
    column_ids: list[str]
    transform_name: str
    task: TaskKind


@dataclass
class FittedTransform:
    name: str
    target_map: TargetMap
    in_names: list[str]
    out_dim: int
    notes: dict
