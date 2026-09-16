"""Shared contract for original data-generating processes.

Every dataset module exposes a ``SPEC`` (or a ``generate(n, seed)`` plus
the column constants below).  Downstream evaluation treats tables as
generic mixed-type frames with:

* a continuous target ``y``
* an optional binary target ``y_class``
* any mix of numeric / categorical / count / binary features

A *general* synthesizer must not hard-code credit-risk column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

GenerateFn = Callable[[int, int], pd.DataFrame]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    generate: GenerateFn
    description: str
    feature_cols: tuple[str, ...] = ()
    target_reg: str = "y"
    target_clf: Optional[str] = "y_class"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def sample(self, n: int = 800, seed: int = 0) -> pd.DataFrame:
        df = self.generate(int(n), int(seed))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{self.name}.generate must return a DataFrame")
        if self.target_reg not in df.columns:
            raise ValueError(f"{self.name} missing target {self.target_reg!r}")
        return df
