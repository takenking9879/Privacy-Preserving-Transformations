"""Public numeric encoding used as the raw-model baseline and attack target.

This is *not* a privacy transform. It is the honest comparison point:
trees/linear models need numbers, so even X_raw is encoded with a
deterministic, non-secret scheme. Privacy transforms must beat this
on concealment while staying close on predictive utility.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.schema import ColumnSpec, RawTable


def _stable_hash_float(value: object) -> float:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64)


class PublicEncoder:
    """Fit on train only. Strings with high cardinality become hash floats."""

    def __init__(self, columns: list[ColumnSpec], max_ordinal_card: int = 64) -> None:
        self.columns = [c for c in columns if not c.identifier]
        self.max_ordinal_card = max_ordinal_card
        self.cat_encoder: OrdinalEncoder | None = None
        self.cat_names: list[str] = []
        self.num_names: list[str] = []
        self.str_names: list[str] = []
        self.out_names: list[str] = []
        self.out_kinds: list[str] = []
        self.sensitive_mask: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "PublicEncoder":
        self.num_names = [c.name for c in self.columns if c.kind == "numeric"]
        self.cat_names = [c.name for c in self.columns if c.kind == "categorical"]
        self.str_names = [c.name for c in self.columns if c.kind == "string"]
        if self.cat_names:
            self.cat_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
            self.cat_encoder.fit(frame.loc[:, self.cat_names].astype(str))
        self.out_names = list(self.num_names) + list(self.cat_names) + list(self.str_names)
        self.out_kinds = (
            ["numeric"] * len(self.num_names)
            + ["categorical"] * len(self.cat_names)
            + ["string"] * len(self.str_names)
        )
        cmap = {c.name: c for c in self.columns}
        self.sensitive_mask = np.array(
            [cmap[n].sensitive for n in self.out_names], dtype=bool
        )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.num_names:
            blocks.append(frame.loc[:, self.num_names].to_numpy(dtype=np.float64))
        if self.cat_names:
            assert self.cat_encoder is not None
            blocks.append(
                self.cat_encoder.transform(frame.loc[:, self.cat_names].astype(str)).astype(
                    np.float64
                )
            )
        if self.str_names:
            hashed = np.empty((len(frame), len(self.str_names)), dtype=np.float64)
            for j, name in enumerate(self.str_names):
                hashed[:, j] = [_stable_hash_float(v) for v in frame[name].tolist()]
            blocks.append(hashed)
        if not blocks:
            return np.zeros((len(frame), 0), dtype=np.float64)
        return np.hstack(blocks)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def encode_raw_table(table: RawTable, train_idx: np.ndarray) -> PublicEncoder:
    enc = PublicEncoder(table.columns)
    enc.fit(table.predictive_frame().iloc[train_idx])
    return enc
