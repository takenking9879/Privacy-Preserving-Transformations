"""Registry of original DGPs used to stress-test general synthesizers."""

from src.datasets.base import DatasetSpec
from src.datasets.registry import DATASETS, available, get

__all__ = ["DATASETS", "DatasetSpec", "available", "get"]
