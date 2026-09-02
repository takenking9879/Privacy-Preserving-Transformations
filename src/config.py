"""Fixed experimental configuration.

The predictive model and its hyperparameters are frozen for every
comparison of M_theta(X) vs M_theta(f(X)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

SEEDS = [0, 1, 2]
N_SEEDS = len(SEEDS)

TRAIN_FRAC = 0.70
TEST_FRAC = 0.15
AUX_FRAC = 0.15

MODEL_NAME = "HistGradientBoostingRegressor"
MODEL_HYPERPARAMS = {
    "max_iter": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "validation_fraction": 0.1,
}

ENCODER_HYPERPARAMS = {
    "hidden": 96,
    "epochs": 25,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "adv_lambda": 0.05,
    "adv_epochs": 25,
}

KNOWN_PAIR_GRID = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
DISTANCE_SUBSAMPLE = 600
NEURAL_ATTACK_SUBSAMPLE = 8000

HEAVY_ATTACK_CONFIGS = {
    "raw",
    "gauss_white_rot",
    "gw_q16_rot",
    "gw_q16_p50_rot",
    "bn50_q16_rot",
}


@dataclass(frozen=True)
class TransformSpec:
    name: str
    gaussianize: bool = False
    whiten: bool = False
    quantize_levels: int | None = None
    rotate: bool = False
    proj_ratio: float | None = None
    bottleneck: bool = False
    bottleneck_ratio: float = 0.5
    adversarial: bool = False

    def output_dim(self, d: int) -> int:
        if self.bottleneck:
            return max(2, int(round(self.bottleneck_ratio * d)))
        if self.proj_ratio is None:
            return d
        return max(2, int(round(self.proj_ratio * d)))


TRANSFORM_SPECS: list[TransformSpec] = [
    TransformSpec("raw"),
    TransformSpec("gauss", gaussianize=True),
    TransformSpec("gauss_white", gaussianize=True, whiten=True),
    TransformSpec("gauss_white_rot", gaussianize=True, whiten=True, rotate=True),
    TransformSpec("gw_q8", gaussianize=True, whiten=True, quantize_levels=8),
    TransformSpec("gw_q16", gaussianize=True, whiten=True, quantize_levels=16),
    TransformSpec("gw_q32", gaussianize=True, whiten=True, quantize_levels=32),
    TransformSpec("gw_q64", gaussianize=True, whiten=True, quantize_levels=64),
    TransformSpec("gw_q16_rot", gaussianize=True, whiten=True, quantize_levels=16, rotate=True),
    TransformSpec(
        "gw_q16_p90_rot",
        gaussianize=True,
        whiten=True,
        quantize_levels=16,
        proj_ratio=0.9,
        rotate=True,
    ),
    TransformSpec(
        "gw_q16_p80_rot",
        gaussianize=True,
        whiten=True,
        quantize_levels=16,
        proj_ratio=0.8,
        rotate=True,
    ),
    TransformSpec(
        "gw_q16_p70_rot",
        gaussianize=True,
        whiten=True,
        quantize_levels=16,
        proj_ratio=0.7,
        rotate=True,
    ),
    TransformSpec(
        "gw_q16_p50_rot",
        gaussianize=True,
        whiten=True,
        quantize_levels=16,
        proj_ratio=0.5,
        rotate=True,
    ),
    TransformSpec("bn50", bottleneck=True, bottleneck_ratio=0.5),
    TransformSpec("bn50_rot", bottleneck=True, bottleneck_ratio=0.5, rotate=True),
    TransformSpec(
        "bn50_q16_rot",
        bottleneck=True,
        bottleneck_ratio=0.5,
        quantize_levels=16,
        rotate=True,
    ),
    TransformSpec(
        "bn50_adv_rot",
        bottleneck=True,
        bottleneck_ratio=0.5,
        rotate=True,
        adversarial=True,
    ),
]


QUICK_SPECS = [
    spec
    for spec in TRANSFORM_SPECS
    if spec.name
    in {
        "raw",
        "gauss",
        "gauss_white",
        "gauss_white_rot",
        "gw_q16",
        "gw_q16_rot",
        "gw_q16_p50_rot",
        "bn50_q16_rot",
    }
]
