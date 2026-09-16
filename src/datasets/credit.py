from src.datasets.base import DatasetSpec
from src.dgp import FEATURE_COLS, generate_original, get_ground_truth_description

SPEC = DatasetSpec(
    name="credit",
    generate=generate_original,
    description=get_ground_truth_description(),
    feature_cols=tuple(FEATURE_COLS),
    target_reg="y",
    target_clf="y_class",
    tags=("mixed", "skew", "interactions", "mar", "legacy"),
)
