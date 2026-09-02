"""Frozen predictive model used for every raw vs transformed comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_HYPERPARAMS, MODEL_NAME


@dataclass
class UtilityResult:
    r2: float
    rmse: float
    mae: float
    train_time_s: float
    infer_time_s: float
    n_train: int
    n_test: int
    n_features: int


def make_model(seed: int) -> HistGradientBoostingRegressor:
    params = dict(MODEL_HYPERPARAMS)
    params["random_state"] = seed
    return HistGradientBoostingRegressor(**params)


def train_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[UtilityResult, np.ndarray]:
    model = make_model(seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    infer_time = time.perf_counter() - t0
    result = UtilityResult(
        r2=float(r2_score(y_test, pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        mae=float(mean_absolute_error(y_test, pred)),
        train_time_s=float(train_time),
        infer_time_s=float(infer_time),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        n_features=int(X_train.shape[1]),
    )
    return result, pred


def model_card() -> dict:
    return {"name": MODEL_NAME, "hyperparameters": dict(MODEL_HYPERPARAMS)}
