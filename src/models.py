"""Multiple model families with frozen hyperparameters per family."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import classification_scores


MODEL_NAMES = ["linear", "hgb", "mlp", "knn"]


@dataclass
class UtilityResult:
    model: str
    task: str
    score: float
    score_name: str
    r2: float = float("nan")
    rmse: float = float("nan")
    mae: float = float("nan")
    accuracy: float = float("nan")
    roc_auc: float = float("nan")
    train_time_s: float = float("nan")
    infer_time_s: float = float("nan")
    n_features: int = 0
    extras: dict = field(default_factory=dict)


def make_model(name: str, task: str, seed: int):
    if name == "linear":
        if task == "regression":
            return Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
            )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=400, C=1.0, random_state=seed),
                ),
            ]
        )
    if name == "hgb":
        common = dict(max_iter=80, max_depth=6, learning_rate=0.1, min_samples_leaf=20)
        if task == "regression":
            return HistGradientBoostingRegressor(random_state=seed, **common)
        return HistGradientBoostingClassifier(random_state=seed, **common)
    if name == "mlp":
        if task == "regression":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(64,),
                            max_iter=120,
                            random_state=seed,
                            early_stopping=True,
                        ),
                    ),
                ]
            )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        max_iter=120,
                        random_state=seed,
                        early_stopping=True,
                    ),
                ),
            ]
        )
    if name == "knn":
        if task == "regression":
            return Pipeline(
                [("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=15))]
            )
        return Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=15))]
        )
    raise KeyError(name)


def train_eval(
    name: str,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[UtilityResult, np.ndarray]:
    model = make_model(name, task, seed)
    n_tr = len(X_train)
    if name == "knn" and n_tr > 1800:
        rng = np.random.RandomState(seed)
        take = rng.choice(n_tr, size=1800, replace=False)
        X_fit, y_fit = X_train[take], y_train[take]
    else:
        X_fit, y_fit = X_train, y_train
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit)
    train_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    infer_t = time.perf_counter() - t0
    result = UtilityResult(
        model=name,
        task=task,
        score=0.0,
        score_name="",
        train_time_s=train_t,
        infer_time_s=infer_t,
        n_features=int(X_train.shape[1]),
    )
    if task == "regression":
        result.r2 = float(r2_score(y_test, pred))
        result.rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        result.mae = float(mean_absolute_error(y_test, pred))
        result.score = result.r2
        result.score_name = "r2"
    else:
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
            except Exception:
                proba = None
        clf = classification_scores(y_test, pred, proba)
        result.accuracy = clf["accuracy"]
        result.roc_auc = clf["roc_auc"]
        result.score = clf["roc_auc"] if np.isfinite(clf["roc_auc"]) else clf["accuracy"]
        result.score_name = "roc_auc" if np.isfinite(clf["roc_auc"]) else "accuracy"
    return result, np.asarray(pred)


def prediction_agreement(task: str, pred_raw: np.ndarray, pred_t: np.ndarray) -> float:
    if task == "regression":
        if pred_raw.std() < 1e-12 or pred_t.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(pred_raw.reshape(-1), pred_t.reshape(-1))[0, 1])
    return float(np.mean(pred_raw.reshape(-1) == pred_t.reshape(-1)))
