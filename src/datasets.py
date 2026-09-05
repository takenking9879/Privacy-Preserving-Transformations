"""Heterogeneous datasets: one real numeric table plus two mixed synthetic banks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from src.schema import ColumnSpec, RawTable


EMPLOYERS = [
    "northwind_bank",
    "helix_logistics",
    "cinder_health",
    "oak_municipal",
    "vertex_retail",
    "plainfield_foods",
    "aurora_energy",
    "self_employed",
    "unemployed",
    "other",
]
REGIONS = ["NE", "MW", "S", "W", "intl"]
PRODUCTS = ["checking", "savings", "credit", "mortgage", "brokerage"]
EDU = ["hs", "assoc", "ba", "ms", "phd"]


def load_california_housing(n_max: int | None = None, seed: int = 0) -> RawTable:
    housing = fetch_california_housing()
    X = np.asarray(housing.data, dtype=np.float64)
    y = np.asarray(housing.target, dtype=np.float64)
    names = list(housing.feature_names)
    if n_max is not None and n_max < len(X):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=n_max, replace=False)
        X, y = X[idx], y[idx]
    frame = pd.DataFrame(X, columns=names)
    columns = [
        ColumnSpec(n, "numeric", sensitive=n in {"MedInc", "Latitude", "Longitude"}, semantic_role=n)
        for n in names
    ]
    return RawTable(
        name="california_housing",
        frame=frame,
        y=y,
        columns=columns,
        task="regression",
        context="US census-derived housing block statistics from California.",
        description=f"sklearn California housing, n={len(frame)}, d={frame.shape[1]}.",
    )


def _zip_from_region(region: str, rng: np.random.RandomState) -> str:
    base = {"NE": 10000, "MW": 40000, "S": 30000, "W": 90000, "intl": 0}[region]
    if region == "intl":
        return f"X{rng.randint(10000, 99999)}"
    return f"{base + rng.randint(0, 8999):05d}"


def make_banking_mixed(
    n: int = 5000,
    seed: int = 0,
    task: str = "regression",
    name: str = "banking_mixed",
) -> RawTable:
    """Bank-like heterogeneous table with sensitive fields and identifier strings.

    A few latents drive both features and the target, so a utility-preserving
    map must keep those latent relationships without exposing raw values.
    """
    rng = np.random.RandomState(seed)
    risk = rng.normal(size=n)
    income_lat = rng.normal(size=n)
    activity = rng.normal(size=n)
    loyalty = rng.normal(size=n)

    income = np.exp(10.4 + 0.55 * income_lat + 0.08 * rng.normal(size=n))
    age = np.clip(rng.normal(42, 12, size=n), 18, 90).round().astype(int)
    tenure = np.clip((age - 18) * rng.uniform(0.1, 0.6, size=n) + rng.normal(0, 1, size=n), 0, 50)
    balance = income * (0.4 + 0.25 * loyalty) + rng.normal(0, 4000, size=n)
    utilization = 1 / (1 + np.exp(-(-0.3 + 0.9 * risk + 0.2 * activity))) + rng.normal(0, 0.03, size=n)
    utilization = np.clip(utilization, 0, 1.5)
    credit = np.clip(
        680 + 40 * income_lat - 35 * risk + 0.2 * (age - 40) + rng.normal(0, 15, size=n),
        300,
        850,
    )
    spend = np.exp(6.5 + 0.4 * activity + 0.25 * income_lat) + rng.normal(0, 50, size=n)
    n_products = np.clip(np.round(2.2 + 0.8 * loyalty + 0.3 * income_lat + rng.normal(0, 0.6, size=n)), 1, 8).astype(int)

    gender = rng.choice(["F", "M"], size=n)
    region = rng.choice(REGIONS, size=n, p=[0.22, 0.20, 0.28, 0.24, 0.06])
    product = rng.choice(PRODUCTS, size=n, p=[0.30, 0.22, 0.25, 0.15, 0.08])
    education = rng.choice(EDU, size=n, p=[0.28, 0.18, 0.32, 0.17, 0.05])
    # Employer is weakly predictive via income/risk coupling.
    emp_idx = np.clip(
        np.round(4 + 1.4 * income_lat - 0.8 * risk + rng.normal(0, 1.0, size=n)).astype(int),
        0,
        len(EMPLOYERS) - 1,
    )
    employer = np.array(EMPLOYERS)[emp_idx]
    zips = np.array([_zip_from_region(r, rng) for r in region])
    customer_id = np.array([f"CUST-{seed:02d}-{i:06d}-{rng.randint(100, 999)}" for i in range(n)])
    email = np.array(
        [
            f"{('a' + str(abs(hash((i, seed))) % 100000))}@{employer[i].split('_')[0]}.example"
            for i in range(n)
        ]
    )

    # Target: expected loss / default-ish score depending on risk, utilization, income.
    latent_y = (
        1.6 * risk
        + 1.1 * utilization
        - 0.55 * np.log(income / 30000)
        - 0.25 * loyalty
        + 0.15 * (product == "credit")
        + 0.08 * (education == "hs")
        + rng.normal(0, 0.35, size=n)
    )
    if task == "classification":
        y = (latent_y > np.quantile(latent_y, 0.72)).astype(int)
        task_kind = "classification"
        desc = "Synthetic bank default classification."
    else:
        y = 2500 * (1 / (1 + np.exp(-latent_y))) + rng.normal(0, 40, size=n)
        task_kind = "regression"
        desc = "Synthetic bank expected-loss regression."

    frame = pd.DataFrame(
        {
            "income": income,
            "balance": balance,
            "utilization": utilization,
            "tenure_years": tenure,
            "age": age,
            "credit_score": credit,
            "monthly_spend": spend,
            "n_products": n_products,
            "gender": gender,
            "region": region,
            "product": product,
            "education": education,
            "employer": employer,
            "zip_code": zips,
            "email": email,
            "customer_id": customer_id,
        }
    )
    columns = [
        ColumnSpec("income", "numeric", True, "income"),
        ColumnSpec("balance", "numeric", True, "account_balance"),
        ColumnSpec("utilization", "numeric", True, "credit_utilization"),
        ColumnSpec("tenure_years", "numeric", False, "tenure"),
        ColumnSpec("age", "numeric", True, "age"),
        ColumnSpec("credit_score", "numeric", True, "credit_score"),
        ColumnSpec("monthly_spend", "numeric", True, "spend"),
        ColumnSpec("n_products", "numeric", False, "product_count"),
        ColumnSpec("gender", "categorical", True, "gender"),
        ColumnSpec("region", "categorical", False, "region"),
        ColumnSpec("product", "categorical", False, "product_type"),
        ColumnSpec("education", "categorical", False, "education"),
        ColumnSpec("employer", "string", False, "employer"),
        ColumnSpec("zip_code", "string", True, "zip"),
        ColumnSpec("email", "string", True, "email"),
        ColumnSpec("customer_id", "string", True, "row_id", identifier=True),
    ]
    return RawTable(
        name=name,
        frame=frame,
        y=y,
        columns=columns,
        task=task_kind,  # type: ignore[arg-type]
        context="This dataset comes from a bank and contains customer-related statistics.",
        description=f"{desc} n={n}. Mixed numeric/categorical/string/sensitive fields.",
    )


@dataclass
class DatasetSpec:
    loader: str
    kwargs: dict


def load_protocol_datasets(quick: bool = False) -> list[RawTable]:
    n_bank = 1800 if quick else 4500
    n_house = 1800 if quick else 6000
    tables = [
        load_california_housing(n_max=n_house, seed=0),
        make_banking_mixed(n=n_bank, seed=1, task="regression", name="banking_mixed_reg"),
        make_banking_mixed(n=n_bank, seed=2, task="classification", name="banking_mixed_clf"),
    ]
    return tables
