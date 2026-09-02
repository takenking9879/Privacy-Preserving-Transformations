"""Experiment orchestration: same splits, same model, same hyperparameters."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.attacks import run_attacks
from src.config import (
    FIGURES_DIR,
    HEAVY_ATTACK_CONFIGS,
    RESULTS_DIR,
    SEEDS,
    TABLES_DIR,
    TRANSFORM_SPECS,
    QUICK_SPECS,
    TransformSpec,
)
from src.datasets import DatasetBundle, load_all_datasets, split_indices
from src.metrics import retention
from src.models import model_card, train_eval
from src.plots import generate_all_plots
from src.transforms import PrivacyTransform


def _json_default(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def run_one(
    bundle: DatasetBundle,
    spec: TransformSpec,
    seed: int,
    idx: dict[str, np.ndarray],
    raw_utility: dict | None,
    run_heavy: bool,
) -> dict[str, Any]:
    X, y = bundle.X, bundle.y
    X_train, y_train = X[idx["train"]], y[idx["train"]]
    X_test, y_test = X[idx["test"]], y[idx["test"]]
    X_aux, y_aux = X[idx["aux"]], y[idx["aux"]]
    del y_aux

    tfm = PrivacyTransform(spec, seed=seed)
    Z_train = tfm.fit_transform(X_train, y_train)
    Z_test = tfm.transform(X_test)
    Z_aux = tfm.transform(X_aux)

    util, _ = train_eval(Z_train, y_train, Z_test, y_test, seed=seed)
    util_d = asdict(util)

    row: dict[str, Any] = {
        "dataset": bundle.name,
        "dataset_kind": bundle.kind,
        "dataset_n": bundle.n,
        "dataset_d": bundle.d,
        "method": spec.name,
        "seed": seed,
        "out_dim": int(Z_train.shape[1]),
        "utility": util_d,
        "model": model_card(),
        "spec": spec.__dict__,
    }
    if raw_utility is not None:
        row["r2_raw"] = raw_utility["r2"]
        row["rmse_raw"] = raw_utility["rmse"]
        row["mae_raw"] = raw_utility["mae"]
        row["r2_retention"] = retention(util_d["r2"], raw_utility["r2"])
        row["delta_r2"] = util_d["r2"] - raw_utility["r2"]
        row["rmse_ratio"] = retention(util_d["rmse"], raw_utility["rmse"])
        row["mae_ratio"] = retention(util_d["mae"], raw_utility["mae"])
    else:
        row["r2_raw"] = util_d["r2"]
        row["rmse_raw"] = util_d["rmse"]
        row["mae_raw"] = util_d["mae"]
        row["r2_retention"] = 1.0
        row["delta_r2"] = 0.0
        row["rmse_ratio"] = 1.0
        row["mae_ratio"] = 1.0

    heavy = run_heavy and spec.name in HEAVY_ATTACK_CONFIGS and seed == 0
    try:
        row["attacks"] = run_attacks(
            X_train,
            y_train,
            X_test,
            y_test,
            X_aux,
            Z_train,
            Z_test,
            Z_aux,
            tfm,
            seed,
            heavy=heavy,
        )
    except Exception as exc:  # noqa: BLE001
        row["attacks"] = {"error": str(exc), "traceback": traceback.format_exc()}
    return row


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    util = row["utility"]
    attacks = row.get("attacks") or {}
    semantic = attacks.get("semantic_matching") or {}
    unpaired = semantic.get("unpaired_moment_match") or {}
    flat = {
        "dataset": row["dataset"],
        "dataset_kind": row["dataset_kind"],
        "dataset_n": row["dataset_n"],
        "dataset_d": row["dataset_d"],
        "method": row["method"],
        "seed": row["seed"],
        "out_dim": row["out_dim"],
        "r2": util["r2"],
        "rmse": util["rmse"],
        "mae": util["mae"],
        "train_time_s": util["train_time_s"],
        "infer_time_s": util["infer_time_s"],
        "r2_raw": row.get("r2_raw"),
        "rmse_raw": row.get("rmse_raw"),
        "mae_raw": row.get("mae_raw"),
        "r2_retention": row.get("r2_retention"),
        "delta_r2": row.get("delta_r2"),
        "rmse_ratio": row.get("rmse_ratio"),
        "mae_ratio": row.get("mae_ratio"),
        "distance_preservation": attacks.get("distance_preservation"),
        "semantic_unpaired_acc": unpaired.get("assignment_accuracy"),
        "semantic_paired_acc": semantic.get("paired_corr_assignment_accuracy"),
        "semantic_max_abs_corr": semantic.get("max_abs_corr"),
        "recon_r2_knownpair_ridge": attacks.get("summary_recon_r2_knownpair_ridge"),
        "worst_attr_knownpair_ridge": attacks.get("summary_worst_attr_knownpair_ridge"),
        "known_pair_robustness": attacks.get("known_pair_robustness"),
        "z_mean_abs_skew": (attacks.get("statistical_inspection") or {}).get("z_mean_abs_skew"),
        "z_offdiag_corr_mean_abs": (attacks.get("statistical_inspection") or {}).get(
            "z_offdiag_corr_mean_abs"
        ),
    }
    aux = attacks.get("auxiliary_unpaired") or {}
    rr = aux.get("random_rotation_moment_match") or {}
    ica = aux.get("ica_moment_align") or {}
    flat["aux_random_rot_r2"] = rr.get("global_r2")
    flat["aux_ica_r2"] = ica.get("global_r2")
    neural_best = attacks.get("neural_best") or {}
    flat["neural_best_r2"] = neural_best.get("global_r2")
    flat["neural_best_worst_attr"] = neural_best.get("max_feature_r2")
    return flat


def aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "method"]
    num_cols = [
        c
        for c in df.columns
        if c not in keys + ["dataset_kind", "known_pair_robustness", "seed"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for (dataset, method), g in df.groupby(keys, sort=False):
        rec = {
            "dataset": dataset,
            "method": method,
            "dataset_kind": g["dataset_kind"].iloc[0],
            "n_runs": len(g),
        }
        for c in num_cols:
            rec[f"{c}_mean"] = g[c].mean()
            rec[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        rec["known_pair_robustness"] = (
            g["known_pair_robustness"].mode().iloc[0]
            if g["known_pair_robustness"].notna().any()
            else ""
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def run_experiments(quick: bool = False) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets(quick=quick)
    specs = QUICK_SPECS if quick else TRANSFORM_SPECS
    seeds = [0] if quick else SEEDS
    save_json(
        RESULTS_DIR / "run_meta.json",
        {
            "quick": quick,
            "seeds": seeds,
            "model": model_card(),
            "n_datasets": len(datasets),
            "n_specs": len(specs),
            "datasets": [
                {
                    "name": d.name,
                    "kind": d.kind,
                    "n": d.n,
                    "d": d.d,
                    "description": d.description,
                }
                for d in datasets
            ],
        },
    )

    flat_rows: list[dict] = []
    raw_records: list[dict] = []

    for bundle in datasets:
        print(f"\n=== Dataset {bundle.name} n={bundle.n} d={bundle.d} ===", flush=True)
        for seed in seeds:
            idx = split_indices(bundle.n, seed)
            save_json(
                RESULTS_DIR / bundle.name / f"split_seed{seed}.json",
                {k: v.tolist() for k, v in idx.items()},
            )
            raw_utility = None
            for spec in specs:
                print(f"  seed={seed} method={spec.name}", flush=True)
                try:
                    row = run_one(
                        bundle,
                        spec,
                        seed,
                        idx,
                        raw_utility,
                        run_heavy=not quick,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"    FAILED: {exc}", flush=True)
                    row = {
                        "dataset": bundle.name,
                        "dataset_kind": bundle.kind,
                        "dataset_n": bundle.n,
                        "dataset_d": bundle.d,
                        "method": spec.name,
                        "seed": seed,
                        "out_dim": spec.output_dim(bundle.d),
                        "utility": {
                            "r2": float("nan"),
                            "rmse": float("nan"),
                            "mae": float("nan"),
                            "train_time_s": float("nan"),
                            "infer_time_s": float("nan"),
                        },
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "r2_retention": float("nan"),
                        "delta_r2": float("nan"),
                        "rmse_ratio": float("nan"),
                        "mae_ratio": float("nan"),
                        "attacks": {},
                    }
                if spec.name == "raw" and "utility" in row:
                    raw_utility = row["utility"]
                save_json(
                    RESULTS_DIR / bundle.name / f"{spec.name}_seed{seed}.json",
                    row,
                )
                raw_records.append(row)
                flat_rows.append(flatten_row(row))

    df = pd.DataFrame(flat_rows)
    df.to_csv(TABLES_DIR / "utility_attacks_runs.csv", index=False)
    agg = aggregate_table(df)
    agg.to_csv(TABLES_DIR / "utility_attacks_summary.csv", index=False)
    save_json(RESULTS_DIR / "all_runs.json", raw_records)
    try:
        generate_all_plots(df, raw_records, FIGURES_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"Plot generation failed: {exc}", flush=True)
        traceback.print_exc()
    return df
