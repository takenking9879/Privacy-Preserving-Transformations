#!/usr/bin/env python3
"""Compare the inference capsule against identity and plain VIB."""

from __future__ import annotations

from src.datasets import load_protocol_datasets
from src.encode import PublicEncoder
from src.experiment import RESULTS, TABLES, flatten_row, run_one, save_json, split_indices
from src.models import MODEL_NAMES, train_eval
import pandas as pd


METHODS = ["identity", "vib", "capsule", "capsule_soft", "capsule_tight"]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    datasets = load_protocol_datasets(quick=False)
    seeds = [0, 1]
    rows = []
    for table in datasets:
        print(f"\n=== {table.name} ===", flush=True)
        for seed in seeds:
            idx = split_indices(table.n, seed)
            encoder = PublicEncoder(table.columns)
            X_all = encoder.fit_transform(table.predictive_frame())
            raw_scores, raw_preds = {}, {}
            X_tr, X_te = X_all[idx["train"]], X_all[idx["test"]]
            y_tr, y_te = table.y[idx["train"]], table.y[idx["test"]]
            for model_name in MODEL_NAMES:
                util, pred = train_eval(model_name, table.task, X_tr, y_tr, X_te, y_te, seed)
                raw_scores[model_name] = {
                    "score": util.score,
                    "r2": util.r2,
                    "accuracy": util.accuracy,
                }
                raw_preds[model_name] = pred
            for method in METHODS:
                print(f"  seed={seed} {method}", flush=True)
                row = run_one(
                    table,
                    method,
                    seed,
                    idx,
                    raw_scores,
                    raw_preds,
                    encoder,
                    X_all,
                    heavy=seed == 0,
                )
                save_json(RESULTS / "capsule" / table.name / f"{method}_seed{seed}.json", row)
                rows.extend(flatten_row(row))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "capsule_runs.csv", index=False)
    print(df.groupby(["dataset", "method", "model"])[["retention", "worst_attr_r2", "recon_r2"]].mean())


if __name__ == "__main__":
    main()
