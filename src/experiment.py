"""Protocol: same splits for every transform; utility then Attacker A/B."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.attacks import attacker_a, attacker_b
from src.datasets import load_protocol_datasets
from src.encode import PublicEncoder
from src.metrics import retention
from src.models import MODEL_NAMES, prediction_agreement, train_eval
from src.properties import analyze_properties
from src.transforms import CANDIDATE_TRANSFORMS, QUICK_TRANSFORMS, build_transform

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"


def _json_default(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_tr = int(0.70 * n)
    n_te = int(0.15 * n)
    return {
        "train": idx[:n_tr],
        "test": idx[n_tr : n_tr + n_te],
        "aux": idx[n_tr + n_te :],
    }


def _roles_kinds(encoder: PublicEncoder) -> tuple[list[str], list[str], np.ndarray, list[int]]:
    cmap = {c.name: c for c in encoder.columns}
    roles = [cmap[n].semantic_role for n in encoder.out_names]
    kinds = list(encoder.out_kinds)
    sens = encoder.sensitive_mask if encoder.sensitive_mask is not None else np.zeros(len(roles), dtype=bool)
    cat_idx = [i for i, k in enumerate(kinds) if k in {"categorical", "string"}]
    return roles, kinds, sens, cat_idx


def run_one(
    table,
    transform_name: str,
    seed: int,
    idx: dict[str, np.ndarray],
    raw_scores: dict[str, dict],
    raw_preds: dict[str, np.ndarray],
    encoder: PublicEncoder,
    X_all: np.ndarray,
    heavy: bool,
) -> dict[str, Any]:
    tfm = build_transform(transform_name, seed)
    tfm.fit(table, idx["train"])
    tr = tfm.apply(table, idx["train"])
    te = tfm.apply(table, idx["test"])
    # Predictions scored in original y-space.
    y_tr = table.y[idx["train"]]
    y_te = table.y[idx["test"]]
    y_tr_model = tr.y_tilde
    # Train on transformed target, invert predictions.
    utilities = []
    preds_orig = {}
    for model_name in MODEL_NAMES:
        util, pred_t = train_eval(model_name, table.task, tr.Z, y_tr_model, te.Z, te.y_tilde, seed)
        pred_orig = tfm.inverse_y(pred_t)
        if table.task == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            util.r2 = float(r2_score(y_te, pred_orig))
            util.rmse = float(np.sqrt(mean_squared_error(y_te, pred_orig)))
            util.mae = float(mean_absolute_error(y_te, pred_orig))
            util.score = util.r2
        else:
            from src.metrics import classification_scores

            clf = classification_scores(y_te, pred_orig, None)
            util.accuracy = clf["accuracy"]
            util.score = clf["accuracy"]
            util.score_name = "accuracy"
        raw = raw_scores.get(model_name)
        if raw:
            util.extras["retention"] = retention(util.score, raw["score"])
            util.extras["agreement"] = prediction_agreement(
                table.task, raw_preds[model_name], pred_orig
            )
        elif transform_name == "identity":
            util.extras["retention"] = 1.0
            util.extras["agreement"] = 1.0
        utilities.append(asdict(util))
        preds_orig[model_name] = pred_orig

    X_tr, X_te = X_all[idx["train"]], X_all[idx["test"]]
    props = analyze_properties(X_tr, tr.Z, seed)
    roles, kinds, sens, cat_idx = _roles_kinds(encoder)
    att_a = attacker_a(
        X_tr,
        tr.Z,
        X_te,
        te.Z,
        y_tr,
        y_te,
        sens,
        cat_idx,
        seed,
        heavy=heavy and seed == 0,
    )
    # Attacker B sees only the transformed matrix the trainer would see, plus context.
    Z_all = np.vstack([tr.Z, te.Z])
    train_mask = np.array([True] * len(tr.Z) + [False] * len(te.Z))
    # Roles/kinds are aligned to X, not Z, so only pass them when dims match.
    roles_b = roles if len(roles) == tr.Z.shape[1] else []
    kinds_b = kinds if len(kinds) == tr.Z.shape[1] else []
    sens_b = sens if len(sens) == tr.Z.shape[1] else np.zeros(tr.Z.shape[1], dtype=bool)
    att_b = attacker_b(
        Z_all,
        table.context,
        roles_b,
        kinds_b,
        sens_b,
        X_aux_public=X_all,
        train_mask=train_mask,
        seed=seed,
    )
    return {
        "dataset": table.name,
        "task": table.task,
        "method": transform_name,
        "family": tfm.family,
        "seed": seed,
        "out_dim": int(tr.Z.shape[1]),
        "in_dim": int(X_tr.shape[1]),
        "column_ids": tr.column_ids,
        "notes": tfm.notes,
        "properties": props,
        "utility": utilities,
        "attacker_a": att_a,
        "attacker_b": att_b,
        "target_transformed": transform_name != "identity",
    }


def flatten_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    a = row.get("attacker_a") or {}
    b = row.get("attacker_b") or {}
    props = row.get("properties") or {}
    out = []
    for u in row.get("utility") or []:
        out.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "method": row["method"],
                "family": row.get("family"),
                "seed": row["seed"],
                "model": u["model"],
                "score": u["score"],
                "score_name": u["score_name"],
                "r2": u.get("r2"),
                "rmse": u.get("rmse"),
                "mae": u.get("mae"),
                "accuracy": u.get("accuracy"),
                "roc_auc": u.get("roc_auc"),
                "retention": (u.get("extras") or {}).get("retention"),
                "agreement": (u.get("extras") or {}).get("agreement"),
                "out_dim": row.get("out_dim"),
                "distance_corr": props.get("distance_corr"),
                "rank_match": props.get("rank_match"),
                "corr_gap": props.get("corr_frobenius_gap"),
                "z_offdiag_corr": props.get("z_offdiag_corr"),
                "recon_r2": a.get("summary_recon_r2"),
                "worst_attr_r2": a.get("summary_worst_attr"),
                "sensitive_attr_r2": a.get("summary_sensitive_attr"),
                "n_pairs_r2_0.5": a.get("n_for_ridge_global_r2_0.5"),
                "known_pair_robustness": a.get("known_pair_robustness"),
                "spearman_leak": a.get("spearman_leak"),
                "semantic_acc": b.get("semantic_top1_accuracy"),
                "kind_acc": b.get("kind_accuracy_if_aligned"),
                "membership_auc": b.get("membership_inference_auc"),
                "linkage": (b.get("linkage") or {}).get("nn_self_match_rate"),
                "sensitive_hit": b.get("sensitive_column_hit_rate"),
            }
        )
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "method", "model"]
    rows = []
    for key, g in df.groupby(keys, sort=False):
        rec = {"dataset": key[0], "method": key[1], "model": key[2], "n_runs": len(g)}
        for c in df.columns:
            if c in keys + ["task", "family", "known_pair_robustness", "score_name", "seed"]:
                continue
            if pd.api.types.is_numeric_dtype(g[c]):
                rec[f"{c}_mean"] = g[c].mean()
                rec[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        rec["known_pair_robustness"] = (
            g["known_pair_robustness"].dropna().mode().iloc[0]
            if g["known_pair_robustness"].notna().any()
            else ""
        )
        rec["task"] = g["task"].iloc[0]
        rec["family"] = g["family"].iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def run_experiments(quick: bool = False) -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    datasets = load_protocol_datasets(quick=quick)
    methods = QUICK_TRANSFORMS if quick else CANDIDATE_TRANSFORMS
    seeds = [0] if quick else [0, 1]
    save_json(
        RESULTS / "run_meta.json",
        {
            "quick": quick,
            "seeds": seeds,
            "methods": methods,
            "models": MODEL_NAMES,
            "datasets": [
                {"name": t.name, "task": t.task, "n": t.n, "d": t.predictive_frame().shape[1]}
                for t in datasets
            ],
        },
    )
    flat_rows: list[dict] = []
    raw_records: list[dict] = []
    for table in datasets:
        print(f"\n=== {table.name} n={table.n} task={table.task} ===", flush=True)
        for seed in seeds:
            idx = split_indices(table.n, seed)
            encoder = PublicEncoder(table.columns)
            X_all = encoder.fit_transform(table.predictive_frame())
            # Identity first so other methods can compute retention.
            raw_scores: dict[str, dict] = {}
            raw_preds: dict[str, np.ndarray] = {}
            for method in methods:
                print(f"  seed={seed} {method}", flush=True)
                try:
                    row = run_one(
                        table,
                        method,
                        seed,
                        idx,
                        raw_scores,
                        raw_preds,
                        encoder,
                        X_all,
                        heavy=not quick,
                    )
                except Exception as exc:
                    print(f"    FAILED: {exc}", flush=True)
                    traceback.print_exc()
                    row = {
                        "dataset": table.name,
                        "task": table.task,
                        "method": method,
                        "seed": seed,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "utility": [],
                        "attacker_a": {},
                        "attacker_b": {},
                    }
                if method == "identity":
                    for u in row.get("utility") or []:
                        raw_scores[u["model"]] = u
                    # Recompute identity preds for agreement: run models on raw X.
                    X_tr = X_all[idx["train"]]
                    X_te = X_all[idx["test"]]
                    y_tr = table.y[idx["train"]]
                    y_te = table.y[idx["test"]]
                    for model_name in MODEL_NAMES:
                        _, pred = train_eval(model_name, table.task, X_tr, y_tr, X_te, y_te, seed)
                        raw_preds[model_name] = pred
                save_json(RESULTS / table.name / f"{method}_seed{seed}.json", row)
                raw_records.append(row)
                flat_rows.extend(flatten_row(row))
    df = pd.DataFrame(flat_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "runs.csv", index=False)
    if len(df):
        agg = aggregate(df)
        agg.to_csv(TABLES / "summary.csv", index=False)
    save_json(RESULTS / "all_runs.json", raw_records)
    try:
        from src.plots import generate_all_plots
        from src.report import write_report

        generate_all_plots(df, raw_records, FIGURES)
        write_report(df, raw_records, RESULTS)
    except Exception as exc:
        print(f"Plot/report failed: {exc}", flush=True)
        traceback.print_exc()
    return df
