"""Figures for the utility / leakage protocol."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, raw_records: list[dict], out: Path) -> None:
    if df is None or len(df) == 0:
        return
    out.mkdir(parents=True, exist_ok=True)
    _utility_bars(df, out / "01_utility_retention.png")
    _pareto(df, out / "02_pareto_utility_leakage.png")
    _known_pairs(raw_records, out / "03_known_pairs.png")
    _structure(df, out / "04_structure_vs_utility.png")
    _attacker_b(df, out / "05_attacker_b.png")


def _utility_bars(df: pd.DataFrame, path: Path) -> None:
    g = df.groupby(["dataset", "method", "model"], as_index=False)["retention"].mean()
    datasets = list(g["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 3.2 * max(len(datasets), 1)), squeeze=False)
    for ax, ds in zip(axes[:, 0], datasets):
        sub = g[g["dataset"] == ds]
        methods = list(sub["method"].unique())
        models = list(sub["model"].unique())
        x = np.arange(len(methods))
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            ys = [
                float(sub[(sub["method"] == m) & (sub["model"] == model)]["retention"].mean())
                for m in methods
            ]
            ax.bar(x + i * width, ys, width, label=model)
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.set_xticks(x + 0.3)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("score retention")
        ax.set_title(ds)
        ax.legend(fontsize=8, ncol=4)
    _save(fig, path)


def _pareto(df: pd.DataFrame, path: Path) -> None:
    # One point per method: mean retention (hgb) vs mean worst-attr leakage.
    rows = []
    for (ds, method), g in df.groupby(["dataset", "method"]):
        hgb = g[g["model"] == "hgb"]
        rows.append(
            {
                "dataset": ds,
                "method": method,
                "retention": hgb["retention"].mean() if len(hgb) else g["retention"].mean(),
                "leak": g["worst_attr_r2"].mean(),
            }
        )
    p = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds, sub in p.groupby("dataset"):
        ax.scatter(sub["leak"], sub["retention"], label=ds, s=40)
        for _, r in sub.iterrows():
            ax.annotate(r["method"], (r["leak"], r["retention"]), fontsize=7, alpha=0.8)
    ax.set_xlabel("known-pair worst-attribute R² (lower is better privacy)")
    ax.set_ylabel("HGB utility retention (higher is better)")
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.legend()
    ax.set_title("Utility / leakage trade-off")
    _save(fig, path)


def _known_pairs(raw_records: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in raw_records:
        curve = (row.get("attacker_a") or {}).get("curve") or []
        if not curve:
            continue
        xs, ys = [], []
        for c in curve:
            r = c.get("ridge") or {}
            if "global_r2" in r:
                xs.append(c["n"])
                ys.append(r["global_r2"])
        if xs:
            ax.plot(xs, ys, alpha=0.35, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("known pairs")
    ax.set_ylabel("ridge reconstruction R²")
    ax.set_title("Attacker A: reconstruction vs pair budget")
    _save(fig, path)


def _structure(df: pd.DataFrame, path: Path) -> None:
    h = df[df["model"] == "hgb"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(h["distance_corr"], h["retention"], c=h["rank_match"], cmap="viridis", s=36)
    ax.set_xlabel("pairwise distance correlation")
    ax.set_ylabel("HGB retention")
    ax.set_title("Geometry vs tree utility")
    _save(fig, path)


def _attacker_b(df: pd.DataFrame, path: Path) -> None:
    g = df.groupby("method", as_index=False)[["semantic_acc", "membership_auc", "linkage"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["semantic_acc"], 0.2, label="semantic acc")
    ax.bar(x, g["membership_auc"], 0.2, label="membership AUC")
    ax.bar(x + 0.2, g["linkage"], 0.2, label="linkage")
    ax.set_xticks(x)
    ax.set_xticklabels(g["method"], rotation=30, ha="right")
    ax.legend()
    ax.set_title("Attacker B summary")
    _save(fig, path)
