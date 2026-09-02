"""Required visualizations for the privacy/utility study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

HIGHLIGHT_METHODS = {
    "raw",
    "gauss",
    "gauss_white_rot",
    "gw_q16_rot",
    "gw_q16_p50_rot",
    "bn50",
    "bn50_q16_rot",
    "bn50_adv_rot",
}


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "method"], sort=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_raw_mean=("r2_raw", "mean"),
            r2_retention_mean=("r2_retention", "mean"),
            r2_retention_std=("r2_retention", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_raw_mean=("rmse_raw", "mean"),
            rmse_ratio_mean=("rmse_ratio", "mean"),
            dist_mean=("distance_preservation", "mean"),
            recon_mean=("recon_r2_knownpair_ridge", "mean"),
            worst_mean=("worst_attr_knownpair_ridge", "mean"),
            neural_mean=("neural_best_r2", "mean"),
            out_dim_mean=("out_dim", "mean"),
            dataset_d=("dataset_d", "mean"),
        )
        .reset_index()
    )


def generate_all_plots(df: pd.DataFrame, raw_records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    agg = _agg(df)
    _plot_raw_vs_transformed(agg, out_dir)
    _plot_retention_vs_strength(agg, out_dir)
    _plot_known_pairs(raw_records, out_dir)
    _plot_per_feature_leakage(raw_records, out_dir)
    _plot_pareto(agg, out_dir)
    _plot_structure_vs_utility(agg, out_dir)
    _plot_dim_effect(agg, out_dir)


def _plot_raw_vs_transformed(agg: pd.DataFrame, out_dir: Path) -> None:
    datasets = list(agg["dataset"].unique())
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.2 * n), squeeze=False)
    for ax, ds in zip(axes[:, 0], datasets):
        g = agg[agg["dataset"] == ds]
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["r2_raw_mean"], width=0.36, label="Raw $R^2$", color="#4c78a8")
        ax.bar(x + 0.18, g["r2_mean"], width=0.36, label="Transformed $R^2$", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(g["method"], rotation=50, ha="right")
        ax.set_ylabel("$R^2$")
        ax.set_title(f"Raw vs transformed predictive $R^2$ — {ds}")
        ax.legend()
        ax.set_ylim(0, max(0.05, float(np.nanmax(g[["r2_raw_mean", "r2_mean"]].values)) * 1.15))
    fig.tight_layout()
    fig.savefig(out_dir / "01_raw_vs_transformed_r2.png")
    plt.close(fig)


def _strength(method: str) -> float:
    order = {
        "raw": 0,
        "gauss": 1,
        "gauss_white": 2,
        "gw_q64": 3,
        "gw_q32": 4,
        "gw_q16": 5,
        "gw_q8": 6,
        "gauss_white_rot": 7,
        "gw_q16_rot": 8,
        "gw_q16_p90_rot": 9,
        "gw_q16_p80_rot": 10,
        "gw_q16_p70_rot": 11,
        "gw_q16_p50_rot": 12,
        "bn50": 13,
        "bn50_rot": 14,
        "bn50_q16_rot": 15,
        "bn50_adv_rot": 16,
    }
    return float(order.get(method, 50))


def _plot_retention_vs_strength(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for ds, g in agg.groupby("dataset"):
        g = g.copy()
        g["s"] = g["method"].map(_strength)
        g = g.sort_values("s")
        ax.errorbar(
            g["s"],
            g["r2_retention_mean"],
            yerr=g["r2_retention_std"].fillna(0),
            marker="o",
            capsize=3,
            label=ds,
        )
    ax.axhline(0.9, color="gray", ls="--", lw=1, label="90% retention target")
    ax.set_xlabel("Transformation strength (protocol order)")
    ax.set_ylabel("$R^2$ retention")
    ax.set_title("Utility retention vs transformation strength")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_retention_vs_strength.png")
    plt.close(fig)


def _plot_known_pairs(raw_records: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), squeeze=False)
    datasets = []
    for rec in raw_records:
        if rec["dataset"] not in datasets:
            datasets.append(rec["dataset"])
    datasets = datasets[:3]
    while len(datasets) < 3:
        datasets.append(None)
    for ax, ds in zip(axes[0], datasets):
        if ds is None:
            ax.axis("off")
            continue
        methods = {}
        for rec in raw_records:
            if rec["dataset"] != ds:
                continue
            curve = ((rec.get("attacks") or {}).get("known_pair") or {}).get("curve") or []
            if not curve:
                continue
            xs, ys = [], []
            for pt in curve:
                ridge = pt.get("ridge") or {}
                if "global_r2" in ridge:
                    xs.append(pt["n"])
                    ys.append(ridge["global_r2"])
            if not xs:
                continue
            methods.setdefault(rec["method"], []).append((xs, ys))
        for method, series in methods.items():
            # average across seeds on shared n
            n_map: dict[int, list[float]] = {}
            for xs, ys in series:
                for n, v in zip(xs, ys):
                    n_map.setdefault(n, []).append(v)
            ns = sorted(n_map)
            mean = [float(np.mean(n_map[n])) for n in ns]
            ax.plot(ns, mean, marker="o", label=method, lw=1.6)
        ax.set_xscale("log")
        ax.set_xlabel("Known pairs")
        ax.set_ylabel("Reconstruction $R^2$ (ridge)")
        ax.set_title(ds)
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Known-pair reconstruction quality")
    fig.tight_layout()
    fig.savefig(out_dir / "03_known_pairs_reconstruction.png")
    plt.close(fig)


def _plot_per_feature_leakage(raw_records: list[dict], out_dir: Path) -> None:
    # Use neural best if present else largest ridge known-pair.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    picked = []
    wanted = ["raw", "gauss_white_rot", "gw_q16_rot", "bn50_q16_rot"]
    by_method = {}
    for rec in raw_records:
        by_method.setdefault(rec["method"], []).append(rec)
    for name in wanted:
        if name in by_method:
            picked.append(by_method[name][0])
    if not picked:
        picked = raw_records[:4]
    for ax, rec in zip(axes, picked):
        r2j = None
        neural = (rec.get("attacks") or {}).get("neural_best")
        if isinstance(neural, dict) and "per_feature_r2" in neural:
            r2j = neural["per_feature_r2"]
        else:
            curve = ((rec.get("attacks") or {}).get("known_pair") or {}).get("curve") or []
            if curve and isinstance(curve[-1].get("ridge"), dict):
                r2j = curve[-1]["ridge"].get("per_feature_r2")
        if not r2j:
            ax.set_title(f"{rec['method']} (no leakage vector)")
            continue
        r2j = np.asarray(r2j, dtype=float)
        ax.bar(np.arange(len(r2j)), r2j, color="#e45756")
        ax.set_title(
            f"{rec['dataset']} / {rec['method']}\nmax $R^2$={np.nanmax(r2j):.3f}, median={np.nanmedian(r2j):.3f}"
        )
        ax.set_xlabel("Original feature index")
        ax.set_ylabel("Reconstruction $R^2$")
        ax.set_ylim(min(-0.2, float(np.nanmin(r2j)) - 0.05), 1.05)
    fig.suptitle("Per-feature reconstruction leakage")
    fig.tight_layout()
    fig.savefig(out_dir / "04_per_feature_leakage.png")
    plt.close(fig)


def _plot_pareto(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for ds, g in agg.groupby("dataset"):
        leak = g["worst_mean"].fillna(g["recon_mean"])
        ax.scatter(leak, g["r2_retention_mean"], s=55, label=ds)
        for _, row in g.iterrows():
            x = row["worst_mean"] if pd.notna(row["worst_mean"]) else row["recon_mean"]
            y = row["r2_retention_mean"]
            if pd.notna(x) and pd.notna(y) and row["method"] in HIGHLIGHT_METHODS:
                ax.annotate(
                    row["method"],
                    (x, y),
                    fontsize=6,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
    ax.set_xlabel("Worst-attribute leakage (known-pair ridge $R^2$)")
    ax.set_ylabel("$R^2$ retention")
    ax.set_title("Privacy / utility Pareto scatter (utility on vertical axis)")
    ax.axhline(0.9, color="gray", ls="--", lw=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_pareto_frontier.png")
    plt.close(fig)


def _plot_structure_vs_utility(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for ds, g in agg.groupby("dataset"):
        ax.scatter(g["dist_mean"], g["r2_retention_mean"], s=60, label=ds)
        for _, row in g.iterrows():
            ax.annotate(row["method"], (row["dist_mean"], row["r2_retention_mean"]), fontsize=6)
    ax.set_xlabel("Pairwise distance preservation (corr)")
    ax.set_ylabel("$R^2$ retention")
    ax.set_title("Structural distance preservation vs predictive retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_structure_vs_utility.png")
    plt.close(fig)


def _plot_dim_effect(agg: pd.DataFrame, out_dir: Path) -> None:
    syn = agg[agg["dataset"].str.contains("banking", na=False)].copy()
    if syn.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method, g in syn.groupby("method"):
        ax.plot(g["dataset_d"], g["recon_mean"], marker="o", label=method)
    ax.set_xlabel("Original dimensionality $d$")
    ax.set_ylabel("Known-pair ridge reconstruction $R^2$")
    ax.set_title("Does increasing dimensionality make reconstruction harder?")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "07_dimensionality_vs_reconstruction.png")
    plt.close(fig)
