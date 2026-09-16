"""Comparison plots for real vs synthetic tables.

All figures are written with the Agg backend so a missing display
server cannot crash the evaluation pipeline.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Stable filenames produced by :func:`save_comparison_plots`.
PLOT_NAMES: tuple[str, ...] = (
    "marginal_y.png",
    "marginal_income.png",
    "marginal_risk_score.png",
    "corr_heatmap_real.png",
    "corr_heatmap_synth.png",
    "scatter_y_vs_risk_score.png",
)

MARGINAL_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("y", ("y", "Y", "target", "expected_loss"), "marginal_y.png"),
    ("income", ("income", "Income", "annual_income"), "marginal_income.png"),
    (
        "risk_score",
        ("risk_score", "riskScore", "risk", "RiskScore"),
        "marginal_risk_score.png",
    ),
)

_REAL_COLOR = "#2c7bb6"
_SYNTH_COLOR = "#d7191c"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.28,
    }
)


def _as_frame(obj, name: str) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    try:
        return pd.DataFrame(obj)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Could not coerce {name} to DataFrame: {exc}")
        return pd.DataFrame()


def _find_col(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    if frame is None or frame.empty:
        return None
    lower = {str(c).lower(): c for c in frame.columns}
    for cand in candidates:
        if cand in frame.columns:
            return cand
        key = str(cand).lower()
        if key in lower:
            return lower[key]
    return None


def _numeric_overlap(real: pd.DataFrame, synth: pd.DataFrame) -> list[str]:
    real_num = set(real.select_dtypes(include=[np.number]).columns)
    synth_num = set(synth.select_dtypes(include=[np.number]).columns)
    cols = [c for c in real.columns if c in real_num and c in synth_num]
    usable: list[str] = []
    for c in cols:
        if real[c].notna().sum() >= 3 and synth[c].notna().sum() >= 3:
            usable.append(c)
    return usable


def _safe_savefig(fig: plt.Figure, path: Path) -> Path | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        return path
    except Exception as exc:
        warnings.warn(f"Failed to write {path}: {exc}")
        return None
    finally:
        plt.close(fig)


def _plot_marginal(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    col: str,
    out_path: Path,
    label: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    r = pd.to_numeric(real[col], errors="coerce").dropna()
    s = pd.to_numeric(synth[col], errors="coerce").dropna()
    if r.empty and s.empty:
        ax.text(0.5, 0.5, f"No numeric values for {label}", ha="center", va="center")
    else:
        bins = 36
        if len(r):
            ax.hist(
                r,
                bins=bins,
                density=True,
                alpha=0.55,
                color=_REAL_COLOR,
                label=f"real (n={len(r)})",
                edgecolor="white",
                linewidth=0.3,
            )
        if len(s):
            ax.hist(
                s,
                bins=bins,
                density=True,
                alpha=0.45,
                color=_SYNTH_COLOR,
                label=f"synth (n={len(s)})",
                edgecolor="white",
                linewidth=0.3,
            )
        ax.legend(frameon=True)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_title(f"Marginal histogram — {label}")
    return _safe_savefig(fig, out_path)


def _plot_corr_heatmap(
    frame: pd.DataFrame,
    cols: list[str],
    out_path: Path,
    title: str,
    vmin: float,
    vmax: float,
) -> Path | None:
    fig_w = max(7.0, 0.55 * len(cols) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.82))
    if not cols:
        ax.text(0.5, 0.5, "No shared numeric columns", ha="center", va="center")
        ax.set_axis_off()
    else:
        corr = frame[cols].corr(method="spearman", numeric_only=True)
        sns.heatmap(
            corr,
            ax=ax,
            cmap="vlag",
            vmin=vmin,
            vmax=vmax,
            center=0.0,
            square=True,
            cbar_kws={"shrink": 0.72, "label": "Spearman ρ"},
            linewidths=0.2,
            linecolor="white",
        )
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.set_title(title)
    return _safe_savefig(fig, out_path)


def _plot_scatter(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    y_col: str,
    risk_col: str,
    out_path: Path,
) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
    pairs = (
        (axes[0], real, "real", _REAL_COLOR),
        (axes[1], synth, "synth", _SYNTH_COLOR),
    )
    for ax, frame, title, color in pairs:
        x = pd.to_numeric(frame[risk_col], errors="coerce")
        y = pd.to_numeric(frame[y_col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "No paired values", ha="center", va="center")
        else:
            ax.scatter(
                x[mask],
                y[mask],
                s=12,
                alpha=0.35,
                c=color,
                edgecolors="none",
                rasterized=True,
            )
        ax.set_title(f"{title}: y vs risk_score")
        ax.set_xlabel(risk_col)
        ax.set_ylabel(y_col)
    fig.suptitle("Scatter — expected loss (y) vs risk_score", y=1.02, fontsize=13)
    return _safe_savefig(fig, out_path)


def save_comparison_plots(real, synth, outdir) -> list[Path]:
    """Write the six comparison PNGs and return the paths that succeeded.

    Parameters
    ----------
    real, synth:
        DataFrame-like tables. Missing columns are skipped rather than
        raising, so the rest of the pipeline can continue.
    outdir:
        Directory for PNG output (created if needed).
    """
    written: list[Path] = []
    try:
        out_dir = Path(outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        real_df = _as_frame(real, "real")
        synth_df = _as_frame(synth, "synth")
        if real_df.empty or synth_df.empty:
            warnings.warn("save_comparison_plots: real or synth table is empty; skipping.")
            return written

        for label, aliases, filename in MARGINAL_SPECS:
            col_r = _find_col(real_df, aliases)
            col_s = _find_col(synth_df, aliases)
            dest = out_dir / filename
            if col_r is None or col_s is None or col_r != col_s:
                # still plot if the same logical column exists under one shared name
                col = col_r or col_s
                if col is None or col not in real_df.columns or col not in synth_df.columns:
                    warnings.warn(f"Skipping marginal {label}: column not found.")
                    continue
            else:
                col = col_r
            path = _plot_marginal(real_df, synth_df, col, dest, label)
            if path is not None:
                written.append(path)

        cols = _numeric_overlap(real_df, synth_df)
        vmin, vmax = -1.0, 1.0
        path = _plot_corr_heatmap(
            real_df,
            cols,
            out_dir / "corr_heatmap_real.png",
            "Correlation heatmap — real",
            vmin,
            vmax,
        )
        if path is not None:
            written.append(path)
        path = _plot_corr_heatmap(
            synth_df,
            cols,
            out_dir / "corr_heatmap_synth.png",
            "Correlation heatmap — synth",
            vmin,
            vmax,
        )
        if path is not None:
            written.append(path)

        y_col = _find_col(real_df, ("y", "Y", "target", "expected_loss"))
        risk_col = _find_col(real_df, ("risk_score", "riskScore", "risk"))
        y_s = _find_col(synth_df, ("y", "Y", "target", "expected_loss"))
        risk_s = _find_col(synth_df, ("risk_score", "riskScore", "risk"))
        if (
            y_col
            and risk_col
            and y_s
            and risk_s
            and y_col in synth_df.columns
            and risk_col in synth_df.columns
        ):
            path = _plot_scatter(
                real_df,
                synth_df,
                y_col,
                risk_col,
                out_dir / "scatter_y_vs_risk_score.png",
            )
            if path is not None:
                written.append(path)
        else:
            warnings.warn("Skipping scatter y vs risk_score: required columns missing.")
    except Exception as exc:
        # Last-resort guard: never take down evaluate / report.
        warnings.warn(f"save_comparison_plots aborted without crashing: {exc}")
    return written


__all__ = ["PLOT_NAMES", "save_comparison_plots"]
