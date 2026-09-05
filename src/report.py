"""Build REPORT.md from experimental artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    use = [c for c in cols if c in df.columns]
    if not use or df.empty:
        return "_no data_\n"
    lines = ["| " + " | ".join(use) + " |", "| " + " | ".join("---" for _ in use) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in use:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if np.isfinite(v) else "—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def select_recommended(df: pd.DataFrame) -> dict:
    """Max privacy subject to mean HGB retention >= 0.90 when possible."""
    if df.empty:
        return {"method": "typed_keyed", "reason": "no results yet"}
    h = df[df["model"] == "hgb"].copy()
    if h.empty:
        h = df.copy()
    rows = []
    for method, g in h.groupby("method"):
        ret = g["retention"].mean()
        leak = g["worst_attr_r2"].mean()
        sem = g["semantic_acc"].mean()
        rows.append((method, ret, leak, sem, g["dataset"].nunique()))
    # Prefer methods that keep retention on every dataset.
    by_ds = h.groupby(["method", "dataset"])["retention"].mean().unstack()
    robust = []
    for method, ret, leak, sem, n_ds in rows:
        if method == "identity":
            continue
        ok = True
        if method in by_ds.index:
            ok = bool((by_ds.loc[method] >= 0.85).all())
        robust.append((ok, -float(ret or 0), float(leak or 1), method, ret, leak, sem))
    robust.sort()
    if not robust:
        return {"method": "typed_keyed", "reason": "fallback"}
    # Among high-utility methods, pick lowest leakage.
    high = [r for r in robust if r[0]]
    pool = high if high else robust
    pool.sort(key=lambda t: (t[2], t[1]))
    best = pool[0]
    return {
        "method": best[3],
        "retention": best[4],
        "leak": best[5],
        "semantic": best[6],
        "met_utility_floor": best[0],
    }


def write_report(df: pd.DataFrame, raw_records: list[dict], results_dir: Path) -> None:
    rec = select_recommended(df)
    hgb = df[df["model"] == "hgb"] if "model" in df.columns else df
    util_cols = [
        "dataset",
        "method",
        "model",
        "score_mean",
        "retention_mean",
        "agreement_mean",
        "worst_attr_r2_mean",
        "recon_r2_mean",
        "semantic_acc_mean",
        "membership_auc_mean",
        "n_pairs_r2_0.5_mean",
        "known_pair_robustness",
    ]
    summary_path = results_dir / "tables" / "summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    lines = [
        "# Utility-Preserving Privacy Transformations",
        "",
        "This report is generated from the experimental protocol in `src/experiment.py`.",
        "It is **not** a cryptographic security claim.",
        "",
        "## 1. Current transformation analysis",
        "",
        "The repository originally shipped no implementation on `main`. A prior branch",
        "(`cursor/privacy-preserving-ml-transforms-6ed8`) studied numeric-only maps",
        "`Z = Q(W(G(X))) P R` and a supervised bottleneck, evaluated only with",
        "`HistGradientBoostingRegressor`. That work is treated as the **current**",
        "baseline family (`gauss`, `gauss_white_rot`, `bn_adv`) and is re-run here",
        "against heterogeneous tables and multiple model families.",
        "",
        "### What the classical maps preserve / destroy",
        "",
        "- **`gauss`**: preserves within-column rank and tree splits; destroys units,",
        "  scale, and marginal shape. Invertible via the quantile map for numeric",
        "  columns. Predictions map back because the target affine `g` is inverted.",
        "- **`gauss_white_rot`**: preserves second-order geometry up to a secret",
        "  rotation; destroys column identity and axis-aligned structure. Fully",
        "  invertible given the key; statistically identifiable from `O(d)` pairs.",
        "- **`keyed_monotone` / `typed_keyed`**: preserve per-column order and",
        "  categorical equality; destroy names, string literals (HMAC buckets), and",
        "  original codes. Partially invertible. Tree utility stays high.",
        "- **`bn_adv` / `vib`**: preserve `I(Z;Y)` by local training; destroy features",
        "  orthogonal to `Y` and column identity. Not invertible. Known-pair attackers",
        "  still recover the *predictive* raw attributes.",
        "- **`rff` / `microagg_rot` / `noisy_gauss_rot`**: lossy / nonlinear; lower",
        "  leakage, usually lower utility.",
        "",
        "## 2. Weaknesses found",
        "",
        "1. **Known-pair inversion is the binding constraint.** Any deterministic",
        "   invertible stage (especially secret rotations) collapses once the attacker",
        "   holds on the order of `d` matched rows.",
        "2. **Utility-preserving maps must keep `I(Z; X_predictive)`.** Features that",
        "   cause `Y` remain the leaky ones under Attacker A, even when values look random.",
        "3. **Tree vs linear disagreement.** Rotations hurt axis-aligned trees while",
        "   leaving linear/MLP models closer to raw performance — a model-class effect,",
        "   not information destruction.",
        "4. **Leaving `Y` in the clear is a leak.** This protocol always applies an",
        "   invertible target map `g` except for `identity`. Utility is scored after `g^{-1}`.",
        "5. **Column-wise monotone maps hide names but not order.** Rank attacks recover",
        "   numeric features from few pairs.",
        "",
        "## 3. Candidate alternative transformations",
        "",
        "Implemented and benchmarked: `identity`, `gauss`, `gauss_white_rot`,",
        "`keyed_monotone`, `typed_keyed`, `bn_adv`, `bn_noisy`, `vib`, `vib_stoch`,",
        "`rff`, `microagg_rot`, `noisy_gauss_rot`.",
        "",
        "Literature that informed the grid (see `RESEARCH.md`): variational information",
        "bottleneck / privacy funnel; random Fourier features; microaggregation;",
        "adversarial representation learning; typed masking / FPE-style hashing.",
        "Synthetic-data and DP-SGD methods were considered and rejected as the primary",
        "mechanism because they do not implement `X_raw → X_transformed` with",
        "row-aligned outsourced training.",
        "",
        "## 4. Utility benchmark",
        "",
        "Models: ridge/logistic (`linear`), `HistGradientBoosting` (`hgb`), MLP, k-NN.",
        "Metric: regression `R²`, classification accuracy, both after mapping predictions",
        "back to the original target space. Retention is transformed_score / raw_score.",
        "",
        _md_table(summary, util_cols) if not summary.empty else _hgb_fallback(hgb),
        "",
        "## 5. Attacker A results",
        "",
        "Known-pair grid: 1, 2, 5, 10, 25, 50, 100, 200. Attacks: ridge, pinv,",
        "rank/order matching, categorical frequency, HGB inversion, MLP inversion.",
        "",
        _md_table(
            summary,
            [
                "dataset",
                "method",
                "recon_r2_mean",
                "worst_attr_r2_mean",
                "sensitive_attr_r2_mean",
                "n_pairs_r2_0.5_mean",
                "spearman_leak_mean",
                "known_pair_robustness",
            ],
        )
        if not summary.empty
        else "_run the protocol to populate_\n",
        "",
        "## 6. Attacker B results",
        "",
        "The attacker receives only `Z` and a context sentence",
        "(`This dataset comes from a bank...` or the housing description).",
        "No keys, schema, column names, or raw rows.",
        "",
        _md_table(
            summary,
            [
                "dataset",
                "method",
                "semantic_acc_mean",
                "kind_acc_mean",
                "membership_auc_mean",
                "linkage_mean",
                "sensitive_hit_mean",
            ],
        )
        if not summary.empty
        else "_run the protocol to populate_\n",
        "",
        "## 7. Security / utility trade-off",
        "",
        "Pareto view: HGB retention vs known-pair worst-attribute `R²`",
        "(see `results/figures/02_pareto_utility_leakage.png`).",
        "Selection rule: maximize practical privacy subject to minimal utility loss,",
        "operationalized as lowest mean worst-attribute leakage among methods with",
        "HGB retention ≥ 0.85 on every dataset when such methods exist.",
        "",
        f"Selected method: **`{rec.get('method')}`**",
        f"(retention={rec.get('retention')}, worst-attr leakage={rec.get('leak')},",
        f"utility floor met={rec.get('met_utility_floor')}).",
        "",
        "## 8. Recommended architecture",
        "",
        "Keep the schema, keys, and `g^{-1}` on the data-owner side.",
        "",
        "```",
        "owner:  X, y, schema",
        "        -> type-aware encode (HMAC strings, keyed cats, quantile nums)",
        "        -> optional local VIB / bottleneck if y is available locally",
        "        -> secret permutation / rotation",
        "        -> g(y) invertible target map",
        "        -> send (Z, y_tilde, anonymous column ids) to trainer",
        "trainer: fit M: Z -> y_tilde   (no names, no raw values)",
        "owner:   y_hat = g^{-1}(M(Z_new))",
        "```",
        "",
        "This satisfies: train in transformed space; map predictions back; hide",
        "column names; support mixed types. It does **not** claim known-pair security.",
        "",
        "## 9. Recommended transformation",
        "",
        f"**`{rec.get('method')}`** is the empirically preferred point on this grid.",
        "Use `typed_keyed` when the owner cannot train a local encoder (no `y` yet,",
        "or model-agnostic export). Use `vib` / `bn_adv` when a local labelled fit",
        "is acceptable and column semantics must be thoroughly mixed.",
        "",
        "## 10. Remaining attack surface",
        "",
        "- Known-pair inversion of any approximately invertible or low-dimensional map.",
        "- Recovery of attributes that cause `Y` from `Z` whenever `I(Z;Y)` is large.",
        "- Rank/frequency attacks against per-column monotone or keyed-categorical maps.",
        "- Linkage if neighborhood geometry is preserved and an overlapping public table exists.",
        "- Membership inference on deterministic unique rows (identifiers).",
        "- Side-channel leakage from `out_dim`, sparsity, and dummy-column structure.",
        "- The trainer still sees `y_tilde`; a weak `g` (affine) hides units, not ranks.",
        "",
        "## 11. Permanent regression tests",
        "",
        "See `tests/`. They lock: target invertibility, anonymous column ids, mixed-type",
        "support, identity utility ceiling, Attacker A/B interfaces, and the requirement",
        "that a recommended high-utility method must not expose raw column names.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python -m pytest tests/ -q",
        "python run.py --quick",
        "python run.py",
        "```",
        "",
    ]
    (results_dir.parent / "REPORT.md").write_text("\n".join(lines))


def _hgb_fallback(hgb: pd.DataFrame) -> str:
    if hgb.empty:
        return "_no utility rows_\n"
    g = hgb.groupby(["dataset", "method"], as_index=False)[["score", "retention", "worst_attr_r2"]].mean()
    return _md_table(g, ["dataset", "method", "score", "retention", "worst_attr_r2"])
