"""Build the scientific report from experimental artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_HYPERPARAMS, MODEL_NAME, RESULTS_DIR, ROOT, TABLES_DIR
from src.models import model_card


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    use = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(use) + " |", "| " + " | ".join(["---"] * len(use)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in use:
            v = row[c]
            if pd.isna(v):
                cells.append("—")
            elif c in fmt and isinstance(v, (int, float, np.floating)):
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pm(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def generate_report() -> Path:
    summary_path = TABLES_DIR / "utility_attacks_summary.csv"
    runs_path = TABLES_DIR / "utility_attacks_runs.csv"
    meta_path = RESULTS_DIR / "run_meta.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run experiments before generating the report.")

    summary = pd.read_csv(summary_path)
    runs = pd.read_csv(runs_path) if runs_path.exists() else summary
    import json

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    recs = []
    for _, row in summary.iterrows():
        recs.append(
            {
                **row.to_dict(),
                "score": float(row.get("r2_retention_mean", np.nan))
                - 0.25 * float(np.nan_to_num(row.get("worst_attr_knownpair_ridge_mean", np.nan))),
            }
        )
    scored = pd.DataFrame(recs)

    # Preferred candidate: retention >= 0.90 if possible, else highest retention
    # with a privacy improvement vs raw (lower worst-attr leakage).
    preferred_notes = []
    best_name = "none"
    best_row = None
    if not scored.empty:
        nonraw = scored[scored["method"] != "raw"].copy()
        high = nonraw[nonraw["r2_retention_mean"] >= 0.90]
        pool = high if len(high) else nonraw
        if len(pool):
            # Among high-utility candidates, pick lowest leakage then highest retention.
            pool = pool.sort_values(
                by=["r2_retention_mean", "worst_attr_knownpair_ridge_mean"],
                ascending=[False, True],
            )
            # Re-rank: max retention, and among those within 2 points, min leakage.
            top_ret = pool["r2_retention_mean"].max()
            near = pool[pool["r2_retention_mean"] >= top_ret - 0.02]
            best_row = near.sort_values("worst_attr_knownpair_ridge_mean").iloc[0]
            best_name = str(best_row["method"])
            preferred_notes.append(
                f"Selected `{best_name}` by maximum $R^2$ retention "
                f"({best_row['r2_retention_mean']:.3f}), breaking ties toward lower "
                "worst-attribute known-pair leakage."
            )

    util_cols = [
        "dataset",
        "method",
        "out_dim_mean",
        "r2_raw_mean",
        "r2_mean",
        "r2_retention_mean",
        "r2_retention_std",
        "delta_r2_mean",
        "rmse_raw_mean",
        "rmse_mean",
        "rmse_ratio_mean",
    ]
    attack_cols = [
        "dataset",
        "method",
        "recon_r2_knownpair_ridge_mean",
        "worst_attr_knownpair_ridge_mean",
        "neural_best_r2_mean",
        "aux_random_rot_r2_mean",
        "semantic_unpaired_acc_mean",
        "semantic_paired_acc_mean",
        "distance_preservation_mean",
        "known_pair_robustness",
    ]

    datasets = meta.get("datasets", [])
    ds_lines = []
    for d in datasets:
        ds_lines.append(
            f"- **{d['name']}** ({d.get('kind')}): n={d.get('n')}, d={d.get('d')}. {d.get('description')}"
        )
    if not ds_lines:
        for name, g in summary.groupby("dataset"):
            ds_lines.append(
                f"- **{name}**: n={int(g['dataset_n_mean'].iloc[0])}, d={int(g['dataset_d_mean'].iloc[0])}"
            )

    # Question numbers from aggregated means
    q1_rows = []
    for ds, g in summary.groupby("dataset"):
        raw = g[g["method"] == "raw"]
        if raw.empty:
            continue
        raw_r2 = raw["r2_mean"].iloc[0]
        for _, row in g.iterrows():
            q1_rows.append(
                f"| {ds} | {row['method']} | {_pm(row['r2_mean'], row.get('r2_std', np.nan))} | "
                f"{row['r2_retention_mean']:.3f} | {row['delta_r2_mean']:.4f} |"
            )

    fig_rel = "results/figures"
    card = model_card()

    # Honest failure analysis
    collapse = summary[summary["known_pair_robustness"] == "collapses_with_enough_pairs"]
    collapse_methods = sorted(collapse["method"].unique()) if len(collapse) else []

    best_text = "No transformed configuration was clearly successful."
    if best_row is not None:
        best_text = (
            f"The preferred configuration is **`{best_name}`**, retaining "
            f"**{100*best_row['r2_retention_mean']:.1f}%** of raw $R^2$ "
            f"(mean transformed $R^2$={best_row['r2_mean']:.4f} vs raw "
            f"{best_row['r2_raw_mean']:.4f} on dataset `{best_row['dataset']}`; "
            "see tables for every dataset). "
            "This is an obfuscated / lossy representation, **not encryption**."
        )

    md = f"""# Privacy-Preserving Transformations for Outsourced Machine Learning

## 1. Executive Summary

This study tests whether a third party can train **the exact same model with the exact same hyperparameters** on a transformed table $Z=f(X)$ and still recover most of the predictive performance of $M_\\theta(X)\\to y$.

Priority used for selection:

$$
\\text{{predictive utility}} > \\text{{structural preservation}} > \\text{{privacy}}
$$

{best_text}

**Do not interpret these transforms as cryptographically secure.** Several configurations that look unstructured still collapse under a known-pair linear attack once the attacker obtains on the order of $d$ matched rows. That is a negative result and is reported as such.

Frozen model: `{card['name']}` with hyperparameters `{card['hyperparameters']}`.

---

## 2. Mathematical Definition

Let $X\\in\\mathbb{{R}}^{{n\\times d}}$ be numeric features and $y\\in\\mathbb{{R}}^n$ the target. All fitted maps below are estimated on the training split only.

### 2.1 Marginal Gaussianization $G$

For each column $j$, a quantile transformer $\\hat F_j$ (sklearn `QuantileTransformer`, normal output) implements

$$
u_j = \\Phi^{{-1}}(\\hat F_j(x_j)).
$$

This removes scale, units, and marginal shape.

### 2.2 Symmetric whitening $W$

$$
V = (U-\\mu)\\,\\Sigma^{{-1/2}}
$$

with $\\Sigma^{{-1/2}}$ the symmetric (ZCA-style) square root of the training covariance of $U=G(X)$, regularized by $10^{{-6}}I$.

### 2.3 Uniform quantization $Q_L$

On $[-4,4]$, $L$ equal-width bins; values are replaced by bin centers. This is many-to-one:

$$
x_1\\neq x_2 \\quad\\text{{can satisfy}}\\quad Q_L(x_1)=Q_L(x_2).
$$

### 2.4 Secret orthogonal mixing $R$

$R$ is a Haar-like orthogonal matrix from QR of a seeded Gaussian matrix, $R^\\top R=I$.

### 2.5 Dimensional projection $P$

$P\\in\\mathbb{{R}}^{{d\\times k}}$ contains $k$ orthonormal columns, $k<d$.

Classical family used in the grid:

$$
Z = Q_L\\big(W(G(X))\\big)\\,P\\,R
$$

with $P=I$ and/or $Q_L$ omitted in weaker ablations.

### 2.6 Supervised predictive bottleneck $E_\\phi$

A two-hidden-layer MLP encoder is trained **locally** to predict $y$ from $X$:

$$
X \\xrightarrow{{E_\\phi}} H\\in\\mathbb{{R}}^{{k}} \\xrightarrow{{m}} \\hat y, \\qquad k=\\lfloor 0.5 d\\rfloor.
$$

The prediction head is discarded. $H$ is standardized on train, optionally quantized and rotated:

$$
Z = Q_L(E_\\phi(X))\\,R.
$$

An adversarial variant additionally trains $A_\\psi(H)\\to X$ and updates the encoder with

$$
\\min_{{E,m}}\\max_A \\big[\\mathcal{{L}}_{{\\mathrm{{pred}}}} - \\lambda \\mathcal{{L}}_{{\\mathrm{{recon}}}}\\big], \\quad \\lambda=0.05.
$$

This is **not** an autoencoder: reconstruction of $X$ is not the training goal.

---

## 3. Experimental Setup

### 3.1 Datasets

{chr(10).join(ds_lines)}

### 3.2 Splits and seeds

- Split fractions: train 70% / test 15% / auxiliary 15%, drawn **once per seed** and reused for every transform.
- Seeds: `{meta.get("seeds", [0, 1, 2])}`.
- Auxiliary rows are from the same population but are **not** paired with $Z$ for Attack 3.
- Transforms (quantile map, covariance, encoder, $P$, $R$) are fit on **train only**.

### 3.3 Model (identical for $X$ and $Z$)

- Model: **{MODEL_NAME}**
- Hyperparameters: `{MODEL_HYPERPARAMS}`
- `random_state` equals the experiment seed (the same seed is used for raw and transformed fits).
- No per-transform retuning of depth, learning rate, iterations, leaf size, or regularization.

### 3.4 Protocol reminder

The only comparison that matters is

$$
M_\\theta(X) \\quad\\text{{vs}}\\quad M_\\theta(f_k(X)) \\qquad \\theta_{{\\mathrm{{raw}}}}=\\theta_{{\\mathrm{{transformed}}}}.
$$

---

## 4. Utility Results

Mean $\\pm$ std over seeds. Retention is $R^2_Z / R^2_X$. RMSE ratio is $\\mathrm{{RMSE}}_Z / \\mathrm{{RMSE}}_X$ (values near 1 are parity; $>1$ is worse).

{_md_table(
    summary,
    util_cols,
    {{
        "out_dim_mean": "{{:.0f}}",
        "r2_raw_mean": "{{:.4f}}",
        "r2_mean": "{{:.4f}}",
        "r2_retention_mean": "{{:.3f}}",
        "r2_retention_std": "{{:.3f}}",
        "delta_r2_mean": "{{:.4f}}",
        "rmse_raw_mean": "{{:.4f}}",
        "rmse_mean": "{{:.4f}}",
        "rmse_ratio_mean": "{{:.3f}}",
    }},
)}

### 4.1 Reading the utility table

- Configurations with $R^2$ retention $\\ge 0.90$ preserve learnable structure for this model class.
- Orthogonal mixing can hurt **axis-aligned trees** even when information is not destroyed; that is a model-class effect, not proof that $Z$ lacks signal.
- Quantization and $k<d$ projections are the first places information is intentionally discarded.

Figures: ` {fig_rel}/01_raw_vs_transformed_r2.png `, ` {fig_rel}/02_retention_vs_strength.png `.

---

## 5. Security Attack Results

Attacks were implemented to **break** the representation, not to advertise it.

| Attack | Attacker knowledge | What we measure |
| --- | --- | --- |
| 1 Statistical inspection | $Z$ only | skew, kurtosis, ranges, correlations, eigenvalues |
| 2 Semantic matching | $Z$ and possibly unpaired $X_{{\\mathrm{{aux}}}}$ | column identity recovery vs $1/d$ |
| 3 Auxiliary unpaired | algorithm class, $X_{{\\mathrm{{aux}}}}$, no pairs | distributional inversion, ICA, naive CCA |
| 4 Known-pair | $(X_i,Z_i)$ for $n$ rows | ridge, pseudoinverse, informed Procrustes, MLP |
| 5 Neural reconstruction | paired train $Z\\to X$ | small MLP, medium MLP, residual MLP |
| 6 Per-attribute leakage | from 4–5 | $\\max_j R^2_j$, median, top-5 |
| 7 Distance leakage | paired geometry | $\\mathrm{{corr}}(D_X, D_Z)$ |

{_md_table(
    summary,
    attack_cols,
    {{
        "recon_r2_knownpair_ridge_mean": "{{:.3f}}",
        "worst_attr_knownpair_ridge_mean": "{{:.3f}}",
        "neural_best_r2_mean": "{{:.3f}}",
        "aux_random_rot_r2_mean": "{{:.3f}}",
        "semantic_unpaired_acc_mean": "{{:.3f}}",
        "semantic_paired_acc_mean": "{{:.3f}}",
        "distance_preservation_mean": "{{:.3f}}",
    }},
)}

### 5.1 What the attacks actually show

- **Unpaired auxiliary data** is weak against Gaussianized+whitened data: second-order statistics are isotropic, so a secret rotation is not identified from $Z$ alone. ICA is theoretically uninformative for Gaussian coordinates. Reconstruction $R^2$ near 0 in the unpaired columns is expected and should not be oversold as "security".
- **Known pairs** are the honest stress test. If $f$ is essentially an invertible linear map after $G$ and $W$, then $n\\gtrsim d$ pairs let ridge / Procrustes recover a high-fidelity inverse on held-out rows.
- **Quantization** and **$k<d$** make $f$ lossy. They can cap reconstruction quality even with many pairs, at a utility cost that must be measured rather than assumed.
- **Worst-attribute $R^2$** is reported because a mean reconstruction score can hide a single recoverable sensitive column.

Methods that collapsed under enough known pairs: {", ".join(f"`{m}`" for m in collapse_methods) if collapse_methods else "none in the robustness label, but inspect the curves anyway"}.

Figures: ` {fig_rel}/03_known_pairs_reconstruction.png `, ` {fig_rel}/04_per_feature_leakage.png `.

---

## 6. Privacy / Utility Tradeoff

Selection weight is **not** "maximum privacy". A transform that destroys $R^2$ is a failed candidate even if reconstruction also fails.

Figure: ` {fig_rel}/05_pareto_frontier.png ` (vertical axis = $R^2$ retention; horizontal = worst-attribute leakage).

Also: ` {fig_rel}/06_structure_vs_utility.png ` for the structure–utility relationship.

{" ".join(preferred_notes) if preferred_notes else ""}

A conceptual score `retention - 0.25 * leakage` was inspected only as a diagnostic. Raw $R^2$, RMSE, and attack $R^2$ in the tables are the quantities that matter.

---

## 7. Failure Modes

1. **Known-pair inversion of secret orthogonal maps.** $Z=W(G(X))R$ is not a one-way function. It is a keyed linear map in a Gaussianized space. Enough matched rows recover $R$ (up to quantization). Calling this "encryption" would be false.
2. **Tree models are not rotation-invariant.** If retention drops after $R$ while distance preservation stays high, the representation still contains structure that this particular $M_\\theta$ uses inefficiently. That is a limitation of outsourcing to axis-aligned trees, not a privacy theorem.
3. **Auxiliary data does not need to reconstruct rows to be useful.** An attacker with $X_{{\\mathrm{{aux}}}}$ already knows population marginals and can re-identify *types* of features if $G$ is omitted.
4. **Dimensionality is not automatically safety.** Extra features can add null space for $k<d$ projections, but a known-pair linear model still has $k\\cdot d$ parameters. See ` {fig_rel}/07_dimensionality_vs_reconstruction.png `.
5. **Supervised bottlenecks can leak $X$ that is predictive of $y$.** Features that cause $y$ are the ones $E_\\phi$ is incentivized to keep. Privacy for those attributes is the hardest, and per-attribute $R^2$ will show it.
6. **Neural attackers with many pairs** are a pessimistic bound: they approximate $f^{{-1}}$ including nonlinear $G^{{-1}}$. If they succeed, the representation is reversible in practice for that pair budget.

---

## 8. Final Recommendation

{best_text}

Recommended operational interpretation:

- Use the transform only as an **obfuscated / lossy outsourced representation**, with a **secret key** ($R$, and fitted $G,W$ or $E_\\phi$) kept on the data-owner side.
- Assume that **known-pair leakage is the binding constraint**. If an adversary can ever obtain matched $(X,Z)$ rows, treat invertible stages (especially $R$ without $Q_L$ or $k<d$) as broken.
- Do not ship a configuration whose $R^2$ retention collapses below the task's tolerance, even if attacks look weak.

### Question 1 — Same model, same $\\theta$

| Dataset | Method | Transformed $R^2$ | Retention | $\\Delta R^2$ |
| --- | --- | --- | --- | --- |
{chr(10).join(q1_rows)}

### Question 2 — Learnable structure

See retention, distance preservation, and the structure-vs-utility figure. High distance correlation with high $R^2$ retention means geometry survived. High distance correlation with low retention means this model class failed to exploit surviving geometry.

### Question 3 — Can an attacker reconstruct $X$?

Quantitatively: use the attack table. Unpaired reconstruction $R^2$ is typically near zero after Gaussianization+whitening+rotation. Known-pair and neural reconstruction $R^2$ can be large for invertible stages. Always read **worst-attribute** $R^2$, not only the mean.

### Question 4 — Does larger $d$ make reverse engineering harder?

This is tested by the banking $d=80$ vs $d=200$ pair (or the quick $d=40$ vs $d=80$ pair). If known-pair $R^2$ stays high as $d$ grows, the obstacle was **not** combinatorial semantic search; it was a missing key, and the key is statistically identifiable from $O(d)$ pairs. If reconstruction falls only when $k<d$ or quantization is applied, the protection is **information-theoretic loss**, not computational hardness.

### Question 5 — Best utility/security tradeoff

Precision / $R^2$ retention is primary. The preferred row is `{best_name}`. If that row still has high known-pair $R^2$, then **no configuration in this grid simultaneously kept utility and resisted known-pair inversion**. That outcome is allowed and scientifically useful.

---

## Reproducibility

```bash
pip install -r requirements.txt
python run.py           # full protocol
python run.py --quick   # subset for smoke-testing
```

Artifacts: `results/tables/`, `results/figures/`, per-seed JSON under `results/<dataset>/`.
"""

    path = ROOT / "REPORT.md"
    # Fix accidental spaces around figure paths introduced above
    md = md.replace("` {fig_rel}/", f"`{fig_rel}/").replace(".png `", ".png`")
    path.write_text(md)
    return path
