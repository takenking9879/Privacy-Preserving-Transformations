# Privacy-Preserving-Transformations

High-fidelity synthetic tabular data: build a complex original table `X` and a
synthetic counterpart `X'` that keeps both the structure of `X` and the map
`X → Y`, so a model trained on the synthetic table transfers to the original.

## Goal

Given original `(X, Y)` and synthetic `(X', Y')`:

1. Statistical patterns among the features of `X` survive in `X'`.
2. The relationship `P(Y | X)` is preserved.
3. **TSTR ≈ TRTR**: a model `M'` trained only on `(X', Y')` scores close to a
   model `M` trained on `(X, Y)` when both are evaluated on a real hold-out.
4. A frozen model trained on real data produces similar *score distributions*
   on `X` and on `X'`.

## What we try / skip

| Try | Why |
| --- | --- |
| Gaussian copula | Marginals + linear/rank correlations |
| Sequential CART (synthpop-style) | Interactions, mixed types |
| Conditional mixture | Multimodality / segment effects |
| **Hybrid (primary bet)** | Copula on `X` + residual-sampling `P(Y\|X)` |

Skip for now: from-scratch GANs / CTGAN / TVAE / diffusion (heavy, unstable,
likely no torch), i.i.d. noise, DP-Laplace on every cell.

See [PLAN.md](PLAN.md) for the full protocol and numeric gates.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m src.evaluate            # default n=1500
python -m src.evaluate --quick    # smoke, n≤200
python -m src.report
```

Artifacts:

- `data/original.csv`, `data/synthetic_<method>.csv`
- `artifacts/evaluation.json`
- `reports/RESULTS.md` and `reports/figures/`

## Success gates

- Best TSTR R² gap vs TRTR ≤ 0.08
- Winning synthesizer `fidelity_score` ≥ 0.70
- Negative control (column shuffle) fails those gates

## Latest run (`n=1200`, seed 42)

| method | fidelity | RF TSTR R² | RF TRTR R² | RF gap | gates |
| --- | --- | --- | --- | --- | --- |
| **cart** (winner) | 0.937 | 0.479 | 0.544 | **0.065** | pass |
| mixture | 0.932 | 0.226 | 0.544 | 0.319 | fail TSTR |
| hybrid | 0.889 | 0.262 | 0.544 | 0.282 | fail TSTR |
| copula | 0.854 | 0.185 | 0.544 | 0.360 | fail TSTR |
| negative control | 0.619 | −0.021 | 0.544 | 0.566 | fail (as required) |

CART also keeps classification: TSTR AUC gap ≈ 0.012 (gate ≤ 0.05).
Full write-up: [reports/RESULTS.md](reports/RESULTS.md).

## Multi-dataset / método general

Credit CART above is one DGP. The portable method is **`AutoSynthesizer`**:
it inspects a generic mixed-type table (no hardcoded credit-risk column
names), diagnoses margins / joints / `P(Y|X)`, and picks or blends
copula, sequential CART, conditional mixture, and hybrid so the same
`fit` / `sample` API travels across datasets.

Ten original DGPs in `src.datasets` stress different failure modes:

| DGP | What it stresses |
| --- | --- |
| `credit` | Mixed types, latent confounder, rare distressed cluster, interactions, hockey-stick `Y`, heteroscedasticity, counts, MAR missingness |
| `healthcare` | Clinical mix (labs, codes, binaries), informative missingness, skewed costs, rare adverse `y_class` |
| `retail` | High-cardinality categoricals, zero-inflated counts, promo / basket effects |
| `insurance` | Heavy-tailed claims, deductibles / thresholds, rare large losses |
| `interactions` | Multiplicative and threshold terms in `P(Y\|X)` — pairwise copulas lose TSTR |
| `multimodal` | Distinct segments / mixture modes — a global Gaussian smears clusters |
| `imbalanced` | Rare positive class; minority joints and TSTR AUC, not accuracy |
| `heavytail` | Pareto / \(t\) margins and tail dependence — Gaussian copula understates extremes |
| `panel` | Within-unit repeated measures — i.i.d. row synthesizers drop serial structure |
| `sparse_linear` | Few linear drivers — copula should suffice; trees can overfit noise |

Cross-dataset eval and report (same gates as the credit run: TSTR gap,
`fidelity_score`, negative control):

```bash
python -m src.eval_multidataset
python -m src.report_multidataset
```

`eval_multidataset` walks the registry, fits each synthesizer (including
`AutoSynthesizer`) per DGP, and writes a multi-dataset artifact.
`report_multidataset` turns that artifact into a cross-DGP leaderboard
so a method is judged by how often it travels, not by credit CART alone.

### Latest suite (`n=280`, seed 0)

A method is **general** only if it passes `interactions` **and** `multimodal`
**and** `imbalanced` (fidelity ≥ 0.70 and TSTR R² gap ≤ 0.10). Credit CART
does not get that seal.

| method | passes | general? |
| --- | --- | --- |
| **auto** | **7/10** | **yes** (the three veto DGPs + credit, insurance, heavytail, sparse_linear) |
| mixture | 6/10 | no (fails multimodal) |
| ensemble | 5/10 | no |
| copula | 5/10 | no (fails all three vetoes) |
| knn | 4/10 | no |
| cart | 3/10 | no (credit specialist) |
| hybrid | 3/10 | no |
| forest | 2/10 | no |
| negative control | 0/10 | — |

Still hard: `healthcare`, `retail`, `panel` — no portable method cleared all three
plus those. Use `AutoSynthesizer` as the default portable API; use CART only
when the table looks like the credit DGP.

Full matrix: [reports/MULTI_DATASET.md](reports/MULTI_DATASET.md) · plan:
[PLAN_GENERAL.md](PLAN_GENERAL.md).
