# Synthetic-data fidelity — RESULTS

_Generated 2026-09-16 18:36 UTC by `python -m src.report`._

**Overall:** ALL SCORED GATES PASSED

## Success gates

Fixed thresholds (do not drift these):

- Best TSTR R² gap vs TRTR **≤ 0.08** (regression; report RF and linear; gate on the better model).
- `fidelity_score` **≥ 0.70** for the winning synthesizer.
- Negative control: `fidelity_score` **< 0.55** **OR** `tstr_gap_r2` **> 0.15**.

| Gate | Status | Detail |
| --- | --- | --- |
| TSTR R² gap ≤ 0.08 (best of RF / linear) | PASS | cart: RF gap=0.0652; linear gap=0.0542; best gap=0.0542 (need ≤ 0.08) |
| fidelity_score ≥ 0.70 (winner) | PASS | cart: fidelity_score=0.9366 (need ≥ 0.70) |
| negative control is weak (fidelity < 0.55 or TSTR gap > 0.15) | PASS | negative_control: fidelity_score=0.6193 (pass if < 0.55) OR tstr_gap_r2=0.1963 (pass if > 0.15). Triggered via fidelity=False, gap=True. |

## Evaluation artifact

Loaded `/workspace/artifacts/evaluation.json` (6 top-level keys).

## Leaderboard

| synthesizer | fidelity | TSTR R² RF | TRTR R² RF | gap RF | TSTR R² linear | TRTR R² linear | gap linear | best gap | neg. ctrl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 1.0000 | 0.5825 | 0.5445 | -0.0380 | 0.1902 | 0.1886 | -0.0016 | -0.0380 | — |
| cart ← winner | 0.9366 | 0.4793 | 0.5445 | 0.0652 | 0.1344 | 0.1886 | 0.0542 | 0.0542 | — |
| mixture | 0.9315 | 0.2256 | 0.5445 | 0.3189 | 0.1502 | 0.1886 | 0.0384 | 0.0384 | — |
| hybrid | 0.8892 | 0.2624 | 0.5445 | 0.2821 | 0.1016 | 0.1886 | 0.0870 | 0.0870 | — |
| copula | 0.8535 | 0.1849 | 0.5445 | 0.3596 | 0.1191 | 0.1886 | 0.0695 | 0.0695 | — |
| negative_control | 0.6193 | -0.0212 | 0.5445 | 0.5657 | -0.0077 | 0.1886 | 0.1963 | 0.1963 | yes |

Gap = TRTR R² − TSTR R². The **best gap** is `min(gap_RF, gap_linear)` and is the number used by the 0.08 gate.

## Winner

**cart**

- fidelity_score = 0.9366 (gate ≥ 0.70)
- RF: TSTR=0.4793 TRTR=0.5445 gap=0.0652
- linear: TSTR=0.1344 TRTR=0.1886 gap=0.0542
- best-model gap = 0.0542 (gate ≤ 0.08)

## Negative control

**negative_control**

- fidelity_score = 0.6193
- tstr_gap_r2 (best model) = 0.1963
- protocol pass if fidelity < 0.55 or gap > 0.15

## Data tables

`data/original.csv`: 1200 rows × 14 columns (`age, income, tenure_months, region, segment, risk_score, num_products, is_premium, usage, engagement, complaint_count, credit_util…`).
`data/synthetic_cart.csv`: 840 rows × 14 columns.
`data/synthetic_copula.csv`: 840 rows × 14 columns.
`data/synthetic_hybrid.csv`: 840 rows × 14 columns.
`data/synthetic_identity.csv`: 840 rows × 14 columns.
`data/synthetic_mixture.csv`: 840 rows × 14 columns.
`data/synthetic_negative_control.csv`: 840 rows × 14 columns.

## Plots

Wrote:

- `/workspace/reports/figures/marginal_y.png`
- `/workspace/reports/figures/marginal_income.png`
- `/workspace/reports/figures/marginal_risk_score.png`
- `/workspace/reports/figures/corr_heatmap_real.png`
- `/workspace/reports/figures/corr_heatmap_synth.png`
- `/workspace/reports/figures/scatter_y_vs_risk_score.png`

## How to interpret (Jorge)

This is a **modeling-substitute** test, not a row-by-row disguise. There is no pairing
between real row `i` and synthetic row `i'`. Looking at a single customer and asking
"did we copy them?" is the wrong question.

1. **`fidelity_score` (0–1, higher is better).** Blend of 1-D margins, Spearman
   correlation structure, and whether the `X → y` relationship survived. The
   winner must be **≥ 0.70**. A pretty histogram is not enough if joints die.
2. **TSTR vs TRTR R² gap.** Train on synthetic, test on a real hold-out (TSTR).
   Compare to train-on-real / test-on-real (TRTR). We report **Random Forest and
   linear/Ridge**. The gate uses the **best** (smaller) of those two gaps and
   requires **≤ 0.08**. If RF gap is 0.05 and linear gap is 0.12, the gate still
   passes — the synthetic table is good enough for at least one frozen model
   family. If *both* gaps are large, `X'` lost the signal that predicts `y`.
3. **Negative control.** A column-wise shuffle (or similar junk generator) must
   look *bad*: `fidelity_score < 0.55` **or** TSTR gap `> 0.15`. If junk data
   passes, the metric is too soft and we should not trust a CART/copula "win".
4. **Plots.** Overlay histograms of `y`, `income`, `risk_score` should sit on
   top of each other. Heatmaps should keep the same red/blue blocks (especially
   `risk_score`–`credit_util`–`y`). The scatter should keep the hockey-stick:
   expected loss rising once `risk_score` is high.

**En corto:** si el winner pasa los tres gates, `X'` sirve para entrenar un
modelo y llevarlo a datos reales sin un desplome grande de R². Si el control
negativo *también* pasa, no celebres — las métricas están mal calibradas.
