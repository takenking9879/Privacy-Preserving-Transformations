# Multi-dataset synthesizer report

_Generated 2026-09-16 18:49 UTC by `python -m src.report_multidataset`._

Suite gates (same numbers as `src.eval_multidataset`):

- `fidelity_score` **≥ 0.70**
- `tstr_gap_r2` **≤ 0.10**
- A pair **passes** only if both gates hold.
- **General** = pass on `interactions` **and** `multimodal` **and** `imbalanced` (not just `credit`). Controls `identity` / `negative_control` are excluded from winners.

## Evaluation artifact

Loaded `/workspace/artifacts/multidataset.json` · n=280 · seed=0 · quick=False · 10 datasets × 10 methods · 100 pair rows.

## Per-dataset winner

| dataset | winner | combined | fidelity | tstr_gap_r2 | pass |
| --- | --- | --- | --- | --- | --- |
| credit | cart | 1.8514 | 0.8851 | -0.0098 | yes |
| healthcare | mixture | 1.7877 | 0.9269 | 0.1788 | no |
| retail | mixture | 1.7440 | 0.8654 | 0.1669 | no |
| insurance | mixture | 1.8277 | 0.8861 | -0.0678 | yes |
| interactions | auto | 1.8343 | 0.8949 | -0.0703 | yes |
| multimodal | auto | 1.8035 | 0.8712 | 0.0856 | yes |
| imbalanced | mixture | 1.8302 | 0.8770 | 0.0192 | yes |
| heavytail | hybrid | 1.6978 | 0.8068 | 0.0323 | yes |
| panel | ensemble | 1.7442 | 0.8448 | 0.1303 | no |
| sparse_linear | mixture | 1.7337 | 0.8093 | 0.0897 | yes |

Winner per DGP = non-control method with the highest `combined_score = fidelity_score + utility_score` (ties: gate pass, then smaller TSTR gap).

## Method × dataset combined scores

| method | credit | healthcare | retail | insurance | interactions | multimodal | imbalanced | heavytail | panel | sparse_linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 |
| copula | 1.6695 | 1.6840 | 1.6539 | 1.7588 | 1.3508 | 1.5608 | 1.3362 | 1.5720 | 1.5702 | 1.5950 |
| cart | 1.8514 | 1.6263 | 1.6400 | 1.7124 | 1.7883 | 1.7437 | 1.3750 | 1.6086 | 1.6254 | 1.6830 |
| mixture | 1.7840 | 1.7877 | 1.7440 | 1.8277 | 1.7413 | 1.6052 | 1.8302 | 1.6322 | 1.5868 | 1.7337 |
| hybrid | 1.8132 | 1.6878 | 1.5637 | 1.7529 | 1.6392 | 1.3884 | 1.3660 | 1.6978 | 1.5506 | 1.6654 |
| forest | 1.5245 | 1.4192 | 1.3535 | 1.2467 | 1.5701 | 1.5274 | 1.6034 | 1.4027 | 1.1405 | 1.3512 |
| knn | 1.8070 | 1.5690 | 1.7306 | 1.7907 | 1.7464 | 1.7617 | 1.7064 | 1.4794 | 1.7401 | 1.6204 |
| auto | 1.6075 | 1.7021 | 1.6801 | 1.8162 | 1.8343 | 1.8035 | 1.6131 | 1.6002 | 1.7216 | 1.7066 |
| ensemble | 1.8345 | 1.7176 | 1.3885 | 1.7997 | 1.7548 | 1.7395 | 1.7373 | 1.2225 | 1.7442 | 1.6906 |
| negative_control | 1.3138 | 1.0140 | 1.1261 | 1.4939 | 1.4972 | 1.5902 | 1.4506 | 1.6528 | 1.1051 | 1.2430 |

`combined_score = fidelity_score + utility_score` (higher is better). Em dash = missing / error pair.

## Median combined score per method

| method | median_combined | n_scored | control |
| --- | --- | --- | --- |
| mixture | 1.7426 | 10 | — |
| ensemble | 1.7384 | 10 | — |
| knn | 1.7353 | 10 | — |
| auto | 1.7043 | 10 | — |
| cart | 1.6615 | 10 | — |
| hybrid | 1.6523 | 10 | — |
| copula | 1.5835 | 10 | — |
| forest | 1.4109 | 10 | — |
| identity | 2.0000 | 10 | yes |
| negative_control | 1.3822 | 10 | yes |

Median is over datasets with a finite `combined_score`. Controls are listed last and never win the suite.

## Pass-count per method

A pair passes when `fidelity_score ≥ 0.70` **and** `tstr_gap_r2 ≤ 0.10`.

| method | pass-count | datasets passed | interactions | multimodal | imbalanced | general? |
| --- | --- | --- | --- | --- | --- | --- |
| auto | 7/10 | credit, insurance, interactions, multimodal, imbalanced, heavytail, sparse_linear | yes | yes | yes | yes |
| mixture | 6/10 | credit, insurance, interactions, imbalanced, heavytail, sparse_linear | yes | no | yes | no |
| ensemble | 5/10 | credit, retail, multimodal, imbalanced, heavytail | no | yes | yes | no |
| copula | 5/10 | credit, healthcare, retail, heavytail, sparse_linear | no | no | no | no |
| knn | 4/10 | credit, interactions, heavytail, panel | yes | no | no | no |
| cart | 3/10 | credit, multimodal, imbalanced | no | yes | yes | no |
| hybrid | 3/10 | credit, interactions, heavytail | yes | no | no | no |
| forest | 2/10 | imbalanced, heavytail | no | no | yes | no |
| identity | 10/10 | credit, healthcare, retail, insurance, interactions, multimodal, imbalanced, heavytail, panel, sparse_linear | yes | yes | yes | no |
| negative_control | 0/10 | — | no | no | no | no |

`general?` is yes only when the method passed **all three** veto DGPs (`interactions`, `multimodal`, `imbalanced`). Missing veto columns stay em dash and block the seal.

## General winner

**auto**

- median combined_score = 1.7043
- pass-count = 7/10
- passed datasets: credit, insurance, interactions, multimodal, imbalanced, heavytail, sparse_linear
- Passes interactions AND multimodal AND imbalanced (not just credit); highest median combined_score among qualifying methods.

## Note for Jorge / Nota para Jorge

**EN.** A method is *general* only if it **passes on `interactions` AND `multimodal` AND `imbalanced`**.
A win on `credit` (CART looking strong on the legacy DGP) is a local result, not transfer.
Do not call the suite winner “general” because it topped the credit column or had a high median
on additive / near-elliptical tables. Those three DGPs are the veto: product / threshold
structure, mixture modes, and a rare class. Fail any one of them and the method is a specialist,
not a portable synthesizer.

**ES.** Un método es *general* solo si **pasa en `interactions` Y `multimodal` Y `imbalanced`**.
Ganar en `credit` no basta: es una victoria local, no transferencia.
No declares ganador general a quien solo brilla en crédito o en tablas casi aditivas.
Esos tres DGPs son el veto (interacciones, modos, clase rara). Si falla uno, es especialista,
no un sintetizador portable.
