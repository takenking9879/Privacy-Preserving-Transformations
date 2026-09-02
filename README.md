# Privacy-Preserving Transformations for Outsourced ML

Experimental study of representations $Z=f(X)$ such that the **same** tabular model $M_\\theta$ can be trained on $Z$ instead of $X$, while making original feature values and column semantics harder to recover.

Priority:

**predictive utility > structural preservation > privacy**

This is **not** encryption. Several transforms that look unstructured still fail known-pair reconstruction attacks. Negative results are reported in `REPORT.md`.

## Protocol (non-negotiable)

- One train/test/aux split per seed, reused for every transform.
- Transforms fit on train only.
- Frozen model: `HistGradientBoostingRegressor` with a single hyperparameter set for raw and transformed data.
- Utility retention $= R^2_Z / R^2_X$.
- Attacks: statistical inspection, semantic matching, unpaired auxiliary inversion, known-pair (ridge / pinv / Procrustes / MLP), neural reconstruction, per-attribute leakage, pairwise distance preservation.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python run.py --quick    # smoke
python run.py            # full grid, 3 seeds, 3 datasets
```

## Layout

| Path | Role |
| --- | --- |
| `src/transforms.py` | $G$, $W$, $Q_L$, $P$, $R$, supervised bottleneck |
| `src/models.py` | Frozen $M_\\theta$ |
| `src/attacks.py` | Attack suite |
| `src/experiment.py` | Orchestration |
| `results/` | JSON, tables, figures |
| `REPORT.md` | Full scientific write-up |

## Datasets

- **A** Real tabular regression (OpenML superconduct or ailerons; California housing + derived census features if OpenML is unavailable).
- **B** Synthetic banking-like data, $d=80$, nonlinear $y$ of a few latents.
- **C** Same generative process at $d=200$.
