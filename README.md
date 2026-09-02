# Privacy-Preserving Transformations for Outsourced ML

Experimental study of representations $Z=f(X)$ such that the **same** tabular model $M_\theta$ can be trained on $Z$ instead of $X$, while making original feature values and column semantics harder to recover.

Priority:

**predictive utility > structural preservation > privacy**

This is **not** encryption. Several transforms that look unstructured still fail known-pair reconstruction attacks. Negative results are reported in [`REPORT.md`](REPORT.md).

## Headline result

With a frozen `HistGradientBoostingRegressor` (identical hyperparameters for $X$ and $Z`):

- A **locally trained supervised bottleneck** (`bn50_adv_rot`: $X\to H\to y$, then secret rotation of $H$) retains **~99–101%** of raw $R^2$ on superconduct ($d=81$), synthetic banking ($d=80$), and the $d=200$ stress test.
- Unpaired reconstruction fails and column-identity matching falls to chance.
- **Known-pair ridge / neural inversion still recovers the attributes that predict $y$** (worst-attribute $R^2 \approx 0.90$–$0.95$). Adversarial reconstruction training did not fix that.
- Invertible Gaussianize+whiten+rotate maps collapse once the attacker has $O(d)$ matched rows. Increasing $d$ from 80 to 200 does **not** make that attack fail.
- Random $k=d/2$ projections lower leakage but destroy tree utility on the synthetic tables (retention $45$–$56\%$).

No configuration in the grid kept $\ge 90\%$ $R^2$ on every dataset **and** drove known-pair worst-attribute $R^2$ below $0.7$.

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
| `src/models.py` | Frozen $M_\theta$ |
| `src/attacks.py` | Attack suite |
| `src/experiment.py` | Orchestration |
| `results/` | JSON, tables, figures |
| `REPORT.md` | Full scientific write-up |

## Datasets

- **A** OpenML superconduct, $n=21263$, $d=81$.
- **B** Synthetic banking-like data, $n=18000$, $d=80$, nonlinear $y$ of a few latents.
- **C** Same generative process at $d=200$.
