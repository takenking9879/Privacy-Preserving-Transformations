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
