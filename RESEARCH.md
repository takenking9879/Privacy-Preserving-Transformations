# Research notes

The goal is unusual: a third party should **train in a transformed space**
`Z = f(X)` and still produce predictions that can be mapped back to the
original problem space, without being shown column names or raw values.

This is **not** the same problem as encryption, synthetic data, or DP-SGD.

## Repository first

`main` contained only a README/LICENSE. A prior agent branch
`cursor/privacy-preserving-ml-transforms-6ed8` implemented numeric maps

```
Z = Q_L(W(G(X))) P R
```

and a deterministic supervised bottleneck `X → H → y`, then secret-rotated `H`.
It evaluated a single frozen `HistGradientBoostingRegressor`. Headline findings
we treat as established and do **not** re-litigate blindly:

- Gaussianization is invisible to histogram boosting (quantile bins).
- Secret rotations preserve distances but hurt axis-aligned trees as `d` grows.
- Invertible `G+W+R` collapses under ridge known-pair attacks at `O(d)` pairs.
  Raising `d` from 80 to 200 does not stop that attack.
- `k < d` projections and quantization cap reconstruction and also cap tree utility.
- The bottleneck kept ~100% `R²` but leaked the attributes that cause `y`
  (worst-attribute known-pair `R² ≈ 0.90–0.95`). Adversarial reconstruction at
  `λ=0.05` did not change that.
- **No configuration hit ≥90% utility on every dataset and worst-attribute
  leakage below 0.7.**

Gaps versus this goal:

- Only numeric `X`; no categoricals, strings, or identifiers.
- Only one model family.
- Target `y` was left in the clear (trainer sees original labels/units).
- Attacker B was mostly moment matching / ICA, not a semantic black-box agent.
- No prediction-remapping protocol (`g` / `g^{-1}`).
- Deterministic encoders only (no variational IB / stochastic Z).

## Families considered (and why they were kept or rejected)

### Kept as candidates

| Family | Why it could satisfy the requirement |
| --- | --- |
| Per-column monotone / quantile maps | Trees keep splits; `g^{-1}` is easy; names can be stripped |
| Typed keyed masking (HMAC buckets, keyed cat codes) | Heterogeneous tables; strings become many-to-one |
| Secret orthogonal mix after whitening | Hides column identity; geometry survives for linear/MLP |
| Supervised bottleneck (prior) | Locally keeps `I(H;Y)`; trainer sees anonymous `H` |
| Variational information bottleneck | Stochastic / compressed `Z`; theoretically shrinks `I(Z;X)` |
| Random Fourier features | Nonlinear secret embedding; linear inversion should fail |
| Microaggregation + rotate | Many-to-one row replacement; caps exact recovery |
| Isotropic noise after rotation | Local-DP flavour; trades utility for inversion error |

### Rejected as the *primary* mechanism

| Family | Why it fails the requirement |
| --- | --- |
| Format-preserving encryption / AES | Trainable numbers lose order/distance; this is encryption |
| Order-preserving encryption | Known weak to sort / frequency / known-plaintext |
| DP-SGD / DP training | Protects the *learning algorithm*, not a published table `Z` |
| Synthetic data / tabular diffusion (DP-TLDM, DP-NTK) | Trainer fits a different sample, not `f(X_i)` per row |
| Homomorphic encryption / MPC | Different systems problem; not a representation `Z` |
| Autoencoders / VAEs trained to reconstruct `X` | Explicitly maximize leakage of `X` |
| Feature hashing without keys | Public and reversible by frequency on low-card cats |
| DELTA / RL feature reprogramming | Needs declared sensitive attributes and a huge search; we test a lighter IB instead |

Relevant papers that shaped the grid (not copied):

- Alemi et al., *Deep Variational Information Bottleneck* — `min I(Z;X) − β I(Z;Y)`.
- Privacy funnel / CPF (ITW 2021) — compress `X` while keeping utility, variationally.
- Adversarial representation learning — min utility loss, max attacker loss (prior `bn_adv`).
- Johnson–Lindenstrauss / random projections — distance preservation, linear invertibility.
- Rahimi & Recht, random Fourier features — secret nonlinear lift.
- Microaggregation / k-anonymity — many-to-one row replacement.
- Known-plaintext attacks on OPE and on affine encodings — assume Attacker A will try them.

## Fundamental limit (do not paper over)

If `Z` is a deterministic function of `X` and `I(Z; Y) ≈ I(X; Y)`, then
`Z` still contains the coordinates of `X` that determine `Y`. An attacker
with pairs `(X, Z)` can learn `E[X_predictive | Z]` to high accuracy.
**Stochasticity, quantization, and dimension reduction are the only
information-theoretic brakes**, and they cost utility.

So “strongest practical transform” here means:

1. Near-raw predictive utility after `g^{-1}`.
2. No names / types / units exposed to the trainer.
3. Unpaired reconstruction and semantic matching near chance.
4. Known-pair recovery delayed or capped — not magically impossible.

The recommended system is therefore an **owner-side architecture**
(typed encode → optional local IB → secret mix → invertible `g`) rather
than a one-way cryptographic claim.
