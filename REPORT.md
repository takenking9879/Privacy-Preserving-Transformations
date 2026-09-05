# Utility-Preserving Privacy Transformations

**Status:** experimental study, not a cryptographic claim.
**Priority:** maximum practical privacy subject to minimal predictive utility loss.
**Protocol:** `python run.py` (2 seeds, 3 datasets, 14 transforms, 4 model families).

`main` had no implementation. A prior branch evaluated numeric-only
`Z = Q(W(G(X)))PR` and a deterministic bottleneck with one tree model.
This study treats those maps as the **current** baseline, then asks whether
anything stronger exists for heterogeneous tables, multiple model families,
and prediction remapping.

---

## 1. Current transformation analysis

### What the trainer is allowed to see

Every non-identity transform sends only:

- a numeric matrix `Z` with anonymous column ids `c000…`
- a target `ỹ = g(y)` where `g` is an invertible secret affine map
  (regression) or label permutation (classification)

Predictions are scored after `ŷ = g^{-1}(M(Z))`. Column names, types, and
schema never leave the owner.

### Properties preserved vs destroyed

Measured on `banking_mixed_reg` (seed 0). Distance = pairwise Euclidean
correlation between `X` and `Z`. Rank = mean best |Spearman| of each `X`
column against any `Z` column.

| Transform | Distance | Rank match | Off-diag corr in Z | Invertible? | Preserves | Destroys |
| --- | --- | --- | --- | --- | --- | --- |
| `identity` | 1.00 | 1.00 | raw | yes | everything | nothing |
| `secret_affine` | 1.00 | 1.00 | same as raw | yes | linear span, order, tree splits | units, names, sign |
| `gauss` / `typed_keyed` | 0.10 | 1.00 | similar | partial | within-column order, cat equality | scale, units, string literals (HMAC) |
| `std_rot` / `gauss_white_rot` | 0.37 | 0.51 | ≈ 0 (whitened) | yes (key) | 2nd-order geometry | column identity, axis-aligned splits |
| `bn_adv` | 0.16 | 0.57 | 0.63 | no | I(Z;Y) | features ⊥ Y, names |
| `vib` | 0.07 | 0.33 | 0.81 | no | I(Z;Y) up to β | fine-grained X |
| `rff` | 0.20 | 0.50 | low | no | approximate kernel geometry | original coordinates |
| `microagg_rot` | 0.19 | 0.39 | mixed | no | coarse neighborhoods | within-cluster identity |

**Relationships / correlations.** Secret affine and per-column monotone maps
keep pairwise comonotonicity (rank = 1). Rotations keep a rotated covariance
but destroy the original correlation *pattern in the named basis*.
Bottlenecks / VIB destroy most pairwise X-structure and keep a low-dimensional
predictive subspace.

**Distributions.** Gaussianization and typed keyed maps remove skew and units.
Identity and secret affine leave skew intact (only scale/shift).

**Sparsity.** None of the maps are designed to preserve exact zeros; dummy
columns in `keyed_monotone` add dense structured noise.

**Categoricals / strings.** `typed_keyed` is the only map that treats types
honestly: keyed permutations for cats, HMAC buckets for strings (many-to-one).
Public encoding of strings is a non-secret hash and is **not** privacy.

**Model-class effect.** Trees barely notice monotone maps (HGB retention ≈ 1.00).
Rotations cost 5–12% HGB retention. Linear models absorb `secret_affine` and
`std_rot` exactly (retention 1.00) and slightly *gain* on VIB/bottleneck
representations (retention ≈ 1.07) because the local encoder is already a
supervised projection.

**Prediction remapping.** Affine `g` is free for every model family tested:
training on `g(y)` and inverting is equivalent to training on `y` up to
numerical error. This hides target units from the trainer without utility loss.

---

## 2. Weaknesses found

1. **Known-pair inversion is the binding constraint.** `secret_affine`,
   `identity`, and `std_rot` reach ridge reconstruction R² ≥ 0.9 by 10–25
   pairs. `gauss_white_rot` does the same by 25 pairs on the banking tables
   (d ≈ 15). Extra dimensions do not make a keyed linear map one-way.
2. **Worst-attribute leakage hides inside “resistant” global scores.**
   `vib` / `bn_adv` never reach global reconstruction R² = 0.5 within 200
   pairs, but the attributes that cause `y` recover to worst-attribute
   R² ≈ 0.88–0.93 on the banking tables from 2–10 pairs. Mean reconstruction
   is the wrong headline.
3. **Utility-preserving ⇒ I(Z; X_predictive) stays large.** This is
   information-theoretic, not an implementation bug. Stochastic VIB
   (`vib_stoch`) is the only increment that lowers that leakage (mean
   worst-attribute 0.76 vs 0.83) and it drops California HGB retention to 0.895.
4. **Rank attacks beat linear inversion on monotone maps.** `typed_keyed`
   needs ~50–100 pairs for global R² = 0.5, but worst-attribute R² is already
   0.98 at 5 pairs via Spearman matching.
5. **Microaggregation does not cap the worst attribute.** Cluster centroids
   still leak extreme / well-separated features (worst-attribute ≈ 1.0) while
   destroying utility (HGB retention 0.16 on banking regression).
6. **Leaving `y` in the clear is a leak.** The prior branch did this. The
   present protocol always applies `g` except for `identity`.
7. **Attacker B on raw encodings is not helpless.** On the banking table it
   flags age, gender, utilization-like rates, and money-like skew
   (semantic hit 0.20, kind 0.73, sensitive-column hit 0.89) from `Z = X`
   plus the sentence *“This dataset comes from a bank…”*.

---

## 3. Candidate alternative transformations

Implemented and measured (not just discussed):

| Name | Family | Local `y` required? |
| --- | --- | --- |
| `identity` | baseline | no |
| `secret_affine` | typed / linear-friendly | no |
| `std_rot` | classical | no |
| `gauss` | classical (prior) | no |
| `gauss_white_rot` | classical (prior) | no |
| `keyed_monotone` | typed | no |
| `typed_keyed` | typed / heterogeneous | no |
| `bn_adv` | learned (prior) | yes |
| `bn_noisy` | learned | yes |
| `vib` | learned (VIB) | yes |
| `vib_stoch` | learned | yes |
| `rff` | nonlinear | no |
| `microagg_rot` | lossy | no |
| `noisy_gauss_rot` | classical + noise | no |

Rejected as the *primary* mechanism (see `RESEARCH.md`): FPE/AES, OPE,
DP-SGD, synthetic diffusion, HE/MPC, reconstructive VAEs. They do not
implement row-aligned `X → Z` training with `g^{-1}` predictions.

Further increments past `vib` (`vib_stoch`, heavier noise, microagg, RFF)
were measured. Gains in leakage are real and **conflict with the 90%
all-dataset utility floor**. Additional search in this family is marginal.

---

## 4. Utility benchmark

HGB score retention (transformed / raw), mean of 2 seeds. Regression uses
R²; classification uses accuracy. All scores are after `g^{-1}`.

| Method | housing | bank-reg | bank-clf | mean | linear mean | all-ds ≥ 0.90? |
| --- | --- | --- | --- | --- | --- | --- |
| `secret_affine` | 1.005 | 0.998 | 1.006 | **1.003** | **1.000** | yes |
| `typed_keyed` | 1.000 | 1.001 | 1.006 | **1.002** | 0.983 | yes |
| `keyed_monotone` | 1.000 | 0.998 | 1.002 | 1.000 | 0.997 | yes |
| `gauss` | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | yes |
| `bn_adv` | 0.917 | 0.990 | 1.002 | 0.970 | 1.075 | yes |
| `vib` | 0.903 | 0.992 | 1.008 | 0.968 | 1.074 | yes |
| `vib_stoch` | 0.895 | 0.985 | 1.004 | 0.961 | 1.072 | no |
| `std_rot` | 0.880 | 0.945 | 0.979 | 0.935 | 1.000 | no |
| `gauss_white_rot` | 0.903 | 0.895 | 0.958 | 0.919 | 0.982 | no |
| `rff` | 0.759 | 0.655 | 0.914 | 0.776 | 0.699 | no |
| `microagg_rot` | 0.571 | 0.159 | 0.790 | 0.507 | 0.531 | no |

Prediction agreement (HGB raw vs transformed, after remap) stays ≥ 0.96 for
every method that clears the 90% floor.

k-NN tracks HGB. MLP retention is noisy and sometimes > 1 because the raw MLP
is under-trained (120 iterations, early stopping); do not over-read MLP ratios.

**Phase 1 conclusion.** Utility is a solved problem for trees and linear
models: `secret_affine` is an exact named-hiding ceiling; `typed_keyed` is
essentially free for HGB and only a 2% linear tax. Learned maps are slightly
lossy for trees on California housing and slightly helpful for linear models.

---

## 5. Attacker A results (known pairs)

Grid: 1, 2, 5, 10, 25, 50, 100, 200 pairs. Attacks: ridge, pinv, rank/order,
categorical frequency, HGB inversion, MLP inversion (heavy, seed 0).

### Pairs to global ridge R² ≥ 0.5

| Method | Typical n | Label |
| --- | --- | --- |
| `std_rot` | ~8–10 | collapses |
| `identity` / `secret_affine` | ~10–25 | collapses |
| `gauss` / `gauss_white_rot` / `keyed_monotone` | ~25 | partial / collapses |
| `typed_keyed` | ~50–100 | partial (global), **worst-attr immediate** |
| `bn_adv` / `vib` / `vib_stoch` / `rff` | > 200 | resistant *globally* |

### Worst-attribute R² at the largest budget (HGB rows, mean)

| Method | Worst-attr R² | Sensitive-attr R² |
| --- | --- | --- |
| `secret_affine` / `identity` / `std_rot` | 1.00 | 1.00 |
| `typed_keyed` / `gauss` / `gauss_white_rot` | 0.97 | 0.97 |
| `bn_adv` | 0.89 | 0.89 |
| `bn_noisy` / `vib` | 0.83 | 0.83 |
| `vib_stoch` | **0.76** | 0.76 |
| `rff` | 0.69 | 0.69 |

On `banking_mixed_reg`, `vib` worst-attribute R² is already 0.89 at **2 pairs**.
The leak is not “the whole table”; it is the coordinates that determine `y`.

Spearman leak (max |ρ| between any X column and any Z column, unpaired) is
1.0 for every monotone / affine map and 0.33–0.89 for learned / rotated maps.

---

## 6. Attacker B results (Z + one context sentence)

No keys, schema, names, or raw rows. Context example:
*“This dataset comes from a bank and contains customer-related statistics.”*

| Method | Semantic top-1 (bank) | Kind inference | Sensitive-col hit (bank) | Membership AUC |
| --- | --- | --- | --- | --- |
| `identity` | 0.20 (age, gender, money, rates) | 0.73 | 0.89 | 0.53 |
| `gauss` | 0.00 | 0.73 | 0.89 | 0.53 |
| `secret_affine` | 0.00 | 0.43–0.57 | 0.50–0.72 | 0.53 |
| `typed_keyed` | **0.00** (all “unknown”) | 0.40–0.53 | 0.61–0.67 | 0.53 |
| `gauss_white_rot` | 0.00 | 0.53 | 0.44 | 0.52 |
| `vib` / `bn_*` | n/a (dims ≠ raw; no column alignment) | n/a | n/a | 0.52 |
| `vib_stoch` | n/a | n/a | n/a | **0.65** (worse) |

Membership inference from nearest-neighbor uniqueness is near chance except
`vib_stoch`, whose deterministic keyed noise is a train-set signature.
That variant is therefore **not** recommended despite slightly better
known-pair numbers.

Linkage via PCA nearest-neighbor against a same-size public-like table did
not produce a reliable self-match once columns were mixed (metric often
undefined when dimensions differ). Neighborhood-preserving maps remain the
theoretical risk; it did not fire cleanly in this protocol.

---

## 7. Security / utility trade-off

Axes used for selection:

- Utility: HGB retention, linear retention, prediction agreement
- Privacy: worst-attribute known-pair R², pairs-to-global-R²=0.5,
  Attacker B semantic accuracy, membership AUC

**Pareto observations**

- Upper-right utility, no privacy: `identity`
- Same utility, hide names/units only: `secret_affine`
- Same tree utility, hide marginals/strings/semantics from Attacker B:
  `typed_keyed` (best model-agnostic point)
- Small tree tax, much lower *global* reconstruction, still high
  worst-attribute leak: `vib`, `bn_adv`
- Lower worst-attribute leak, breaks the 90% floor or raises MI:
  `vib_stoch`, `rff`, `microagg_rot`

**No tested point simultaneously keeps ≥ 90% HGB retention on every
dataset and drives worst-attribute known-pair R² below 0.7.**
That matches the prior numeric-only study and survives the broader grid.

Figure: `results/figures/02_pareto_utility_leakage.png`.

---

## 8. Recommended architecture

```
owner holds schema, keys, g, optional local encoder
  1. drop row identifiers
  2. strings      -> HMAC-SHA256 buckets (many-to-one)
  3. categoricals -> keyed code permutation
  4. numerics     -> train-only quantile map
  5. optional     -> VIB / bottleneck if labelled train data is local
  6. secret permutation or rotation of the resulting columns
  7. ỹ = g(y)     -> invertible affine / label permutation
  8. send (Z, ỹ, ids c000…) to the trainer

trainer fits any model M: Z -> ỹ   (no names, no raw values, no schema)

owner / client returns ŷ = g^{-1}(M(Z_new))
```

This is the unusual requirement, met:

```
train in transformed space
and still obtain useful predictions in the original problem space
```

It is an obfuscated / lossy outsourced representation with an owner-side key.
It is not encryption.

---

## 9. Recommended transformation

**Default (model-agnostic, no local encoder): `typed_keyed`.**

- HGB retention 1.00 on every dataset; linear 0.98
- Attacker B cannot name columns (semantic 0.00 vs 0.20 on raw)
- Strings are many-to-one; names are stripped; target units are hidden
- Known-pair worst-attribute recovery remains trivial — disclose this

**When the owner can fit a local labelled encoder and the threat includes
unpaired / low-pair reconstruction of the whole table: `vib`.**

- Only learned map that clears the 90% HGB floor on every dataset
- Global reconstruction stays below 0.5 through 200 pairs
- Worst-attribute leak of predictive fields remains (~0.83 mean, ~0.90 on banking)
- Linear models are as good or better than raw

**Do not ship `vib_stoch`** (membership AUC 0.65) or `microagg_rot` /
`rff` as defaults (utility collapse).

**Do not ship `gauss_white_rot` as “encrypted features.”** It is a keyed
linear map and falls at `O(d)` pairs.

---

## 10. Remaining attack surface

- Known-pair / chosen-plaintext inversion of any approximately invertible stage
- Recovery of `X` coordinates that cause `Y` whenever `I(Z;Y)` is large
- Rank and frequency attacks on monotone or keyed-categorical columns
- Linkage if an overlapping public table exists and neighborhoods survive
- Membership inference on deterministic unique rows and on keyed noise
- Side channels: `out_dim`, dummy-column count, sparsity, training-time encoder
- `g` hides units and label ids, not the rank order of `y`
- Trainer-side model inversion against released `M` is out of scope here
  and remains possible if `M` is published

---

## 11. Permanent regression tests

```
python -m pytest tests/ -q
```

| Test | Locks |
| --- | --- |
| `test_target_map_*_roundtrip` | `g^{-1}∘g = id` |
| `test_transformed_columns_are_anonymous` | no raw names in `Z` ids |
| `test_typed_keyed_handles_mixed_types` | num/cat/string tables |
| `test_gauss_is_rank_preserving_on_numeric` | monotone numeric property |
| `test_attacker_a_identity_recovers_with_enough_pairs` | A is not a no-op |
| `test_attacker_a_rotation_needs_pairs` | secret rotations are identifiable |
| `test_attacker_b_does_not_need_schema` | B runs on Z + context only |
| `test_unique_continuous_is_not_identifier` | B is not artificially confused |
| `test_age_like_column_is_guessed` / money | B actually searches for semantics |
| `test_predictions_remap_after_target_transform` | original-space predictions |

Re-run `python run.py` after changing a transform; compare
`results/tables/summary.csv` against this report.

---

## 12. Inference capsule (follow-up)

The black-box attacker (no pairs) already failed against learned summaries.
The follow-up hardens that family against the pair attacker without giving
up the 90% utility floor.

`capsule` = typed keyed front-end → adversarial VIB → 12-level quantize → secret rotation,
plus reversible `g` on the target and an optional RSA envelope to the trainer.

HGB retention: housing 0.91, bank-reg 0.98, bank-clf 0.96.
Global reconstruction stays ~0.08 on the banking tables at 200 pairs (was ~0.36 for plain `vib`).
Worst-attribute leakage of the columns that cause `y` is still practical at **2 pairs** (~0.75–0.88).
Attacker B still cannot name columns. Membership stays near a coin flip.

Public/private keys wrap the released file; they do not stop a trainer who already sees `Z`.
The encoder must stay an owner secret. See `GUIA_DISFRAZ.md`.

## Reproducibility

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python run.py --quick
python run.py
```

Artifacts: `results/tables/`, `results/figures/`, per-seed JSON under
`results/<dataset>/`. Auto-generated tables also land in
`results/GENERATED_REPORT.md`.
