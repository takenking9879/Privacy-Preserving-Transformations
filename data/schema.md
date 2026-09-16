# Original customer / credit-risk DGP

Source of record for the **observed** table produced by
`src.dgp.generate_original(n=4000, seed=7)`. Latent variables are **not**
written to disk; they exist only inside the generator.

Default size: **n = 4000**. Default seed: **7**.

---

## Columns and dtypes

| Column | Role | Observed type | Support / notes |
|---|---|---|---|
| `age` | X | float64 | Near-normal, truncated to [18, 79] |
| `income` | X | float64 | Right-skewed (log-normal). **MAR NaNs** |
| `tenure_months` | X | int64 | Rounded gamma, [0, 360] |
| `region` | X | string | 4 levels, unequal frequencies |
| `segment` | X | string | 3 levels, unequal frequencies |
| `risk_score` | X | float64 | Near-normal mixture on [5, 99]; higher = riskier |
| `num_products` | X | int64 | Poisson counts, clipped [0, 14] |
| `is_premium` | X | int64 | Binary 0/1 |
| `usage` | X | float64 | Right-skewed spend-like intensity |
| `engagement` | X | float64 | Near-normal on [0, 100]. **MAR NaNs** |
| `complaint_count` | X | int64 | Negative-binomial counts |
| `credit_util` | X | float64 | Bounded (0, 1), clipped [0.01, 0.99] |
| `y` | regression target | float64 | Expected loss (USD), right-skewed |
| `y_class` | classification target | int64 | Binary 0/1 from a **logistic risk index** (not a y-quantile) |

Exports from `src.dgp`:

- `generate_original(n: int = 4000, seed: int = 7) -> pandas.DataFrame`
- `FEATURE_COLS` — the 12 X columns above
- `TARGET_REG = ["y"]`
- `TARGET_CLF = ["y_class"]`
- `get_ground_truth_description() -> str`

`region` levels: `Northeast`, `Midwest`, `South`, `West`.
`segment` levels: `Mass`, `Affluent`, `Private`.

---

## Latent variables (never returned)

Two confounders are drawn first and then fan out into many X columns **and** `y`:

1. **Z — financial stress**, `Z ~ Normal(0, 1)`.
2. **H — distressed / high-risk cluster**, `H ~ Bernoulli(0.04)` (~3–5% tail event).
3. **Wealth propensity** (auxiliary): `W = 0.75·ε − 0.25·Z`, `ε ~ N(0,1)`.
   Anti-correlated with Z; drives segment, income, usage, and premium status.

`H = 1` jointly raises utilization, risk score, complaints, and `y`, and
lowers income, engagement, tenure, and P(premium). A synthesizer that
matches one-dimensional tails but not this **joint** 4% cluster has failed.

---

## Feature-generating equations (high level)

Notation: `σ(·)` is the logistic function. Clipping bounds are applied after
the structural mean plus noise.

### Age (near-normal)

```
age = clip( N(42, 12.5²), 18, 79 )
```

### Segment (3 levels, unequal)

Softmax logits, then one draw per row:

```
ℓ_Mass     =  1.25 − 0.95 W
ℓ_Affluent =  0.05 + 0.35 W
ℓ_Private  = −1.25 + 1.15 W + 0.018 (age − 40)
```

Intended marginals: Mass ≈ 0.60, Affluent ≈ 0.28, Private ≈ 0.12.

### Region (4 levels, unequal)

```
ℓ_NE = 0.55
ℓ_MW = 0.18
ℓ_SO = 0.42 + 0.12 Z
ℓ_WE = −0.55 + 0.28 Z + 0.35 H
```

Intended marginals: Northeast ≈ 0.34, Midwest ≈ 0.24, South ≈ 0.30, West ≈ 0.12.
West/South pick up a mild stress tilt.

### Income (continuous, right-skewed)

```
log income = 10.38 + 0.52 W + 0.38 I_Private + 0.16 I_Affluent
             + 0.011 (age − 40) − 0.16 Z − 0.58 H + N(0, 0.36²)
income     = clip( exp(log income), 14_000, 420_000 )
```

### Tenure (skewed, age-linked)

```
μ_ten = clip( 10 + 0.85 (age − 18) + 8 I_Private + 3.5 I_Affluent − 6 H + 1.5 W,
              2, 300 )
tenure_months = clip( round( Gamma(shape=5, scale=μ_ten/5) ), 0, 360 )
```

### Premium flag

```
P(is_premium = 1) = σ( −2.85 + 2.25 I_Private + 1.05 I_Affluent
                       + 0.50 W + 0.28 (log income − 10.4) − 0.45 H )
```

### Usage (spend-like, right-skewed)

```
log usage = 2.75 + 0.32 (log income − 10.4) + 0.28 is_premium
            + 0.12 Z + 0.38 W + 0.15 I_Private + N(0, 0.50²)
usage     = clip( exp(log usage), 0.15, 180 )
```

### Credit utilization (bounded [0, 1])

```
credit_util = clip( σ( −0.55 + 0.52 Z + 1.55 H − 0.22 (log income − 10.4)
                       + 0.10 (age − 40)/20 − 0.18 is_premium + N(0, 0.62²) ),
                    0.01, 0.99 )
```

### Risk score (near-normal mixture)

```
risk_score = clip( 47.5 + 7.5 Z + 21 H + 16 credit_util + 0.10 (age − 40)
                   − 3.8 is_premium − 2.6 W + N(0, 6.8²),
                   5, 99 )
```

### Engagement (near-normal)

```
engagement = clip( 61 − 5.5 Z − 15 H + 7.5 is_premium + 0.055 tenure
                   + 2.4 (log income − 10.4) + N(0, 9.5²),
                   0, 100 )
```

### Product count (Poisson)

```
λ_prod = clip( 1.05 + 0.95 is_premium + 0.0075 tenure + 0.75 I_Private
               + 0.32 I_Affluent + 0.018 usage − 0.25 H, 0.15, 12 )
num_products ~ Poisson(λ_prod), then clip to [0, 14]
```

### Complaint count (NegBin / overdispersed)

Mean–dispersion form `Var = μ + α μ²` with `α = 1.15`:

```
μ_cmp = clip( 0.22 + 0.32 max(Z, 0) + 1.90 H + 0.95 credit_util
              + 0.016 max(risk_score − 60, 0) + 0.08 I_Mass, 0.02, 18 )
complaint_count ~ NegBin(μ_cmp, α=1.15), clip to [0, 30]
```

---

## Target equations

### `y` — expected loss (USD, continuous, right-skewed)

Log-normal with **heteroscedastic** scale. Segment intercepts, linear terms,
two interactions, a tenure square-root, a hockey-stick threshold on risk, and
direct latent effects:

```
η = 3.55
    + 0.18 I_Affluent + 0.40 I_Private          # segment-specific intercepts
    + 1.10 credit_util
    + 0.016 risk_score
    + 0.07 (log income − 10.4)
    + 0.085 complaint_count
    − 0.14 is_premium
    + 0.58 (credit_util · is_premium)           # interaction
    + 0.14 (log income − 10.4) · (usage / 15)   # interaction
    − 0.075 √tenure_months                      # nonlinear
    + 0.52 max(risk_score − 68, 0) / 10         # threshold / hockey-stick
    + 0.26 Z + 0.80 H                           # latent confounder + cluster

σ = 0.26 · ( 1 + 0.95 credit_util + 0.75 H + 0.20 I(risk_score > 68) )

y = clip( exp(η + σ · ε), 1, 25_000 ),   ε ~ N(0, 1)
```

Implications:

- `Var(y | X)` grows with utilization and in the distressed cluster.
- Private / Affluent have higher expected-loss *levels* (larger balances)
  even after controls.
- Premium customers have a **negative** main effect but a **positive**
  `credit_util × is_premium` interaction (more dollars at risk when they
  revolve).
- Loss jumps once `risk_score` clears 68.

### `y_class` — high-loss flag (binary)

**Not** a quantile cut on `y`. It is a Bernoulli draw from a logistic of a
structural risk index that **shares** Z, H, utilization, and the risk-score
threshold with `y`, so the two targets are associated but not a deterministic
function of each other:

```
risk_index = −2.35
             + 2.05 credit_util
             + 0.032 (risk_score − 50)
             + 1.15 H + 0.38 Z
             + 0.14 complaint_count
             − 0.28 is_premium
             + 0.45 I(risk_score > 68)

y_class ~ Bernoulli( σ(risk_index) )
```

Typical class balance is about 22–28% ones (seed- and n-dependent; ≈0.27 at n=4000, seed=7).

---

## Missingness (MAR)

NaNs are left in place so downstream code can impute. Mechanisms depend only
on **fully observed** columns, so complete-case rows are a selected but
identifiable subsample (MAR, not MCAR).

```
P(income missing)      = σ( −3.35 + 0.85 I_Mass + 0.018 (risk_score − 50) )
P(engagement missing)  = σ( −3.55 + 0.38 complaint_count )
```

Expected missingness is moderate (roughly high-single-digit to low-teens
percent for income; slightly lower for engagement), so dropping NaNs still
leaves a large complete-case table.

No other columns are missing.

---

## Expected correlations a synthesizer must recover

Signs are for the **observed** associations (some are partly confounded by Z/H).

| Pair | Expected direction | Why |
|---|---|---|
| `credit_util`, `risk_score` | strong + | Direct path + shared Z, H |
| `credit_util`, `y` | strong + | Linear term + heteroscedasticity |
| `risk_score`, `y` | strong + | Linear + threshold at 68 + Z, H |
| `complaint_count`, `y` / `risk_score` | + | NegBin mean depends on util/risk/H |
| `engagement`, `risk_score` / `y` | − | Z and H push them opposite ways |
| `income`, `credit_util` | − | Direct + Z (stress lowers income, raises util) |
| `income`, `usage` | + | usage log-mean includes log income and W |
| `income`, `segment=Private` | + | Wealth and segment shift |
| `is_premium`, `segment=Private` | + | Logistic shift +2.25 |
| `is_premium`, `num_products` | + | Poisson mean |
| `tenure_months`, `age` | + | Tenure mean rises with age |
| `tenure_months`, `y` | mild − | `−0.075 √tenure` in η |
| residual (`credit_util`, `engagement`, `income`, `y`) | induced by Z | not explained by observed covariates alone |
| joint tail of (high util, high risk, high complaints, high y, low income) | ~4% | H cluster |

Nonlinear / interaction structure that **marginal matching is not enough** for:

1. `E[y | X]` contains `credit_util × is_premium`.
2. `E[y | X]` contains `log(income) × usage`.
3. Hockey-stick: extra loss only for `risk_score > 68`.
4. Segment intercepts (Private / Affluent ≠ Mass).
5. `Var(y)` increases in `credit_util` and `H`.
6. `y_class` is a noisy logistic of the shared risk index, not `I(y > q)`.

---

## What a good synthesizer MUST capture

A fidelity-passing synthesizer is expected to reproduce, at least:

1. **Mixed-type margins**: skewed `income` / `usage` / `y`; near-normal `age`,
   `risk_score`, `engagement`; [0, 1] `credit_util`; integer Poisson /
   NegBin counts; binary `is_premium` / `y_class`.
2. **Unequal categorical frequencies** for `region` (4) and `segment` (3),
   including the rarer `West` and `Private` cells.
3. **The ~4% distressed cluster**: a small group that is jointly extreme on
   utilization, risk, complaints, loss, and income — not just a fat
   one-dimensional tail.
4. **Latent-confounder residual dependence** among `income`, `credit_util`,
   `risk_score`, `complaint_count`, `engagement`, and `y` after conditioning
   on the obvious observed drivers.
5. **Target structure**: linear effects, both named interactions, the
   `√tenure` term, the risk-score threshold, and segment-specific intercepts.
6. **Heteroscedasticity** of `y` (wider spread at high utilization / cluster).
7. **Classification mechanism**: `y_class` correlated with `y` and the risk
   index, with similar prevalence, but **not** a hard cutoff of `y`.
8. **MAR missingness** on `income` and `engagement` (more missing for Mass /
   high `risk_score` and for high `complaint_count`). Complete-case rates
   and the missingness predictors should be close to the original.

Matching means and pairwise Pearson correlations alone is **not** sufficient.
Thresholds, interactions, the rare cluster, count overdispersion, and the
MAR pattern are the intended failure modes for weak synthesizers.

---

## Complete-case vs. impute-ready

The returned frame **is** impute-ready: NaNs appear only in `income` and
`engagement`. For complete-case work, drop those two columns' NaNs
(`DataFrame.dropna(subset=["income", "engagement"])`). The remaining
columns are always fully observed, and the MAR mechanism is a function of
`segment`, `risk_score`, and `complaint_count`, all of which survive the drop.
