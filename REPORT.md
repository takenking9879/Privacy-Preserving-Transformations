# Privacy-Preserving Transformations for Outsourced Machine Learning

## 1. Executive Summary

This study tests whether a third party can train **the exact same model with the exact same hyperparameters** on a transformed table $Z=f(X)$ and still recover most of the predictive performance of $M_\theta(X)\to y$.

Priority used for selection:

$$
\text{predictive utility} > \text{structural preservation} > \text{privacy}
$$

The preferred configuration is **`bn50_adv_rot`**, retaining on average **100.1%** of raw $R^2$ across datasets (mean transformed $R^2$=0.9531 vs raw 0.9519). Worst-attribute known-pair leakage remains **0.930**. This is an obfuscated / lossy representation, **not encryption**, and it does **not** resist known-pair inversion of the attributes that predict $y$.

29 dataset-method cells reached the 90% retention guideline. Methods with $\ge 90\%$ on every dataset: `bn50`, `bn50_adv_rot`, `bn50_q16_rot`, `bn50_rot`, `gauss`

**Do not interpret these transforms as cryptographically secure.** Several configurations that look unstructured still collapse under a known-pair linear attack once the attacker obtains on the order of $d$ matched rows. That is a negative result and is reported as such.

Frozen model: `HistGradientBoostingRegressor` with hyperparameters `{'max_iter': 200, 'max_depth': 6, 'learning_rate': 0.1, 'min_samples_leaf': 20, 'l2_regularization': 0.0, 'early_stopping': False, 'validation_fraction': 0.1}`.

### Key empirical findings

1. **Same frozen `HistGradientBoostingRegressor` can learn from $Z$.** Gaussianization alone is invisible to this model (identical $R^2$ to raw on every dataset), because histogram boosting already quantile-bins each axis.
2. **Secret rotation is not free for trees.** `gauss_white_rot` retention: openml_superconduct 0.959, banking_d80 0.832, banking_d200 0.684. Distance geometry is preserved (whitening+rotation is isometric up to scale), but axis-aligned splits become less efficient as $d$ grows. That is a model-class effect, not information destruction.
3. **Random $k<d$ projections buy known-pair resistance and spend utility.** `gw_q16_p50_rot` retention: openml_superconduct 0.908, banking_d80 0.564, banking_d200 0.448. Worst-attribute leakage: openml_superconduct 0.638, banking_d80 0.593, banking_d200 0.483. On the real superconduct table this still clears 90% retention; on the synthetic banking tables it does not.
4. **Besides monotone Gaussianization (which preserves paired column identity), the supervised bottleneck (`bn50_adv_rot`) is the only family that keeps (or slightly beats) raw $R^2$ on every dataset.** Retention: openml_superconduct 0.990, banking_d80 1.006, banking_d200 1.007. Unpaired reconstruction $R^2$ is negative. Paired semantic matching accuracy: openml_superconduct 0.000, banking_d80 0.017, banking_d200 0.003 (near chance). Worst-attribute known-pair $R^2$: openml_superconduct 0.901, banking_d80 0.936, banking_d200 0.952. The encoder is trained to keep $I(H;Y)$, so the features that cause $y$ remain the leaky ones. Adversarial reconstruction training at $\lambda=0.05$ did not produce a qualitatively different privacy outcome.
5. **Larger $d$ does not make an invertible secret rotation hard.** Known-pair ridge $R^2$ for `gauss_white_rot` is 0.923 at $d=80$ and 0.915 at $d=200$. The map is statistically identifiable from $O(d)$ pairs. Extra dimensions *do* hurt tree utility under rotation, and *do* lower reconstruction when combined with $k<d$ lossy projection. Those are different mechanisms: keyed linear identifiability vs information-theoretic loss.
6. **No tested configuration simultaneously (a) kept $\ge 90\%$ $R^2$ retention on all three datasets and (b) drove known-pair worst-attribute $R^2$ below $0.7$.** If both constraints are required, this grid does not contain a solution. Utility-first selection therefore returns `bn50_adv_rot` and records the known-pair failure in the open.

---

## 2. Mathematical Definition

Let $X\in\mathbb{R}^{n\times d}$ be numeric features and $y\in\mathbb{R}^n$ the target. All fitted maps below are estimated on the training split only.

### 2.1 Marginal Gaussianization $G$

For each column $j$, a quantile transformer $\hat F_j$ (sklearn `QuantileTransformer`, normal output) implements

$$
u_j = \Phi^{-1}(\hat F_j(x_j)).
$$

This removes scale, units, and marginal shape. For `HistGradientBoosting`, which bins features by quantiles internally, $G$ can leave predictions almost unchanged — that is a model-class fact, not a privacy result.

### 2.2 Symmetric whitening $W$

$$
V = (U-\mu)\,\Sigma^{-1/2}
$$

with $\Sigma^{-1/2}$ the symmetric (ZCA-style) square root of the training covariance of $U=G(X)$, regularized by $10^{-6}I$.

### 2.3 Uniform quantization $Q_L$

On $[-4,4]$, $L$ equal-width bins; values are replaced by bin centers. This is many-to-one:

$$
x_1\neq x_2 \quad\text{can satisfy}\quad Q_L(x_1)=Q_L(x_2).
$$

### 2.4 Secret orthogonal mixing $R$

$R$ is a Haar-like orthogonal matrix from QR of a seeded Gaussian matrix, $R^\top R=I$.

### 2.5 Dimensional projection $P$

$P\in\mathbb{R}^{d\times k}$ contains $k$ orthonormal columns, $k<d$.

Classical family used in the grid:

$$
Z = Q_L\big(W(G(X))\big)\,P\,R
$$

with $P=I$ and/or $Q_L$ omitted in weaker ablations.

### 2.6 Supervised predictive bottleneck $E_\phi$

A two-hidden-layer MLP encoder is trained **locally** to predict $y$ from $X$:

$$
X \xrightarrow{E_\phi} H\in\mathbb{R}^{k} \xrightarrow{m} \hat y, \qquad k=\lfloor 0.5 d\rfloor.
$$

The prediction head is discarded. $H$ is standardized on train, optionally quantized and rotated:

$$
Z = Q_L(E_\phi(X))\,R.
$$

An adversarial variant additionally trains $A_\psi(H)\to X$ and updates the encoder with

$$
\min_{E,m}\max_A \big[\mathcal{L}_{\mathrm{pred}} - \lambda \mathcal{L}_{\mathrm{recon}}\big], \quad \lambda=0.05.
$$

This is **not** an autoencoder: reconstruction of $X$ is not the training goal.

---

## 3. Experimental Setup

### 3.1 Datasets

- **openml_superconduct** (real): n=21263, d=81. OpenML 'superconduct' regression, n=21263, d=81. Real tabular features; target is the dataset default.
- **banking_d80** (synthetic): n=18000, d=80. Synthetic banking-like tabular data n=18000, d=80, q=12 latents. Heterogeneous observation maps; y is a nonlinear function of 6 latents.
- **banking_d200** (synthetic): n=18000, d=200. Synthetic banking-like tabular data n=18000, d=200, q=12 latents. Heterogeneous observation maps; y is a nonlinear function of 6 latents.

### 3.2 Splits and seeds

- Split fractions: train 70% / test 15% / auxiliary 15%, drawn **once per seed** and reused for every transform.
- Seeds: `[0, 1, 2]`.
- Auxiliary rows are from the same population but are **not** paired with $Z$ for Attack 3.
- Transforms (quantile map, covariance, encoder, $P$, $R$) are fit on **train only**.

### 3.3 Model (identical for $X$ and $Z$)

- Model: **HistGradientBoostingRegressor**
- Hyperparameters: `{'max_iter': 200, 'max_depth': 6, 'learning_rate': 0.1, 'min_samples_leaf': 20, 'l2_regularization': 0.0, 'early_stopping': False, 'validation_fraction': 0.1}`
- `random_state` equals the experiment seed (the same seed is used for raw and transformed fits).
- No per-transform retuning of depth, learning rate, iterations, leaf size, or regularization.

### 3.4 Protocol reminder

The only comparison that matters is

$$
M_\theta(X) \quad\text{vs}\quad M_\theta(f_k(X)) \qquad \theta_{\mathrm{raw}}=\theta_{\mathrm{transformed}}.
$$

---

## 4. Utility Results

Mean $\pm$ std over seeds. Retention is $R^2_Z / R^2_X$. RMSE ratio is $\mathrm{RMSE}_Z / \mathrm{RMSE}_X$ (values near 1 are parity; $>1$ is worse).

| dataset | method | out_dim_mean | r2_raw_mean | r2_mean | r2_retention_mean | r2_retention_std | delta_r2_mean | rmse_raw_mean | rmse_mean | rmse_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openml_superconduct | raw | 81 | 0.9129 | 0.9129 | 1.000 | 0.000 | 0.0000 | 10.0915 | 10.0915 | 1.000 |
| openml_superconduct | gauss | 81 | 0.9129 | 0.9129 | 1.000 | 0.000 | 0.0000 | 10.0915 | 10.0915 | 1.000 |
| openml_superconduct | gauss_white | 81 | 0.9129 | 0.8940 | 0.979 | 0.003 | -0.0189 | 10.0915 | 11.1341 | 1.103 |
| openml_superconduct | gauss_white_rot | 81 | 0.9129 | 0.8754 | 0.959 | 0.006 | -0.0375 | 10.0915 | 12.0681 | 1.196 |
| openml_superconduct | gw_q8 | 81 | 0.9129 | 0.8759 | 0.959 | 0.002 | -0.0370 | 10.0915 | 12.0445 | 1.194 |
| openml_superconduct | gw_q16 | 81 | 0.9129 | 0.8853 | 0.970 | 0.003 | -0.0277 | 10.0915 | 11.5829 | 1.148 |
| openml_superconduct | gw_q32 | 81 | 0.9129 | 0.8888 | 0.974 | 0.003 | -0.0241 | 10.0915 | 11.4018 | 1.130 |
| openml_superconduct | gw_q64 | 81 | 0.9129 | 0.8906 | 0.976 | 0.002 | -0.0223 | 10.0915 | 11.3106 | 1.121 |
| openml_superconduct | gw_q16_rot | 81 | 0.9129 | 0.8668 | 0.949 | 0.008 | -0.0461 | 10.0915 | 12.4759 | 1.236 |
| openml_superconduct | gw_q16_p90_rot | 73 | 0.9129 | 0.8606 | 0.943 | 0.002 | -0.0524 | 10.0915 | 12.7701 | 1.266 |
| openml_superconduct | gw_q16_p80_rot | 65 | 0.9129 | 0.8556 | 0.937 | 0.002 | -0.0573 | 10.0915 | 12.9968 | 1.288 |
| openml_superconduct | gw_q16_p70_rot | 57 | 0.9129 | 0.8528 | 0.934 | 0.002 | -0.0601 | 10.0915 | 13.1223 | 1.301 |
| openml_superconduct | gw_q16_p50_rot | 40 | 0.9129 | 0.8288 | 0.908 | 0.008 | -0.0841 | 10.0915 | 14.1497 | 1.402 |
| openml_superconduct | bn50 | 40 | 0.9129 | 0.8988 | 0.985 | 0.004 | -0.0142 | 10.0915 | 10.8800 | 1.078 |
| openml_superconduct | bn50_rot | 40 | 0.9129 | 0.9037 | 0.990 | 0.002 | -0.0092 | 10.0915 | 10.6115 | 1.052 |
| openml_superconduct | bn50_q16_rot | 40 | 0.9129 | 0.8927 | 0.978 | 0.004 | -0.0203 | 10.0915 | 11.2022 | 1.110 |
| openml_superconduct | bn50_adv_rot | 40 | 0.9129 | 0.9039 | 0.990 | 0.003 | -0.0090 | 10.0915 | 10.5985 | 1.050 |
| banking_d80 | raw | 80 | 0.9723 | 0.9723 | 1.000 | 0.000 | 0.0000 | 0.4863 | 0.4863 | 1.000 |
| banking_d80 | gauss | 80 | 0.9723 | 0.9723 | 1.000 | 0.000 | 0.0000 | 0.4863 | 0.4863 | 1.000 |
| banking_d80 | gauss_white | 80 | 0.9723 | 0.8779 | 0.903 | 0.001 | -0.0944 | 0.4863 | 1.0227 | 2.104 |
| banking_d80 | gauss_white_rot | 80 | 0.9723 | 0.8089 | 0.832 | 0.005 | -0.1634 | 0.4863 | 1.2798 | 2.633 |
| banking_d80 | gw_q8 | 80 | 0.9723 | 0.8433 | 0.867 | 0.006 | -0.1290 | 0.4863 | 1.1582 | 2.383 |
| banking_d80 | gw_q16 | 80 | 0.9723 | 0.8734 | 0.898 | 0.002 | -0.0989 | 0.4863 | 1.0413 | 2.142 |
| banking_d80 | gw_q32 | 80 | 0.9723 | 0.8800 | 0.905 | 0.001 | -0.0923 | 0.4863 | 1.0139 | 2.086 |
| banking_d80 | gw_q64 | 80 | 0.9723 | 0.8818 | 0.907 | 0.002 | -0.0906 | 0.4863 | 1.0066 | 2.072 |
| banking_d80 | gw_q16_rot | 80 | 0.9723 | 0.7904 | 0.813 | 0.007 | -0.1819 | 0.4863 | 1.3398 | 2.756 |
| banking_d80 | gw_q16_p90_rot | 72 | 0.9723 | 0.7539 | 0.775 | 0.010 | -0.2184 | 0.4863 | 1.4518 | 2.986 |
| banking_d80 | gw_q16_p80_rot | 64 | 0.9723 | 0.7270 | 0.748 | 0.018 | -0.2453 | 0.4863 | 1.5284 | 3.143 |
| banking_d80 | gw_q16_p70_rot | 56 | 0.9723 | 0.6848 | 0.704 | 0.037 | -0.2875 | 0.4863 | 1.6404 | 3.373 |
| banking_d80 | gw_q16_p50_rot | 40 | 0.9723 | 0.5486 | 0.564 | 0.028 | -0.4237 | 0.4863 | 1.9660 | 4.044 |
| banking_d80 | bn50 | 40 | 0.9723 | 0.9789 | 1.007 | 0.001 | 0.0066 | 0.4863 | 0.4250 | 0.874 |
| banking_d80 | bn50_rot | 40 | 0.9723 | 0.9786 | 1.006 | 0.001 | 0.0063 | 0.4863 | 0.4276 | 0.879 |
| banking_d80 | bn50_q16_rot | 40 | 0.9723 | 0.9769 | 1.005 | 0.001 | 0.0045 | 0.4863 | 0.4450 | 0.915 |
| banking_d80 | bn50_adv_rot | 40 | 0.9723 | 0.9781 | 1.006 | 0.002 | 0.0058 | 0.4863 | 0.4327 | 0.890 |
| banking_d200 | raw | 200 | 0.9704 | 0.9704 | 1.000 | 0.000 | 0.0000 | 0.4936 | 0.4936 | 1.000 |
| banking_d200 | gauss | 200 | 0.9704 | 0.9704 | 1.000 | 0.000 | 0.0000 | 0.4936 | 0.4936 | 1.000 |
| banking_d200 | gauss_white | 200 | 0.9704 | 0.8368 | 0.862 | 0.002 | -0.1336 | 0.4936 | 1.1580 | 2.347 |
| banking_d200 | gauss_white_rot | 200 | 0.9704 | 0.6642 | 0.684 | 0.011 | -0.3062 | 0.4936 | 1.6615 | 3.366 |
| banking_d200 | gw_q8 | 200 | 0.9704 | 0.8002 | 0.825 | 0.007 | -0.1701 | 0.4936 | 1.2810 | 2.596 |
| banking_d200 | gw_q16 | 200 | 0.9704 | 0.8308 | 0.856 | 0.002 | -0.1395 | 0.4936 | 1.1789 | 2.389 |
| banking_d200 | gw_q32 | 200 | 0.9704 | 0.8397 | 0.865 | 0.004 | -0.1307 | 0.4936 | 1.1476 | 2.326 |
| banking_d200 | gw_q64 | 200 | 0.9704 | 0.8371 | 0.863 | 0.004 | -0.1332 | 0.4936 | 1.1566 | 2.344 |
| banking_d200 | gw_q16_rot | 200 | 0.9704 | 0.6311 | 0.650 | 0.014 | -0.3392 | 0.4936 | 1.7410 | 3.528 |
| banking_d200 | gw_q16_p90_rot | 180 | 0.9704 | 0.5922 | 0.610 | 0.024 | -0.3781 | 0.4936 | 1.8299 | 3.709 |
| banking_d200 | gw_q16_p80_rot | 160 | 0.9704 | 0.5666 | 0.584 | 0.036 | -0.4037 | 0.4936 | 1.8867 | 3.822 |
| banking_d200 | gw_q16_p70_rot | 140 | 0.9704 | 0.5344 | 0.551 | 0.029 | -0.4360 | 0.4936 | 1.9563 | 3.962 |
| banking_d200 | gw_q16_p50_rot | 100 | 0.9704 | 0.4351 | 0.448 | 0.036 | -0.5352 | 0.4936 | 2.1544 | 4.364 |
| banking_d200 | bn50 | 100 | 0.9704 | 0.9781 | 1.008 | 0.002 | 0.0077 | 0.4936 | 0.4241 | 0.860 |
| banking_d200 | bn50_rot | 100 | 0.9704 | 0.9782 | 1.008 | 0.001 | 0.0078 | 0.4936 | 0.4236 | 0.859 |
| banking_d200 | bn50_q16_rot | 100 | 0.9704 | 0.9767 | 1.006 | 0.001 | 0.0063 | 0.4936 | 0.4380 | 0.888 |
| banking_d200 | bn50_adv_rot | 100 | 0.9704 | 0.9773 | 1.007 | 0.000 | 0.0070 | 0.4936 | 0.4318 | 0.875 |

### 4.1 Reading the utility table

- Configurations with $R^2$ retention $\ge 0.90$ preserve learnable structure for this model class.
- Orthogonal mixing can hurt **axis-aligned trees** even when information is not destroyed; that is a model-class effect, not proof that $Z$ lacks signal.
- Quantization and $k<d$ projections are the first places information is intentionally discarded.
- Gaussianization alone can be nearly invisible to histogram-based trees.

Figures: `results/figures/01_raw_vs_transformed_r2.png`, `results/figures/02_retention_vs_strength.png`.

---

## 5. Security Attack Results

Attacks were implemented to **break** the representation, not to advertise it.

| Attack | Attacker knowledge | What we measure |
| --- | --- | --- |
| 1 Statistical inspection | $Z$ only | skew, kurtosis, ranges, correlations, eigenvalues |
| 2 Semantic matching | $Z$ and possibly unpaired $X_{\mathrm{aux}}$ | column identity recovery vs $1/d$ |
| 3 Auxiliary unpaired | algorithm class, $X_{\mathrm{aux}}$, no pairs | distributional inversion, ICA, naive CCA |
| 4 Known-pair | $(X_i,Z_i)$ for $n$ rows | ridge, pseudoinverse, informed Procrustes, MLP |
| 5 Neural reconstruction | paired train $Z\to X$ | small MLP, medium MLP, residual MLP |
| 6 Per-attribute leakage | from 4–5 | $\max_j R^2_j$, median, top-5 |
| 7 Distance leakage | paired geometry | $\mathrm{corr}(D_X, D_Z)$ |

| dataset | method | recon_r2_knownpair_ridge_mean | worst_attr_knownpair_ridge_mean | neural_best_r2_mean | aux_random_rot_r2_mean | semantic_unpaired_acc_mean | semantic_paired_acc_mean | distance_preservation_mean | known_pair_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openml_superconduct | raw | 1.000 | 1.000 | 0.990 | -0.961 | 0.901 | 1.000 | 1.000 | collapses_with_enough_pairs |
| openml_superconduct | gauss | 0.949 | 0.991 | — | -0.987 | 0.029 | 1.000 | 0.548 | collapses_with_enough_pairs |
| openml_superconduct | gauss_white | 0.949 | 0.991 | — | -0.999 | 0.004 | 1.000 | 0.350 | collapses_with_enough_pairs |
| openml_superconduct | gauss_white_rot | 0.949 | 0.991 | 0.979 | -1.011 | 0.008 | 0.025 | 0.350 | collapses_with_enough_pairs |
| openml_superconduct | gw_q8 | 0.854 | 0.899 | — | -0.999 | 0.000 | 1.000 | 0.410 | partial_recovery_with_enough_pairs |
| openml_superconduct | gw_q16 | 0.916 | 0.960 | — | -0.998 | 0.000 | 1.000 | 0.413 | collapses_with_enough_pairs |
| openml_superconduct | gw_q32 | 0.934 | 0.978 | — | -0.998 | 0.004 | 1.000 | 0.414 | collapses_with_enough_pairs |
| openml_superconduct | gw_q64 | 0.938 | 0.983 | — | -0.998 | 0.000 | 1.000 | 0.414 | collapses_with_enough_pairs |
| openml_superconduct | gw_q16_rot | 0.916 | 0.960 | 0.966 | -1.013 | 0.016 | 0.025 | 0.413 | collapses_with_enough_pairs |
| openml_superconduct | gw_q16_p90_rot | 0.826 | 0.917 | — | -0.999 | 0.018 | 0.014 | 0.410 | partial_recovery_with_enough_pairs |
| openml_superconduct | gw_q16_p80_rot | 0.720 | 0.845 | — | -0.997 | 0.021 | 0.015 | 0.407 | partial_recovery_with_enough_pairs |
| openml_superconduct | gw_q16_p70_rot | 0.652 | 0.788 | — | -1.013 | 0.029 | 0.023 | 0.399 | partial_recovery_with_enough_pairs |
| openml_superconduct | gw_q16_p50_rot | 0.455 | 0.638 | 0.888 | -0.991 | 0.025 | 0.008 | 0.383 | resistant_in_tested_budget |
| openml_superconduct | bn50 | 0.818 | 0.935 | — | -0.995 | 0.025 | 0.017 | 0.255 | partial_recovery_with_enough_pairs |
| openml_superconduct | bn50_rot | 0.818 | 0.935 | — | -1.005 | 0.025 | 0.025 | 0.255 | partial_recovery_with_enough_pairs |
| openml_superconduct | bn50_q16_rot | 0.757 | 0.913 | 0.898 | -1.004 | 0.017 | 0.025 | 0.252 | partial_recovery_with_enough_pairs |
| openml_superconduct | bn50_adv_rot | 0.727 | 0.901 | — | -0.955 | 0.008 | 0.000 | 0.242 | partial_recovery_with_enough_pairs |
| banking_d80 | raw | 1.000 | 1.000 | 0.987 | -1.010 | 0.129 | 1.000 | 1.000 | collapses_with_enough_pairs |
| banking_d80 | gauss | 0.923 | 1.000 | — | -1.000 | 0.004 | 1.000 | 0.857 | collapses_with_enough_pairs |
| banking_d80 | gauss_white | 0.923 | 1.000 | — | -0.989 | 0.000 | 1.000 | 0.517 | collapses_with_enough_pairs |
| banking_d80 | gauss_white_rot | 0.923 | 1.000 | 0.973 | -0.979 | 0.008 | 0.017 | 0.517 | collapses_with_enough_pairs |
| banking_d80 | gw_q8 | 0.812 | 0.872 | — | -0.991 | 0.012 | 1.000 | 0.503 | partial_recovery_with_enough_pairs |
| banking_d80 | gw_q16 | 0.887 | 0.953 | — | -0.989 | 0.029 | 1.000 | 0.526 | partial_recovery_with_enough_pairs |
| banking_d80 | gw_q32 | 0.905 | 0.972 | — | -0.989 | 0.008 | 1.000 | 0.529 | collapses_with_enough_pairs |
| banking_d80 | gw_q64 | 0.909 | 0.978 | — | -0.989 | 0.017 | 1.000 | 0.528 | collapses_with_enough_pairs |
| banking_d80 | gw_q16_rot | 0.887 | 0.953 | 0.938 | -0.981 | 0.000 | 0.008 | 0.526 | partial_recovery_with_enough_pairs |
| banking_d80 | gw_q16_p90_rot | 0.799 | 0.896 | — | -1.001 | 0.009 | 0.009 | 0.521 | partial_recovery_with_enough_pairs |
| banking_d80 | gw_q16_p80_rot | 0.709 | 0.806 | — | -0.990 | 0.021 | 0.026 | 0.510 | partial_recovery_with_enough_pairs |
| banking_d80 | gw_q16_p70_rot | 0.626 | 0.752 | — | -0.995 | 0.030 | 0.018 | 0.497 | partial_recovery_with_enough_pairs |
| banking_d80 | gw_q16_p50_rot | 0.440 | 0.593 | 0.634 | -0.990 | 0.008 | 0.050 | 0.448 | resistant_in_tested_budget |
| banking_d80 | bn50 | 0.835 | 0.951 | — | -0.899 | 0.033 | 0.025 | 0.732 | partial_recovery_with_enough_pairs |
| banking_d80 | bn50_rot | 0.835 | 0.951 | — | -0.930 | 0.033 | 0.025 | 0.732 | partial_recovery_with_enough_pairs |
| banking_d80 | bn50_q16_rot | 0.750 | 0.925 | 0.830 | -0.928 | 0.033 | 0.033 | 0.709 | partial_recovery_with_enough_pairs |
| banking_d80 | bn50_adv_rot | 0.785 | 0.936 | — | -0.948 | 0.017 | 0.017 | 0.630 | partial_recovery_with_enough_pairs |
| banking_d200 | raw | 0.999 | 1.000 | 0.986 | -1.007 | 0.070 | 1.000 | 1.000 | collapses_with_enough_pairs |
| banking_d200 | gauss | 0.917 | 1.000 | — | -0.990 | 0.002 | 1.000 | 0.865 | collapses_with_enough_pairs |
| banking_d200 | gauss_white | 0.915 | 1.000 | — | -1.036 | 0.002 | 1.000 | 0.518 | collapses_with_enough_pairs |
| banking_d200 | gauss_white_rot | 0.915 | 1.000 | 0.970 | -1.021 | 0.005 | 0.005 | 0.518 | collapses_with_enough_pairs |
| banking_d200 | gw_q8 | 0.745 | 0.831 | — | -1.038 | 0.002 | 1.000 | 0.459 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q16 | 0.853 | 0.913 | — | -1.037 | 0.003 | 1.000 | 0.488 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q32 | 0.874 | 0.937 | — | -1.037 | 0.002 | 1.000 | 0.491 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q64 | 0.880 | 0.945 | — | -1.037 | 0.003 | 1.000 | 0.493 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q16_rot | 0.853 | 0.913 | 0.908 | -1.023 | 0.003 | 0.005 | 0.488 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q16_p90_rot | 0.748 | 0.837 | — | -1.038 | 0.002 | 0.007 | 0.481 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q16_p80_rot | 0.645 | 0.736 | — | -1.030 | 0.006 | 0.002 | 0.472 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q16_p70_rot | 0.550 | 0.649 | — | -1.036 | 0.005 | 0.010 | 0.457 | partial_recovery_with_enough_pairs |
| banking_d200 | gw_q16_p50_rot | 0.370 | 0.483 | 0.515 | -1.041 | 0.007 | 0.003 | 0.429 | resistant_in_tested_budget |
| banking_d200 | bn50 | 0.881 | 0.970 | — | -1.039 | 0.010 | 0.017 | 0.786 | partial_recovery_with_enough_pairs |
| banking_d200 | bn50_rot | 0.881 | 0.970 | — | -1.057 | 0.007 | 0.010 | 0.786 | partial_recovery_with_enough_pairs |
| banking_d200 | bn50_q16_rot | 0.819 | 0.956 | 0.899 | -1.059 | 0.013 | 0.007 | 0.754 | partial_recovery_with_enough_pairs |
| banking_d200 | bn50_adv_rot | 0.851 | 0.952 | — | -1.036 | 0.017 | 0.003 | 0.640 | partial_recovery_with_enough_pairs |

### 5.1 What the attacks actually show

- **Unpaired auxiliary data** is weak against Gaussianized+whitened data: second-order statistics are isotropic, so a secret rotation is not identified from $Z$ alone. ICA is theoretically uninformative for Gaussian coordinates. Reconstruction $R^2$ near 0 (or negative) in the unpaired columns is expected and should not be oversold as security.
- **Known pairs** are the honest stress test. If $f$ is essentially an invertible linear map after $G$ and $W$, then $n \gtrsim d$ pairs let ridge / Procrustes recover a high-fidelity inverse on held-out rows.
- **Quantization** and **$k<d$** make $f$ lossy. They can cap reconstruction quality even with many pairs, at a utility cost that must be measured rather than assumed.
- **Worst-attribute $R^2$** is reported because a mean reconstruction score can hide a single recoverable sensitive column.

Methods labeled as collapsing under enough known pairs: `gauss`, `gauss_white`, `gauss_white_rot`, `gw_q16`, `gw_q16_rot`, `gw_q32`, `gw_q64`, `raw`.

Figures: `results/figures/03_known_pairs_reconstruction.png`, `results/figures/04_per_feature_leakage.png`.

---

## 6. Privacy / Utility Tradeoff

Selection weight is **not** "maximum privacy". A transform that destroys $R^2$ is a failed candidate even if reconstruction also fails.

Figure: `results/figures/05_pareto_frontier.png` (vertical axis = $R^2$ retention; horizontal = worst-attribute leakage).

Also: `results/figures/06_structure_vs_utility.png` for the structure–utility relationship.

Selected `bn50_adv_rot` among methods with $R^2$ retention $\ge 0.90$ on **every** dataset (min retention=0.990, mean retention=1.001), then lowest mean worst-attribute known-pair leakage (0.930). Methods meeting the all-dataset 90% bar: `bn50`, `bn50_adv_rot`, `bn50_q16_rot`, `bn50_rot`, `gauss`.

A conceptual score `retention - 0.25 * leakage` was inspected only as a diagnostic. Raw $R^2$, RMSE, and attack $R^2$ in the tables are the quantities that matter.

---

## 7. Failure Modes

1. **Known-pair inversion of secret orthogonal maps.** $Z=W(G(X))R$ is not a one-way function. It is a keyed linear map in a Gaussianized space. Enough matched rows recover $R$ (up to quantization). Calling this encryption would be false.
2. **Tree models are not rotation-invariant.** If retention drops after $R$ while distance preservation stays high, the representation still contains structure that this particular $M_\theta$ uses inefficiently. That is a limitation of outsourcing to axis-aligned trees, not a privacy theorem.
3. **Auxiliary data does not need to reconstruct rows to be useful.** An attacker with $X_{\mathrm{aux}}$ already knows population marginals and can re-identify *types* of features if $G$ is omitted.
4. **Dimensionality is not automatically safety.** Extra features can add null space for $k<d$ projections, but a known-pair linear model still has parameters scaling as $O(kd)$. See `results/figures/07_dimensionality_vs_reconstruction.png`.
5. **Supervised bottlenecks can leak $X$ that is predictive of $y$.** Features that cause $y$ are the ones $E_\phi$ is incentivized to keep. Privacy for those attributes is the hardest, and per-attribute $R^2$ will show it.
6. **Neural attackers with many pairs** are a pessimistic bound: they approximate $f^{-1}$ including nonlinear $G^{-1}$. If they succeed, the representation is reversible in practice for that pair budget.

---

## 8. Final Recommendation

The preferred configuration is **`bn50_adv_rot`**, retaining on average **100.1%** of raw $R^2$ across datasets (mean transformed $R^2$=0.9531 vs raw 0.9519). Worst-attribute known-pair leakage remains **0.930**. This is an obfuscated / lossy representation, **not encryption**, and it does **not** resist known-pair inversion of the attributes that predict $y$.

Recommended operational interpretation:

- Use the transform only as an **obfuscated / lossy outsourced representation**, with a **secret key** ($R$, and fitted $G,W$ or $E_\phi$) kept on the data-owner side.
- Assume that **known-pair leakage is the binding constraint**. If an adversary can ever obtain matched $(X,Z)$ rows, treat invertible stages (especially $R$ without $Q_L$ or $k<d$) as broken.
- Do not ship a configuration whose $R^2$ retention collapses below the task's tolerance, even if attacks look weak.

### Question 1 — Same model, same $\theta$

| Dataset | Method | Transformed $R^2$ | Retention | $\Delta R^2$ |
| --- | --- | --- | --- | --- |
| openml_superconduct | raw | 0.9129 ± 0.0040 | 1.000 | 0.0000 |
| openml_superconduct | gauss | 0.9129 ± 0.0040 | 1.000 | 0.0000 |
| openml_superconduct | gauss_white | 0.8940 ± 0.0062 | 0.979 | -0.0189 |
| openml_superconduct | gauss_white_rot | 0.8754 ± 0.0087 | 0.959 | -0.0375 |
| openml_superconduct | gw_q8 | 0.8759 ± 0.0058 | 0.959 | -0.0370 |
| openml_superconduct | gw_q16 | 0.8853 ± 0.0067 | 0.970 | -0.0277 |
| openml_superconduct | gw_q32 | 0.8888 ± 0.0054 | 0.974 | -0.0241 |
| openml_superconduct | gw_q64 | 0.8906 ± 0.0056 | 0.976 | -0.0223 |
| openml_superconduct | gw_q16_rot | 0.8668 ± 0.0104 | 0.949 | -0.0461 |
| openml_superconduct | gw_q16_p90_rot | 0.8606 ± 0.0053 | 0.943 | -0.0524 |
| openml_superconduct | gw_q16_p80_rot | 0.8556 ± 0.0022 | 0.937 | -0.0573 |
| openml_superconduct | gw_q16_p70_rot | 0.8528 ± 0.0033 | 0.934 | -0.0601 |
| openml_superconduct | gw_q16_p50_rot | 0.8288 ± 0.0082 | 0.908 | -0.0841 |
| openml_superconduct | bn50 | 0.8988 ± 0.0048 | 0.985 | -0.0142 |
| openml_superconduct | bn50_rot | 0.9037 ± 0.0038 | 0.990 | -0.0092 |
| openml_superconduct | bn50_q16_rot | 0.8927 ± 0.0065 | 0.978 | -0.0203 |
| openml_superconduct | bn50_adv_rot | 0.9039 ± 0.0066 | 0.990 | -0.0090 |
| banking_d80 | raw | 0.9723 ± 0.0025 | 1.000 | 0.0000 |
| banking_d80 | gauss | 0.9723 ± 0.0025 | 1.000 | 0.0000 |
| banking_d80 | gauss_white | 0.8779 ± 0.0025 | 0.903 | -0.0944 |
| banking_d80 | gauss_white_rot | 0.8089 ± 0.0059 | 0.832 | -0.1634 |
| banking_d80 | gw_q8 | 0.8433 ± 0.0074 | 0.867 | -0.1290 |
| banking_d80 | gw_q16 | 0.8734 ± 0.0039 | 0.898 | -0.0989 |
| banking_d80 | gw_q32 | 0.8800 ± 0.0027 | 0.905 | -0.0923 |
| banking_d80 | gw_q64 | 0.8818 ± 0.0011 | 0.907 | -0.0906 |
| banking_d80 | gw_q16_rot | 0.7904 ± 0.0085 | 0.813 | -0.1819 |
| banking_d80 | gw_q16_p90_rot | 0.7539 ± 0.0119 | 0.775 | -0.2184 |
| banking_d80 | gw_q16_p80_rot | 0.7270 ± 0.0190 | 0.748 | -0.2453 |
| banking_d80 | gw_q16_p70_rot | 0.6848 ± 0.0376 | 0.704 | -0.2875 |
| banking_d80 | gw_q16_p50_rot | 0.5486 ± 0.0285 | 0.564 | -0.4237 |
| banking_d80 | bn50 | 0.9789 ± 0.0018 | 1.007 | 0.0066 |
| banking_d80 | bn50_rot | 0.9786 ± 0.0017 | 1.006 | 0.0063 |
| banking_d80 | bn50_q16_rot | 0.9769 ± 0.0019 | 1.005 | 0.0045 |
| banking_d80 | bn50_adv_rot | 0.9781 ± 0.0012 | 1.006 | 0.0058 |
| banking_d200 | raw | 0.9704 ± 0.0009 | 1.000 | 0.0000 |
| banking_d200 | gauss | 0.9704 ± 0.0009 | 1.000 | 0.0000 |
| banking_d200 | gauss_white | 0.8368 ± 0.0021 | 0.862 | -0.1336 |
| banking_d200 | gauss_white_rot | 0.6642 ± 0.0111 | 0.684 | -0.3062 |
| banking_d200 | gw_q8 | 0.8002 ± 0.0067 | 0.825 | -0.1701 |
| banking_d200 | gw_q16 | 0.8308 ± 0.0018 | 0.856 | -0.1395 |
| banking_d200 | gw_q32 | 0.8397 ± 0.0036 | 0.865 | -0.1307 |
| banking_d200 | gw_q64 | 0.8371 ± 0.0040 | 0.863 | -0.1332 |
| banking_d200 | gw_q16_rot | 0.6311 ± 0.0131 | 0.650 | -0.3392 |
| banking_d200 | gw_q16_p90_rot | 0.5922 ± 0.0228 | 0.610 | -0.3781 |
| banking_d200 | gw_q16_p80_rot | 0.5666 ± 0.0354 | 0.584 | -0.4037 |
| banking_d200 | gw_q16_p70_rot | 0.5344 ± 0.0284 | 0.551 | -0.4360 |
| banking_d200 | gw_q16_p50_rot | 0.4351 ± 0.0348 | 0.448 | -0.5352 |
| banking_d200 | bn50 | 0.9781 ± 0.0007 | 1.008 | 0.0077 |
| banking_d200 | bn50_rot | 0.9782 ± 0.0004 | 1.008 | 0.0078 |
| banking_d200 | bn50_q16_rot | 0.9767 ± 0.0003 | 1.006 | 0.0063 |
| banking_d200 | bn50_adv_rot | 0.9773 ± 0.0005 | 1.007 | 0.0070 |

### Question 2 — Learnable structure

See retention, distance preservation, and the structure-vs-utility figure. High distance correlation with high $R^2$ retention means geometry survived. High distance correlation with low retention means this model class failed to exploit surviving geometry.

### Question 3 — Can an attacker reconstruct $X$?

Quantitatively: use the attack table. Unpaired reconstruction $R^2$ is typically near zero or negative after Gaussianization+whitening+rotation. Known-pair and neural reconstruction $R^2$ can be large for invertible stages. Always read **worst-attribute** $R^2$, not only the mean.

### Question 4 — Does larger $d$ make reverse engineering harder?

This is tested by the banking $d=80$ vs $d=200$ pair (or the quick $d=40$ vs $d=80$ pair). If known-pair $R^2$ stays high as $d$ grows, the obstacle was **not** combinatorial semantic search; it was a missing key, and the key is statistically identifiable from $O(d)$ pairs. If reconstruction falls only when $k<d$ or quantization is applied, the protection is **information-theoretic loss**, not computational hardness. Figure: `results/figures/07_dimensionality_vs_reconstruction.png`.

### Question 5 — Best utility/security tradeoff

Precision / $R^2$ retention is primary. The preferred method is `bn50_adv_rot`. If that method still has high known-pair $R^2$, then **no configuration in this grid simultaneously kept utility and resisted known-pair inversion**. That outcome is allowed and scientifically useful.

---

## Reproducibility

```bash
pip install -r requirements.txt
python run.py           # full protocol
python run.py --quick   # subset for smoke-testing
```

Artifacts: `results/tables/`, `results/figures/`, per-seed JSON under `results/<dataset>/`.
