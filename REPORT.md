# Utility-Preserving Privacy Transformations

This report is generated from the experimental protocol in `src/experiment.py`.
It is **not** a cryptographic security claim.

## 1. Current transformation analysis

The repository originally shipped no implementation on `main`. A prior branch
(`cursor/privacy-preserving-ml-transforms-6ed8`) studied numeric-only maps
`Z = Q(W(G(X))) P R` and a supervised bottleneck, evaluated only with
`HistGradientBoostingRegressor`. That work is treated as the **current**
baseline family (`gauss`, `gauss_white_rot`, `bn_adv`) and is re-run here
against heterogeneous tables and multiple model families.

### What the classical maps preserve / destroy

- **`gauss`**: preserves within-column rank and tree splits; destroys units,
  scale, and marginal shape. Invertible via the quantile map for numeric
  columns. Predictions map back because the target affine `g` is inverted.
- **`gauss_white_rot`**: preserves second-order geometry up to a secret
  rotation; destroys column identity and axis-aligned structure. Fully
  invertible given the key; statistically identifiable from `O(d)` pairs.
- **`keyed_monotone` / `typed_keyed`**: preserve per-column order and
  categorical equality; destroy names, string literals (HMAC buckets), and
  original codes. Partially invertible. Tree utility stays high.
- **`bn_adv` / `vib`**: preserve `I(Z;Y)` by local training; destroy features
  orthogonal to `Y` and column identity. Not invertible. Known-pair attackers
  still recover the *predictive* raw attributes.
- **`rff` / `microagg_rot` / `noisy_gauss_rot`**: lossy / nonlinear; lower
  leakage, usually lower utility.

## 2. Weaknesses found

1. **Known-pair inversion is the binding constraint.** Any deterministic
   invertible stage (especially secret rotations) collapses once the attacker
   holds on the order of `d` matched rows.
2. **Utility-preserving maps must keep `I(Z; X_predictive)`.** Features that
   cause `Y` remain the leaky ones under Attacker A, even when values look random.
3. **Tree vs linear disagreement.** Rotations hurt axis-aligned trees while
   leaving linear/MLP models closer to raw performance — a model-class effect,
   not information destruction.
4. **Leaving `Y` in the clear is a leak.** This protocol always applies an
   invertible target map `g` except for `identity`. Utility is scored after `g^{-1}`.
5. **Column-wise monotone maps hide names but not order.** Rank attacks recover
   numeric features from few pairs.

## 3. Candidate alternative transformations

Implemented and benchmarked: `identity`, `gauss`, `gauss_white_rot`,
`keyed_monotone`, `typed_keyed`, `bn_adv`, `bn_noisy`, `vib`, `vib_stoch`,
`rff`, `microagg_rot`, `noisy_gauss_rot`.

Literature that informed the grid (see `RESEARCH.md`): variational information
bottleneck / privacy funnel; random Fourier features; microaggregation;
adversarial representation learning; typed masking / FPE-style hashing.
Synthetic-data and DP-SGD methods were considered and rejected as the primary
mechanism because they do not implement `X_raw → X_transformed` with
row-aligned outsourced training.

## 4. Utility benchmark

Models: ridge/logistic (`linear`), `HistGradientBoosting` (`hgb`), MLP, k-NN.
Metric: regression `R²`, classification accuracy, both after mapping predictions
back to the original target space. Retention is transformed_score / raw_score.

| dataset | method | model | score_mean | retention_mean | agreement_mean | worst_attr_r2_mean | recon_r2_mean | semantic_acc_mean | membership_auc_mean | n_pairs_r2_0.5_mean | known_pair_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| california_housing | identity | linear | -336.897 | 1.000 | 1.000 | 1.000 | 0.994 | 0.000 | 0.501 | 25.000 | collapses_with_enough_pairs |
| california_housing | identity | hgb | 0.811 | 1.000 | 1.000 | 1.000 | 0.994 | 0.000 | 0.501 | 25.000 | collapses_with_enough_pairs |
| california_housing | identity | mlp | -50.470 | 1.000 | 1.000 | 1.000 | 0.994 | 0.000 | 0.501 | 25.000 | collapses_with_enough_pairs |
| california_housing | identity | knn | 0.747 | 1.000 | 1.000 | 1.000 | 0.994 | 0.000 | 0.501 | 25.000 | collapses_with_enough_pairs |
| california_housing | gauss | linear | 0.647 | -0.002 | 0.012 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | hgb | 0.811 | 1.000 | 1.000 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | mlp | 0.701 | -0.014 | 0.166 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | knn | 0.713 | 0.954 | 0.918 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | linear | 0.647 | -0.002 | 0.012 | 0.912 | 0.584 | 0.000 | 0.530 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | hgb | 0.776 | 0.956 | 0.931 | 0.912 | 0.584 | 0.000 | 0.530 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | mlp | 0.698 | -0.014 | 0.158 | 0.912 | 0.584 | 0.000 | 0.530 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | knn | 0.713 | 0.954 | 0.900 | 0.912 | 0.584 | 0.000 | 0.530 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | linear | 0.659 | -0.002 | -0.013 | 0.996 | 0.599 | — | 0.502 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | hgb | 0.807 | 0.995 | 0.993 | 0.996 | 0.599 | — | 0.502 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | mlp | 0.729 | -0.014 | 0.206 | 0.996 | 0.599 | — | 0.502 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | knn | 0.698 | 0.935 | 0.914 | 0.996 | 0.599 | — | 0.502 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | linear | 0.647 | -0.002 | 0.012 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | hgb | 0.811 | 1.000 | 1.000 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | mlp | 0.700 | -0.014 | 0.161 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | knn | 0.713 | 0.954 | 0.918 | 0.912 | 0.584 | 0.000 | 0.526 | 10.000 | partial_recovery_with_enough_pairs |
| california_housing | bn_adv | linear | -31.138 | 0.092 | -0.977 | 0.145 | -135.328 | — | 0.510 | — | resistant_in_tested_budget |
| california_housing | bn_adv | hgb | 0.730 | 0.900 | 0.948 | 0.145 | -135.328 | — | 0.510 | — | resistant_in_tested_budget |
| california_housing | bn_adv | mlp | -72.771 | 1.442 | 1.000 | 0.145 | -135.328 | — | 0.510 | — | resistant_in_tested_budget |
| california_housing | bn_adv | knn | 0.784 | 1.050 | 0.929 | 0.145 | -135.328 | — | 0.510 | — | resistant_in_tested_budget |
| california_housing | vib | linear | -12.781 | 0.038 | -0.956 | -0.016 | -623.204 | — | 0.500 | — | resistant_in_tested_budget |
| california_housing | vib | hgb | 0.762 | 0.939 | 0.944 | -0.016 | -623.204 | — | 0.500 | — | resistant_in_tested_budget |
| california_housing | vib | mlp | -38.787 | 0.769 | 0.999 | -0.016 | -623.204 | — | 0.500 | — | resistant_in_tested_budget |
| california_housing | vib | knn | 0.729 | 0.977 | 0.914 | -0.016 | -623.204 | — | 0.500 | — | resistant_in_tested_budget |
| california_housing | rff | linear | 0.348 | -0.001 | 0.055 | 0.923 | 0.495 | — | 0.515 | — | resistant_in_tested_budget |
| california_housing | rff | hgb | 0.605 | 0.746 | 0.832 | 0.923 | 0.495 | — | 0.515 | — | resistant_in_tested_budget |
| california_housing | rff | mlp | 0.606 | -0.012 | 0.084 | 0.923 | 0.495 | — | 0.515 | — | resistant_in_tested_budget |
| california_housing | rff | knn | 0.537 | 0.719 | 0.838 | 0.923 | 0.495 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_reg | identity | linear | 0.879 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.530 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | hgb | 0.919 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.530 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | mlp | -3.346 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.530 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | knn | 0.723 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.530 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | gauss | linear | 0.852 | 0.969 | 0.991 | 1.000 | 0.872 | 0.133 | 0.560 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | hgb | 0.919 | 1.000 | 1.000 | 1.000 | 0.872 | 0.133 | 0.560 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | mlp | -3.641 | 1.088 | 0.938 | 1.000 | 0.872 | 0.133 | 0.560 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | knn | 0.688 | 0.951 | 0.939 | 1.000 | 0.872 | 0.133 | 0.560 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | linear | 0.852 | 0.969 | 0.991 | 1.000 | 0.872 | 0.067 | 0.545 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | hgb | 0.803 | 0.873 | 0.935 | 1.000 | 0.872 | 0.067 | 0.545 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | mlp | -3.689 | 1.102 | 0.766 | 1.000 | 0.872 | 0.067 | 0.545 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | knn | 0.643 | 0.889 | 0.896 | 1.000 | 0.872 | 0.067 | 0.545 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | linear | 0.911 | 1.036 | 0.987 | 1.000 | 0.927 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | hgb | 0.917 | 0.997 | 0.997 | 1.000 | 0.927 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | mlp | -3.546 | 1.060 | 0.904 | 1.000 | 0.927 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | knn | 0.713 | 0.985 | 0.932 | 1.000 | 0.927 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | typed_keyed | linear | 0.851 | 0.968 | 0.991 | 1.000 | 0.636 | 0.000 | 0.515 | 50.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | hgb | 0.921 | 1.001 | 0.997 | 1.000 | 0.636 | 0.000 | 0.515 | 50.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | mlp | -3.630 | 1.085 | 0.911 | 1.000 | 0.636 | 0.000 | 0.515 | 50.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | knn | 0.699 | 0.966 | 0.921 | 1.000 | 0.636 | 0.000 | 0.515 | 50.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | bn_adv | linear | 0.904 | 1.029 | 0.976 | 0.915 | 0.282 | — | 0.541 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | hgb | 0.899 | 0.978 | 0.973 | 0.915 | 0.282 | — | 0.541 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | mlp | -3.235 | 0.967 | 0.680 | 0.915 | 0.282 | — | 0.541 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | knn | 0.905 | 1.251 | 0.919 | 0.915 | 0.282 | — | 0.541 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | linear | 0.901 | 1.025 | 0.978 | 0.916 | 0.385 | — | 0.540 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | hgb | 0.907 | 0.986 | 0.975 | 0.916 | 0.385 | — | 0.540 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | mlp | -3.362 | 1.005 | 0.585 | 0.916 | 0.385 | — | 0.540 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | knn | 0.914 | 1.264 | 0.925 | 0.916 | 0.385 | — | 0.540 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | linear | 0.565 | 0.643 | 0.786 | 0.606 | 0.458 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | hgb | 0.564 | 0.613 | 0.783 | 0.606 | 0.458 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | mlp | -3.449 | 1.031 | 0.627 | 0.606 | 0.458 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | knn | 0.505 | 0.699 | 0.777 | 0.606 | 0.458 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_clf | identity | linear | 0.915 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.511 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | hgb | 0.911 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.511 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | mlp | 0.878 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.511 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | knn | 0.848 | 1.000 | 1.000 | 1.000 | 0.996 | 0.467 | 0.511 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | gauss | linear | 0.915 | 1.000 | 0.985 | 1.000 | 0.866 | 0.133 | 0.506 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | hgb | 0.911 | 1.000 | 1.000 | 1.000 | 0.866 | 0.133 | 0.506 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | mlp | 0.885 | 1.008 | 0.948 | 1.000 | 0.866 | 0.133 | 0.506 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | knn | 0.815 | 0.961 | 0.922 | 1.000 | 0.866 | 0.133 | 0.506 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | linear | 0.922 | 1.008 | 0.985 | 1.000 | 0.866 | 0.067 | 0.509 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | hgb | 0.837 | 0.919 | 0.896 | 1.000 | 0.866 | 0.067 | 0.509 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | mlp | 0.889 | 1.013 | 0.937 | 1.000 | 0.866 | 0.067 | 0.509 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | knn | 0.789 | 0.930 | 0.881 | 1.000 | 0.866 | 0.067 | 0.509 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | linear | 0.919 | 1.004 | 0.989 | 1.000 | 0.944 | — | 0.541 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | hgb | 0.900 | 0.988 | 0.974 | 1.000 | 0.944 | — | 0.541 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | mlp | 0.822 | 0.937 | 0.900 | 1.000 | 0.944 | — | 0.541 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | knn | 0.822 | 0.969 | 0.900 | 1.000 | 0.944 | — | 0.541 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | typed_keyed | linear | 0.904 | 0.988 | 0.981 | 1.000 | 0.611 | 0.000 | 0.517 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | hgb | 0.896 | 0.984 | 0.978 | 1.000 | 0.611 | 0.000 | 0.517 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | mlp | 0.900 | 1.025 | 0.948 | 1.000 | 0.611 | 0.000 | 0.517 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | knn | 0.830 | 0.978 | 0.907 | 1.000 | 0.611 | 0.000 | 0.517 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | bn_adv | linear | 0.885 | 0.968 | 0.963 | 0.914 | 0.257 | — | 0.507 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | hgb | 0.885 | 0.972 | 0.944 | 0.914 | 0.257 | — | 0.507 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | mlp | 0.904 | 1.030 | 0.930 | 0.914 | 0.257 | — | 0.507 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | knn | 0.896 | 1.057 | 0.900 | 0.914 | 0.257 | — | 0.507 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | linear | 0.900 | 0.984 | 0.963 | 0.908 | 0.305 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | hgb | 0.870 | 0.955 | 0.930 | 0.908 | 0.305 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | mlp | 0.885 | 1.008 | 0.911 | 0.908 | 0.305 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | knn | 0.889 | 1.048 | 0.900 | 0.908 | 0.305 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | linear | 0.811 | 0.887 | 0.837 | 0.626 | 0.342 | — | 0.507 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | hgb | 0.785 | 0.862 | 0.830 | 0.626 | 0.342 | — | 0.507 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | mlp | 0.770 | 0.878 | 0.826 | 0.626 | 0.342 | — | 0.507 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | knn | 0.800 | 0.943 | 0.848 | 0.626 | 0.342 | — | 0.507 | — | resistant_in_tested_budget |


## 5. Attacker A results

Known-pair grid: 1, 2, 5, 10, 25, 50, 100, 200. Attacks: ridge, pinv,
rank/order matching, categorical frequency, HGB inversion, MLP inversion.

| dataset | method | recon_r2_mean | worst_attr_r2_mean | sensitive_attr_r2_mean | n_pairs_r2_0.5_mean | spearman_leak_mean | known_pair_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| california_housing | identity | 0.994 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | identity | 0.994 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | identity | 0.994 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | identity | 0.994 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | gauss | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.584 | 0.912 | 0.912 | 10.000 | 0.774 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.584 | 0.912 | 0.912 | 10.000 | 0.774 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.584 | 0.912 | 0.912 | 10.000 | 0.774 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.584 | 0.912 | 0.912 | 10.000 | 0.774 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.599 | 0.996 | 0.946 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.599 | 0.996 | 0.946 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.599 | 0.996 | 0.946 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.599 | 0.996 | 0.946 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.584 | 0.912 | 0.912 | 10.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | bn_adv | -135.328 | 0.145 | -40.832 | — | 0.840 | resistant_in_tested_budget |
| california_housing | bn_adv | -135.328 | 0.145 | -40.832 | — | 0.840 | resistant_in_tested_budget |
| california_housing | bn_adv | -135.328 | 0.145 | -40.832 | — | 0.840 | resistant_in_tested_budget |
| california_housing | bn_adv | -135.328 | 0.145 | -40.832 | — | 0.840 | resistant_in_tested_budget |
| california_housing | vib | -623.204 | -0.016 | -2.587 | — | 0.815 | resistant_in_tested_budget |
| california_housing | vib | -623.204 | -0.016 | -2.587 | — | 0.815 | resistant_in_tested_budget |
| california_housing | vib | -623.204 | -0.016 | -2.587 | — | 0.815 | resistant_in_tested_budget |
| california_housing | vib | -623.204 | -0.016 | -2.587 | — | 0.815 | resistant_in_tested_budget |
| california_housing | rff | 0.495 | 0.923 | 0.923 | — | 0.829 | resistant_in_tested_budget |
| california_housing | rff | 0.495 | 0.923 | 0.923 | — | 0.829 | resistant_in_tested_budget |
| california_housing | rff | 0.495 | 0.923 | 0.923 | — | 0.829 | resistant_in_tested_budget |
| california_housing | rff | 0.495 | 0.923 | 0.923 | — | 0.829 | resistant_in_tested_budget |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | gauss | 0.872 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | 0.872 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | 0.872 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | 0.872 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.872 | 1.000 | 1.000 | 25.000 | 0.637 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.872 | 1.000 | 1.000 | 25.000 | 0.637 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.872 | 1.000 | 1.000 | 25.000 | 0.637 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.872 | 1.000 | 1.000 | 25.000 | 0.637 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.927 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.927 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.927 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.927 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.636 | 1.000 | 1.000 | 50.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.636 | 1.000 | 1.000 | 50.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.636 | 1.000 | 1.000 | 50.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.636 | 1.000 | 1.000 | 50.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | bn_adv | 0.282 | 0.915 | 0.915 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | 0.282 | 0.915 | 0.915 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | 0.282 | 0.915 | 0.915 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | 0.282 | 0.915 | 0.915 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.385 | 0.916 | 0.916 | — | 0.952 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.385 | 0.916 | 0.916 | — | 0.952 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.385 | 0.916 | 0.916 | — | 0.952 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.385 | 0.916 | 0.916 | — | 0.952 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.458 | 0.606 | 0.606 | — | 0.676 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.458 | 0.606 | 0.606 | — | 0.676 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.458 | 0.606 | 0.606 | — | 0.676 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.458 | 0.606 | 0.606 | — | 0.676 | resistant_in_tested_budget |
| banking_mixed_clf | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | gauss | 0.866 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | 0.866 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | 0.866 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | 0.866 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.866 | 1.000 | 1.000 | 25.000 | 0.651 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.866 | 1.000 | 1.000 | 25.000 | 0.651 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.866 | 1.000 | 1.000 | 25.000 | 0.651 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.866 | 1.000 | 1.000 | 25.000 | 0.651 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.944 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.944 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.944 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.944 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.611 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.611 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.611 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.611 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | bn_adv | 0.257 | 0.914 | 0.914 | — | 0.962 | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | 0.257 | 0.914 | 0.914 | — | 0.962 | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | 0.257 | 0.914 | 0.914 | — | 0.962 | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | 0.257 | 0.914 | 0.914 | — | 0.962 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.305 | 0.908 | 0.908 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.305 | 0.908 | 0.908 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.305 | 0.908 | 0.908 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.305 | 0.908 | 0.908 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.342 | 0.626 | 0.558 | — | 0.641 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.342 | 0.626 | 0.558 | — | 0.641 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.342 | 0.626 | 0.558 | — | 0.641 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.342 | 0.626 | 0.558 | — | 0.641 | resistant_in_tested_budget |


## 6. Attacker B results

The attacker receives only `Z` and a context sentence
(`This dataset comes from a bank...` or the housing description).
No keys, schema, column names, or raw rows.

| dataset | method | semantic_acc_mean | kind_acc_mean | membership_auc_mean | linkage_mean | sensitive_hit_mean |
| --- | --- | --- | --- | --- | --- | --- |
| california_housing | identity | 0.000 | 1.000 | 0.501 | — | 0.000 |
| california_housing | identity | 0.000 | 1.000 | 0.501 | — | 0.000 |
| california_housing | identity | 0.000 | 1.000 | 0.501 | — | 0.000 |
| california_housing | identity | 0.000 | 1.000 | 0.501 | — | 0.000 |
| california_housing | gauss | 0.000 | 1.000 | 0.526 | — | 0.000 |
| california_housing | gauss | 0.000 | 1.000 | 0.526 | — | 0.000 |
| california_housing | gauss | 0.000 | 1.000 | 0.526 | — | 0.000 |
| california_housing | gauss | 0.000 | 1.000 | 0.526 | — | 0.000 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.530 | — | 0.667 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.530 | — | 0.667 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.530 | — | 0.667 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.530 | — | 0.667 |
| california_housing | keyed_monotone | — | — | 0.502 | — | — |
| california_housing | keyed_monotone | — | — | 0.502 | — | — |
| california_housing | keyed_monotone | — | — | 0.502 | — | — |
| california_housing | keyed_monotone | — | — | 0.502 | — | — |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.526 | — | 0.333 |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.526 | — | 0.333 |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.526 | — | 0.333 |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.526 | — | 0.333 |
| california_housing | bn_adv | — | — | 0.510 | — | — |
| california_housing | bn_adv | — | — | 0.510 | — | — |
| california_housing | bn_adv | — | — | 0.510 | — | — |
| california_housing | bn_adv | — | — | 0.510 | — | — |
| california_housing | vib | — | — | 0.500 | — | — |
| california_housing | vib | — | — | 0.500 | — | — |
| california_housing | vib | — | — | 0.500 | — | — |
| california_housing | vib | — | — | 0.500 | — | — |
| california_housing | rff | — | — | 0.515 | — | — |
| california_housing | rff | — | — | 0.515 | — | — |
| california_housing | rff | — | — | 0.515 | — | — |
| california_housing | rff | — | — | 0.515 | — | — |
| banking_mixed_reg | identity | 0.467 | 0.733 | 0.530 | — | 0.889 |
| banking_mixed_reg | identity | 0.467 | 0.733 | 0.530 | — | 0.889 |
| banking_mixed_reg | identity | 0.467 | 0.733 | 0.530 | — | 0.889 |
| banking_mixed_reg | identity | 0.467 | 0.733 | 0.530 | — | 0.889 |
| banking_mixed_reg | gauss | 0.133 | 0.733 | 0.560 | — | 0.889 |
| banking_mixed_reg | gauss | 0.133 | 0.733 | 0.560 | — | 0.889 |
| banking_mixed_reg | gauss | 0.133 | 0.733 | 0.560 | — | 0.889 |
| banking_mixed_reg | gauss | 0.133 | 0.733 | 0.560 | — | 0.889 |
| banking_mixed_reg | gauss_white_rot | 0.067 | 0.533 | 0.545 | — | 0.444 |
| banking_mixed_reg | gauss_white_rot | 0.067 | 0.533 | 0.545 | — | 0.444 |
| banking_mixed_reg | gauss_white_rot | 0.067 | 0.533 | 0.545 | — | 0.444 |
| banking_mixed_reg | gauss_white_rot | 0.067 | 0.533 | 0.545 | — | 0.444 |
| banking_mixed_reg | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_reg | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_reg | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_reg | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_reg | typed_keyed | 0.000 | 0.467 | 0.515 | — | 0.667 |
| banking_mixed_reg | typed_keyed | 0.000 | 0.467 | 0.515 | — | 0.667 |
| banking_mixed_reg | typed_keyed | 0.000 | 0.467 | 0.515 | — | 0.667 |
| banking_mixed_reg | typed_keyed | 0.000 | 0.467 | 0.515 | — | 0.667 |
| banking_mixed_reg | bn_adv | — | — | 0.541 | — | — |
| banking_mixed_reg | bn_adv | — | — | 0.541 | — | — |
| banking_mixed_reg | bn_adv | — | — | 0.541 | — | — |
| banking_mixed_reg | bn_adv | — | — | 0.541 | — | — |
| banking_mixed_reg | vib | — | — | 0.540 | — | — |
| banking_mixed_reg | vib | — | — | 0.540 | — | — |
| banking_mixed_reg | vib | — | — | 0.540 | — | — |
| banking_mixed_reg | vib | — | — | 0.540 | — | — |
| banking_mixed_reg | rff | — | — | 0.515 | — | — |
| banking_mixed_reg | rff | — | — | 0.515 | — | — |
| banking_mixed_reg | rff | — | — | 0.515 | — | — |
| banking_mixed_reg | rff | — | — | 0.515 | — | — |
| banking_mixed_clf | identity | 0.467 | 0.733 | 0.511 | — | 0.889 |
| banking_mixed_clf | identity | 0.467 | 0.733 | 0.511 | — | 0.889 |
| banking_mixed_clf | identity | 0.467 | 0.733 | 0.511 | — | 0.889 |
| banking_mixed_clf | identity | 0.467 | 0.733 | 0.511 | — | 0.889 |
| banking_mixed_clf | gauss | 0.133 | 0.733 | 0.506 | — | 0.889 |
| banking_mixed_clf | gauss | 0.133 | 0.733 | 0.506 | — | 0.889 |
| banking_mixed_clf | gauss | 0.133 | 0.733 | 0.506 | — | 0.889 |
| banking_mixed_clf | gauss | 0.133 | 0.733 | 0.506 | — | 0.889 |
| banking_mixed_clf | gauss_white_rot | 0.067 | 0.533 | 0.509 | — | 0.444 |
| banking_mixed_clf | gauss_white_rot | 0.067 | 0.533 | 0.509 | — | 0.444 |
| banking_mixed_clf | gauss_white_rot | 0.067 | 0.533 | 0.509 | — | 0.444 |
| banking_mixed_clf | gauss_white_rot | 0.067 | 0.533 | 0.509 | — | 0.444 |
| banking_mixed_clf | keyed_monotone | — | — | 0.541 | — | — |
| banking_mixed_clf | keyed_monotone | — | — | 0.541 | — | — |
| banking_mixed_clf | keyed_monotone | — | — | 0.541 | — | — |
| banking_mixed_clf | keyed_monotone | — | — | 0.541 | — | — |
| banking_mixed_clf | typed_keyed | 0.000 | 0.467 | 0.517 | — | 0.667 |
| banking_mixed_clf | typed_keyed | 0.000 | 0.467 | 0.517 | — | 0.667 |
| banking_mixed_clf | typed_keyed | 0.000 | 0.467 | 0.517 | — | 0.667 |
| banking_mixed_clf | typed_keyed | 0.000 | 0.467 | 0.517 | — | 0.667 |
| banking_mixed_clf | bn_adv | — | — | 0.507 | — | — |
| banking_mixed_clf | bn_adv | — | — | 0.507 | — | — |
| banking_mixed_clf | bn_adv | — | — | 0.507 | — | — |
| banking_mixed_clf | bn_adv | — | — | 0.507 | — | — |
| banking_mixed_clf | vib | — | — | 0.506 | — | — |
| banking_mixed_clf | vib | — | — | 0.506 | — | — |
| banking_mixed_clf | vib | — | — | 0.506 | — | — |
| banking_mixed_clf | vib | — | — | 0.506 | — | — |
| banking_mixed_clf | rff | — | — | 0.507 | — | — |
| banking_mixed_clf | rff | — | — | 0.507 | — | — |
| banking_mixed_clf | rff | — | — | 0.507 | — | — |
| banking_mixed_clf | rff | — | — | 0.507 | — | — |


## 7. Security / utility trade-off

Pareto view: HGB retention vs known-pair worst-attribute `R²`
(see `results/figures/02_pareto_utility_leakage.png`).
Selection rule: maximize practical privacy subject to minimal utility loss,
operationalized as lowest mean worst-attribute leakage among methods with
HGB retention ≥ 0.85 on every dataset when such methods exist.

Selected method: **`vib`**
(retention=0.9602526890116941, worst-attr leakage=0.6026803622381801,
utility floor met=True).

## 8. Recommended architecture

Keep the schema, keys, and `g^{-1}` on the data-owner side.

```
owner:  X, y, schema
        -> type-aware encode (HMAC strings, keyed cats, quantile nums)
        -> optional local VIB / bottleneck if y is available locally
        -> secret permutation / rotation
        -> g(y) invertible target map
        -> send (Z, y_tilde, anonymous column ids) to trainer
trainer: fit M: Z -> y_tilde   (no names, no raw values)
owner:   y_hat = g^{-1}(M(Z_new))
```

This satisfies: train in transformed space; map predictions back; hide
column names; support mixed types. It does **not** claim known-pair security.

## 9. Recommended transformation

**`vib`** is the empirically preferred point on this grid.
Use `typed_keyed` when the owner cannot train a local encoder (no `y` yet,
or model-agnostic export). Use `vib` / `bn_adv` when a local labelled fit
is acceptable and column semantics must be thoroughly mixed.

## 10. Remaining attack surface

- Known-pair inversion of any approximately invertible or low-dimensional map.
- Recovery of attributes that cause `Y` from `Z` whenever `I(Z;Y)` is large.
- Rank/frequency attacks against per-column monotone or keyed-categorical maps.
- Linkage if neighborhood geometry is preserved and an overlapping public table exists.
- Membership inference on deterministic unique rows (identifiers).
- Side-channel leakage from `out_dim`, sparsity, and dummy-column structure.
- The trainer still sees `y_tilde`; a weak `g` (affine) hides units, not ranks.

## 11. Permanent regression tests

See `tests/`. They lock: target invertibility, anonymous column ids, mixed-type
support, identity utility ceiling, Attacker A/B interfaces, and the requirement
that a recommended high-utility method must not expose raw column names.

## Reproducibility

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python run.py --quick
python run.py
```
