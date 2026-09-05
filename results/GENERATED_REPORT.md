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
| california_housing | identity | linear | 0.611 | 1.000 | 1.000 | 1.000 | 0.999 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | identity | hgb | 0.789 | 1.000 | 1.000 | 1.000 | 0.999 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | identity | mlp | 0.643 | 1.000 | 1.000 | 1.000 | 0.999 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | identity | knn | 0.644 | 1.000 | 1.000 | 1.000 | 0.999 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | linear | 0.611 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | hgb | 0.792 | 1.005 | 0.995 | 1.000 | 1.000 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | mlp | 0.584 | 0.903 | 0.982 | 1.000 | 1.000 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | knn | 0.644 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.545 | 10.000 | collapses_with_enough_pairs |
| california_housing | std_rot | linear | 0.611 | 1.000 | 1.000 | 1.000 | 0.961 | 0.000 | 0.549 | 7.500 | collapses_with_enough_pairs |
| california_housing | std_rot | hgb | 0.694 | 0.880 | 0.932 | 1.000 | 0.961 | 0.000 | 0.549 | 7.500 | collapses_with_enough_pairs |
| california_housing | std_rot | mlp | 0.665 | 1.044 | 0.967 | 1.000 | 0.961 | 0.000 | 0.549 | 7.500 | collapses_with_enough_pairs |
| california_housing | std_rot | knn | 0.640 | 0.993 | 0.993 | 1.000 | 0.961 | 0.000 | 0.549 | 7.500 | collapses_with_enough_pairs |
| california_housing | gauss | linear | 0.589 | 0.964 | 0.895 | 0.913 | 0.492 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | gauss | hgb | 0.789 | 1.000 | 1.000 | 0.913 | 0.492 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | gauss | mlp | 0.726 | 1.141 | 0.918 | 0.913 | 0.492 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | gauss | knn | 0.666 | 1.037 | 0.896 | 0.913 | 0.492 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | linear | 0.589 | 0.964 | 0.895 | 0.913 | 0.488 | 0.000 | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | hgb | 0.712 | 0.903 | 0.936 | 0.913 | 0.488 | 0.000 | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | mlp | 0.722 | 1.136 | 0.917 | 0.913 | 0.488 | 0.000 | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | knn | 0.663 | 1.032 | 0.868 | 0.913 | 0.488 | 0.000 | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | linear | 0.586 | 0.960 | 0.862 | 0.993 | 0.596 | — | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | hgb | 0.789 | 1.000 | 0.994 | 0.993 | 0.596 | — | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | mlp | 0.701 | 1.103 | 0.898 | 0.993 | 0.596 | — | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | knn | 0.629 | 0.978 | 0.885 | 0.993 | 0.596 | — | 0.540 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | linear | 0.589 | 0.964 | 0.895 | 0.912 | 0.489 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | hgb | 0.789 | 1.000 | 1.000 | 0.912 | 0.489 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | mlp | 0.718 | 1.130 | 0.917 | 0.912 | 0.489 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | knn | 0.666 | 1.037 | 0.896 | 0.912 | 0.489 | 0.000 | 0.543 | 17.500 | partial_recovery_with_enough_pairs |
| california_housing | bn_adv | linear | 0.720 | 1.178 | 0.927 | 0.810 | 0.209 | — | 0.537 | — | resistant_in_tested_budget |
| california_housing | bn_adv | hgb | 0.723 | 0.917 | 0.941 | 0.810 | 0.209 | — | 0.537 | — | resistant_in_tested_budget |
| california_housing | bn_adv | mlp | 0.719 | 1.133 | 0.929 | 0.810 | 0.209 | — | 0.537 | — | resistant_in_tested_budget |
| california_housing | bn_adv | knn | 0.711 | 1.105 | 0.916 | 0.810 | 0.209 | — | 0.537 | — | resistant_in_tested_budget |
| california_housing | bn_noisy | linear | 0.712 | 1.165 | 0.925 | 0.678 | 0.034 | — | 0.517 | — | resistant_in_tested_budget |
| california_housing | bn_noisy | hgb | 0.701 | 0.889 | 0.933 | 0.678 | 0.034 | — | 0.517 | — | resistant_in_tested_budget |
| california_housing | bn_noisy | mlp | 0.707 | 1.115 | 0.929 | 0.678 | 0.034 | — | 0.517 | — | resistant_in_tested_budget |
| california_housing | bn_noisy | knn | 0.688 | 1.068 | 0.907 | 0.678 | 0.034 | — | 0.517 | — | resistant_in_tested_budget |
| california_housing | vib | linear | 0.718 | 1.175 | 0.936 | 0.654 | -0.052 | — | 0.521 | — | resistant_in_tested_budget |
| california_housing | vib | hgb | 0.712 | 0.903 | 0.937 | 0.654 | -0.052 | — | 0.521 | — | resistant_in_tested_budget |
| california_housing | vib | mlp | 0.720 | 1.134 | 0.934 | 0.654 | -0.052 | — | 0.521 | — | resistant_in_tested_budget |
| california_housing | vib | knn | 0.706 | 1.098 | 0.913 | 0.654 | -0.052 | — | 0.521 | — | resistant_in_tested_budget |
| california_housing | vib_stoch | linear | 0.716 | 1.172 | 0.933 | 0.642 | -0.448 | — | 0.609 | — | resistant_in_tested_budget |
| california_housing | vib_stoch | hgb | 0.706 | 0.895 | 0.933 | 0.642 | -0.448 | — | 0.609 | — | resistant_in_tested_budget |
| california_housing | vib_stoch | mlp | 0.716 | 1.126 | 0.941 | 0.642 | -0.448 | — | 0.609 | — | resistant_in_tested_budget |
| california_housing | vib_stoch | knn | 0.701 | 1.090 | 0.901 | 0.642 | -0.448 | — | 0.609 | — | resistant_in_tested_budget |
| california_housing | rff | linear | 0.398 | 0.654 | 0.755 | 0.915 | 0.422 | — | 0.541 | 100.000 | partial_recovery_with_enough_pairs |
| california_housing | rff | hgb | 0.598 | 0.759 | 0.876 | 0.915 | 0.422 | — | 0.541 | 100.000 | partial_recovery_with_enough_pairs |
| california_housing | rff | mlp | 0.612 | 0.963 | 0.870 | 0.915 | 0.422 | — | 0.541 | 100.000 | partial_recovery_with_enough_pairs |
| california_housing | rff | knn | 0.528 | 0.826 | 0.890 | 0.915 | 0.422 | — | 0.541 | 100.000 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | linear | 0.393 | 0.644 | 0.714 | 0.786 | 0.446 | 0.000 | 0.501 | 75.000 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | hgb | 0.450 | 0.571 | 0.753 | 0.786 | 0.446 | 0.000 | 0.501 | 75.000 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | mlp | 0.444 | 0.696 | 0.760 | 0.786 | 0.446 | 0.000 | 0.501 | 75.000 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | knn | 0.424 | 0.662 | 0.745 | 0.786 | 0.446 | 0.000 | 0.501 | 75.000 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | linear | 0.585 | 0.957 | 0.893 | 0.908 | 0.487 | 0.000 | 0.542 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | hgb | 0.705 | 0.895 | 0.937 | 0.908 | 0.487 | 0.000 | 0.542 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | mlp | 0.721 | 1.132 | 0.912 | 0.908 | 0.487 | 0.000 | 0.542 | 25.000 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | knn | 0.655 | 1.018 | 0.869 | 0.908 | 0.487 | 0.000 | 0.542 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | identity | linear | 0.879 | 1.000 | 1.000 | 1.000 | 0.996 | 0.200 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | hgb | 0.923 | 1.000 | 1.000 | 1.000 | 0.996 | 0.200 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | mlp | -0.605 | 1.000 | 1.000 | 1.000 | 0.996 | 0.200 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | knn | 0.753 | 1.000 | 1.000 | 1.000 | 0.996 | 0.200 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | linear | 0.879 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | hgb | 0.921 | 0.998 | 0.999 | 1.000 | 1.000 | 0.000 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | mlp | -1.375 | 2.300 | 0.985 | 1.000 | 1.000 | 0.000 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | knn | 0.753 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.529 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | linear | 0.879 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.529 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | hgb | 0.872 | 0.945 | 0.963 | 1.000 | 1.000 | 0.000 | 0.529 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | mlp | -1.168 | 1.934 | 0.879 | 1.000 | 1.000 | 0.000 | 0.529 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | knn | 0.763 | 1.013 | 0.987 | 1.000 | 1.000 | 0.000 | 0.529 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_reg | gauss | linear | 0.861 | 0.979 | 0.990 | 1.000 | 0.870 | 0.000 | 0.513 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | hgb | 0.923 | 1.000 | 1.000 | 1.000 | 0.870 | 0.000 | 0.513 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | mlp | -1.347 | 2.242 | 0.941 | 1.000 | 0.870 | 0.000 | 0.513 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | knn | 0.723 | 0.960 | 0.941 | 1.000 | 0.870 | 0.000 | 0.513 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | linear | 0.861 | 0.979 | 0.990 | 1.000 | 0.870 | 0.000 | 0.512 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | hgb | 0.826 | 0.895 | 0.941 | 1.000 | 0.870 | 0.000 | 0.512 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | mlp | -1.462 | 2.445 | 0.779 | 1.000 | 0.870 | 0.000 | 0.512 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | knn | 0.668 | 0.888 | 0.904 | 1.000 | 0.870 | 0.000 | 0.512 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | linear | 0.908 | 1.032 | 0.988 | 1.000 | 0.938 | — | 0.532 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | hgb | 0.921 | 0.998 | 0.999 | 1.000 | 0.938 | — | 0.532 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | mlp | -1.034 | 1.691 | 0.927 | 1.000 | 0.938 | — | 0.532 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | knn | 0.749 | 0.995 | 0.951 | 1.000 | 0.938 | — | 0.532 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_reg | typed_keyed | linear | 0.862 | 0.981 | 0.989 | 1.000 | 0.589 | 0.000 | 0.520 | 75.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | hgb | 0.924 | 1.001 | 0.998 | 1.000 | 0.589 | 0.000 | 0.520 | 75.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | mlp | -1.327 | 2.217 | 0.944 | 1.000 | 0.589 | 0.000 | 0.520 | 75.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | knn | 0.728 | 0.967 | 0.917 | 1.000 | 0.589 | 0.000 | 0.520 | 75.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | bn_adv | linear | 0.917 | 1.043 | 0.975 | 0.934 | 0.393 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | hgb | 0.914 | 0.990 | 0.989 | 0.934 | 0.393 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | mlp | -0.503 | 0.822 | 0.700 | 0.934 | 0.393 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | knn | 0.913 | 1.213 | 0.930 | 0.934 | 0.393 | — | 0.506 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | linear | 0.913 | 1.039 | 0.973 | 0.921 | 0.288 | — | 0.523 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | hgb | 0.910 | 0.986 | 0.986 | 0.921 | 0.288 | — | 0.523 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | mlp | -0.456 | 0.727 | 0.694 | 0.921 | 0.288 | — | 0.523 | — | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | knn | 0.911 | 1.210 | 0.928 | 0.921 | 0.288 | — | 0.523 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | linear | 0.918 | 1.044 | 0.975 | 0.923 | 0.362 | — | 0.518 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | hgb | 0.915 | 0.992 | 0.990 | 0.923 | 0.362 | — | 0.518 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | mlp | -0.614 | 0.993 | 0.686 | 0.923 | 0.362 | — | 0.518 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib | knn | 0.914 | 1.215 | 0.930 | 0.923 | 0.362 | — | 0.518 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | linear | 0.916 | 1.042 | 0.975 | 0.871 | -0.186 | — | 0.641 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | hgb | 0.909 | 0.985 | 0.989 | 0.871 | -0.186 | — | 0.641 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | mlp | -0.541 | 0.861 | 0.734 | 0.871 | -0.186 | — | 0.641 | — | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | knn | 0.906 | 1.203 | 0.921 | 0.871 | -0.186 | — | 0.641 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | linear | 0.485 | 0.552 | 0.703 | 0.564 | 0.402 | — | 0.536 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | hgb | 0.604 | 0.655 | 0.806 | 0.564 | 0.402 | — | 0.536 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | mlp | -1.097 | 1.835 | 0.522 | 0.564 | 0.402 | — | 0.536 | — | resistant_in_tested_budget |
| banking_mixed_reg | rff | knn | 0.554 | 0.734 | 0.799 | 0.564 | 0.402 | — | 0.536 | — | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | linear | 0.153 | 0.174 | 0.409 | 1.000 | 0.292 | 0.000 | 0.503 | — | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | hgb | 0.147 | 0.159 | 0.403 | 1.000 | 0.292 | 0.000 | 0.503 | — | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | mlp | -1.523 | 2.550 | 0.550 | 1.000 | 0.292 | 0.000 | 0.503 | — | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | knn | 0.096 | 0.127 | 0.411 | 1.000 | 0.292 | 0.000 | 0.503 | — | resistant_in_tested_budget |
| banking_mixed_reg | noisy_gauss_rot | linear | 0.859 | 0.977 | 0.988 | 0.996 | 0.860 | 0.000 | 0.514 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | noisy_gauss_rot | hgb | 0.825 | 0.894 | 0.939 | 0.996 | 0.860 | 0.000 | 0.514 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | noisy_gauss_rot | mlp | -1.454 | 2.432 | 0.778 | 0.996 | 0.860 | 0.000 | 0.514 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | noisy_gauss_rot | knn | 0.670 | 0.890 | 0.900 | 0.996 | 0.860 | 0.000 | 0.514 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | identity | linear | 0.937 | 1.000 | 1.000 | 1.000 | 0.995 | 0.200 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | hgb | 0.931 | 1.000 | 1.000 | 1.000 | 0.995 | 0.200 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | mlp | 0.919 | 1.000 | 1.000 | 1.000 | 0.995 | 0.200 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | knn | 0.860 | 1.000 | 1.000 | 1.000 | 0.995 | 0.200 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | linear | 0.937 | 1.000 | 1.000 | 1.000 | 0.997 | 0.000 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | hgb | 0.937 | 1.006 | 0.987 | 1.000 | 0.997 | 0.000 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | mlp | 0.920 | 1.001 | 0.958 | 1.000 | 0.997 | 0.000 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | knn | 0.860 | 1.000 | 1.000 | 1.000 | 0.997 | 0.000 | 0.522 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | linear | 0.936 | 0.999 | 0.999 | 1.000 | 1.000 | 0.000 | 0.522 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | hgb | 0.912 | 0.979 | 0.927 | 1.000 | 1.000 | 0.000 | 0.522 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | mlp | 0.933 | 1.015 | 0.961 | 1.000 | 1.000 | 0.000 | 0.522 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | knn | 0.866 | 1.007 | 0.964 | 1.000 | 1.000 | 0.000 | 0.522 | 10.000 | collapses_with_enough_pairs |
| banking_mixed_clf | gauss | linear | 0.941 | 1.005 | 0.982 | 1.000 | 0.873 | 0.000 | 0.520 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | hgb | 0.931 | 1.000 | 1.000 | 1.000 | 0.873 | 0.000 | 0.520 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | mlp | 0.927 | 1.009 | 0.950 | 1.000 | 0.873 | 0.000 | 0.520 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | knn | 0.873 | 1.016 | 0.936 | 1.000 | 0.873 | 0.000 | 0.520 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | linear | 0.940 | 1.003 | 0.984 | 1.000 | 0.873 | 0.000 | 0.516 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | hgb | 0.892 | 0.958 | 0.912 | 1.000 | 0.873 | 0.000 | 0.516 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | mlp | 0.921 | 1.002 | 0.945 | 1.000 | 0.873 | 0.000 | 0.516 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | knn | 0.852 | 0.991 | 0.918 | 1.000 | 0.873 | 0.000 | 0.516 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | linear | 0.937 | 1.000 | 0.985 | 1.000 | 0.934 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | hgb | 0.933 | 1.002 | 0.986 | 1.000 | 0.934 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | mlp | 0.924 | 1.005 | 0.953 | 1.000 | 0.934 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | knn | 0.869 | 1.010 | 0.926 | 1.000 | 0.934 | — | 0.517 | 25.000 | collapses_with_enough_pairs |
| banking_mixed_clf | typed_keyed | linear | 0.940 | 1.003 | 0.979 | 1.000 | 0.583 | 0.000 | 0.511 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | hgb | 0.936 | 1.006 | 0.974 | 1.000 | 0.583 | 0.000 | 0.511 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | mlp | 0.919 | 0.999 | 0.927 | 1.000 | 0.583 | 0.000 | 0.511 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | knn | 0.864 | 1.004 | 0.906 | 1.000 | 0.583 | 0.000 | 0.511 | 100.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | bn_adv | linear | 0.939 | 1.002 | 0.967 | 0.928 | 0.327 | — | 0.524 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | hgb | 0.933 | 1.002 | 0.964 | 0.928 | 0.327 | — | 0.524 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | mlp | 0.936 | 1.019 | 0.950 | 0.928 | 0.327 | — | 0.524 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | knn | 0.939 | 1.091 | 0.886 | 0.928 | 0.327 | — | 0.524 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | linear | 0.934 | 0.997 | 0.961 | 0.891 | 0.140 | — | 0.514 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | hgb | 0.930 | 0.999 | 0.953 | 0.891 | 0.140 | — | 0.514 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | mlp | 0.928 | 1.010 | 0.936 | 0.891 | 0.140 | — | 0.514 | — | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | knn | 0.930 | 1.082 | 0.884 | 0.891 | 0.140 | — | 0.514 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | linear | 0.939 | 1.002 | 0.968 | 0.902 | 0.317 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | hgb | 0.939 | 1.008 | 0.961 | 0.902 | 0.317 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | mlp | 0.933 | 1.015 | 0.934 | 0.902 | 0.317 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib | knn | 0.934 | 1.086 | 0.883 | 0.902 | 0.317 | — | 0.515 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | linear | 0.940 | 1.003 | 0.964 | 0.769 | -0.310 | — | 0.697 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | hgb | 0.935 | 1.004 | 0.959 | 0.769 | -0.310 | — | 0.697 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | mlp | 0.936 | 1.018 | 0.942 | 0.769 | -0.310 | — | 0.697 | — | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | knn | 0.938 | 1.090 | 0.879 | 0.769 | -0.310 | — | 0.697 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | linear | 0.836 | 0.892 | 0.850 | 0.597 | 0.386 | — | 0.528 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | hgb | 0.851 | 0.914 | 0.853 | 0.597 | 0.386 | — | 0.528 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | mlp | 0.851 | 0.926 | 0.877 | 0.597 | 0.386 | — | 0.528 | — | resistant_in_tested_budget |
| banking_mixed_clf | rff | knn | 0.830 | 0.966 | 0.886 | 0.597 | 0.386 | — | 0.528 | — | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | linear | 0.726 | 0.775 | 0.744 | 1.000 | 0.298 | 0.000 | 0.502 | — | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | hgb | 0.736 | 0.790 | 0.751 | 1.000 | 0.298 | 0.000 | 0.502 | — | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | mlp | 0.734 | 0.799 | 0.770 | 1.000 | 0.298 | 0.000 | 0.502 | — | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | knn | 0.713 | 0.830 | 0.804 | 1.000 | 0.298 | 0.000 | 0.502 | — | resistant_in_tested_budget |
| banking_mixed_clf | noisy_gauss_rot | linear | 0.931 | 0.994 | 0.967 | 0.995 | 0.866 | 0.000 | 0.519 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | noisy_gauss_rot | hgb | 0.893 | 0.959 | 0.912 | 0.995 | 0.866 | 0.000 | 0.519 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | noisy_gauss_rot | mlp | 0.920 | 1.001 | 0.940 | 0.995 | 0.866 | 0.000 | 0.519 | 25.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | noisy_gauss_rot | knn | 0.845 | 0.983 | 0.916 | 0.995 | 0.866 | 0.000 | 0.519 | 25.000 | partial_recovery_with_enough_pairs |


## 5. Attacker A results

Known-pair grid: 1, 2, 5, 10, 25, 50, 100, 200. Attacks: ridge, pinv,
rank/order matching, categorical frequency, HGB inversion, MLP inversion.

| dataset | method | recon_r2_mean | worst_attr_r2_mean | sensitive_attr_r2_mean | n_pairs_r2_0.5_mean | spearman_leak_mean | known_pair_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| california_housing | identity | 0.999 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | identity | 0.999 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | identity | 0.999 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | identity | 0.999 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | 1.000 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | 1.000 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | 1.000 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | secret_affine | 1.000 | 1.000 | 1.000 | 10.000 | 1.000 | collapses_with_enough_pairs |
| california_housing | std_rot | 0.961 | 1.000 | 1.000 | 7.500 | 0.943 | collapses_with_enough_pairs |
| california_housing | std_rot | 0.961 | 1.000 | 1.000 | 7.500 | 0.943 | collapses_with_enough_pairs |
| california_housing | std_rot | 0.961 | 1.000 | 1.000 | 7.500 | 0.943 | collapses_with_enough_pairs |
| california_housing | std_rot | 0.961 | 1.000 | 1.000 | 7.500 | 0.943 | collapses_with_enough_pairs |
| california_housing | gauss | 0.492 | 0.913 | 0.913 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | 0.492 | 0.913 | 0.913 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | 0.492 | 0.913 | 0.913 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss | 0.492 | 0.913 | 0.913 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.488 | 0.913 | 0.913 | 25.000 | 0.773 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.488 | 0.913 | 0.913 | 25.000 | 0.773 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.488 | 0.913 | 0.913 | 25.000 | 0.773 | partial_recovery_with_enough_pairs |
| california_housing | gauss_white_rot | 0.488 | 0.913 | 0.913 | 25.000 | 0.773 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.596 | 0.993 | 0.950 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.596 | 0.993 | 0.950 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.596 | 0.993 | 0.950 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | keyed_monotone | 0.596 | 0.993 | 0.950 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.489 | 0.912 | 0.912 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.489 | 0.912 | 0.912 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.489 | 0.912 | 0.912 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | typed_keyed | 0.489 | 0.912 | 0.912 | 17.500 | 1.000 | partial_recovery_with_enough_pairs |
| california_housing | bn_adv | 0.209 | 0.810 | 0.810 | — | 0.792 | resistant_in_tested_budget |
| california_housing | bn_adv | 0.209 | 0.810 | 0.810 | — | 0.792 | resistant_in_tested_budget |
| california_housing | bn_adv | 0.209 | 0.810 | 0.810 | — | 0.792 | resistant_in_tested_budget |
| california_housing | bn_adv | 0.209 | 0.810 | 0.810 | — | 0.792 | resistant_in_tested_budget |
| california_housing | bn_noisy | 0.034 | 0.678 | 0.678 | — | 0.826 | resistant_in_tested_budget |
| california_housing | bn_noisy | 0.034 | 0.678 | 0.678 | — | 0.826 | resistant_in_tested_budget |
| california_housing | bn_noisy | 0.034 | 0.678 | 0.678 | — | 0.826 | resistant_in_tested_budget |
| california_housing | bn_noisy | 0.034 | 0.678 | 0.678 | — | 0.826 | resistant_in_tested_budget |
| california_housing | vib | -0.052 | 0.654 | 0.654 | — | 0.752 | resistant_in_tested_budget |
| california_housing | vib | -0.052 | 0.654 | 0.654 | — | 0.752 | resistant_in_tested_budget |
| california_housing | vib | -0.052 | 0.654 | 0.654 | — | 0.752 | resistant_in_tested_budget |
| california_housing | vib | -0.052 | 0.654 | 0.654 | — | 0.752 | resistant_in_tested_budget |
| california_housing | vib_stoch | -0.448 | 0.642 | 0.642 | — | 0.805 | resistant_in_tested_budget |
| california_housing | vib_stoch | -0.448 | 0.642 | 0.642 | — | 0.805 | resistant_in_tested_budget |
| california_housing | vib_stoch | -0.448 | 0.642 | 0.642 | — | 0.805 | resistant_in_tested_budget |
| california_housing | vib_stoch | -0.448 | 0.642 | 0.642 | — | 0.805 | resistant_in_tested_budget |
| california_housing | rff | 0.422 | 0.915 | 0.915 | 100.000 | 0.830 | partial_recovery_with_enough_pairs |
| california_housing | rff | 0.422 | 0.915 | 0.915 | 100.000 | 0.830 | partial_recovery_with_enough_pairs |
| california_housing | rff | 0.422 | 0.915 | 0.915 | 100.000 | 0.830 | partial_recovery_with_enough_pairs |
| california_housing | rff | 0.422 | 0.915 | 0.915 | 100.000 | 0.830 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | 0.446 | 0.786 | 0.786 | 75.000 | 0.746 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | 0.446 | 0.786 | 0.786 | 75.000 | 0.746 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | 0.446 | 0.786 | 0.786 | 75.000 | 0.746 | partial_recovery_with_enough_pairs |
| california_housing | microagg_rot | 0.446 | 0.786 | 0.786 | 75.000 | 0.746 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | 0.487 | 0.908 | 0.908 | 25.000 | 0.765 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | 0.487 | 0.908 | 0.908 | 25.000 | 0.765 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | 0.487 | 0.908 | 0.908 | 25.000 | 0.765 | partial_recovery_with_enough_pairs |
| california_housing | noisy_gauss_rot | 0.487 | 0.908 | 0.908 | 25.000 | 0.765 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | identity | 0.996 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | 1.000 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | 1.000 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | 1.000 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | secret_affine | 1.000 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.789 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.789 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.789 | collapses_with_enough_pairs |
| banking_mixed_reg | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.789 | collapses_with_enough_pairs |
| banking_mixed_reg | gauss | 0.870 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | 0.870 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | 0.870 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss | 0.870 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.870 | 1.000 | 1.000 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.870 | 1.000 | 1.000 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.870 | 1.000 | 1.000 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | gauss_white_rot | 0.870 | 1.000 | 1.000 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.938 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.938 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.938 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | keyed_monotone | 0.938 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.589 | 1.000 | 1.000 | 75.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.589 | 1.000 | 1.000 | 75.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.589 | 1.000 | 1.000 | 75.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | typed_keyed | 0.589 | 1.000 | 1.000 | 75.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | bn_adv | 0.393 | 0.934 | 0.934 | — | 0.976 | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | 0.393 | 0.934 | 0.934 | — | 0.976 | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | 0.393 | 0.934 | 0.934 | — | 0.976 | resistant_in_tested_budget |
| banking_mixed_reg | bn_adv | 0.393 | 0.934 | 0.934 | — | 0.976 | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | 0.288 | 0.921 | 0.921 | — | 0.971 | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | 0.288 | 0.921 | 0.921 | — | 0.971 | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | 0.288 | 0.921 | 0.921 | — | 0.971 | resistant_in_tested_budget |
| banking_mixed_reg | bn_noisy | 0.288 | 0.921 | 0.921 | — | 0.971 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.362 | 0.923 | 0.923 | — | 0.960 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.362 | 0.923 | 0.923 | — | 0.960 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.362 | 0.923 | 0.923 | — | 0.960 | resistant_in_tested_budget |
| banking_mixed_reg | vib | 0.362 | 0.923 | 0.923 | — | 0.960 | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | -0.186 | 0.871 | 0.871 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | -0.186 | 0.871 | 0.871 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | -0.186 | 0.871 | 0.871 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | vib_stoch | -0.186 | 0.871 | 0.871 | — | 0.953 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.402 | 0.564 | 0.562 | — | 0.629 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.402 | 0.564 | 0.562 | — | 0.629 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.402 | 0.564 | 0.562 | — | 0.629 | resistant_in_tested_budget |
| banking_mixed_reg | rff | 0.402 | 0.564 | 0.562 | — | 0.629 | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | 0.292 | 1.000 | 1.000 | — | 0.723 | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | 0.292 | 1.000 | 1.000 | — | 0.723 | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | 0.292 | 1.000 | 1.000 | — | 0.723 | resistant_in_tested_budget |
| banking_mixed_reg | microagg_rot | 0.292 | 1.000 | 1.000 | — | 0.723 | resistant_in_tested_budget |
| banking_mixed_reg | noisy_gauss_rot | 0.860 | 0.996 | 0.996 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | noisy_gauss_rot | 0.860 | 0.996 | 0.996 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | noisy_gauss_rot | 0.860 | 0.996 | 0.996 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_reg | noisy_gauss_rot | 0.860 | 0.996 | 0.996 | 25.000 | 0.665 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | identity | 0.995 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | 0.995 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | 0.995 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | identity | 0.995 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | 0.997 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | 0.997 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | 0.997 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | secret_affine | 0.997 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.714 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.714 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.714 | collapses_with_enough_pairs |
| banking_mixed_clf | std_rot | 1.000 | 1.000 | 1.000 | 10.000 | 0.714 | collapses_with_enough_pairs |
| banking_mixed_clf | gauss | 0.873 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | 0.873 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | 0.873 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss | 0.873 | 1.000 | 1.000 | 25.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.873 | 1.000 | 1.000 | 25.000 | 0.589 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.873 | 1.000 | 1.000 | 25.000 | 0.589 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.873 | 1.000 | 1.000 | 25.000 | 0.589 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | gauss_white_rot | 0.873 | 1.000 | 1.000 | 25.000 | 0.589 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.934 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.934 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.934 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | keyed_monotone | 0.934 | 1.000 | 1.000 | 25.000 | 1.000 | collapses_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.583 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.583 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.583 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | typed_keyed | 0.583 | 1.000 | 1.000 | 100.000 | 1.000 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | bn_adv | 0.327 | 0.928 | 0.928 | — | 0.966 | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | 0.327 | 0.928 | 0.928 | — | 0.966 | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | 0.327 | 0.928 | 0.928 | — | 0.966 | resistant_in_tested_budget |
| banking_mixed_clf | bn_adv | 0.327 | 0.928 | 0.928 | — | 0.966 | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | 0.140 | 0.891 | 0.891 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | 0.140 | 0.891 | 0.891 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | 0.140 | 0.891 | 0.891 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | bn_noisy | 0.140 | 0.891 | 0.891 | — | 0.942 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.317 | 0.902 | 0.902 | — | 0.945 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.317 | 0.902 | 0.902 | — | 0.945 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.317 | 0.902 | 0.902 | — | 0.945 | resistant_in_tested_budget |
| banking_mixed_clf | vib | 0.317 | 0.902 | 0.902 | — | 0.945 | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | -0.310 | 0.769 | 0.769 | — | 0.940 | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | -0.310 | 0.769 | 0.769 | — | 0.940 | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | -0.310 | 0.769 | 0.769 | — | 0.940 | resistant_in_tested_budget |
| banking_mixed_clf | vib_stoch | -0.310 | 0.769 | 0.769 | — | 0.940 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.386 | 0.597 | 0.597 | — | 0.587 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.386 | 0.597 | 0.597 | — | 0.587 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.386 | 0.597 | 0.597 | — | 0.587 | resistant_in_tested_budget |
| banking_mixed_clf | rff | 0.386 | 0.597 | 0.597 | — | 0.587 | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | 0.298 | 1.000 | 1.000 | — | 0.775 | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | 0.298 | 1.000 | 1.000 | — | 0.775 | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | 0.298 | 1.000 | 1.000 | — | 0.775 | resistant_in_tested_budget |
| banking_mixed_clf | microagg_rot | 0.298 | 1.000 | 1.000 | — | 0.775 | resistant_in_tested_budget |
| banking_mixed_clf | noisy_gauss_rot | 0.866 | 0.995 | 0.995 | 25.000 | 0.579 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | noisy_gauss_rot | 0.866 | 0.995 | 0.995 | 25.000 | 0.579 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | noisy_gauss_rot | 0.866 | 0.995 | 0.995 | 25.000 | 0.579 | partial_recovery_with_enough_pairs |
| banking_mixed_clf | noisy_gauss_rot | 0.866 | 0.995 | 0.995 | 25.000 | 0.579 | partial_recovery_with_enough_pairs |


## 6. Attacker B results

The attacker receives only `Z` and a context sentence
(`This dataset comes from a bank...` or the housing description).
No keys, schema, column names, or raw rows.

| dataset | method | semantic_acc_mean | kind_acc_mean | membership_auc_mean | linkage_mean | sensitive_hit_mean |
| --- | --- | --- | --- | --- | --- | --- |
| california_housing | identity | 0.000 | 1.000 | 0.545 | — | 0.000 |
| california_housing | identity | 0.000 | 1.000 | 0.545 | — | 0.000 |
| california_housing | identity | 0.000 | 1.000 | 0.545 | — | 0.000 |
| california_housing | identity | 0.000 | 1.000 | 0.545 | — | 0.000 |
| california_housing | secret_affine | 0.000 | 1.000 | 0.545 | — | 0.333 |
| california_housing | secret_affine | 0.000 | 1.000 | 0.545 | — | 0.333 |
| california_housing | secret_affine | 0.000 | 1.000 | 0.545 | — | 0.333 |
| california_housing | secret_affine | 0.000 | 1.000 | 0.545 | — | 0.333 |
| california_housing | std_rot | 0.000 | 1.000 | 0.549 | — | 0.667 |
| california_housing | std_rot | 0.000 | 1.000 | 0.549 | — | 0.667 |
| california_housing | std_rot | 0.000 | 1.000 | 0.549 | — | 0.667 |
| california_housing | std_rot | 0.000 | 1.000 | 0.549 | — | 0.667 |
| california_housing | gauss | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | gauss | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | gauss | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | gauss | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.540 | — | 0.667 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.540 | — | 0.667 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.540 | — | 0.667 |
| california_housing | gauss_white_rot | 0.000 | 1.000 | 0.540 | — | 0.667 |
| california_housing | keyed_monotone | — | — | 0.540 | — | — |
| california_housing | keyed_monotone | — | — | 0.540 | — | — |
| california_housing | keyed_monotone | — | — | 0.540 | — | — |
| california_housing | keyed_monotone | — | — | 0.540 | — | — |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | typed_keyed | 0.000 | 1.000 | 0.543 | — | 0.333 |
| california_housing | bn_adv | — | — | 0.537 | — | — |
| california_housing | bn_adv | — | — | 0.537 | — | — |
| california_housing | bn_adv | — | — | 0.537 | — | — |
| california_housing | bn_adv | — | — | 0.537 | — | — |
| california_housing | bn_noisy | — | — | 0.517 | — | — |
| california_housing | bn_noisy | — | — | 0.517 | — | — |
| california_housing | bn_noisy | — | — | 0.517 | — | — |
| california_housing | bn_noisy | — | — | 0.517 | — | — |
| california_housing | vib | — | — | 0.521 | — | — |
| california_housing | vib | — | — | 0.521 | — | — |
| california_housing | vib | — | — | 0.521 | — | — |
| california_housing | vib | — | — | 0.521 | — | — |
| california_housing | vib_stoch | — | — | 0.609 | — | — |
| california_housing | vib_stoch | — | — | 0.609 | — | — |
| california_housing | vib_stoch | — | — | 0.609 | — | — |
| california_housing | vib_stoch | — | — | 0.609 | — | — |
| california_housing | rff | — | — | 0.541 | — | — |
| california_housing | rff | — | — | 0.541 | — | — |
| california_housing | rff | — | — | 0.541 | — | — |
| california_housing | rff | — | — | 0.541 | — | — |
| california_housing | microagg_rot | 0.000 | 1.000 | 0.501 | — | 0.667 |
| california_housing | microagg_rot | 0.000 | 1.000 | 0.501 | — | 0.667 |
| california_housing | microagg_rot | 0.000 | 1.000 | 0.501 | — | 0.667 |
| california_housing | microagg_rot | 0.000 | 1.000 | 0.501 | — | 0.667 |
| california_housing | noisy_gauss_rot | 0.000 | 1.000 | 0.542 | — | 0.667 |
| california_housing | noisy_gauss_rot | 0.000 | 1.000 | 0.542 | — | 0.667 |
| california_housing | noisy_gauss_rot | 0.000 | 1.000 | 0.542 | — | 0.667 |
| california_housing | noisy_gauss_rot | 0.000 | 1.000 | 0.542 | — | 0.667 |
| banking_mixed_reg | identity | 0.200 | 0.733 | 0.529 | — | 0.889 |
| banking_mixed_reg | identity | 0.200 | 0.733 | 0.529 | — | 0.889 |
| banking_mixed_reg | identity | 0.200 | 0.733 | 0.529 | — | 0.889 |
| banking_mixed_reg | identity | 0.200 | 0.733 | 0.529 | — | 0.889 |
| banking_mixed_reg | secret_affine | 0.000 | 0.433 | 0.529 | — | 0.722 |
| banking_mixed_reg | secret_affine | 0.000 | 0.433 | 0.529 | — | 0.722 |
| banking_mixed_reg | secret_affine | 0.000 | 0.433 | 0.529 | — | 0.722 |
| banking_mixed_reg | secret_affine | 0.000 | 0.433 | 0.529 | — | 0.722 |
| banking_mixed_reg | std_rot | 0.000 | 0.533 | 0.529 | — | 0.444 |
| banking_mixed_reg | std_rot | 0.000 | 0.533 | 0.529 | — | 0.444 |
| banking_mixed_reg | std_rot | 0.000 | 0.533 | 0.529 | — | 0.444 |
| banking_mixed_reg | std_rot | 0.000 | 0.533 | 0.529 | — | 0.444 |
| banking_mixed_reg | gauss | 0.000 | 0.733 | 0.513 | — | 0.889 |
| banking_mixed_reg | gauss | 0.000 | 0.733 | 0.513 | — | 0.889 |
| banking_mixed_reg | gauss | 0.000 | 0.733 | 0.513 | — | 0.889 |
| banking_mixed_reg | gauss | 0.000 | 0.733 | 0.513 | — | 0.889 |
| banking_mixed_reg | gauss_white_rot | 0.000 | 0.533 | 0.512 | — | 0.444 |
| banking_mixed_reg | gauss_white_rot | 0.000 | 0.533 | 0.512 | — | 0.444 |
| banking_mixed_reg | gauss_white_rot | 0.000 | 0.533 | 0.512 | — | 0.444 |
| banking_mixed_reg | gauss_white_rot | 0.000 | 0.533 | 0.512 | — | 0.444 |
| banking_mixed_reg | keyed_monotone | — | — | 0.532 | — | — |
| banking_mixed_reg | keyed_monotone | — | — | 0.532 | — | — |
| banking_mixed_reg | keyed_monotone | — | — | 0.532 | — | — |
| banking_mixed_reg | keyed_monotone | — | — | 0.532 | — | — |
| banking_mixed_reg | typed_keyed | 0.000 | 0.533 | 0.520 | — | 0.611 |
| banking_mixed_reg | typed_keyed | 0.000 | 0.533 | 0.520 | — | 0.611 |
| banking_mixed_reg | typed_keyed | 0.000 | 0.533 | 0.520 | — | 0.611 |
| banking_mixed_reg | typed_keyed | 0.000 | 0.533 | 0.520 | — | 0.611 |
| banking_mixed_reg | bn_adv | — | — | 0.506 | — | — |
| banking_mixed_reg | bn_adv | — | — | 0.506 | — | — |
| banking_mixed_reg | bn_adv | — | — | 0.506 | — | — |
| banking_mixed_reg | bn_adv | — | — | 0.506 | — | — |
| banking_mixed_reg | bn_noisy | — | — | 0.523 | — | — |
| banking_mixed_reg | bn_noisy | — | — | 0.523 | — | — |
| banking_mixed_reg | bn_noisy | — | — | 0.523 | — | — |
| banking_mixed_reg | bn_noisy | — | — | 0.523 | — | — |
| banking_mixed_reg | vib | — | — | 0.518 | — | — |
| banking_mixed_reg | vib | — | — | 0.518 | — | — |
| banking_mixed_reg | vib | — | — | 0.518 | — | — |
| banking_mixed_reg | vib | — | — | 0.518 | — | — |
| banking_mixed_reg | vib_stoch | — | — | 0.641 | — | — |
| banking_mixed_reg | vib_stoch | — | — | 0.641 | — | — |
| banking_mixed_reg | vib_stoch | — | — | 0.641 | — | — |
| banking_mixed_reg | vib_stoch | — | — | 0.641 | — | — |
| banking_mixed_reg | rff | — | — | 0.536 | — | — |
| banking_mixed_reg | rff | — | — | 0.536 | — | — |
| banking_mixed_reg | rff | — | — | 0.536 | — | — |
| banking_mixed_reg | rff | — | — | 0.536 | — | — |
| banking_mixed_reg | microagg_rot | 0.000 | 0.533 | 0.503 | — | 0.444 |
| banking_mixed_reg | microagg_rot | 0.000 | 0.533 | 0.503 | — | 0.444 |
| banking_mixed_reg | microagg_rot | 0.000 | 0.533 | 0.503 | — | 0.444 |
| banking_mixed_reg | microagg_rot | 0.000 | 0.533 | 0.503 | — | 0.444 |
| banking_mixed_reg | noisy_gauss_rot | 0.000 | 0.533 | 0.514 | — | 0.444 |
| banking_mixed_reg | noisy_gauss_rot | 0.000 | 0.533 | 0.514 | — | 0.444 |
| banking_mixed_reg | noisy_gauss_rot | 0.000 | 0.533 | 0.514 | — | 0.444 |
| banking_mixed_reg | noisy_gauss_rot | 0.000 | 0.533 | 0.514 | — | 0.444 |
| banking_mixed_clf | identity | 0.200 | 0.733 | 0.522 | — | 0.889 |
| banking_mixed_clf | identity | 0.200 | 0.733 | 0.522 | — | 0.889 |
| banking_mixed_clf | identity | 0.200 | 0.733 | 0.522 | — | 0.889 |
| banking_mixed_clf | identity | 0.200 | 0.733 | 0.522 | — | 0.889 |
| banking_mixed_clf | secret_affine | 0.000 | 0.567 | 0.522 | — | 0.500 |
| banking_mixed_clf | secret_affine | 0.000 | 0.567 | 0.522 | — | 0.500 |
| banking_mixed_clf | secret_affine | 0.000 | 0.567 | 0.522 | — | 0.500 |
| banking_mixed_clf | secret_affine | 0.000 | 0.567 | 0.522 | — | 0.500 |
| banking_mixed_clf | std_rot | 0.000 | 0.533 | 0.522 | — | 0.444 |
| banking_mixed_clf | std_rot | 0.000 | 0.533 | 0.522 | — | 0.444 |
| banking_mixed_clf | std_rot | 0.000 | 0.533 | 0.522 | — | 0.444 |
| banking_mixed_clf | std_rot | 0.000 | 0.533 | 0.522 | — | 0.444 |
| banking_mixed_clf | gauss | 0.000 | 0.733 | 0.520 | — | 0.889 |
| banking_mixed_clf | gauss | 0.000 | 0.733 | 0.520 | — | 0.889 |
| banking_mixed_clf | gauss | 0.000 | 0.733 | 0.520 | — | 0.889 |
| banking_mixed_clf | gauss | 0.000 | 0.733 | 0.520 | — | 0.889 |
| banking_mixed_clf | gauss_white_rot | 0.000 | 0.533 | 0.516 | — | 0.444 |
| banking_mixed_clf | gauss_white_rot | 0.000 | 0.533 | 0.516 | — | 0.444 |
| banking_mixed_clf | gauss_white_rot | 0.000 | 0.533 | 0.516 | — | 0.444 |
| banking_mixed_clf | gauss_white_rot | 0.000 | 0.533 | 0.516 | — | 0.444 |
| banking_mixed_clf | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_clf | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_clf | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_clf | keyed_monotone | — | — | 0.517 | — | — |
| banking_mixed_clf | typed_keyed | 0.000 | 0.400 | 0.511 | — | 0.667 |
| banking_mixed_clf | typed_keyed | 0.000 | 0.400 | 0.511 | — | 0.667 |
| banking_mixed_clf | typed_keyed | 0.000 | 0.400 | 0.511 | — | 0.667 |
| banking_mixed_clf | typed_keyed | 0.000 | 0.400 | 0.511 | — | 0.667 |
| banking_mixed_clf | bn_adv | — | — | 0.524 | — | — |
| banking_mixed_clf | bn_adv | — | — | 0.524 | — | — |
| banking_mixed_clf | bn_adv | — | — | 0.524 | — | — |
| banking_mixed_clf | bn_adv | — | — | 0.524 | — | — |
| banking_mixed_clf | bn_noisy | — | — | 0.514 | — | — |
| banking_mixed_clf | bn_noisy | — | — | 0.514 | — | — |
| banking_mixed_clf | bn_noisy | — | — | 0.514 | — | — |
| banking_mixed_clf | bn_noisy | — | — | 0.514 | — | — |
| banking_mixed_clf | vib | — | — | 0.515 | — | — |
| banking_mixed_clf | vib | — | — | 0.515 | — | — |
| banking_mixed_clf | vib | — | — | 0.515 | — | — |
| banking_mixed_clf | vib | — | — | 0.515 | — | — |
| banking_mixed_clf | vib_stoch | — | — | 0.697 | — | — |
| banking_mixed_clf | vib_stoch | — | — | 0.697 | — | — |
| banking_mixed_clf | vib_stoch | — | — | 0.697 | — | — |
| banking_mixed_clf | vib_stoch | — | — | 0.697 | — | — |
| banking_mixed_clf | rff | — | — | 0.528 | — | — |
| banking_mixed_clf | rff | — | — | 0.528 | — | — |
| banking_mixed_clf | rff | — | — | 0.528 | — | — |
| banking_mixed_clf | rff | — | — | 0.528 | — | — |
| banking_mixed_clf | microagg_rot | 0.000 | 0.533 | 0.502 | — | 0.444 |
| banking_mixed_clf | microagg_rot | 0.000 | 0.533 | 0.502 | — | 0.444 |
| banking_mixed_clf | microagg_rot | 0.000 | 0.533 | 0.502 | — | 0.444 |
| banking_mixed_clf | microagg_rot | 0.000 | 0.533 | 0.502 | — | 0.444 |
| banking_mixed_clf | noisy_gauss_rot | 0.000 | 0.533 | 0.519 | — | 0.444 |
| banking_mixed_clf | noisy_gauss_rot | 0.000 | 0.533 | 0.519 | — | 0.444 |
| banking_mixed_clf | noisy_gauss_rot | 0.000 | 0.533 | 0.519 | — | 0.444 |
| banking_mixed_clf | noisy_gauss_rot | 0.000 | 0.533 | 0.519 | — | 0.444 |


## 7. Security / utility trade-off

Pareto view: HGB retention vs known-pair worst-attribute `R²`
(see `results/figures/02_pareto_utility_leakage.png`).
Selection rule: maximize practical privacy subject to minimal utility loss,
operationalized as lowest mean worst-attribute leakage among methods with
HGB retention ≥ 0.85 on every dataset when such methods exist.

Selected method: **`vib_stoch`**
(retention=0.9613317368375568, worst-attr leakage=0.7604996421881113,
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

**`vib_stoch`** is the empirically preferred point on this grid.
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
