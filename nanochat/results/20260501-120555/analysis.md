# Analysis: 20260501-120555

- **Source**: `out/nanochat/d8/20260501-120555/history.pkl`
- **Smoothing λ**: 10.0
- **Hessian freq**: 50
- **Entropy freq**: 50

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.8146   | 0.8404  |
| H vs H_VV        | 0.4144   | 0.2826  |
| H vs GN          | 0.8640   | 0.5080  |
| H vs Diag_H      | 0.7029   | 0.7646  |
| H vs Fisher      | 0.8398   | 0.4470  |
| Prec_H vs H_VV   | 0.3515   | 0.2985  |
| Prec_H vs GN     | 0.6221   | 0.3111  |
| Prec_H vs Diag_H | 0.6411   | 0.7236  |
| Prec_H vs Fisher | 0.6680   | 0.3810  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.8914   | 0.8913  |
| H vs H_VV        | 0.6399   | 0.6338  |
| H vs GN          | 0.9248   | 0.8706  |
| H vs Diag_H      | 0.8486   | 0.8861  |
| H vs Fisher      | 0.9414   | 0.9215  |
| Prec_H vs H_VV   | 0.6910   | 0.8098  |
| Prec_H vs GN     | 0.7378   | 0.5940  |
| Prec_H vs Diag_H | 0.8282   | 0.7924  |
| Prec_H vs Fisher | 0.7892   | 0.7014  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | -0.0269  | -0.0276 |
| H vs Entropy(avg)      | -0.0270  | -0.0354 |
| Prec_H vs Entropy(L0)  | -0.0403  | -0.0418 |
| Prec_H vs Entropy(avg) | -0.0404  | -0.0497 |
| H_VV vs Entropy(L0)    | -0.0168  | 0.0042  |
| GN vs Entropy(L0)      | -0.0257  | -0.0326 |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 84         | 59      | 0.702        |
| H vs H_VV   | 84         | 14      | 0.167        |
| H vs GN     | 84         | 26      | 0.310        |
| H vs Diag_H | 84         | 28      | 0.333        |
| H vs Fisher | 84         | 39      | 0.464        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 68         | 48      | 0.706        |
| H vs H_VV   | 69         | 10      | 0.145        |
| H vs GN     | 68         | 12      | 0.176        |
| H vs Diag_H | 68         | 17      | 0.250        |
| H vs Fisher | 68         | 30      | 0.441        |
