# Analysis: 20260430-164837

- **Source**: `out/nanochat/d8/20260430-164837/history.pkl`
- **Smoothing λ**: 10.0
- **Hessian freq**: 50
- **Entropy freq**: 50

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.6991   | 0.6929  |
| H vs H_VV        | 0.0779   | 0.2281  |
| H vs GN          | 0.7702   | 0.5973  |
| H vs Diag_H      | 0.4175   | 0.4464  |
| H vs Fisher      | 0.7504   | 0.5264  |
| Prec_H vs H_VV   | 0.0751   | 0.2354  |
| Prec_H vs GN     | 0.4704   | 0.2904  |
| Prec_H vs Diag_H | 0.3282   | 0.5341  |
| Prec_H vs Fisher | 0.5443   | 0.5903  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.6968   | 0.8891  |
| H vs H_VV        | 0.0699   | 0.4925  |
| H vs GN          | 0.8484   | 0.7428  |
| H vs Diag_H      | 0.5109   | 0.6833  |
| H vs Fisher      | 0.9040   | 0.8227  |
| Prec_H vs H_VV   | 0.3516   | 0.5561  |
| Prec_H vs GN     | 0.4317   | 0.5814  |
| Prec_H vs Diag_H | 0.4421   | 0.6960  |
| Prec_H vs Fisher | 0.6063   | 0.7077  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | -0.0814  | -0.0834 |
| H vs Entropy(avg)      | -0.0817  | -0.0888 |
| Prec_H vs Entropy(L0)  | -0.0597  | -0.0834 |
| Prec_H vs Entropy(avg) | -0.0600  | -0.0887 |
| H_VV vs Entropy(L0)    | -0.0403  | -0.0510 |
| GN vs Entropy(L0)      | -0.0512  | -0.0666 |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 69         | 43      | 0.623        |
| H vs H_VV   | 69         | 15      | 0.217        |
| H vs GN     | 69         | 33      | 0.478        |
| H vs Diag_H | 69         | 17      | 0.246        |
| H vs Fisher | 69         | 35      | 0.507        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 55         | 34      | 0.618        |
| H vs H_VV   | 55         | 13      | 0.236        |
| H vs GN     | 55         | 18      | 0.327        |
| H vs Diag_H | 55         | 11      | 0.200        |
| H vs Fisher | 55         | 29      | 0.527        |
