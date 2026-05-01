# Analysis: 20260430-005608

- **Source**: `out/cifar100/vitb16/20260430-005608/history.pkl`
- **Smoothing λ**: 10.0
- **Hessian freq**: 50
- **Entropy freq**: 50

## Training Configuration

### Model

| Parameter       | Value                |
| --------------- | -------------------- |
| model_name      | vit_base_patch16_224 |
| pretrained      | False                |
| num_classes     | 100                  |
| img_size        | 32                   |
| depth           | 12                   |
| num_heads       | 12                   |
| embed_dim       | 768                  |
| patch_size      | 4                    |
| init_std        | 0.02                 |
| use_scaled_init | True                 |
| qk_norm         | False                |
| label_smoothing | 0.1                  |

### Data

| Parameter   | Value    |
| ----------- | -------- |
| dataset     | cifar100 |
| data_dir    | ./data   |
| batch_size  | 256      |
| num_workers | 8        |

### Optimiser

| Parameter     | Value  |
| ------------- | ------ |
| optimizer     | adamw  |
| learning_rate | 0.0003 |
| max_iters     | 20000  |
| weight_decay  | 0.05   |
| beta1         | 0.9    |
| beta2         | 0.999  |
| grad_clip     | 1.0    |
| eps           | 1e-08  |

### LR Schedule

| Parameter      | Value |
| -------------- | ----- |
| decay_lr       | True  |
| warmup_iters   | 2000  |
| lr_decay_iters | 20000 |
| min_lr         | 3e-06 |

### Hessian

| Parameter          | Value |
| ------------------ | ----- |
| hessian_intv       | 50    |
| hessian_max_iter   | 10    |
| hessian_batch_size | 128   |
| compute_fd         | False |

### Entropy

| Parameter    | Value |
| ------------ | ----- |
| entropy_intv | 50    |

### Intervention

| Parameter         | Value |
| ----------------- | ----- |
| temp_shift_step   | 15000 |
| temp_shift_factor | 0.25  |

### Compute

| Parameter | Value    |
| --------- | -------- |
| device    | cuda     |
| compile   | False    |
| dtype     | bfloat16 |
| seed      | 1337     |

### I/O

| Parameter           | Value               |
| ------------------- | ------------------- |
| out_dir             | out/cifar100/vitb16 |
| eval_interval       | 500                 |
| log_interval        | 1                   |
| checkpoint_interval | -1                  |
| save_checkpoint     | False               |
| init_from           | scratch             |

### W&B

| Parameter      | Value                     |
| -------------- | ------------------------- |
| wandb_log      | True                      |
| wandb_project  | entropy-collapse-cifar100 |
| wandb_run_name | 20260430-005606           |

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.8342   | 0.8089  |
| H vs H_VV        | 0.7271   | 0.7895  |
| H vs GN          | 0.8861   | 0.7773  |
| H vs Diag_H      | 0.6810   | 0.7356  |
| H vs Fisher      | 0.6701   | 0.6994  |
| Prec_H vs H_VV   | 0.7758   | 0.8647  |
| Prec_H vs GN     | 0.8034   | 0.4300  |
| Prec_H vs Diag_H | 0.6154   | 0.5017  |
| Prec_H vs Fisher | 0.6731   | 0.4791  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.9001   | 0.6551  |
| H vs H_VV        | 0.8110   | 0.6674  |
| H vs GN          | 0.9509   | 0.9387  |
| H vs Diag_H      | 0.7863   | 0.8775  |
| H vs Fisher      | 0.7830   | 0.9008  |
| Prec_H vs H_VV   | 0.8396   | 0.8628  |
| Prec_H vs GN     | 0.9071   | 0.4699  |
| Prec_H vs Diag_H | 0.7292   | 0.4145  |
| Prec_H vs Fisher | 0.7689   | 0.6070  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | 0.0461   | 0.1092  |
| H vs Entropy(avg)      | 0.0460   | 0.1091  |
| Prec_H vs Entropy(L0)  | -0.0247  | -0.0228 |
| Prec_H vs Entropy(avg) | -0.0244  | -0.0229 |
| H_VV vs Entropy(L0)    | -0.0014  | -0.0188 |
| GN vs Entropy(L0)      | 0.0169   | 0.0973  |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 47         | 23      | 0.489        |
| H vs H_VV   | 47         | 25      | 0.532        |
| H vs GN     | 47         | 18      | 0.383        |
| H vs Diag_H | 47         | 13      | 0.277        |
| H vs Fisher | 47         | 14      | 0.298        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 31         | 19      | 0.613        |
| H vs H_VV   | 31         | 19      | 0.613        |
| H vs GN     | 31         | 9       | 0.290        |
| H vs Diag_H | 31         | 7       | 0.226        |
| H vs Fisher | 31         | 8       | 0.258        |
