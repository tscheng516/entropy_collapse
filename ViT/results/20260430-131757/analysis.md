# Analysis: 20260430-131757

- **Source**: `out/cifar100/vitb16/20260430-131757/history.pkl`
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
| learning_rate | 0.0006 |
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
| temp_shift_step   | -1    |
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
| wandb_run_name | 20260430-131755           |

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.9033   | 0.9853  |
| H vs H_VV        | 0.5256   | 0.6649  |
| H vs GN          | 0.7487   | 0.6278  |
| H vs Diag_H      | 0.9230   | 0.9967  |
| H vs Fisher      | 0.7283   | 0.3355  |
| Prec_H vs H_VV   | 0.6386   | 0.7167  |
| Prec_H vs GN     | 0.7005   | 0.5110  |
| Prec_H vs Diag_H | 0.8410   | 0.9915  |
| Prec_H vs Fisher | 0.6823   | 0.1822  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.9450   | 0.9331  |
| H vs H_VV        | 0.7754   | 0.6480  |
| H vs GN          | 0.8198   | 0.8531  |
| H vs Diag_H      | 0.9596   | 0.9544  |
| H vs Fisher      | 0.7824   | 0.7022  |
| Prec_H vs H_VV   | 0.8243   | 0.7764  |
| Prec_H vs GN     | 0.7699   | 0.7785  |
| Prec_H vs Diag_H | 0.8938   | 0.8881  |
| Prec_H vs Fisher | 0.7181   | 0.6157  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | -0.0369  | -0.0289 |
| H vs Entropy(avg)      | -0.0368  | -0.0292 |
| Prec_H vs Entropy(L0)  | -0.0539  | -0.0358 |
| Prec_H vs Entropy(avg) | -0.0537  | -0.0357 |
| H_VV vs Entropy(L0)    | -0.0400  | 0.0073  |
| GN vs Entropy(L0)      | -0.0665  | -0.0620 |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 42         | 35      | 0.833        |
| H vs H_VV   | 42         | 15      | 0.357        |
| H vs GN     | 42         | 32      | 0.762        |
| H vs Diag_H | 42         | 37      | 0.881        |
| H vs Fisher | 42         | 24      | 0.571        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 27         | 24      | 0.889        |
| H vs H_VV   | 27         | 7       | 0.259        |
| H vs GN     | 27         | 24      | 0.889        |
| H vs Diag_H | 27         | 26      | 0.963        |
| H vs Fisher | 27         | 19      | 0.704        |
