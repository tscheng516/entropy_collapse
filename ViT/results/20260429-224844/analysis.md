# Analysis: 20260429-224844

- **Source**: `out/cifar100/vitb16/20260429-224844/history.pkl`
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
| learning_rate | 0.0001 |
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
| wandb_run_name | 20260429-224842           |

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.5773   | 0.6515  |
| H vs H_VV        | 0.5862   | 0.7559  |
| H vs GN          | 0.9105   | 0.9693  |
| H vs Diag_H      | 0.7204   | 0.8569  |
| H vs Fisher      | 0.6440   | 0.7777  |
| Prec_H vs H_VV   | 0.7622   | 0.5948  |
| Prec_H vs GN     | 0.5558   | 0.6524  |
| Prec_H vs Diag_H | 0.7305   | 0.5953  |
| Prec_H vs Fisher | 0.8446   | 0.8037  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.4834   | 0.6232  |
| H vs H_VV        | 0.5067   | 0.7490  |
| H vs GN          | 0.9680   | 0.9932  |
| H vs Diag_H      | 0.5713   | 0.9082  |
| H vs Fisher      | 0.6169   | 0.8380  |
| Prec_H vs H_VV   | 0.7894   | 0.6078  |
| Prec_H vs GN     | 0.4780   | 0.6414  |
| Prec_H vs Diag_H | 0.8200   | 0.5348  |
| Prec_H vs Fisher | 0.8912   | 0.8849  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | 0.0131   | 0.0260  |
| H vs Entropy(avg)      | 0.0129   | 0.0260  |
| Prec_H vs Entropy(L0)  | -0.0366  | -0.0135 |
| Prec_H vs Entropy(avg) | -0.0365  | -0.0134 |
| H_VV vs Entropy(L0)    | -0.0225  | -0.0320 |
| GN vs Entropy(L0)      | 0.0229   | 0.0185  |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 47         | 15      | 0.319        |
| H vs H_VV   | 47         | 19      | 0.404        |
| H vs GN     | 47         | 23      | 0.489        |
| H vs Diag_H | 47         | 12      | 0.255        |
| H vs Fisher | 47         | 10      | 0.213        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 27         | 7       | 0.259        |
| H vs H_VV   | 27         | 12      | 0.444        |
| H vs GN     | 27         | 13      | 0.481        |
| H vs Diag_H | 27         | 7       | 0.259        |
| H vs Fisher | 27         | 7       | 0.259        |
