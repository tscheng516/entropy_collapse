# Analysis: 20260429-162644

- **Source**: `out/cifar100/vitb16/20260429-162644/history.pkl`
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
| wandb_run_name | 20260429-162642           |

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.7231   | 0.6651  |
| H vs H_VV        | 0.5632   | 0.5924  |
| H vs GN          | 0.8452   | 0.9198  |
| H vs Diag_H      | 0.4666   | 0.7405  |
| H vs Fisher      | 0.4842   | 0.7691  |
| Prec_H vs H_VV   | 0.6295   | 0.5575  |
| Prec_H vs GN     | 0.7223   | 0.5305  |
| Prec_H vs Diag_H | 0.3263   | 0.2930  |
| Prec_H vs Fisher | 0.4630   | 0.3606  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.8784   | 0.6081  |
| H vs H_VV        | 0.6727   | 0.5598  |
| H vs GN          | 0.9511   | 0.9558  |
| H vs Diag_H      | 0.6482   | 0.8654  |
| H vs Fisher      | 0.6118   | 0.8559  |
| Prec_H vs H_VV   | 0.6786   | 0.6144  |
| Prec_H vs GN     | 0.8843   | 0.5283  |
| Prec_H vs Diag_H | 0.5349   | 0.1921  |
| Prec_H vs Fisher | 0.5488   | 0.3353  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | 0.0294   | 0.1228  |
| H vs Entropy(avg)      | 0.0293   | 0.1227  |
| Prec_H vs Entropy(L0)  | -0.0560  | -0.0366 |
| Prec_H vs Entropy(avg) | -0.0557  | -0.0366 |
| H_VV vs Entropy(L0)    | -0.0086  | -0.0252 |
| GN vs Entropy(L0)      | 0.0114   | 0.1024  |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 34         | 10      | 0.294        |
| H vs H_VV   | 34         | 14      | 0.412        |
| H vs GN     | 34         | 14      | 0.412        |
| H vs Diag_H | 34         | 6       | 0.176        |
| H vs Fisher | 34         | 11      | 0.324        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 15         | 7       | 0.467        |
| H vs H_VV   | 15         | 7       | 0.467        |
| H vs GN     | 15         | 7       | 0.467        |
| H vs Diag_H | 15         | 2       | 0.133        |
| H vs Fisher | 15         | 3       | 0.200        |
