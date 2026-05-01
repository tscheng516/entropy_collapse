# Analysis: 20260429-183406

- **Source**: `out/cifar100/vitb16/20260429-183406/history.pkl`
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
| wandb_run_name | 20260429-183404           |

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.2688   | 0.5065  |
| H vs H_VV        | 0.2825   | 0.7126  |
| H vs GN          | 0.8204   | 0.9755  |
| H vs Diag_H      | 0.4451   | 0.8573  |
| H vs Fisher      | 0.3234   | 0.7663  |
| Prec_H vs H_VV   | 0.6331   | 0.4429  |
| Prec_H vs GN     | 0.2816   | 0.5287  |
| Prec_H vs Diag_H | 0.4717   | 0.4175  |
| Prec_H vs Fisher | 0.7432   | 0.6727  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.2214   | 0.5232  |
| H vs H_VV        | 0.2429   | 0.7152  |
| H vs GN          | 0.9294   | 0.9932  |
| H vs Diag_H      | 0.2855   | 0.8984  |
| H vs Fisher      | 0.3661   | 0.8723  |
| Prec_H vs H_VV   | 0.7235   | 0.4721  |
| Prec_H vs GN     | 0.2582   | 0.5389  |
| Prec_H vs Diag_H | 0.7337   | 0.3825  |
| Prec_H vs Fisher | 0.8809   | 0.7699  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | -0.0390  | 0.0120  |
| H vs Entropy(avg)      | -0.0391  | 0.0119  |
| Prec_H vs Entropy(L0)  | -0.0604  | -0.0669 |
| Prec_H vs Entropy(avg) | -0.0603  | -0.0668 |
| H_VV vs Entropy(L0)    | -0.0248  | -0.0392 |
| GN vs Entropy(L0)      | -0.0017  | 0.0064  |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 45         | 10      | 0.222        |
| H vs H_VV   | 45         | 12      | 0.267        |
| H vs GN     | 45         | 24      | 0.533        |
| H vs Diag_H | 45         | 10      | 0.222        |
| H vs Fisher | 45         | 6       | 0.133        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 25         | 5       | 0.200        |
| H vs H_VV   | 25         | 8       | 0.320        |
| H vs GN     | 25         | 11      | 0.440        |
| H vs Diag_H | 25         | 4       | 0.160        |
| H vs Fisher | 25         | 2       | 0.080        |
