# Analysis: 20260430-111011

- **Source**: `out/cifar100/vitb16/20260430-111011/history.pkl`
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
| wandb_run_name | 20260430-111009           |

## Raw Correlations

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.9620   | 0.9796  |
| H vs H_VV        | 0.8173   | 0.4532  |
| H vs GN          | 0.9068   | 0.8706  |
| H vs Diag_H      | 0.9716   | 0.9844  |
| H vs Fisher      | 0.8934   | 0.9128  |
| Prec_H vs H_VV   | 0.8629   | 0.5221  |
| Prec_H vs GN     | 0.8967   | 0.8882  |
| Prec_H vs Diag_H | 0.9445   | 0.9695  |
| Prec_H vs Fisher | 0.8800   | 0.9464  |

## Smoothed Correlations (λ=10.0)

| Pair             | Spearman | Pearson |
| ---------------- | -------- | ------- |
| H vs Prec_H      | 0.9794   | 0.9560  |
| H vs H_VV        | 0.9256   | 0.8510  |
| H vs GN          | 0.9532   | 0.8243  |
| H vs Diag_H      | 0.9872   | 0.9901  |
| H vs Fisher      | 0.9327   | 0.8718  |
| Prec_H vs H_VV   | 0.9464   | 0.9112  |
| Prec_H vs GN     | 0.9464   | 0.8194  |
| Prec_H vs Diag_H | 0.9695   | 0.9498  |
| Prec_H vs Fisher | 0.9228   | 0.8654  |

## Proxy vs Entropy (smoothed, λ=10.0)

| Pair                   | Spearman | Pearson |
| ---------------------- | -------- | ------- |
| H vs Entropy(L0)       | -0.0479  | -0.0523 |
| H vs Entropy(avg)      | -0.0479  | -0.0526 |
| Prec_H vs Entropy(L0)  | -0.0542  | -0.0483 |
| Prec_H vs Entropy(avg) | -0.0541  | -0.0487 |
| H_VV vs Entropy(L0)    | -0.0386  | -0.0479 |
| GN vs Entropy(L0)      | -0.0516  | -0.0463 |

## Spike Co-occurrence (z=1.5)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 49         | 43      | 0.878        |
| H vs H_VV   | 49         | 18      | 0.367        |
| H vs GN     | 49         | 37      | 0.755        |
| H vs Diag_H | 49         | 47      | 0.959        |
| H vs Fisher | 49         | 27      | 0.551        |

## Spike Co-occurrence (z=2.0)

| Pair        | n_H_spikes | n_joint | P(Y|X spike) |
| ----------- | ---------- | ------- | ------------ |
| H vs Prec_H | 33         | 27      | 0.818        |
| H vs H_VV   | 33         | 11      | 0.333        |
| H vs GN     | 33         | 23      | 0.697        |
| H vs Diag_H | 33         | 33      | 1.000        |
| H vs Fisher | 33         | 19      | 0.576        |
