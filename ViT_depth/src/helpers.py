"""
Depth-specific curvature helpers and attention-entropy utilities.

All second-order quantities are estimated without materialising the full
Hessian matrix; instead we use power iteration on Hessian-vector products
(HVPs) computed via ``torch.autograd.grad``.

Functions
---------
scale_invariant_log_loss          — Batch-mean scale-invariant log loss (SILog).
scale_invariant_log_loss_per_sample — Per-sample SILog for Fisher estimation.
get_VV_subspace_mask              — Binary mask selecting value-projection params.
get_curvature_metrics             — All nine sharpness proxies (spectral norms).
get_attention_entropy             — Per-layer Shannon entropy of cached attention.

The nine curvature proxies are identical to those in ``ViT/src/helpers.py``:
  hessian, prec_h, hessian_vv, gn, bfgs, fd, diag_h, fisher, kfac.
The only change is that cross-entropy is replaced by SILog wherever the
loss needs to be recomputed internally (GN, Fisher, BFGS, FD).
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — add project root so ``from common.helpers import ...``
# resolves regardless of the working directory.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_FOLDER_DIR = os.path.dirname(_SRC_DIR)
_PROJECT_ROOT = os.path.dirname(_FOLDER_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.autograd import functional as autograd_functional

from common.helpers import smooth_log_trend, get_VV_subspace_mask  # noqa: F401


# ======================================================================
# Scale-invariant log loss  (Eigen et al., 2014 — NYU Depth v2)
# ======================================================================

_SILOG_LAMBDA = 0.5   # variance-balancing term (standard value)
_SILOG_EPS    = 1e-6  # floor for log() to avoid -inf


def scale_invariant_log_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = _SILOG_LAMBDA,
    eps: float = _SILOG_EPS,
) -> torch.Tensor:
    """
    Scale-invariant log loss (SILog) averaged over the batch.

        L = (1/n) Σ d_i²  −  (λ/n²) (Σ d_i)²
        where  d_i = log(pred_i) − log(gt_i)

    Only valid (positive ground-truth) pixels contribute; zero/negative GT
    pixels are treated as invalid and silently ignored.

    Args:
        pred:   ``(B, 1, H, W)`` predicted depth in metres (must be positive;
                the model applies Softplus so this is guaranteed during
                normal operation).
        target: ``(B, 1, H, W)`` or ``(B, H, W)`` ground-truth depth in metres.
                Pixels with ``target <= 0`` are invalid and masked out.
        lam:    Variance-balancing coefficient (default 0.5).
        eps:    Floor for ``log()`` to avoid numerical issues.

    Returns:
        Scalar loss tensor.
    """
    pred   = pred.squeeze(1)     # (B, H, W)
    target = target.squeeze(1)   # (B, H, W)

    valid = (target > 0.0) & torch.isfinite(target)   # (B, H, W)

    d = torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))
    d = d * valid.float()

    n = valid.float().sum(dim=(-2, -1)).clamp(min=1.0)  # (B,)

    term1 = (d ** 2).sum(dim=(-2, -1)) / n                       # (B,)
    term2 = lam * (d.sum(dim=(-2, -1)) / n) ** 2                 # (B,)
    per_sample = term1 - term2                                    # (B,)
    return per_sample.mean()


def scale_invariant_log_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = _SILOG_LAMBDA,
    eps: float = _SILOG_EPS,
) -> torch.Tensor:
    """
    Per-sample SILog losses for empirical Fisher estimation.

    Args:
        pred:   ``(B, 1, H, W)`` predicted depth.
        target: ``(B, 1, H, W)`` or ``(B, H, W)`` ground-truth depth.
        lam:    Variance-balancing coefficient (default 0.5).
        eps:    Floor for ``log()``.

    Returns:
        ``(B,)`` tensor of per-sample SILog values.
    """
    pred   = pred.squeeze(1)
    target = target.squeeze(1)

    valid = (target > 0.0) & torch.isfinite(target)

    d = torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))
    d = d * valid.float()

    n = valid.float().sum(dim=(-2, -1)).clamp(min=1.0)

    term1 = (d ** 2).sum(dim=(-2, -1)) / n
    term2 = lam * (d.sum(dim=(-2, -1)) / n) ** 2
    return term1 - term2   # (B,)


# ======================================================================
# Smooth log-trend extraction  (delegated to common.helpers)
# ======================================================================
# smooth_log_trend and get_VV_subspace_mask are imported from
# common.helpers above; get_VV_subspace_mask auto-detects the timm ViT
# fused-qkv architecture used by ViT_depth.


# ======================================================================
# All curvature metrics
# ======================================================================


def get_curvature_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss: torch.Tensor,
    vv_mask: torch.Tensor,
    max_iter: int = 10,
    compute_fd: bool = False,
) -> dict[str, float]:
    """
    Compute nine sharpness proxies for the current model state.

    Identical in structure to ``ViT/src/helpers.py:get_curvature_metrics``
    except that cross-entropy is replaced by ``scale_invariant_log_loss``
    wherever the loss is recomputed internally (Gauss-Newton, Fisher, BFGS,
    FD sections).

    Args:
        model:      ``HookedViTDepth`` model with a live computation graph.
        optimizer:  Current optimiser (used for Adam preconditioner).
        X:          Batch of input images ``(B, 3, H, W)``.
        Y:          Batch of ground-truth depths ``(B, 1, H, W)``.
        loss:       Scalar SILog loss with a live computation graph.
        vv_mask:    Output of ``get_VV_subspace_mask(model)``.
        max_iter:   Power-iteration steps for λ_max estimation.
        compute_fd: Whether to compute finite-difference proxies (BFGS, FD).

    Returns:
        Dict with keys: ``hessian``, ``prec_h``, ``hessian_vv``, ``gn``,
        ``fd``, ``diag_h``, ``fisher``, ``bfgs``, ``kfac``.
    """
    grads = torch.autograd.grad(
        loss, model.parameters(), create_graph=True, retain_graph=True
    )
    flat_grads = torch.cat([g.reshape(-1) for g in grads])

    # ------------------------------------------------------------------ #
    # 1. Exact Hessian Power Iteration (H)
    # ------------------------------------------------------------------ #
    v_h = torch.randn_like(flat_grads)
    v_h = v_h / (v_h.norm() + 1e-9)
    flat_hvp: torch.Tensor | None = None
    for _ in range(max_iter):
        hvp = torch.autograd.grad(
            flat_grads, model.parameters(), grad_outputs=v_h, retain_graph=True
        )
        flat_hvp = torch.cat([g.contiguous().reshape(-1) for g in hvp])
        v_h = flat_hvp / (flat_hvp.norm() + 1e-9)
    hessian_norm = torch.dot(v_h, flat_hvp).item() if flat_hvp is not None else 0.0

    # ------------------------------------------------------------------ #
    # 2. Value-Subspace Power Iteration (H_VV)
    # ------------------------------------------------------------------ #
    vv_mask_dev = vv_mask.to(flat_grads.device)
    v_vv = torch.randn_like(flat_grads) * vv_mask_dev
    v_vv = v_vv / (v_vv.norm() + 1e-9)
    flat_hvp_vv: torch.Tensor | None = None
    for _ in range(max_iter):
        hvp_vv = torch.autograd.grad(
            flat_grads, model.parameters(), grad_outputs=v_vv, retain_graph=True
        )
        flat_hvp_vv = torch.cat([g.contiguous().reshape(-1) for g in hvp_vv])
        v_vv = flat_hvp_vv * vv_mask_dev
        v_vv = v_vv / (v_vv.norm() + 1e-9)
    hessian_vv_norm = (
        torch.dot(v_vv, flat_hvp_vv).item() if flat_hvp_vv is not None else 0.0
    )

    # ------------------------------------------------------------------ #
    # 3. Preconditioned Hessian Power Iteration (H_tilde)
    # ------------------------------------------------------------------ #
    D_inv_sqrt_parts = []
    is_adam = False
    for param in model.parameters():
        state = optimizer.state.get(param, {})
        if "exp_avg_sq" in state and state["exp_avg_sq"].numel() > 0:
            is_adam = True
            v_sq = state["exp_avg_sq"]
            step = state.get("step", 1)
            step = step.item() if isinstance(step, torch.Tensor) else step
            beta2 = optimizer.param_groups[0]["betas"][1]
            bias_corr2 = 1.0 - beta2 ** step
            v_hat = v_sq / bias_corr2
            adam_epsilon = optimizer.param_groups[0].get("eps", 1e-8)
            P = torch.sqrt(v_hat) + adam_epsilon
            D_inv_sqrt_parts.append((1.0 / torch.sqrt(P)).reshape(-1))
        else:
            D_inv_sqrt_parts.append(torch.ones(param.numel(), device=param.device))

    D_inv_sqrt = torch.cat(D_inv_sqrt_parts)

    if is_adam:
        v_prec = torch.randn_like(flat_grads)
        v_prec = v_prec / (v_prec.norm() + 1e-9)
        step3: torch.Tensor | None = None
        for _ in range(max_iter):
            step1 = D_inv_sqrt * v_prec
            hvp_prec = torch.autograd.grad(
                flat_grads, model.parameters(), grad_outputs=step1, retain_graph=True
            )
            flat_hvp_prec = torch.cat([g.contiguous().reshape(-1) for g in hvp_prec])
            step3 = D_inv_sqrt * flat_hvp_prec
            v_prec = step3 / (step3.norm() + 1e-9)
        prec_hessian_norm = (
            torch.dot(v_prec, step3).item() if step3 is not None else 0.0
        )
    else:
        prec_hessian_norm = hessian_norm

    del grads, flat_grads, v_h, flat_hvp, v_vv, flat_hvp_vv, D_inv_sqrt
    if is_adam:
        del v_prec, step3
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 4. Gauss-Newton  (H_GN = J^T H_L J)
    #    Loss function: scale_invariant_log_loss (replaces cross-entropy).
    # ------------------------------------------------------------------ #
    params_keys = [k for k, _ in model.named_parameters()]
    params_vals = [p for _, p in model.named_parameters()]

    def _fwd_tuple(*p_vals: torch.Tensor) -> torch.Tensor:
        p_dict = {k: v for k, v in zip(params_keys, p_vals)}
        return functional_call(model, p_dict, (X,))   # (B, 1, H, W)

    flat_params = torch.cat([p.reshape(-1) for p in params_vals])
    v_g = torch.randn_like(flat_params)
    v_g = v_g / (v_g.norm() + 1e-9)
    flat_gn_v: torch.Tensor | None = None
    for _ in range(max_iter):
        tangents_list: list[torch.Tensor] = []
        offset = 0
        for p in params_vals:
            numel = p.numel()
            tangents_list.append(v_g[offset : offset + numel].view_as(p))
            offset += numel

        out_depth, Jv = autograd_functional.jvp(
            _fwd_tuple, tuple(params_vals), tuple(tangents_list), create_graph=False
        )

        if not out_depth.requires_grad:
            out_depth = out_depth.detach().requires_grad_(True)

        l = scale_invariant_log_loss(out_depth, Y)
        grad_l = torch.autograd.grad(l, out_depth, create_graph=True)[0]
        H_L_Jv = torch.autograd.grad(grad_l, out_depth, grad_outputs=Jv.detach())[0]

        _, gn_tuple = autograd_functional.vjp(
            _fwd_tuple, tuple(params_vals), H_L_Jv.detach()
        )

        flat_gn_v = torch.cat(
            [g.contiguous().reshape(-1) for g in gn_tuple]
        )
        v_g = flat_gn_v / (flat_gn_v.norm() + 1e-9)

        del out_depth, Jv, l, grad_l, H_L_Jv, gn_tuple, tangents_list
        torch.cuda.empty_cache()

    gn_norm = torch.dot(v_g, flat_gn_v).item() if flat_gn_v is not None else 0.0
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 5. Diagonal Hessian  max(diag(H))  — Bekas–Kokiopoulou–Saad
    # ------------------------------------------------------------------ #
    diag_h_norm = 0.0
    n_hutchinson = max(1, max_iter // 2)
    try:
        grads_diag = torch.autograd.grad(
            loss, model.parameters(), create_graph=True, retain_graph=True
        )
        flat_grads_diag = torch.cat([g.reshape(-1) for g in grads_diag])
        diag_acc = torch.zeros_like(flat_grads_diag)
        for _ in range(n_hutchinson):
            z = (
                torch.randint(0, 2, flat_grads_diag.shape, device=flat_grads_diag.device)
                .float() * 2.0 - 1.0
            )
            hz = torch.autograd.grad(
                flat_grads_diag, model.parameters(), grad_outputs=z, retain_graph=True
            )
            flat_hz = torch.cat([g.contiguous().reshape(-1) for g in hz])
            diag_acc += z * flat_hz
        diag_acc /= n_hutchinson
        diag_h_norm = diag_acc.max().item()
        del grads_diag, flat_grads_diag, diag_acc
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 6. Empirical Fisher  λ_max(F)  via power iteration
    #    Per-sample loss: scale_invariant_log_loss_per_sample.
    # ------------------------------------------------------------------ #
    fisher_norm = 0.0
    try:
        def _fwd_loss_tuple(*p_vals: torch.Tensor) -> torch.Tensor:
            p_dict = {k: v for k, v in zip(params_keys, p_vals)}
            depth_pred = functional_call(model, p_dict, (X,))   # (B, 1, H, W)
            return scale_invariant_log_loss_per_sample(depth_pred, Y)  # (B,)

        B = X.size(0)
        v_f = torch.randn_like(flat_params)
        v_f = v_f / (v_f.norm() + 1e-9)
        flat_fv: torch.Tensor | None = None

        for _ in range(max_iter):
            tangents_f: list[torch.Tensor] = []
            offset = 0
            for p in params_vals:
                numel = p.numel()
                tangents_f.append(v_f[offset : offset + numel].view_as(p))
                offset += numel

            _, Jv_f = autograd_functional.jvp(
                _fwd_loss_tuple, tuple(params_vals), tuple(tangents_f),
                create_graph=False,
            )

            _, vjp_f = autograd_functional.vjp(
                _fwd_loss_tuple, tuple(params_vals), Jv_f.detach(),
            )

            flat_fv = torch.cat(
                [g.contiguous().reshape(-1) for g in vjp_f]
            ) / B
            v_f = flat_fv / (flat_fv.norm() + 1e-9)

            del Jv_f, vjp_f, tangents_f
            torch.cuda.empty_cache()

        fisher_norm = torch.dot(v_f, flat_fv).item() if flat_fv is not None else 0.0
        del v_f
        if flat_fv is not None:
            del flat_fv
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 7. BFGS (Central-Difference Spectral) — λ_max(H) via finite diffs
    # ------------------------------------------------------------------ #
    bfgs_norm = 0.0
    if compute_fd:
        try:
            eps_cd = 1e-4
            v_b = torch.randn_like(flat_params)
            v_b = v_b / (v_b.norm() + 1e-9)
            flat_bv: torch.Tensor | None = None

            for _ in range(max_iter):
                offset = 0
                for p in params_vals:
                    numel = p.numel()
                    p.data.add_(eps_cd * v_b[offset : offset + numel].view_as(p))
                    offset += numel

                optimizer.zero_grad()
                loss_bp = scale_invariant_log_loss(model(X), Y)
                loss_bp.backward()
                g_plus = torch.cat(
                    [p.grad.reshape(-1).clone() for p in model.parameters()]
                )

                offset = 0
                for p in params_vals:
                    numel = p.numel()
                    p.data.add_(-2.0 * eps_cd * v_b[offset : offset + numel].view_as(p))
                    offset += numel

                optimizer.zero_grad()
                loss_bm = scale_invariant_log_loss(model(X), Y)
                loss_bm.backward()
                g_minus = torch.cat(
                    [p.grad.reshape(-1).clone() for p in model.parameters()]
                )

                offset = 0
                for p in params_vals:
                    numel = p.numel()
                    p.data.add_(eps_cd * v_b[offset : offset + numel].view_as(p))
                    offset += numel

                flat_bv = (g_plus - g_minus) / (2.0 * eps_cd)
                v_b = flat_bv / (flat_bv.norm() + 1e-9)

                del g_plus, g_minus
                torch.cuda.empty_cache()

            bfgs_norm = torch.dot(v_b, flat_bv).item() if flat_bv is not None else 0.0
            optimizer.zero_grad()
            del v_b
            if flat_bv is not None:
                del flat_bv
        except Exception:
            pass
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 8. K-FAC Proxy  (Kronecker-Factored Approximate Curvature)
    # ------------------------------------------------------------------ #
    kfac_norm = 0.0
    try:
        a_cache: dict[str, torch.Tensor] = {}
        g_cache: dict[str, torch.Tensor] = {}
        handles = []

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                _name = name

                def _fwd_hook(m, inp, out, _n=_name):
                    a_cache[_n] = inp[0].detach()

                def _bwd_hook(m, grad_in, grad_out, _n=_name):
                    g_cache[_n] = grad_out[0].detach()

                handles.append(mod.register_forward_hook(_fwd_hook))
                handles.append(mod.register_full_backward_hook(_bwd_hook))

        optimizer.zero_grad()
        loss_kfac = scale_invariant_log_loss(model(X), Y)
        loss_kfac.backward()
        optimizer.zero_grad()

        for h in handles:
            h.remove()

        max_kfac = 0.0
        for name in a_cache:
            if name not in g_cache:
                continue
            a = a_cache[name]
            g = g_cache[name]
            if a.ndim == 3:
                a = a.reshape(-1, a.size(-1))
            if g.ndim == 3:
                g = g.reshape(-1, g.size(-1))
            n_samples = a.size(0)

            def _top_eig(M: torch.Tensor, iters: int = 5) -> float:
                v = torch.randn(M.size(0), 1, device=M.device)
                for _ in range(iters):
                    v = M @ v
                    v = v / (v.norm() + 1e-9)
                return (v.t() @ M @ v).item()

            A_cov = (a.t() @ a) / n_samples
            G_cov = (g.t() @ g) / n_samples
            lam_A = _top_eig(A_cov)
            lam_G = _top_eig(G_cov)
            max_kfac = max(max_kfac, lam_A * lam_G)

        kfac_norm = max_kfac
        del a_cache, g_cache
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 9. FD (Forward-Difference Spectral) — λ_max(H) via finite diffs
    # ------------------------------------------------------------------ #
    fd_norm = 0.0
    if compute_fd:
        try:
            eps_fd = 1e-4
            optimizer.zero_grad()
            loss_f0 = scale_invariant_log_loss(model(X), Y)
            loss_f0.backward()
            g_base = torch.cat(
                [p.grad.reshape(-1).clone() for p in model.parameters()]
            )

            v_fd = torch.randn_like(flat_params)
            v_fd = v_fd / (v_fd.norm() + 1e-9)
            flat_fdv: torch.Tensor | None = None

            for _ in range(max_iter):
                offset = 0
                for p in params_vals:
                    numel = p.numel()
                    p.data.add_(eps_fd * v_fd[offset : offset + numel].view_as(p))
                    offset += numel

                optimizer.zero_grad()
                loss_fp = scale_invariant_log_loss(model(X), Y)
                loss_fp.backward()
                g_pert = torch.cat(
                    [p.grad.reshape(-1).clone() for p in model.parameters()]
                )

                offset = 0
                for p in params_vals:
                    numel = p.numel()
                    p.data.add_(-eps_fd * v_fd[offset : offset + numel].view_as(p))
                    offset += numel

                flat_fdv = (g_pert - g_base) / eps_fd
                v_fd = flat_fdv / (flat_fdv.norm() + 1e-9)

                del g_pert
                torch.cuda.empty_cache()

            fd_norm = (
                torch.dot(v_fd, flat_fdv).item() if flat_fdv is not None else 0.0
            )
            optimizer.zero_grad()
            del v_fd, g_base
            if flat_fdv is not None:
                del flat_fdv
        except Exception:
            pass
        torch.cuda.empty_cache()

    return {
        "hessian":    hessian_norm,
        "prec_h":     prec_hessian_norm,
        "hessian_vv": hessian_vv_norm,
        "gn":         gn_norm,
        "fd":         fd_norm,
        "diag_h":     diag_h_norm,
        "fisher":     fisher_norm,
        "bfgs":       bfgs_norm,
        "kfac":       kfac_norm,
    }


# ======================================================================
# Attention Entropy  (identical to ViT/src/helpers.py)
# ======================================================================


def get_attention_entropy(model: torch.nn.Module) -> list[float]:
    """
    Compute the mean Shannon entropy of the attention distribution for
    each transformer layer.

    Requires that ``_cache_attn=True`` was set on the attention blocks
    before the forward pass so that ``last_att`` is populated.

    Args:
        model: A ``HookedViTDepth`` model (or its DDP wrapper) whose
               attention blocks have a cached ``last_att`` tensor.

    Returns:
        List of per-layer mean entropy values (nats).
    """
    layer_entropies: list[float] = []
    for block in model.blocks:
        if hasattr(block.attn, "last_att"):
            att = block.attn.last_att  # (B, n_head, N, N)
            entropy = -torch.sum(att * torch.log(att + 1e-9), dim=-1)
            layer_entropies.append(entropy.mean().item())
        else:
            layer_entropies.append(0.0)
    return layer_entropies
