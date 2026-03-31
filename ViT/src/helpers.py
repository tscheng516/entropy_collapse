"""
ViT-specific curvature helpers and attention-entropy utilities.

All second-order quantities are estimated without materialising the full
Hessian matrix; instead we use power iteration on Hessian-vector products
(HVPs) computed cheaply via PyTorch's ``autograd.grad``.

Functions
---------
get_VV_subspace_mask     — Binary mask selecting value-projection params.
get_curvature_metrics    — All five sharpness proxies in one pass:
                             H (exact), H_tilde (preconditioned),
                             H_VV (value subspace), H_GN (Gauss-Newton),
                             FD (finite-difference).
get_attention_entropy    — Per-layer Shannon entropy of cached attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp


# ======================================================================
# Value-subspace mask
# ======================================================================


def get_VV_subspace_mask(model: torch.nn.Module) -> torch.Tensor:
    """
    Build a flat binary mask that selects only the value-projection
    parameters of every attention layer (the W_V slice of the fused
    ``qkv`` linear layer).

    timm ViT fuses Q, K, V into a single ``attn.qkv`` linear layer of
    shape ``(3*dim, dim)``.  The value weights occupy the last third of
    the output dimension (rows ``2*d : 3*d``).

    Args:
        model: A timm ViT (HookedViT) model.

    Returns:
        1-D float tensor on CPU, same length as the flattened parameter
        vector.
    """
    mask_parts = []
    for name, param in model.named_parameters():
        if name.endswith(".attn.qkv.weight"):
            m = torch.zeros_like(param)
            d = param.size(0) // 3
            m[2 * d :, :] = 1.0
            mask_parts.append(m.reshape(-1))
        elif name.endswith(".attn.qkv.bias"):
            m = torch.zeros_like(param)
            d = param.size(0) // 3
            m[2 * d :] = 1.0
            mask_parts.append(m.reshape(-1))
        else:
            mask_parts.append(torch.zeros_like(param).reshape(-1))
    return torch.cat(mask_parts)


# ======================================================================
# All curvature metrics in one function
# ======================================================================


def get_curvature_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss: torch.Tensor,
    vv_mask: torch.Tensor,
    max_iter: int = 10,
    compute_fd: bool = True,
) -> dict[str, float]:
    """
    Compute five sharpness proxies for the current model state.

    The proxies are (all estimated via power iteration unless stated):

    * ``hessian``    — λ_max of the full loss Hessian H.
    * ``prec_h``     — λ_max of the Adam-preconditioned Hessian
                       D^{-½} H D^{-½}  (where D = diag of Adam's v̂).
                       Falls back to ``hessian`` for SGD.
    * ``hessian_vv`` — λ_max of H restricted to the value-projection
                       subspace (H_VV).
    * ``gn``         — λ_max of the Gauss-Newton matrix H_GN, computed
                       via JVP / VJP factorisation.
    * ``fd``         — Finite-difference sharpness proxy
                       ‖Δg‖ / ‖Δw‖  between two consecutive steps.
                       Only computed when ``compute_fd=True``.

    Args:
        model:      Model with a live computation graph.
        optimizer:  Current optimiser (used to read Adam's second moment).
        X:          Batch of input images  (B, C, H, W).
        Y:          Batch of class labels  (B,) — integer class indices.
        loss:       Scalar loss with a live graph (must have been computed
                    with ``create_graph=True`` or retain_graph available).
        vv_mask:    Output of ``get_VV_subspace_mask(model)``.
        max_iter:   Power-iteration steps.
        compute_fd: Whether to compute the finite-difference proxy.

    Returns:
        Dict with keys: ``hessian``, ``prec_h``, ``hessian_vv``,
        ``gn``, ``fd``.
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
    # ------------------------------------------------------------------ #
    params_dict = dict(model.named_parameters())

    def _fwd(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pure-function forward: returns logits (B, num_classes)."""
        return functional_call(model, p_dict, (X,))

    flat_params = torch.cat([p.reshape(-1) for p in model.parameters()])
    v_g = torch.randn_like(flat_params)
    v_g = v_g / (v_g.norm() + 1e-9)
    flat_gn_v: torch.Tensor | None = None
    for _ in range(max_iter):
        tangents: dict[str, torch.Tensor] = {}
        offset = 0
        for k, p in params_dict.items():
            numel = p.numel()
            tangents[k] = v_g[offset : offset + numel].view_as(p)
            offset += numel

        out_logits, Jv = jvp(_fwd, (params_dict,), (tangents,))
        if not out_logits.requires_grad:
            out_logits = out_logits.requires_grad_(True)

        num_classes = out_logits.size(-1)
        l = F.cross_entropy(out_logits.reshape(-1, num_classes), Y.reshape(-1))
        grad_l = torch.autograd.grad(l, out_logits, create_graph=True)[0]
        H_L_Jv = torch.autograd.grad(grad_l, out_logits, grad_outputs=Jv)[0]

        _, vjp_fn = vjp(_fwd, params_dict)
        gn_v_dict = vjp_fn(H_L_Jv.detach())[0]

        flat_gn_v = torch.cat(
            [gn_v_dict[k].contiguous().reshape(-1) for k in params_dict.keys()]
        )
        v_g = flat_gn_v / (flat_gn_v.norm() + 1e-9)

        del out_logits, Jv, l, grad_l, H_L_Jv, gn_v_dict, tangents

    gn_norm = torch.dot(v_g, flat_gn_v).item() if flat_gn_v is not None else 0.0
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 5. Finite-Difference proxy  FD = ‖Δg‖ / ‖Δw‖
    # ------------------------------------------------------------------ #
    fd_norm = 0.0
    if compute_fd:
        w_k = torch.cat([p.data.reshape(-1) for p in model.parameters()])

        optimizer.zero_grad()
        logits_fd = model(X)
        loss_fd = F.cross_entropy(logits_fd, Y)
        loss_fd.backward()
        g_k = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        w_k1 = torch.cat([p.data.reshape(-1) for p in model.parameters()])

        optimizer.zero_grad()
        logits_next = model(X)
        loss_next = F.cross_entropy(logits_next, Y)
        loss_next.backward()
        g_k1 = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

        dw = w_k1 - w_k
        dg = g_k1 - g_k
        fd_norm = (dg.norm() / (dw.norm() + 1e-9)).item()
        optimizer.zero_grad()

    return {
        "hessian": hessian_norm,
        "prec_h": prec_hessian_norm,
        "hessian_vv": hessian_vv_norm,
        "gn": gn_norm,
        "fd": fd_norm,
    }


# ======================================================================
# Attention Entropy
# ======================================================================


def get_attention_entropy(model: torch.nn.Module) -> list[float]:
    """
    Compute the mean Shannon entropy of the attention distribution for
    each transformer layer, using the cached ``last_att`` tensors written
    by the patched attention forward pass.

    Entropy is computed in nats (natural log) as:
        H = -Σ p·ln(p)  over the key dimension.

    Args:
        model: A HookedViT model whose attention blocks have already
               executed a forward pass (so ``last_att`` is populated).

    Returns:
        List of length ``depth`` (number of transformer blocks), each
        entry the mean entropy (float) averaged over batch, heads, and
        sequence (patch) positions.
    """
    layer_entropies: list[float] = []
    for block in model.blocks:
        if hasattr(block.attn, "last_att"):
            att = block.attn.last_att  # (B, n_head, N, N)
            entropy = -torch.sum(att * torch.log(att + 1e-9), dim=-1)  # (B, n_head, N)
            layer_entropies.append(entropy.mean().item())
        else:
            layer_entropies.append(0.0)
    return layer_entropies
