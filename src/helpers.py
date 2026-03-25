"""
Helper functions for curvature analysis of NanoGPT.

All second-order quantities are estimated without materialising the full
Hessian matrix; instead we use power iteration on Hessian-vector
products (HVPs) computed cheaply via PyTorch's ``autograd.grad``.

Functions
---------
power_iteration          — λ_max of the full Hessian via HVP iteration.
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
# Power Iteration — λ_max of the Hessian
# ======================================================================


def power_iteration(
    loss: torch.Tensor | None,
    model: torch.nn.Module,
    max_iter: int = 20,
    tol: float = 1e-4,
    vv_mask: torch.Tensor | None = None,
    D_inv_sqrt: torch.Tensor | None = None,
    hvp_fn: callable | None = None,
) -> float:
    """
    Estimate the largest eigenvalue (λ_max) of the loss Hessian via
    the Lanczos / power-iteration method using Hessian-vector products.

    The graph must still be live when this function is called, so pass
    ``create_graph=True`` and ``retain_graph=True`` to the upstream
    ``loss.backward()`` (or use ``autograd.grad`` as done in
    ``get_curvature_metrics``).

    Args:
        loss:     Scalar loss tensor (graph must be retained).
        model:    The model whose parameters define the Hessian.
        max_iter: Number of power-iteration steps.
        tol:      Early-stop threshold on the relative eigenvalue change.
        vv_mask:  Optional flat mask selecting a subspace (same length
                  as the flattened parameter vector). When provided the
                  iteration projects iterates into this subspace.
        D_inv_sqrt: Optional preconditioner vector (flat) implementing
                  elementwise multiplication by D^{-1/2}. When provided
                  runs power-iteration for D^{-1/2} H D^{-1/2}.

    Returns:
        Estimated λ_max as a Python float.
    """
    # If an explicit H*v operator is provided, use it; otherwise build the
    # default Hessian H*v operator from `loss` via autograd.
    params = tuple(model.parameters())

    if hvp_fn is None:
        if loss is None:
            raise ValueError("loss must be provided when hvp_fn is None")
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True
        )
        flat_grads = torch.cat([g.reshape(-1) for g in grads])

        def _hvp(v: torch.Tensor) -> torch.Tensor:
            hvp_tuple = torch.autograd.grad(flat_grads, params, grad_outputs=v, retain_graph=True)
            return torch.cat([g.contiguous().reshape(-1) for g in hvp_tuple])

        hvp_callable = _hvp
    else:
        hvp_callable = hvp_fn

    # --- Full / masked / preconditioned iterations using hvp_callable ---
    # Choose initial vector size from one model parameter
    sample_param = next(model.parameters())
    n = sum(p.numel() for p in params)

    if vv_mask is None and D_inv_sqrt is None:
        v = torch.randn(n, device=sample_param.device)
        v = v / (v.norm() + 1e-9)
        prev_lambda = None
        flat_hvp = None
        for _ in range(max_iter):
            flat_hvp = hvp_callable(v)
            lam = torch.dot(v, flat_hvp).item()
            v = flat_hvp / (flat_hvp.norm() + 1e-9)
            if prev_lambda is not None and abs(lam - prev_lambda) / (abs(prev_lambda) + 1e-9) < tol:
                break
            prev_lambda = lam
        return torch.dot(v, flat_hvp).item() if flat_hvp is not None else 0.0

    if vv_mask is not None:
        v = torch.randn(n, device=sample_param.device) * vv_mask.to(sample_param.device)
        v = v / (v.norm() + 1e-9)
        flat_hvp = None
        for _ in range(max_iter):
            flat_hvp = hvp_callable(v)
            v = (flat_hvp * vv_mask.to(flat_hvp.device))
            v = v / (v.norm() + 1e-9)
        return torch.dot(v, flat_hvp).item() if flat_hvp is not None else 0.0

    if D_inv_sqrt is not None:
        D_inv_sqrt = D_inv_sqrt.to(sample_param.device)
        v_prec = torch.randn(n, device=sample_param.device)
        v_prec = v_prec / (v_prec.norm() + 1e-9)
        step3 = None
        for _ in range(max_iter):
            step1 = D_inv_sqrt * v_prec
            flat_hvp_prec = hvp_callable(step1)
            step3 = D_inv_sqrt * flat_hvp_prec
            v_prec = step3 / (step3.norm() + 1e-9)
        return torch.dot(v_prec, step3).item() if step3 is not None else 0.0

    return 0.0


# ======================================================================
# Value-subspace mask
# ======================================================================


def get_VV_subspace_mask(model: torch.nn.Module) -> torch.Tensor:
    """
    Build a flat binary mask that selects only the value-projection
    parameters of every attention layer (the ``W_V`` and corresponding
    bias slice of the fused ``c_attn`` layer).

    NanoGPT fuses Q, K, V into a single ``c_attn`` linear layer of shape
    ``(3*n_embd, n_embd)``.  The value weights occupy the last third of
    the output dimension (rows ``2*d : 3*d``).

    Args:
        model: A NanoGPT ``GPT`` (or ``HookedGPT``) model.

    Returns:
        1-D float tensor on CPU, same length as ``flat_params``.
    """
    mask_parts = []
    for name, param in model.named_parameters():
        if "c_attn.weight" in name:
            m = torch.zeros_like(param)
            d = param.size(0) // 3
            m[2 * d :, :] = 1.0
            mask_parts.append(m.reshape(-1))
        elif "c_attn.bias" in name:
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

    * ``hessian``   — λ_max of the full loss Hessian H.
    * ``prec_h``    — λ_max of the Adam-preconditioned Hessian
                      D^{-½} H D^{-½}  (where D = diag of Adam's v̂).
                      Falls back to ``hessian`` for SGD.
    * ``hessian_vv``— λ_max of H restricted to the value-projection
                      subspace (H_VV).
    * ``gn``        — λ_max of the Gauss-Newton matrix H_GN, computed
                      via JVP / VJP factorisation.
    * ``fd``        — Finite-difference sharpness proxy
                      ‖Δg‖ / ‖Δw‖  between two consecutive steps.
                      Only computed when ``compute_fd=True``.

    Args:
        model:      Model with live computation graph (no zero_grad called).
        optimizer:  Current optimiser (used to read Adam's second moment).
        X, Y:       Current training batch.
        loss:       Scalar loss with a live graph (create_graph must have
                    been used when computing gradients).
        vv_mask:    Output of ``get_VV_subspace_mask(model)``.
        max_iter:   Power-iteration steps.
        compute_fd: Whether to compute the finite-difference proxy
                    (requires an extra forward/backward pass).

    Returns:
        Dict with keys: ``hessian``, ``prec_h``, ``hessian_vv``,
        ``gn``, ``fd``.
    """
    # Use the unified power_iteration helper for H, H_VV and preconditioned H
    hessian_norm = power_iteration(loss, model, max_iter=max_iter)

    # Value-subspace estimate
    hessian_vv_norm = power_iteration(loss, model, max_iter=max_iter, vv_mask=vv_mask)

    # Build D^{-1/2} preconditioner from optimizer state (if Adam-like)
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
            adam_epsilon = 1e-7
            P = torch.sqrt(v_hat) + adam_epsilon
            D_inv_sqrt_parts.append((1.0 / torch.sqrt(P)).reshape(-1))
        else:
            D_inv_sqrt_parts.append(torch.ones(param.numel(), device=param.device))

    D_inv_sqrt = torch.cat(D_inv_sqrt_parts)

    if is_adam:
        prec_hessian_norm = power_iteration(
            loss, model, max_iter=max_iter, D_inv_sqrt=D_inv_sqrt
        )
    else:
        prec_hessian_norm = hessian_norm

    # Free second-order graph before Gauss-Newton (uses its own graph)
    # cleanup: avoid deleting names that aren't in this scope (leftover from
    # earlier refactor). Just clear CUDA cache to free temporary memory.
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 4. Gauss-Newton  (H_GN = J^T H_L J)
    # ------------------------------------------------------------------ #
    params_dict = dict(model.named_parameters())

    def _fwd(p_dict):
        logits, _ = functional_call(model, p_dict, (X, Y))
        return logits

    # Build an H*v operator for the Gauss-Newton matrix: given a flat vector
    # v (over params), return J^T H_L J v as a flat tensor.
    def gn_hvp(v_flat: torch.Tensor) -> torch.Tensor:
        tangents: dict[str, torch.Tensor] = {}
        offset = 0
        for k, p in params_dict.items():
            numel = p.numel()
            tangents[k] = v_flat[offset : offset + numel].view_as(p)
            offset += numel

        out_logits, Jv = jvp(_fwd, (params_dict,), (tangents,))
        if not out_logits.requires_grad:
            out_logits = out_logits.requires_grad_(True)

        vocab_size = out_logits.size(-1)
        l = F.cross_entropy(out_logits.reshape(-1, vocab_size), Y.reshape(-1))
        grad_l = torch.autograd.grad(l, out_logits, create_graph=True)[0]
        H_L_Jv = torch.autograd.grad(grad_l, out_logits, grad_outputs=Jv)[0]

        _, vjp_fn = vjp(_fwd, params_dict)
        gn_v_dict = vjp_fn(H_L_Jv.detach())[0]

        flat = torch.cat([gn_v_dict[k].contiguous().reshape(-1) for k in params_dict.keys()])
        return flat

    gn_norm = power_iteration(None, model, max_iter=max_iter, hvp_fn=gn_hvp)
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 5. Finite-Difference proxy  FD = ‖Δg‖ / ‖Δw‖
    # ------------------------------------------------------------------ #
    fd_norm = 0.0
    if compute_fd:
        w_k = torch.cat([p.data.reshape(-1) for p in model.parameters()])

        optimizer.zero_grad()
        _, loss_fd = model(X, Y)
        loss_fd.backward()
        g_k = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        w_k1 = torch.cat([p.data.reshape(-1) for p in model.parameters()])

        optimizer.zero_grad()
        _, loss_next = model(X, Y)
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
        model: A ``HookedGPT`` model whose attention blocks have already
               executed a forward pass (so ``last_att`` is populated).

    Returns:
        List of length ``n_layer``, each entry the mean entropy (float)
        averaged over batch, heads, and sequence position.
    """
    layer_entropies: list[float] = []
    for block in model.transformer.h:
        if hasattr(block.attn, "last_att"):
            att = block.attn.last_att  # (B, n_head, T, T)
            entropy = -torch.sum(att * torch.log(att + 1e-9), dim=-1)  # (B, n_head, T)
            layer_entropies.append(entropy.mean().item())
        else:
            layer_entropies.append(0.0)
    return layer_entropies
