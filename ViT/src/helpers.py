"""
ViT-specific curvature helpers and attention-entropy utilities.

All second-order quantities are estimated without materialising the full
Hessian matrix; instead we use power iteration on Hessian-vector products
(HVPs) computed cheaply via PyTorch's ``autograd.grad``.

Functions
---------
get_VV_subspace_mask     — Binary mask selecting value-projection params.
get_curvature_metrics    — All nine sharpness proxies in one pass:
                             H (exact), H_tilde (preconditioned),
                             H_VV (value subspace), H_GN (Gauss-Newton),
                             FD (finite-difference), diag_h (diagonal
                             Hessian), fisher (empirical Fisher), bfgs
                             (L-BFGS curvature), kfac (K-FAC curvature).
get_attention_entropy    — Per-layer Shannon entropy of cached attention.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.autograd import functional as autograd_functional


# ======================================================================
# Smooth log-trend extraction
# ======================================================================


def _second_difference_matrix(n: int) -> np.ndarray:
    """
    Build the (n-2)×n second-difference operator D such that
    (D m)_i = m_i − 2 m_{i+1} + m_{i+2}.
    """
    if n < 3:
        return np.zeros((0, n), dtype=np.float64)
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i : i + 3] = [1.0, -2.0, 1.0]
    return D


def smooth_log_trend(
    y: np.ndarray | list,
    lam: float = 100.0,
    eps: float = 1e-12,
    use_abs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a smooth trend from a positive time-series by solving a
    regularised least-squares problem in log-space:

        min_m  ‖log(y) − m‖²  +  λ ‖D m‖²

    where D is the second-difference operator (discrete second derivative).
    The regularisation parameter *lam* controls the smoothness of the trend
    (higher → smoother).

    This is the same technique used in the LLM notebook
    (``NanoGPT_shakespear.ipynb``) to compare curvature-proxy dynamics
    without relying solely on spike-detection.

    Args:
        y:       1-D array of (ideally positive) measurements.
        lam:     Smoothing strength (λ).  100 is a good default.
        eps:     Floor value to avoid log(0).
        use_abs: If True, take |y| before logging (useful when the
                 estimator is conceptually non-negative but may produce
                 small negative values due to numerical noise).

    Returns:
        trend_raw:  exp(m) — smooth trend on the original scale.
        trend_log:  m — smooth trend in log-space.
        log_y:      log(y_safe) — the raw log-series.
    """
    y = np.asarray(y, dtype=np.float64)
    y_safe = np.abs(y) + eps if use_abs else np.maximum(y, eps)

    log_y = np.log(y_safe)
    n = len(log_y)

    if n < 3:
        return y_safe.copy(), log_y.copy(), log_y.copy()

    D = _second_difference_matrix(n)
    A = np.eye(n, dtype=np.float64) + lam * (D.T @ D)

    trend_log = np.linalg.solve(A, log_y)
    trend_raw = np.exp(trend_log)

    return trend_raw, trend_log, log_y


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
    Compute nine sharpness proxies for the current model state.

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
    * ``diag_h``     — Trace of the diagonal Hessian, estimated via
                       Hutchinson's stochastic trace estimator using
                       Rademacher random vectors.
    * ``fisher``     — Trace of the empirical Fisher information matrix,
                       Tr(F) = ‖g‖², where g is the mini-batch gradient.
    * ``bfgs``       — L-BFGS curvature estimate s^T y / s^T s from
                       the most recent parameter / gradient displacement
                       pair (finite-difference secant curvature).
    * ``kfac``       — Kronecker-Factored Approximate Curvature (K-FAC)
                       proxy: max Kronecker-factor eigenvalue product
                       across all linear layers, estimated from cached
                       input activations and output-gradient covariances.

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
        ``gn``, ``fd``, ``diag_h``, ``fisher``, ``bfgs``, ``kfac``.
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
    #
    # torch.func.jvp / vjp require pure, side-effect-free functions.
    # _patched_attn_forward writes self.last_att = attn.detach() on every
    # call, which corrupts the functorch tracing state and eventually
    # causes CUDA context corruption.  torch.autograd.functional.jvp /
    # vjp run through the standard autograd engine and tolerate such
    # module-state side effects.
    # ------------------------------------------------------------------ #
    params_keys = [k for k, _ in model.named_parameters()]
    params_vals = [p for _, p in model.named_parameters()]

    def _fwd_tuple(*p_vals: torch.Tensor) -> torch.Tensor:
        """Forward pass with parameters supplied as a flat tuple."""
        p_dict = {k: v for k, v in zip(params_keys, p_vals)}
        return functional_call(model, p_dict, (X,))

    flat_params = torch.cat([p.reshape(-1) for p in params_vals])
    v_g = torch.randn_like(flat_params)
    v_g = v_g / (v_g.norm() + 1e-9)
    flat_gn_v: torch.Tensor | None = None
    for _ in range(max_iter):
        # ---- JVP: Jv = J @ v ----
        tangents_list: list[torch.Tensor] = []
        offset = 0
        for p in params_vals:
            numel = p.numel()
            tangents_list.append(v_g[offset : offset + numel].view_as(p))
            offset += numel

        out_logits, Jv = autograd_functional.jvp(
            _fwd_tuple, tuple(params_vals), tuple(tangents_list), create_graph=False
        )

        # Ensure out_logits is a leaf tensor so torch.autograd.grad can
        # differentiate w.r.t. it.  detach() first in case jvp produced a
        # non-leaf with requires_grad=False (double-backward internals).
        if not out_logits.requires_grad:
            out_logits = out_logits.detach().requires_grad_(True)

        num_classes = out_logits.size(-1)
        l = F.cross_entropy(out_logits.reshape(-1, num_classes), Y.reshape(-1))
        grad_l = torch.autograd.grad(l, out_logits, create_graph=True)[0]
        H_L_Jv = torch.autograd.grad(grad_l, out_logits, grad_outputs=Jv.detach())[0]

        # ---- VJP: J^T @ (H_L @ Jv) ----
        _, gn_tuple = autograd_functional.vjp(
            _fwd_tuple, tuple(params_vals), H_L_Jv.detach()
        )

        flat_gn_v = torch.cat(
            [g.contiguous().reshape(-1) for g in gn_tuple]
        )
        v_g = flat_gn_v / (flat_gn_v.norm() + 1e-9)

        del out_logits, Jv, l, grad_l, H_L_Jv, gn_tuple, tangents_list
        torch.cuda.empty_cache()

    gn_norm = torch.dot(v_g, flat_gn_v).item() if flat_gn_v is not None else 0.0
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 5. Diagonal Hessian  (Hutchinson trace estimator)
    #
    # Tr(H) = E_z[z^T H z]  where z ~ Rademacher(±1).
    # We average over a few random vectors for a cheap trace estimate.
    #
    # NOTE: This MUST run before the FD / BFGS sections because those
    # call optimizer.step(), which modifies parameters in-place and
    # invalidates the computation graph of ``loss``.
    # ------------------------------------------------------------------ #
    diag_h_norm = 0.0
    n_hutchinson = max(1, max_iter // 2)
    try:
        grads_diag = torch.autograd.grad(
            loss, model.parameters(), create_graph=True, retain_graph=True
        )
        flat_grads_diag = torch.cat([g.reshape(-1) for g in grads_diag])
        trace_sum = 0.0
        for _ in range(n_hutchinson):
            z = torch.randint(0, 2, flat_grads_diag.shape, device=flat_grads_diag.device).float() * 2.0 - 1.0
            hz = torch.autograd.grad(
                flat_grads_diag, model.parameters(), grad_outputs=z, retain_graph=True
            )
            flat_hz = torch.cat([g.contiguous().reshape(-1) for g in hz])
            trace_sum += torch.dot(z, flat_hz).item()
        diag_h_norm = trace_sum / n_hutchinson
        del grads_diag, flat_grads_diag
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 6. Finite-Difference proxy  FD = ‖Δg‖ / ‖Δw‖
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

    # ------------------------------------------------------------------ #
    # 7. Empirical Fisher Information Matrix  Tr(F) = ‖g‖²
    #
    # The empirical Fisher is F = g g^T, so Tr(F) = ‖g‖².
    # We compute the gradient on the current mini-batch.
    # ------------------------------------------------------------------ #
    fisher_norm = 0.0
    try:
        optimizer.zero_grad()
        logits_f = model(X)
        loss_fisher = F.cross_entropy(logits_f, Y)
        loss_fisher.backward()
        fisher_g = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        fisher_norm = fisher_g.norm().square().item()
        optimizer.zero_grad()
        del fisher_g
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 8. L-BFGS Curvature Estimate  s^T y / s^T s
    #
    # Secant-equation curvature: take one optimiser step and compute
    # the curvature along the parameter displacement via the change
    # in gradients: (y = g_{k+1} - g_k, s = w_{k+1} - w_k).
    # ------------------------------------------------------------------ #
    bfgs_norm = 0.0
    try:
        w0 = torch.cat([p.data.reshape(-1).clone() for p in model.parameters()])
        optimizer.zero_grad()
        logits_b0 = model(X)
        loss_b0 = F.cross_entropy(logits_b0, Y)
        loss_b0.backward()
        g0 = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        w1 = torch.cat([p.data.reshape(-1) for p in model.parameters()])
        optimizer.zero_grad()
        logits_b1 = model(X)
        loss_b1 = F.cross_entropy(logits_b1, Y)
        loss_b1.backward()
        g1 = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

        s = w1 - w0
        y_bfgs = g1 - g0
        sts = torch.dot(s, s)
        bfgs_norm = (torch.dot(s, y_bfgs) / (sts + 1e-9)).item()
        optimizer.zero_grad()
        del w0, w1, g0, g1, s, y_bfgs
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 9. K-FAC Proxy  (Kronecker-Factored Approximate Curvature)
    #
    # For each Linear layer we estimate two small covariance factors:
    #   A = (1/B) a a^T   (input activations, registered via forward hook)
    #   G = (1/B) δ δ^T   (output gradients, registered via backward hook)
    # The K-FAC block curvature is  A ⊗ G  whose largest eigenvalue is
    # λ_max(A) · λ_max(G).  We report the maximum across layers.
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
        logits_kfac = model(X)
        loss_kfac = F.cross_entropy(logits_kfac, Y)
        loss_kfac.backward()
        optimizer.zero_grad()

        for h in handles:
            h.remove()

        max_kfac = 0.0
        for name in a_cache:
            if name not in g_cache:
                continue
            a = a_cache[name]  # (B, ..., d_in)
            g = g_cache[name]  # (B, ..., d_out)
            if a.ndim == 3:
                a = a.reshape(-1, a.size(-1))
            if g.ndim == 3:
                g = g.reshape(-1, g.size(-1))
            n_samples = a.size(0)

            # Covariance factor eigenvalues via power iteration (cheap)
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

    return {
        "hessian": hessian_norm,
        "prec_h": prec_hessian_norm,
        "hessian_vv": hessian_vv_norm,
        "gn": gn_norm,
        "fd": fd_norm,
        "diag_h": diag_h_norm,
        "fisher": fisher_norm,
        "bfgs": bfgs_norm,
        "kfac": kfac_norm,
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
