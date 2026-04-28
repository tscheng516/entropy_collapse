"""
ViT-specific curvature helpers and attention-entropy utilities.

All second-order quantities are estimated without materialising the full
Hessian matrix; instead we use power iteration on Hessian-vector products
(HVPs) computed cheaply via PyTorch's ``autograd.grad``.

Functions
---------
get_VV_subspace_mask     — Binary mask selecting value-projection params.
get_curvature_metrics    — Six always-active sharpness proxies plus three
                             optional (compute_fd=True) ones:
                             hessian (exact H), prec_h (preconditioned H),
                             hessian_vv (value-projection subspace H),
                             gn (Gauss-Newton), diag_h (diagonal Hessian),
                             fisher (empirical Fisher)
                             (+ bfgs, fd, kfac when compute_fd=True).
get_attention_entropy    — Per-layer Shannon entropy of cached attention.
"""

from __future__ import annotations

from contextlib import nullcontext
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
    lam: float = 10.0,
    eps: float = 1e-12,
    use_abs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a smooth trend from a positive time-series via regularised
    least-squares in log-space:

        min_m  ‖log(y) − m‖²  +  λ ‖D m‖²

    D is the second-difference operator; lam controls smoothness (higher → smoother).
    lam=10 works well for typical curvature traces.

    Args:
        y:       1-D array of (ideally positive) measurements.
        lam:     Smoothing strength (λ).  10 is a good default.
        eps:     Floor added before logging to avoid log(0).
        use_abs: If True, take |y| before logging (for estimators that
                 may produce small negative values due to numerical noise).

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
    vv_mask: torch.Tensor,
    max_iter: int = 10,
    compute_fd: bool = False,
    hessian_batch_size: int = 128,
    use_grad_ckpt: bool = False,  # kept for API compatibility; has no effect
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    """
    Compute sharpness proxies for the current model state.

    Always-active proxies (six):
      * ``hessian``    — λ_max(H) via power iteration on the full Hessian.
      * ``prec_h``     — λ_max(D^{-½} H D^{-½}) Adam-preconditioned Hessian.
                         Falls back to ``hessian`` for SGD.
      * ``hessian_vv`` — λ_max(H_VV), H restricted to value-projection subspace.
      * ``gn``         — λ_max(H_GN = J^T H_L J) Gauss-Newton matrix.
      * ``diag_h``     — max(diag(H)) via Bekas–Kokiopoulou–Saad Hutchinson
                         diagonal estimator.
      * ``fisher``     — λ_max(F) empirical Fisher, via JVP/VJP power iteration.

    Optional proxies (compute_fd=True):
      * ``bfgs``  — λ_max(H) via central-difference finite differences (O(ε²)).
      * ``fd``    — λ_max(H) via forward-difference finite differences (O(ε)).
      * ``kfac``  — K-FAC proxy: max λ_max(A)·λ_max(G) across all Linear layers.

    Memory optimisations vs original:
      1. ``hessian_batch_size`` samples are sliced from X/Y before any
         computation.  All proxies share this smaller diagnostic batch, keeping
         peak activation memory (O(batch × seq_len²)) bounded.
      2. ``hessian``, ``hessian_vv``, ``prec_h``, and ``diag_h`` share a
         **single** ``create_graph=True`` forward pass and first-order gradient
         graph.  The graph is released once, after all four proxies finish.
      3. K-FAC is gated behind ``compute_fd`` (disabled by default).

    Note on gradient checkpointing: timm's ``set_grad_checkpointing`` is
    incompatible with ``autograd.grad(..., create_graph=True)`` because
    timm imports ``torch.utils.checkpoint.checkpoint`` into its own module
    namespace at load time (``from torch.utils.checkpoint import checkpoint``),
    and that local binding uses ``use_reentrant=True`` by default, which
    forbids higher-order autograd.  The ``use_grad_ckpt`` parameter is
    therefore retained only for API compatibility and has no effect.
    The ``hessian_batch_size`` slice is the primary memory-reduction lever:
    128 samples reduce peak activation memory ~8× vs. a typical bs=1024
    training batch, making the HVP forward pass feasible on ViT-L/H without
    any block-level checkpointing.

    Args:
        model:              Model, unwrapped from DDP.
        optimizer:          Current optimiser (used to read Adam second moment).
        X:                  Training-batch images  (B, C, H, W).
        Y:                  Training-batch labels  (B,).
        vv_mask:            Output of ``get_VV_subspace_mask(model)``.
        max_iter:           Power-iteration steps for λ_max estimation.
        compute_fd:         Enable finite-difference proxies (bfgs, fd) and
                            K-FAC.  Each costs at least one extra forward/
                            backward pass.
        hessian_batch_size: Number of samples drawn from X/Y for curvature
                            estimation.  Reduces peak GPU memory in proportion
                            to batch reduction; 128 is sufficient for λ_max
                            trend monitoring.
        use_grad_ckpt:      Unused — kept for API compatibility only. See note
                            in docstring.
        label_smoothing:    Applied to the diagnostic CE loss (should match
                            cfg.label_smoothing for consistent curvature scale).

    Returns:
        Dict with keys: ``hessian``, ``prec_h``, ``hessian_vv``, ``gn``,
        ``fd``, ``diag_h``, ``fisher``, ``bfgs``, ``kfac``
        (``fd`` / ``bfgs`` / ``kfac`` are 0.0 when ``compute_fd=False``).
    """
    # ------------------------------------------------------------------ #
    # Diagnostic mini-batch.
    # All proxies operate on this slice so that the retained activation
    # memory (proportional to batch size) stays bounded.
    # ------------------------------------------------------------------ #
    Xc = X[:hessian_batch_size].detach()
    Yc = Y[:hessian_batch_size].detach()

    # Shared parameter containers used by GN and Fisher later.
    params_keys = [k for k, _ in model.named_parameters()]
    params_vals = [p for _, p in model.named_parameters()]
    flat_params = torch.cat([p.reshape(-1) for p in params_vals])

    # ------------------------------------------------------------------ #
    # Shared forward pass for HVP-based proxies (H, H_VV, prec_H, diag_H).
    #
    # Flash / fused attention is disabled so that second-order autograd can
    # traverse the explicit softmax graph in _patched_attn_forward.
    # ------------------------------------------------------------------ #
    _sdp_ctx = (
        torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
        if Xc.is_cuda
        else nullcontext()
    )
    with _sdp_ctx:
        logits_h = model(Xc)
    loss = F.cross_entropy(logits_h, Yc, label_smoothing=label_smoothing)
    del logits_h  # free; the scalar loss node retains the graph

    # ------------------------------------------------------------------ #
    # Shared first-order gradient graph.
    # H, H_VV, prec_H, and diag_H all reuse this single graph via
    # retain_graph=True and release it together after diag_H finishes.
    # ------------------------------------------------------------------ #
    grads = torch.autograd.grad(
        loss, model.parameters(), create_graph=True, retain_graph=True
    )
    flat_grads = torch.cat([g.reshape(-1) for g in grads])

    # ------------------------------------------------------------------ #
    # 1. Hessian Power Iteration  λ_max(H)
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
    # 2. Value-subspace Power Iteration  λ_max(H_VV)
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
    # 3. Adam-preconditioned Hessian  λ_max(D^{-½} H D^{-½})
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

    # ------------------------------------------------------------------ #
    # 4. Diagonal Hessian  max(diag(H))  — Hutchinson estimator
    #
    # Runs here (before releasing flat_grads) so it shares the single
    # computation graph with proxies 1–3. This avoids a second forward
    # pass for this proxy.
    # ------------------------------------------------------------------ #
    diag_h_norm = 0.0
    n_hutchinson = max(1, max_iter // 2)
    try:
        diag_acc = torch.zeros_like(flat_grads)
        for _ in range(n_hutchinson):
            z = (
                torch.randint(
                    0, 2, flat_grads.shape, device=flat_grads.device
                ).float()
                * 2.0
                - 1.0
            )
            hz = torch.autograd.grad(
                flat_grads, model.parameters(), grad_outputs=z, retain_graph=True
            )
            flat_hz = torch.cat([g.contiguous().reshape(-1) for g in hz])
            diag_acc += z * flat_hz  # z_i · (Hz)_i ≈ H_ii
        diag_acc /= n_hutchinson
        diag_h_norm = diag_acc.max().item()
        del diag_acc
    except Exception:
        pass

    # Release the shared computation graph — all four HVP proxies are done.
    del grads, flat_grads, v_h, flat_hvp, v_vv, flat_hvp_vv, D_inv_sqrt, loss
    if is_adam:
        del v_prec, step3
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 5. Gauss-Newton  λ_max(H_GN = J^T H_L J)
    #
    # Uses autograd_functional.jvp/vjp (not torch.func) to tolerate the
    # module-state side-effects of the patched attention forward.
    # ------------------------------------------------------------------ #
    def _fwd_tuple(*p_vals: torch.Tensor) -> torch.Tensor:
        """Forward pass with parameters supplied as a flat tuple."""
        p_dict = {k: v for k, v in zip(params_keys, p_vals)}
        return functional_call(model, p_dict, (Xc,))

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

        out_logits, Jv = autograd_functional.jvp(
            _fwd_tuple, tuple(params_vals), tuple(tangents_list), create_graph=False
        )

        if not out_logits.requires_grad:
            out_logits = out_logits.detach().requires_grad_(True)

        num_classes = out_logits.size(-1)
        l = F.cross_entropy(out_logits.reshape(-1, num_classes), Yc.reshape(-1))
        grad_l = torch.autograd.grad(l, out_logits, create_graph=True)[0]
        H_L_Jv = torch.autograd.grad(grad_l, out_logits, grad_outputs=Jv.detach())[0]

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
    # 6. Empirical Fisher  λ_max(F)  via power iteration
    #
    # F = (1/B) Σ_i g_i g_i^T  where g_i = ∂ℓ_i/∂θ.
    # Fisher-vector product: Fv = (1/B) J_ℓ^T (J_ℓ v).
    # ------------------------------------------------------------------ #
    fisher_norm = 0.0
    try:
        def _fwd_loss_tuple(*p_vals: torch.Tensor) -> torch.Tensor:
            """Per-sample cross-entropy (reduction='none')."""
            p_dict = {k: v for k, v in zip(params_keys, p_vals)}
            logits = functional_call(model, p_dict, (Xc,))
            return F.cross_entropy(logits, Yc, reduction="none")  # (B,)

        B = Xc.size(0)
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
            )  # (B,)

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
    # 7. BFGS (Central-Difference)  λ_max(H) — only when compute_fd=True
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
                loss_bp = F.cross_entropy(model(Xc), Yc)
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
                loss_bm = F.cross_entropy(model(Xc), Yc)
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
    # 8. K-FAC Proxy — only when compute_fd=True
    #
    # Estimates max λ_max(A) · λ_max(G) across all Linear layers.
    # ------------------------------------------------------------------ #
    kfac_norm = 0.0
    if compute_fd:
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
            logits_kfac = model(Xc)
            loss_kfac = F.cross_entropy(logits_kfac, Yc)
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
    # 9. FD (Forward-Difference)  λ_max(H) — only when compute_fd=True
    # ------------------------------------------------------------------ #
    fd_norm = 0.0
    if compute_fd:
        try:
            eps_fd = 1e-4
            optimizer.zero_grad()
            loss_f0 = F.cross_entropy(model(Xc), Yc)
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
                loss_fp = F.cross_entropy(model(Xc), Yc)
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
