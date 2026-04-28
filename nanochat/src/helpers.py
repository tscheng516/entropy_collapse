"""
nanochat curvature helpers and attention-entropy utilities.

Adapted from ViT/src/helpers.py.  The curvature mathematics are identical
(power iteration on Hessian-vector products); only the model-specific wiring
changes:

  * ``get_VV_subspace_mask`` — selects the separate ``c_v.weight`` parameters
    (nanochat uses distinct c_q / c_k / c_v projections rather than a fused
    qkv matrix as in ViT / nanoGPT).

  * ``get_attention_entropy`` — iterates ``model.transformer.h`` (nanochat's
    block list) instead of ``model.blocks``.

  * ``get_curvature_metrics`` — reshapes language-model logits
    ``(B, T, vocab_size)`` to ``(B*T, vocab_size)`` before computing CE loss,
    consistent with how nanochat's own forward computes the training loss.

Functions
---------
get_VV_subspace_mask     — Binary mask selecting c_v parameters.
get_curvature_metrics    — Nine sharpness proxies (HVP-based + optional FD).
get_attention_entropy    — Per-layer Shannon entropy from cached attention.
smooth_log_trend         — Whittaker–Henderson smoother in log-space.
"""

from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.autograd import functional as autograd_functional


# ======================================================================
# Smooth log-trend extraction  (identical to ViT/src/helpers.py)
# ======================================================================


def _second_difference_matrix(n: int) -> np.ndarray:
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
    Whittaker–Henderson smoother in log-space.

        min_m  ‖log(y) − m‖²  +  λ ‖D m‖²

    Returns (trend_raw, trend_log, log_y).
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
    parameters of every attention layer.

    nanochat uses **separate** ``c_q``, ``c_k``, ``c_v`` linear layers
    (unlike timm ViT, which fuses all three into a single ``qkv``).  The
    value projection is therefore any parameter whose name ends with
    ``.attn.c_v.weight``.

    Args:
        model: A HookedGPT model (nanochat GPT with patched attention).

    Returns:
        1-D float tensor on CPU, same length as the flattened parameter
        vector, with 1s at ``c_v.weight`` positions and 0s elsewhere.
    """
    mask_parts = []
    for name, param in model.named_parameters():
        if name.endswith(".attn.c_v.weight"):
            mask_parts.append(torch.ones_like(param).reshape(-1))
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
    hessian_batch_size: int = 2,
    use_grad_ckpt: bool = False,  # API-compat; no effect
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    """
    Compute sharpness proxies for the current nanochat model state.

    Identical to ViT/src/helpers.py except:
      * ``X`` is token indices ``(B, T)`` dtype ``torch.long``.
      * ``Y`` is target indices ``(B, T)`` dtype ``torch.long``.
      * Model output ``logits`` has shape ``(B, T, vocab_size)`` and is
        reshaped to ``(B*T, vocab_size)`` before every CE call.
      * hessian_batch_size defaults to 2 (large T → O(B·T²) attention mem).

    See ViT/src/helpers.py for full docstring of each proxy.

    Returns:
        Dict with keys: ``hessian``, ``prec_h``, ``hessian_vv``, ``gn``,
        ``fd``, ``diag_h``, ``fisher``, ``bfgs``, ``kfac``.
    """
    # ------------------------------------------------------------------ #
    # Diagnostic mini-batch — token ids, no dtype issue.
    # ------------------------------------------------------------------ #
    Xc = X[:hessian_batch_size].detach()
    Yc = Y[:hessian_batch_size].detach()

    # LM loss helper: handles (B, T, V) → (B*T, V) reshaping.
    def _ce(logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            label_smoothing=label_smoothing,
            **kwargs,
        )

    params_keys = [k for k, _ in model.named_parameters()]
    params_vals = [p for _, p in model.named_parameters()]
    flat_params = torch.cat([p.reshape(-1) for p in params_vals])

    # ------------------------------------------------------------------ #
    # Shared forward for HVP proxies.
    # Disable SDPA optimisations so the autograd graph is fully explicit.
    # (The patched attention already avoids flash_attn/SDPA, so this is
    # belt-and-suspenders only.)
    # ------------------------------------------------------------------ #
    _sdp_ctx = (
        torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
        if Xc.is_cuda
        else nullcontext()
    )
    with _sdp_ctx:
        logits_h = model(Xc)          # (B, T, vocab_size) — no targets arg
    loss = _ce(logits_h, Yc)
    del logits_h

    grads = torch.autograd.grad(
        loss, model.parameters(), create_graph=True, retain_graph=True
    )
    flat_grads = torch.cat([g.reshape(-1) for g in grads])

    # ------------------------------------------------------------------ #
    # 1. Hessian λ_max(H)
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
    # 2. Value-subspace λ_max(H_VV)
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
    # 3. Adam-preconditioned Hessian λ_max(D^{-½} H D^{-½})
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
        prec_hessian_norm = hessian_norm  # Muon groups: fall back to H

    # ------------------------------------------------------------------ #
    # 4. Diagonal Hessian  max(diag(H)) — Hutchinson estimator
    # ------------------------------------------------------------------ #
    diag_h_norm = 0.0
    n_hutchinson = max(1, max_iter // 2)
    try:
        diag_acc = torch.zeros_like(flat_grads)
        for _ in range(n_hutchinson):
            z = (
                torch.randint(0, 2, flat_grads.shape, device=flat_grads.device).float()
                * 2.0 - 1.0
            )
            hz = torch.autograd.grad(
                flat_grads, model.parameters(), grad_outputs=z, retain_graph=True
            )
            flat_hz = torch.cat([g.contiguous().reshape(-1) for g in hz])
            diag_acc += z * flat_hz
        diag_acc /= n_hutchinson
        diag_h_norm = diag_acc.max().item()
        del diag_acc
    except Exception:
        pass

    del grads, flat_grads, v_h, flat_hvp, v_vv, flat_hvp_vv, D_inv_sqrt, loss
    if is_adam:
        del v_prec, step3
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 5. Gauss-Newton  λ_max(H_GN = J^T H_L J)
    # ------------------------------------------------------------------ #
    def _fwd_tuple(*p_vals: torch.Tensor) -> torch.Tensor:
        p_dict = {k: v for k, v in zip(params_keys, p_vals)}
        return functional_call(model, p_dict, (Xc,))  # → (B, T, V)

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

        # Reshape (B, T, V) → (B*T, V) for CE
        num_classes = out_logits.size(-1)
        l = F.cross_entropy(out_logits.reshape(-1, num_classes), Yc.reshape(-1))
        grad_l = torch.autograd.grad(l, out_logits, create_graph=True)[0]
        H_L_Jv = torch.autograd.grad(grad_l, out_logits, grad_outputs=Jv.detach())[0]

        _, gn_tuple = autograd_functional.vjp(
            _fwd_tuple, tuple(params_vals), H_L_Jv.detach()
        )

        flat_gn_v = torch.cat([g.contiguous().reshape(-1) for g in gn_tuple])
        v_g = flat_gn_v / (flat_gn_v.norm() + 1e-9)

        del out_logits, Jv, l, grad_l, H_L_Jv, gn_tuple, tangents_list
        torch.cuda.empty_cache()

    gn_norm = torch.dot(v_g, flat_gn_v).item() if flat_gn_v is not None else 0.0
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 6. Empirical Fisher  λ_max(F)
    # ------------------------------------------------------------------ #
    fisher_norm = 0.0
    try:
        def _fwd_loss_tuple(*p_vals: torch.Tensor) -> torch.Tensor:
            p_dict = {k: v for k, v in zip(params_keys, p_vals)}
            logits = functional_call(model, p_dict, (Xc,))  # (B, T, V)
            # reduction='none' → (B*T,); consistent with per-token Fisher
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                Yc.view(-1),
                reduction="none",
            )

        B_tok = Xc.size(0) * Xc.size(1)  # total tokens in diagnostic batch
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
            )  # (B*T,)

            _, vjp_f = autograd_functional.vjp(
                _fwd_loss_tuple, tuple(params_vals), Jv_f.detach(),
            )

            flat_fv = torch.cat([g.contiguous().reshape(-1) for g in vjp_f]) / B_tok
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
    # 7. BFGS (Central-Difference) — only when compute_fd=True
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
                g_plus = torch.cat(
                    [p.grad.reshape(-1).clone() for p in model.parameters()]
                )
                _loss = _ce(model(Xc), Yc)
                _loss.backward()
                g_plus = torch.cat(
                    [p.grad.reshape(-1).clone() for p in model.parameters()]
                )

                offset = 0
                for p in params_vals:
                    numel = p.numel()
                    p.data.add_(-2.0 * eps_cd * v_b[offset : offset + numel].view_as(p))
                    offset += numel
                optimizer.zero_grad()
                _loss = _ce(model(Xc), Yc)
                _loss.backward()
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
        except Exception:
            pass
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 8. K-FAC Proxy — only when compute_fd=True
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
            _kfac_loss = _ce(model(Xc), Yc)
            _kfac_loss.backward()
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
    # 9. FD (Forward-Difference) — only when compute_fd=True
    # ------------------------------------------------------------------ #
    fd_norm = 0.0
    if compute_fd:
        try:
            eps_fd = 1e-4
            optimizer.zero_grad()
            _fd_loss0 = _ce(model(Xc), Yc)
            _fd_loss0.backward()
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
                _fd_loss1 = _ce(model(Xc), Yc)
                _fd_loss1.backward()
                g_perturbed = torch.cat(
                    [p.grad.reshape(-1).clone() for p in model.parameters()]
                )
                offset = 0
                for p in params_vals:
                    numel = p.numel()
                    p.data.add_(-eps_fd * v_fd[offset : offset + numel].view_as(p))
                    offset += numel

                flat_fdv = (g_perturbed - g_base) / eps_fd
                v_fd = flat_fdv / (flat_fdv.norm() + 1e-9)
                del g_perturbed
                torch.cuda.empty_cache()

            fd_norm = torch.dot(v_fd, flat_fdv).item() if flat_fdv is not None else 0.0
            optimizer.zero_grad()
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
# Attention entropy
# ======================================================================


def get_attention_entropy(model: torch.nn.Module) -> list[float]:
    """
    Compute the mean Shannon entropy of the attention distribution for
    each transformer layer, using the cached ``last_att`` tensors written
    by the patched attention forward.

    Entropy is computed in nats:
        H = −Σ p·ln(p)  over the key (sequence) dimension.

    Args:
        model: A HookedGPT model whose attention blocks have already
               executed a forward pass with ``_cache_attn=True``.

    Returns:
        List of length ``n_layer``, each the mean entropy (float) averaged
        over batch, heads, and sequence positions.
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
