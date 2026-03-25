"""
Power iteration for estimating the largest eigenvalue of the loss Hessian.

All second-order quantities are estimated without materialising the full
Hessian matrix; instead we use power iteration on Hessian-vector
products (HVPs) computed cheaply via PyTorch's ``autograd.grad``.

Functions
---------
power_iteration   — λ_max of the full Hessian (or a subspace / preconditioned
                    variant) via HVP iteration.
"""

from __future__ import annotations

import torch


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
        hvp_fn:   Optional explicit Hessian-vector product callable
                  ``hvp_fn(v) -> Hv``. When provided, ``loss`` is not
                  used and may be ``None``.

    Returns:
        Estimated λ_max as a Python float.
    """
    params = tuple(model.parameters())

    if hvp_fn is None:
        if loss is None:
            raise ValueError("loss must be provided when hvp_fn is None")
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True
        )
        flat_grads = torch.cat([g.reshape(-1) for g in grads])

        def _hvp(v: torch.Tensor) -> torch.Tensor:
            hvp_tuple = torch.autograd.grad(
                flat_grads, params, grad_outputs=v, retain_graph=True
            )
            return torch.cat([g.contiguous().reshape(-1) for g in hvp_tuple])

        hvp_callable = _hvp
    else:
        hvp_callable = hvp_fn

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
            v = flat_hvp * vv_mask.to(flat_hvp.device)
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
