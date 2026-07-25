"""Faithful torch port of the official BeyondMuon spectral transforms.

Source: marcotchen/BeyondMuon optimizers/spectral_ns.py (ICML 2026 Spotlight).
The only change is that ``torch.compile`` is made optional (disabled on CPU /
when unavailable) so the exact same math runs on CPU without a CUDA graph.
"""
from __future__ import annotations
import functools
import torch

_COMPILE = torch.cuda.is_available() and hasattr(torch, "compile")


def _maybe_compile(fn):
    return torch.compile(fn) if _COMPILE else fn


@_maybe_compile
def zero_power_via_quintic_ns(matrix: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """Approximate Psi_0(O)=UV^T with the Muon Newton-Schulz quintic (Algorithm 2)."""
    if matrix.ndim != 2:
        raise ValueError("zero_power_via_quintic_ns expects a 2D tensor")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix / (matrix.norm() + eps)
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + b * gram @ x + c * gram @ gram @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def muon_update(matrix: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """Apply the zero-power/polar transform to a matrix update (paper Muon step)."""
    update = zero_power_via_quintic_ns(matrix, steps=ns_steps)
    update = update * max(1.0, matrix.size(-2) / matrix.size(-1)) ** 0.5
    return update


@_maybe_compile
@torch.no_grad()
def coupled_newton_schulz_sqrt_invsqrt(matrix: torch.Tensor, num_iters: int = 15,
                                        eps: float = 1e-6):
    """Compute X^{1/2} and X^{-1/2} with coupled Newton-Schulz (Algorithm 1)."""
    device, dtype = matrix.device, matrix.dtype
    size = matrix.shape[-1]
    eye = torch.eye(size, device=device, dtype=dtype)
    trace = torch.trace(matrix)
    regularized = matrix + (eps + 1e-4 * trace / size) * eye
    scale = torch.norm(regularized, p=2)
    y = regularized / scale
    z = eye.clone()
    sym_freq = max(1, min(num_iters // 3, 5))
    for idx in range(num_iters):
        transform = 0.5 * (3.0 * eye - z @ y)
        y = y @ transform
        z = transform @ z
        if idx > 0 and idx % sym_freq == 0:
            y = 0.5 * (y + y.mT)
            z = 0.5 * (z + z.mT)
    y = 0.5 * (y + y.mT)
    z = 0.5 * (z + z.mT)
    sqrt_scale = torch.sqrt(scale)
    return sqrt_scale * y, z / sqrt_scale


@_maybe_compile
def inverse_fourth_root_via_ns(matrix: torch.Tensor, ns_iters: int = 15, eps: float = 1e-6):
    """Compute X^{-1/4} by applying coupled NS twice (Appendix D, Algorithm 3)."""
    _, inv_sqrt = coupled_newton_schulz_sqrt_invsqrt(matrix, num_iters=ns_iters, eps=eps)
    inv_fourth, _ = coupled_newton_schulz_sqrt_invsqrt(inv_sqrt, num_iters=ns_iters, eps=eps)
    return inv_fourth


@_maybe_compile
def spectral_half_power_via_ns(update: torch.Tensor, ns_iters: int = 15, eps: float = 1e-6,
                                prefer: str = "auto"):
    """Compute Psi_{1/2}(O)=U Sigma^{1/2} V^T without an explicit SVD."""
    if update.ndim != 2:
        raise ValueError("spectral_half_power_via_ns expects a 2D update matrix")
    rows, cols = update.shape
    use_right = (cols <= rows) if prefer == "auto" else (prefer == "right")
    if use_right:
        gram = update.mT @ update
        inv_fourth = inverse_fourth_root_via_ns(gram, ns_iters, eps)
        return update @ inv_fourth
    gram = update @ update.mT
    inv_fourth = inverse_fourth_root_via_ns(gram, ns_iters, eps)
    return inv_fourth @ update


def apply_spectral_transform(update: torch.Tensor, spectral_exponent: float, *,
                              ns_iters: int, split_qkv_updates: bool) -> torch.Tensor:
    """Apply Psi_p to a matrix for p in {1, 1/2, 1/4, 0} (official dispatch)."""
    if update.ndim != 2:
        raise ValueError("apply_spectral_transform expects a 2D update matrix")
    if split_qkv_updates and update.size(0) == 3 * update.size(1):
        return torch.cat([
            apply_spectral_transform(part, spectral_exponent,
                                     ns_iters=ns_iters, split_qkv_updates=False)
            for part in update.split(update.size(1))
        ])
    if spectral_exponent == 1.0:
        return update
    if spectral_exponent == 0.0:
        return muon_update(update, ns_steps=ns_iters)
    if spectral_exponent not in {0.5, 0.25}:
        raise ValueError(f"Unsupported spectral exponent p={spectral_exponent}")
    half_steps = 1 if spectral_exponent == 0.5 else 2
    for _ in range(half_steps):
        update = spectral_half_power_via_ns(update, ns_iters)
    return update
