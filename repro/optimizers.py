"""Faithful torch port of the official BeyondMuon optimizers + factory.

Source: marcotchen/BeyondMuon optimizers/{adamw_ns,sgdw_ns,sgdw}.py and
optimizer_factory.py. Implements the 8 paper variants over two input families:

  mSGD family (input = first-moment momentum M_t):  mSGD, mSGDS, mSGDQ, mSGDZ
  Adam family (input = RMS-normalized M_t/sqrt(V_t)): Adam, AdamS, AdamQ, AdamZ

mSGDZ == Muon (Psi_0 on momentum). Vector/non-matrix params always use AdamW
at the shared vector lr, exactly as in the paper's controlled setup (Sec 4.1).
"""
from __future__ import annotations
from typing import Iterable, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .spectral_torch import apply_spectral_transform


# --- decoupled-weight-decay momentum SGD (timm SGDW, simplified single-tensor) -
class SGDW(Optimizer):
    """Decoupled-weight-decay momentum SGD (p=1 mSGD matrix baseline)."""

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0,
                 nesterov=False, maximize=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = -p.grad if group["maximize"] else p.grad
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                update = self._momentum(p, grad, group)
                p.add_(update, alpha=-group["lr"])
        return loss

    def _momentum(self, p, grad, group):
        m = group["momentum"]
        if m == 0:
            return grad
        state = self.state[p]
        buf = state.get("momentum_buffer")
        if buf is None:
            buf = torch.clone(grad).detach()
            state["momentum_buffer"] = buf
        else:
            buf.mul_(m).add_(grad, alpha=1.0 - group["dampening"])
        if group["nesterov"]:
            return grad.add(buf, alpha=m)
        return buf


class SGDW_NS(SGDW):
    """SGDW + Psi_p(M_t) on matrix params (mSGDS/mSGDQ/mSGDZ)."""

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0,
                 nesterov=False, maximize=False, ns_iters=15, split_qkv_updates=False,
                 spectral_exponent=0.5):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov, maximize)
        self.ns_iters = ns_iters
        self.split_qkv_updates = split_qkv_updates
        self.spectral_exponent = spectral_exponent

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = -p.grad if group["maximize"] else p.grad
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                update = self._momentum(p, grad, group)
                if p.ndim > 1:
                    update = apply_spectral_transform(
                        update, self.spectral_exponent,
                        ns_iters=self.ns_iters, split_qkv_updates=self.split_qkv_updates)
                p.add_(update, alpha=-group["lr"])
        return loss


class AdamW_NS(Optimizer):
    """AdamW + Psi_p(M_t/sqrt(V_t)) on matrix params (Adam/AdamS/AdamQ/AdamZ)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 ns_iters=15, split_qkv_updates=False, spectral_exponent=0.5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.ns_iters = ns_iters
        self.split_qkv_updates = split_qkv_updates
        self.spectral_exponent = spectral_exponent

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                bc1 = 1.0 - b1 ** step
                bc2 = 1.0 - b2 ** step
                rms = (v / bc2).sqrt().add_(eps)
                update = (m / bc1) / rms
                if p.ndim > 1:
                    update = apply_spectral_transform(
                        update, self.spectral_exponent,
                        ns_iters=self.ns_iters, split_qkv_updates=self.split_qkv_updates)
                p.add_(update, alpha=-lr)
        return loss


VARIANTS = {
    "Adam":  ("adam", 1.0),  "AdamS": ("adam", 0.5), "AdamQ": ("adam", 0.25), "AdamZ": ("adam", 0.0),
    "mSGD":  ("msgd", 1.0),  "mSGDS": ("msgd", 0.5), "mSGDQ": ("msgd", 0.25), "mSGDZ": ("msgd", 0.0),
}


def build_optimizers(model, variant: str, lr_matrix: float, lr_vector: float,
                     betas, weight_decay, sgd_momentum, ns_iters,
                     split_qkv_updates=False) -> list[Optimizer]:
    """Build the matrix + vector optimizers for one of the 8 variants."""
    variant = variant.replace("_", "").replace("-", "")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown optimizer variant: {variant}; choose {list(VARIANTS)}")
    family, exponent = VARIANTS[variant]

    matrix_params = [p for p in model.parameters() if p.dim() >= 2]
    vector_params = [p for p in model.parameters() if p.dim() < 2]
    # vector params always use AdamW at the shared vector lr (paper controlled setup)
    vec_opt = torch.optim.AdamW(vector_params, lr=lr_vector, betas=betas, weight_decay=0.0)

    if family == "adam":
        if variant == "Adam":
            mat_opt = torch.optim.AdamW(matrix_params, lr=lr_matrix, betas=betas,
                                        weight_decay=weight_decay)
        else:
            mat_opt = AdamW_NS(matrix_params, lr=lr_matrix, betas=betas,
                               weight_decay=weight_decay, ns_iters=ns_iters,
                               split_qkv_updates=split_qkv_updates, spectral_exponent=exponent)
        return [vec_opt, mat_opt]

    if variant == "mSGD":
        mat_opt = SGDW(matrix_params, lr=lr_matrix, momentum=sgd_momentum, weight_decay=weight_decay)
    else:
        mat_opt = SGDW_NS(matrix_params, lr=lr_matrix, momentum=sgd_momentum,
                          weight_decay=weight_decay, ns_iters=ns_iters,
                          split_qkv_updates=split_qkv_updates, spectral_exponent=exponent)
    return [vec_opt, mat_opt]


def set_lrs(opts: list[Optimizer], matrix_lr: float, vector_lr: float):
    """Apply the cosine schedule to matrix (group 0) and vector (group 0) opts."""
    opts[1].param_groups[0]["lr"] = matrix_lr   # matrix optimizer
    opts[0].param_groups[0]["lr"] = vector_lr   # vector optimizer
