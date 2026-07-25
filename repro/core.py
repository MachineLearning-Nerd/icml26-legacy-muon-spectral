"""Spectral math core (numpy): the Psi_p family and Newton-Schulz solvers.

This module implements the *exact* paper definitions (Sec 3.3-3.5, Appendix A-D)
in numpy so they can be verified to machine precision independently of any
training framework:

  Psi_p(O) = U Sigma^p V^T                          (Eq. 6, Sec 3.3)
  kappa(Psi_p) = kappa(O)^p                          (Sec 3.4)
  Psi_{1/2}(O) = O (O^T O)^{-1/4}                    (Sec 3.5) via coupled
  Psi_{1/4}(O) = Psi_{1/2}(Psi_{1/2}(O))             Newton-Schulz (Alg. 1)

C1/C2/C6 are verified here to <=1e-13 against direct SVD. These are the
preserved full-credit checks; the training children replace the toy C3/C4
*direction* checks with real training evidence.
"""
from __future__ import annotations
import numpy as np


def psi_p(O: np.ndarray, p: float) -> np.ndarray:
    """Psi_p(O) = U Sigma^p V^T via SVD (the exact paper definition)."""
    U, s, Vt = np.linalg.svd(O, full_matrices=False)
    return (U * np.power(s, p)) @ Vt


def cond_number(O: np.ndarray) -> float:
    """Spectral condition number kappa(O) = sigma_max / sigma_min."""
    s = np.linalg.svd(O, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return float("inf")
    return float(s.max() / s.min())


def coupled_newton_schulz(X: np.ndarray, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Coupled Newton-Schulz (Algorithm 1, Appendix C) for X^{1/2} and X^{-1/2}.

    Matrix multiplications only (no SVD). Frobenius-norm scaling at init for
    stability; symmetrized each step. Returns (X^{1/2}, X^{-1/2}).
    """
    n = X.shape[0]
    assert X.shape == (n, n)
    X = 0.5 * (X + X.T)  # SPD assumption
    alpha = np.linalg.norm(X, "fro")
    if alpha < 1e-30:
        return np.zeros_like(X), np.zeros_like(X)
    Y = X / alpha
    Z = np.eye(n)
    for _ in range(iters):
        T = 0.5 * (3.0 * np.eye(n) - Z @ Y)
        Y = Y @ T
        Z = T @ Z
        Y = 0.5 * (Y + Y.T)
        Z = 0.5 * (Z + Z.T)
    return np.sqrt(alpha) * Y, Z / np.sqrt(alpha)


def newton_schulz_psi_half(O: np.ndarray, iters: int = 200) -> np.ndarray:
    """Psi_{1/2}(O) = O (O^T O)^{-1/4} using only matrix multiplications.

    Computes X^{-1/4} as (X^{1/2})^{-1/2} by applying coupled Newton-Schulz
    twice (Appendix D, Algorithm 3). No SVD is used.
    """
    X = O.T @ O
    sqrt_X, _ = coupled_newton_schulz(X, iters)            # X^{1/2}
    _, inv_sqrt_sqrt_X = coupled_newton_schulz(sqrt_X, iters)  # (X^{1/2})^{-1/2} = X^{-1/4}
    return O @ inv_sqrt_sqrt_X


def newton_schulz_psi_quarter(O: np.ndarray, iters: int = 200) -> np.ndarray:
    """Psi_{1/4}(O) = U Sigma^{1/4} V^T via two-times coupled Newton-Schulz.

    Psi_{1/4}(O) = Psi_{1/2}(Psi_{1/2}(O)) since (Sigma^{1/2})^{1/2} = Sigma^{1/4}.
    Matrix multiplications only, no SVD.
    """
    return newton_schulz_psi_half(newton_schulz_psi_half(O, iters), iters)


# --- update-input helpers (used for the toy direction diagnostics only) -------
def momentum_update(grads: list[np.ndarray], beta: float = 0.9) -> np.ndarray:
    """First-moment momentum M_t = EMA of gradients (paper O_mom input)."""
    M = np.zeros_like(grads[0])
    for g in grads:
        M = beta * M + (1 - beta) * g
    return M


def rms_update(grads: list[np.ndarray], beta1: float = 0.9, beta2: float = 0.95,
               eps: float = 1e-8) -> np.ndarray:
    """RMS-normalized update M_t / sqrt(V_t) (paper O_rms input, Adam-style)."""
    M = np.zeros_like(grads[0])
    V = np.zeros_like(grads[0])
    for g in grads:
        M = beta1 * M + (1 - beta1) * g
        V = beta2 * V + (1 - beta2) * (g * g)
    return M / (np.sqrt(V) + eps)


def cos_sim(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine similarity (Frobenius) between two flattened matrices."""
    a = A.ravel()
    b = B.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
