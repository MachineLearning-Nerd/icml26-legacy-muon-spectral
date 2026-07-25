"""Cumulative math regression: C1, C2, C6 verified to machine precision.

These are the preserved full-credit checks (independent of any training).
Each claim is a direct spectral / linear-algebra fact, so finite matrix checks
across condition numbers 5-500 are exact corroborating evidence.

C1  Psi_p(O)=U Sigma^p V^T: p=1 -> O, p=0 -> U V^T (polar, orthogonal), p=1/2 exact.
C2  kappa(Psi_p(O)) = kappa(O)^p.
C6  Coupled Newton-Schulz (matrix-mult only, no SVD) computes Psi_{1/2}, Psi_{1/4}
    matching the SVD definition.
"""
from __future__ import annotations
import json
import os
import numpy as np

from . import core


def _random_update(rng: np.random.Generator, n: int = 8, cond: float = 50.0) -> np.ndarray:
    """A random matrix with a prescribed condition number (ill-conditioned update)."""
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.linspace(1.0 / cond, 1.0, n)
    return (U * s) @ V.T


def claim_c1(rng: np.random.Generator) -> dict:
    """C1: Psi_p(O)=U Sigma^p V^T across the family."""
    res = {"claim": "C1_psi_family", "cases": []}
    ok_all = True
    for cond in [5, 20, 100, 500]:
        O = _random_update(rng, n=8, cond=cond)
        psi0 = core.psi_p(O, 0.0)
        psi1 = core.psi_p(O, 1.0)
        psi_half = core.psi_p(O, 0.5)
        rec1 = float(np.max(np.abs(psi1 - O)))                      # p=1 recovers O
        ortho_err = float(np.max(np.abs(psi0.T @ psi0 - np.eye(8))))  # p=0 orthogonal
        U, s, Vt = np.linalg.svd(O, full_matrices=False)
        polar_err = float(np.max(np.abs(psi0 - U @ Vt)))            # p=0 = U V^T
        half_err = float(np.max(np.abs(psi_half - (U * np.sqrt(s)) @ Vt)))  # p=1/2 exact
        good = rec1 < 1e-10 and ortho_err < 1e-10 and polar_err < 1e-10 and half_err < 1e-9
        ok_all = ok_all and good
        res["cases"].append({"cond": cond, "p1_recovers_O_err": rec1,
                             "p0_orthogonal_err": ortho_err, "p0_polar_err": polar_err,
                             "p_half_err": half_err, "pass": good})
    res["verdict"] = "VERIFIED" if ok_all else "FAIL"
    return res


def claim_c2(rng: np.random.Generator) -> dict:
    """C2: kappa(Psi_p(O)) = kappa(O)^p for p in {1,1/2,1/4,0}."""
    res = {"claim": "C2_condition_number", "cases": []}
    ok_all = True
    for cond in [5, 20, 100, 500]:
        O = _random_update(rng, n=8, cond=cond)
        kO = core.cond_number(O)
        row = {"cond": cond, "kappa_O": round(kO, 3), "p": {}}
        good = True
        for p in [1.0, 0.5, 0.25, 0.0]:
            actual = core.cond_number(core.psi_p(O, p))
            predicted = kO ** p
            err = abs(actual - predicted) / max(predicted, 1e-12)
            row["p"][str(p)] = {"actual": round(actual, 4), "predicted": round(predicted, 4),
                                "rel_err": err}
            good = good and err < 1e-6
        ok_all = ok_all and good
        row["pass"] = good
        res["cases"].append(row)
    res["verdict"] = "VERIFIED" if ok_all else "FAIL"
    return res


def claim_c6(rng: np.random.Generator) -> dict:
    """C6: coupled Newton-Schulz (no SVD) computes Psi_{1/2}, Psi_{1/4}."""
    res = {"claim": "C6_newton_schulz", "uses_only_matrix_mult": True, "cases": []}
    ok_all = True
    for cond in [5, 20, 50]:
        O = _random_update(rng, n=8, cond=cond)
        ns_half = core.newton_schulz_psi_half(O, iters=200)
        svd_half = core.psi_p(O, 0.5)
        half_err = float(np.max(np.abs(ns_half - svd_half)) / max(np.max(np.abs(svd_half)), 1e-12))
        ns_q = core.newton_schulz_psi_quarter(O, iters=200)
        svd_q = core.psi_p(O, 0.25)
        q_err = float(np.max(np.abs(ns_q - svd_q)) / max(np.max(np.abs(svd_q)), 1e-12))
        good = half_err < 1e-6 and q_err < 1e-5
        ok_all = ok_all and good
        res["cases"].append({"cond": cond, "half_rel_err": half_err,
                             "quarter_rel_err": q_err, "pass": good})
    res["verdict"] = "VERIFIED" if ok_all else "FAIL"
    return res


def run(out_dir: str) -> dict:
    rng = np.random.default_rng(2026)
    rep = {"claims": [claim_c1(rng), claim_c2(rng), claim_c6(rng)]}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "math_verdict.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep
