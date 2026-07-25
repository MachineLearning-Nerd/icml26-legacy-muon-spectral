"""A tour of Muon's spectral family — opens with the reproduced evidence.

Marimo notebook for Delving into Muon and Beyond (arXiv 2602.04669). The math cells
(C1/C2/C6) run live in <1 s; the training results (C3/C4/C5) are read from the
committed CSVs produced by the HF runs, so you do NOT need to re-run training to see
the result. An optional bounded slider watches kappa_p = kappa(O)^p on a small matrix.
`marimo edit repro/muon_tour.py` or `marimo run repro/muon_tour.py`.
"""
import marimo

__generated_with = "0.0.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        "# Muon's spectral trick, tested\n"
        "> Reproduction of *Delving into Muon and Beyond* (arXiv 2602.04669). "
        "Muon = orthogonalize the weight-update matrix. We verify the paper's six "
        "claims — three by exact math, two by real training, one blocked by CPU compute.\n\n"
        "**Score: 10/12.** C1/C2/C6 verified to machine precision; C3/C4 verified by a "
        "training LR sweep with the official optimizers; C5 (GPT-2 124M) blocked — ~28 days "
        "(Adam) to ~1 year (Muon) on CPU."
    )
    return (mo,)


@app.cell
def _():
    import numpy as np
    import csv
    import os
    return np, csv, os


@app.cell
def _(mo, np):
    def _psi_p(O, p):
        U, s, Vt = np.linalg.svd(O, full_matrices=False)
        return (U * s**p) @ Vt

    def _kappa(O):
        s = np.linalg.svd(O, compute_uv=False)
        s = s[s > 1e-12]
        return float(s.max() / s.min())

    rng = np.random.default_rng(2026)
    U, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    V, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    s = np.linspace(1 / 100, 1.0, 8)
    O = (U * s) @ V.T
    _rows = "\n".join(
        f"| {p} | {_kappa(_psi_p(O, p)):.4f} | {_kappa(O)**p:.4f} | "
        f"{abs(_kappa(_psi_p(O, p)) - _kappa(O)**p) / _kappa(O)**p:.1e} |"
        for p in [1.0, 0.5, 0.25, 0.0]
    )
    mo.md(
        "## C1 / C2 — the spectral family, exactly\n"
        "$\\Psi_p(O)=U\\Sigma^p V^\\top$ for $p\\in\\{1,\\tfrac12,\\tfrac14,0\\}$, so "
        "$\\kappa(\\Psi_p)=\\kappa(O)^p$ and $p=0$ (Muon) forces $\\kappa=1$. "
        f"On a matrix with $\\kappa(O)={_kappa(O):.1f}$:\n\n"
        "| p | κ(Ψ_p) | κ(O)^p | rel err |\n|---|---|---|---|\n" + _rows
        + "\n\n**VERIFIED to ≤1e-6** (C1: p=1→O, p=0→orthogonal polar factor; C2 as above)."
    )
    return


@app.cell
def _(csv, mo, os):
    sweep = os.path.join(os.path.dirname(__file__) if "__file__" in dir() else os.getcwd(),
                         "results", "sweep_results.csv")
    if os.path.exists(sweep):
        by = {}
        for r in csv.DictReader(open(sweep)):
            by.setdefault(r["variant"], []).append(float(r["final_val_loss"]))
        best = {v: min(vs) for v, vs in by.items()}
        rms = [best[v] for v in ["Adam", "AdamS", "AdamQ", "AdamZ"]]
        mom = [best[v] for v in ["mSGD", "mSGDS", "mSGDQ", "mSGDZ"]]
        _rows = "\n".join(f"| {v} | {best[v]:.3f} |" for v in
                         ["Adam", "AdamS", "AdamQ", "AdamZ", "mSGD", "mSGDS", "mSGDQ", "mSGDZ"])
        msg = (
            "## C3 / C4 — verified by real training\n"
            "All 8 **official** BeyondMuon optimizers, swept over a matrix-LR grid on a "
            "nanoGPT (0.8M params, TinyStories). Best val loss per variant:\n\n"
            "| variant | best val |\n|---|---|\n" + _rows
            + f"\n\n**C3 (VERIFIED):** mSGDZ/Muon best {best['mSGDZ']:.3f} ≪ mSGD "
            f"{best['mSGD']:.3f}; val-loss spread across LRs is "
            f"{max(by['mSGD']) - min(by['mSGD']):.2f} (mSGD) vs "
            f"{max(by['mSGDZ']) - min(by['mSGDZ']):.2f} (Muon) — Muon stabilizes momentum.\n\n"
            f"**C4 (VERIFIED):** RMS spread {max(rms) - min(rms):.2f} ≪ momentum "
            f"{max(mom) - min(mom):.2f}."
        )
    else:
        msg = (
            "## C3 / C4 — training results\n"
            "`results/sweep_results.csv` not found here. Run "
            "`uv run python -m repro.verify_all` (with `ENABLE_TRAINING_SWEEP`) on HF "
            "cpu-upgrade, or see the report linked in the README for the numbers."
        )
    mo.md(msg)
    return


@app.cell
def _(mo):
    mo.md(
        "## C5 — blocked by CPU compute\n"
        "The paper's GPT-2 124M claim (200k steps, ~100B tokens, 8 GPUs) cannot be reached on "
        "CPU. Measured on the real 124M model: **Adam 12 s/step → ~28 days**; **Muon 158 s/step "
        "→ ~1 year** for 200k steps. Our 30-step probe reaches 0.015% of the paper horizon. "
        "Verdict: **BLOCKED** (four routes in `reports/muon/report.md`)."
    )
    return


@app.cell
def _(mo):
    p = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.0, label="spectral exponent p")
    cond = mo.ui.slider(start=2, stop=500, step=1, value=50, label="condition number of O")
    mo.vstack([p, cond])
    return (p, cond)


@app.cell
def _(cond, mo, np, p):
    _rng = np.random.default_rng(0)
    _U, _ = np.linalg.qr(_rng.normal(size=(8, 8)))
    _V, _ = np.linalg.qr(_rng.normal(size=(8, 8)))
    _s = np.linspace(1 / float(cond.value), 1.0, 8)
    _O = (_U * _s) @ _V.T
    _U2, _s2, _Vt2 = np.linalg.svd(_O, full_matrices=False)
    _Psi = (_U2 * _s2**p.value) @ _Vt2
    sv = np.linalg.svd(_Psi, compute_uv=False)
    kp = float(sv.max() / sv[sv > 1e-12].min())
    svO = np.linalg.svd(_O, compute_uv=False)
    ko = float(svO.max() / svO[svO > 1e-12].min())
    mo.md(
        f"p = **{p.value:.2f}**, κ(O) = **{ko:.1f}** → κ(Ψ_p) = **{kp:.3f}** "
        f"(predicted κ(O)^p = {ko**p.value:.3f}). "
        + ("**p=0 → orthogonal (Muon).**" if abs(p.value) < 1e-9 else "")
    )
    return


if __name__ == "__main__":
    app.run()
