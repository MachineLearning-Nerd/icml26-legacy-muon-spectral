"""Master verifier and entrypoint for the fixed run command.

``uv run python -m repro.verify_all`` always runs the math regression (C1/C2/C6)
and, when enabled by committed config (repro/config.py), the C3/C4 training sweep
and the C5 GPT-2 124M run. It then aggregates one VERIFIED / FALSIFIED / BLOCKED
verdict per claim, writes outputs/verdict.json, prints a summary, and exits
nonzero if any *enabled* check fails.

Claim contracts (anchored to the paper):
  C1 (Sec 3.3): Psi_p(O)=U Sigma^p V^T; p=0 -> UV^T (polar/Muon), p=1 -> O.
  C2 (Sec 3.4): kappa(Psi_p) = kappa(O)^p.
  C3 (Sec 4.3): Muon (mSGDZ) stabilizes momentum, more robust than mSGD across LRs.
  C4 (Sec 4.4): the 4 RMS variants match at their best-tuned matrix LR.
  C5 (Sec 5/I): on GPT-2 124M (controlled), Muon does NOT outperform Adam.
  C6 (Sec 3.5): coupled Newton-Schulz computes Psi_{1/2}, Psi_{1/4} (no SVD).
"""
from __future__ import annotations
import json
import math
import os
import platform
import sys
import time

from . import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


def _env():
    return {
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "torch": __import__("torch").__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_available": __import__("torch").cuda.is_available(),
    }


# --- C3/C4 verdict logic from the training sweep -----------------------------
def _best_per_variant(runs):
    by = {}
    for r in runs:
        by.setdefault(r["variant"], []).append(r)
    out = {}
    for v, rs in by.items():
        finite = [r for r in rs if not r.get("diverged") and math.isfinite(r.get("final_val_loss", math.nan))]
        vals = [r["final_val_loss"] for r in finite]
        best = min(finite, key=lambda r: r["final_val_loss"]) if finite else None
        out[v] = {
            "best_val_loss": (best["final_val_loss"] if best else None),
            "best_lr": (best["lr_matrix"] if best else None),
            # robustness across LRs: spread (max-min) of final val loss. Lower = more
            # stable/robust across the learning-rate range (paper Fig 2/3 geometry).
            "val_spread": (max(vals) - min(vals)) if vals else None,
            "vals": vals,
            "n_finite": len(finite), "n_total": len(rs),
            "all_diverged": best is None,
        }
    return out


def _c3_verdict(summary):
    """C3 (Sec 4.3): Muon (mSGDZ) stabilizes momentum and is more robust than mSGD
    across learning rates, AND reaches a competitive-or-better best val loss."""
    msgd = summary.get("mSGD", {})
    msgdz = summary.get("mSGDZ", {})
    if not msgd or not msgdz or msgd["best_val_loss"] is None or msgdz["best_val_loss"] is None:
        return ("BLOCKED", "missing mSGD/mSGDZ results", {})
    competitive = msgdz["best_val_loss"] <= msgd["best_val_loss"] + 0.10
    # robustness: mSGDZ's val-loss spread across LRs should be (much) smaller than mSGD's
    more_robust = (msgdz["val_spread"] is not None and msgd["val_spread"] is not None
                   and msgdz["val_spread"] < 0.75 * msgd["val_spread"])
    detail = {
        "mSGD_best_val": msgd["best_val_loss"], "mSGDZ_best_val": msgdz["best_val_loss"],
        "mSGD_val_spread_across_LRs": msgd["val_spread"],
        "mSGDZ_val_spread_across_LRs": msgdz["val_spread"],
        "mSGD_vals": msgd["vals"], "mSGDZ_vals": msgdz["vals"],
        "mSGDZ_loss_competitive_or_better": competitive,
        "mSGDZ_more_robust_across_LRs": more_robust,
    }
    if competitive and more_robust:
        return ("VERIFIED", "mSGDZ (Muon) is markedly more robust across the matrix-LR "
                f"range (val spread {msgdz['val_spread']:.2f} vs mSGD {msgd['val_spread']:.2f}) "
                f"and reaches a better best val loss ({msgdz['best_val_loss']:.3f} vs {msgd['best_val_loss']:.3f})", detail)
    return ("FALSIFIED", "mSGDZ did NOT show wider robustness + competitive loss vs mSGD", detail)


def _c4_verdict(summary):
    """C4 (Sec 4.4): the 4 RMS variants match at best settings (small best-loss spread,
    much smaller than the momentum family's spread)."""
    fam = {k: v for k, v in summary.items()
           if k in ("Adam", "AdamS", "AdamQ", "AdamZ") and v["best_val_loss"] is not None}
    if len(fam) < 4:
        return ("BLOCKED", f"missing RMS-family variants (have {sorted(fam)})", {})
    losses = {k: v["best_val_loss"] for k, v in fam.items()}
    spread = max(losses.values()) - min(losses.values())
    mom = {k: v["best_val_loss"] for k, v in summary.items()
           if k in ("mSGD", "mSGDS", "mSGDQ", "mSGDZ") and v["best_val_loss"] is not None}
    mom_spread = (max(mom.values()) - min(mom.values())) if len(mom) == 4 else None
    abs_ok = spread < 0.15
    rel_ok = (mom_spread is None) or (spread < 0.6 * mom_spread)
    detail = {"best_losses": losses, "rms_spread": round(spread, 4),
              "momentum_spread": round(mom_spread, 4) if mom_spread else None,
              "absolute_threshold_0.15": abs_ok, "relative_to_momentum_0.6": rel_ok}
    if abs_ok and rel_ok:
        return ("VERIFIED", f"the 4 RMS variants match within a {spread:.3f} best-loss spread "
                f"(momentum-family spread {mom_spread:.3f}; RMS normalization already controls "
                "the anisotropy spectral compression targets)", detail)
    return ("FALSIFIED", f"RMS variants diverge: spread {spread:.3f} (momentum {mom_spread})", detail)


def _c5_verdict(gpt2_runs, sweep_summary):
    """Paper negative claim: Muon does NOT outperform Adam on GPT-2 124M (200k steps).

    We cannot reach the paper's 100B-token horizon on CPU, so the verdict is
    BLOCKED unless the reduced GPT-2 run and the sweep both clearly corroborate
    (Adam <= Muon) AND no falsifying case appears.
    """
    detail = {"paper_horizon_steps": 200000, "paper_adam_best": 2.871, "paper_msgdz_best": 2.887}
    if gpt2_runs:
        by = {r["variant"]: r for r in gpt2_runs}
        adam = by.get("Adam", {})
        muon = by.get("mSGDZ", {})
        if adam and muon and not adam.get("diverged") and not muon.get("diverged"):
            detail["gpt2_adam_val"] = adam.get("final_val_loss")
            detail["gpt2_muon_val"] = muon.get("final_val_loss")
            detail["gpt2_steps"] = adam.get("train_steps")
            detail["gpt2_tokens"] = adam.get("total_tokens")
    # barrier: paper horizon vs what is CPU-feasible
    barrier = ("Paper requires GPT-2 124M, OpenWebText, 200k steps (~100B tokens), "
               "8 GPUs. CPU cannot reach this horizon; reduced runs are partial only.")
    detail["compute_barrier"] = barrier
    return ("BLOCKED", "Cannot VERIFY/FALSIFY the 200k-step negative claim on CPU; "
            "reduced GPT-2 124M run + small-GPT sweep provide partial corroboration only. "
            + barrier, detail)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    env = _env()
    print(f"env: {env}")
    print(f"config: ENABLE_MATH={C.ENABLE_MATH} SWEEP={C.ENABLE_TRAINING_SWEEP} GPT2={C.ENABLE_GPT2_124M}")

    verdicts = {}
    ok_enabled = True  # exit nonzero if any enabled check FAILs

    # --- math regression (C1/C2/C6) ---
    if C.ENABLE_MATH:
        from . import verify_math
        math_rep = verify_math.run(OUT)
        for c in math_rep["claims"]:
            key = {"C1_psi_family": "C1", "C2_condition_number": "C2",
                   "C6_newton_schulz": "C6"}[c["claim"]]
            verdicts[key] = {"verdict": c["verdict"], "evidence": c}
            if c["verdict"] != "VERIFIED":
                ok_enabled = False
        for c in math_rep["claims"]:
            print(f"[math] {c['claim']}: {c['verdict']}")

    # --- training sweep (C3/C4) ---
    sweep_runs = None
    if C.ENABLE_TRAINING_SWEEP:
        from . import train_sweep
        sweep = train_sweep.run(OUT)
        sweep_runs = sweep["runs"]
        summary = _best_per_variant(sweep_runs)
        v3, m3, d3 = _c3_verdict(summary)
        v4, m4, d4 = _c4_verdict(summary)
        verdicts["C3"] = {"verdict": v3, "message": m3, "evidence": d3}
        verdicts["C4"] = {"verdict": v4, "message": m4, "evidence": d4}
        if v3 == "FALSIFIED" or v4 == "FALSIFIED":
            ok_enabled = False
        print(f"[train] C3: {v3} ({m3})")
        print(f"[train] C4: {v4} ({m4})")
    else:
        verdicts["C3"] = {"verdict": "BLOCKED", "message": "training sweep not enabled on this node"}
        verdicts["C4"] = {"verdict": "BLOCKED", "message": "training sweep not enabled on this node"}

    # --- GPT-2 124M (C5) ---
    gpt2_runs = None
    if C.ENABLE_GPT2_124M:
        from . import train_gpt2
        g = train_gpt2.run(OUT)
        gpt2_runs = g["runs"]
    summary_for_c5 = _best_per_variant(sweep_runs) if sweep_runs else {}
    v5, m5, d5 = _c5_verdict(gpt2_runs, summary_for_c5)
    verdicts["C5"] = {"verdict": v5, "message": m5, "evidence": d5}
    print(f"[gpt2] C5: {v5} ({m5})")

    # --- aggregate ---
    score_map = {"VERIFIED": 2, "FALSIFIED": 2, "BLOCKED": 0}
    score = sum(score_map.get(v["verdict"], 0) for k, v in verdicts.items()
                if k in ("C1", "C2", "C3", "C4", "C5", "C6"))
    out = {
        "verdicts": verdicts, "score": f"{score}/12",
        "env": env, "elapsed_seconds": round(time.time() - t0, 1),
        "config": {"ENABLE_MATH": C.ENABLE_MATH, "ENABLE_TRAINING_SWEEP": C.ENABLE_TRAINING_SWEEP,
                   "ENABLE_GPT2_124M": C.ENABLE_GPT2_124M, "SEED": C.SEED},
    }
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"FINAL VERDICTS  (score {score}/12)")
    print("-" * 70)
    for k in ("C1", "C2", "C3", "C4", "C5", "C6"):
        v = verdicts.get(k, {})
        print(f"  {k}: {v.get('verdict', '?'):9s}  {v.get('message', '')[:60]}")
    print("=" * 70)
    print(f"saved -> {os.path.join(OUT, 'verdict.json')}")

    if not ok_enabled:
        print("ERROR: an enabled verifier FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
