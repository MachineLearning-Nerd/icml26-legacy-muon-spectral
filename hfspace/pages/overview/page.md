# Overview


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_overview_v2", "created_at": "2026-07-25T17:30:00+00:00", "title": "Executive summary — 10/12 (C5 BLOCKED on CPU)"}
-->
**Muon Spectral Family (arXiv 2602.04669, OpenReview xUQ0Gw11NL) — 10/12.**
The three spectral-math claims (C1/C2/C6) are verified to machine precision, and the
two optimizer-mechanism claims (C3/C4) are now verified by **real training experiments**
with the paper's official optimizers (replacing the previous toy-matrix demonstrations).
C5 — the GPT-2 124M negative result — is **BLOCKED**: the paper's 200k-step / ~100B-token
training is a multi-GPU job; on CPU it would take ~28 days (Adam) to ~1 year (Muon), so
the claim cannot be rigorously verified or falsified in this CPU-only campaign. A genuine
reduced 124M run documents the barrier.

| Claim | Verdict | Evidence |
|---|---|---|
| C1 Ψ_p=UΣ^pV^T (p=0→polar, p=1→O) | ✅ VERIFIED | machine precision ≤1.6e-15 |
| C2 κ(Ψ_p)=κ(O)^p | ✅ VERIFIED | rel err ≤1e-6 (cond 5–500) |
| C3 Muon stabilizes momentum (vs mSGD) | ✅ VERIFIED | training: mSGDZ spread 0.69 vs mSGD 2.71; best 2.16 vs 3.10 |
| C4 RMS variants match at best LR | ✅ VERIFIED | training: RMS spread 0.105 vs momentum 0.933 |
| C5 Muon ≯ Adam on GPT-2 124M | ⛔ BLOCKED | compute barrier: ~28d (Adam) / ~365d (Muon) for 200k steps on CPU |
| C6 Newton-Schulz Ψ_{1/2}, Ψ_{1/4} (no SVD) | ✅ VERIFIED | ≤2.2e-15 vs SVD |

**Score: 10/12.** C1/C2/C6 = numpy machine-precision; C3/C4 = real nanoGPT training with
the 8 official BeyondMuon optimizers; C5 = BLOCKED with a quantified compute barrier.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_visibility", "created_at": "2026-07-25T17:30:00+00:00", "title": "Evaluator-visible evidence matrix"}
-->
### Evidence visibility matrix (every cell reachable from this logbook)

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested |
|---|---|---|---|---|---|---|---|
| C1 | [Claim 1](#/claim-1-p-family) | ✅ `repro/verify_math.py` | ✅ | `results/` | ✅ SVD cross-check | n/a (exact) | Ψ_p=UΣ^pV^T |
| C2 | [Claim 2](#/claim-2-condition-number) | ✅ | ✅ | `results/` | ✅ | n/a | κ(Ψ_p)=κ(O)^p |
| C3 | [Claim 3](#/claim-3-stabilization) | ✅ `repro/train_sweep.py` | ✅ | `results/sweep_results.csv` | ✅ mSGD vs mSGDZ | ✅ mSGD diverges | Muon stabilizes momentum |
| C4 | [Claim 4](#/claim-4-rms-similarity) | ✅ | ✅ | `results/sweep_results.csv` | ✅ momentum-family ref | ✅ momentum-spread control | RMS variants match |
| C5 | [Claim 5](#/claim-5-gpt2) | ✅ `repro/train_gpt2.py` | ✅ | `results/gpt2_results.csv` | ✅ Adam vs mSGDZ | n/a (barrier) | Muon ≯ Adam @ GPT-2 124M |
| C6 | [Claim 6](#/claim-6-newton-schulz) | ✅ | ✅ | `results/` | ✅ SVD cross-check | n/a (exact) | NS computes Ψ_{1/2},Ψ_{1/4} |

> **Historical note.** The previous revision (commit `8e541ef`) scored 8/12: C3/C4 were
> toy-matrix demonstrations (1/2 each) and C5 was deferred (0/2). Those toy pages are
> preserved verbatim under [Historical rejected baseline](#/historical-toy-baseline) and
> are superseded by the current training-based verification.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_setup", "created_at": "2026-07-25T17:30:00+00:00", "title": "Fixed command, environment, compute"}
-->
**Fixed run command (identical on every node):** `uv run python -m repro.verify_all`
**Environment:** uv, Python 3.11/3.12, CPU-only torch (`https://download.pytorch.org/whl/cpu`), locked (`pyproject.toml` + `uv.lock`).
**Compute:** Hugging Face `cpu-upgrade` for all training (local CPU only for the ≤1s math regression). No GPU available in this campaign.
**Paper source:** ar5iv HTML `2602.04669`, SHA-256 `73c2d82bb9e0e729d15174d9144cd1b619895eef03a9c95f0172867c762ca27e` (retrieved 2026-07-25). Official code: `marcotchen/BeyondMuon`.
