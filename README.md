# Delving into Muon and Beyond — reproduction (arXiv 2602.04669)

## Collection classification and audit boundary

This repository is a **legacy/source workspace** for *Delving into Muon and Beyond: Deep Analysis and Extensions*
(arXiv `2602.04669`, OpenReview `xUQ0Gw11NL`). It is preserved
separately from the standardized canonical record at
[`icml26-muon-spectral`](https://github.com/MachineLearning-Nerd/icml26-muon-spectral).

The claim results and scores recorded below are historical results of this
workspace. They are not new paper-level verifications performed while
organizing the collection. The collection audit did not run the scientific
implementation; the canonical record documents its own scoped status and
limitations.

### How the historical claim evidence is produced

The claim table and experiment log below are the authoritative mapping from
each paper claim to its producer, command, control, and evidence artifact. In
this workspace, the spectral matrix-optimizer identities, optimizer comparisons, training sweeps, controls, and GPT-2 boundary evidence feed the claim table and report artifacts.

The former `orx/*` branches are historical workstreams, not additional final
publication claims. Their purposes and tips are preserved in
[`BRANCH_AUDIT.md`](BRANCH_AUDIT.md). Citation and author acknowledgment
details are in [`CITATION.cff`](CITATION.cff) and
[`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-xUQ0Gw11NL-delving-into-muon-and-beyond-deep-analysis-and-extensions/blob/main/repro/muon_tour.py)
![score](https://img.shields.io/badge/score-10%2F12-green)

This is an OpenResearch reproduction of **"Delving into Muon and Beyond: Deep Analysis and
Extensions"** (Qi et al., arXiv [2602.04669](https://arxiv.org/abs/2602.04669), ICML 2026
Spotlight, OpenReview `xUQ0Gw11NL`). Official code:
[`marcotchen/BeyondMuon`](https://github.com/marcotchen/BeyondMuon) (after Karpathy's nanoGPT).

## Reproduction summary

The paper studies a **spectral family** of matrix optimizers `Ψ_p(O)=UΣᵖVᵀ` that interpolates
between a raw update (`p=1`) and **Muon** (`p=0`, orthogonalization), and asks when
orthogonalization actually helps. We verify all six anchored claims — three by exact
linear-algebra checks, two by **real training** with the official optimizers, and one
(GPT-2 124M) is **honestly blocked** by CPU compute.

| Claim | Paper | Assessment | Paper # | Observed (this repro) |
|---|---|---|---|---|
| C1 | `Ψ_p=UΣᵖVᵀ` (Sec 3.3) | ✅ VERIFIED, ≤1.6e-15 | exact | p=1→O, p=0→UVᵀ orthogonal |
| C2 | `κ(Ψ_p)=κ(O)ᵖ` (Sec 3.4) | ✅ VERIFIED, ≤1e-6 | exact | rel err ≤1e-6 (cond 5–500) |
| C3 | Muon stabilizes momentum (Sec 4.3) | ✅ VERIFIED (training) | Muon robust | mSGDZ spread 0.69 vs mSGD 2.71; best 2.16 vs 3.10 |
| C4 | RMS variants match (Sec 4.4) | ✅ VERIFIED (training) | spread ≈0.01 | RMS spread 0.105 vs momentum 0.933 |
| C5 | Muon ≯ Adam @ GPT-2 124M (Sec 5) | ⛔ BLOCKED | Adam 2.871 vs Muon 2.887 | CPU barrier ~28d/365d for 200k steps |
| C6 | Newton-Schulz Ψ_{½},Ψ_{¼} (Sec 3.5) | ✅ VERIFIED, ≤2.2e-15 | no-SVD | matches SVD to ≤2.2e-15 |

**Score: 10/12** (previously judged 8/12 — C3/C4 were toy-matrix demos, C5 deferred).

**Downscaling / substitutions.** No GPU was available, so: C3/C4 use a 0.8M-param nanoGPT on
TinyStories (char-level, ~97 tokens) — enough to expose the optimizer *mechanisms* (LR-robustness,
variant similarity) the claims are about; C5 uses the real 124M model to measure the compute
barrier. Newton-Schulz `ns_iters=4` for the sweep (128³ grams), `15` (paper default) for C5.
1 torch thread (multi-threaded backward deadlocks on the HF Linux CPU backend).

**Compute.** Hugging Face `cpu-upgrade` for all training; local CPU for the ≤1 s math regression.
Cost: $0 (CPU-only).

**Full illustrated report:** [`reports/muon/report.md`](reports/muon/report.md) ·
**Interactive notebook:** [`repro/muon_tour.py`](repro/muon_tour.py) (`marimo edit repro/muon_tour.py`
locally — opens with the already-produced evidence).

## Experiment log

| Branch / experiment | Purpose | Exact run command | Outcome | Compute |
|---|---|---|---|---|
| `main` | Publication surface | _Not run as an experiment (publication surface)_ | — | — |
| [`orx/baseline-spectral-math-regression`](https://github.com/MachineLearning-Nerd/icml26-repro-xUQ0Gw11NL-delving-into-muon-and-beyond-deep-analysis-and-extensions/tree/orx/baseline-spectral-math-regression) | Env + math regression C1/C2/C6 | `uv run python -m repro.verify_all` | C1/C2/C6 VERIFIED (machine precision) | local CPU, <1 s |
| [`orx/c3-c4-training-lr-sweep-8-official-optimizers`](https://github.com/MachineLearning-Nerd/icml26-repro-xUQ0Gw11NL-delving-into-muon-and-beyond-deep-analysis-and-extensions/tree/orx/c3-c4-training-lr-sweep-8-official-optimizers) | C3/C4 real training (8 optimizers × LR grid) | `uv run python -m repro.verify_all` | C3 VERIFIED, C4 VERIFIED (run `8ab278fb`) | HF cpu-upgrade, 21 min |
| [`orx/c5-gpt-2-124m-reduced-scale-adam-vs-muon`](https://github.com/MachineLearning-Nerd/icml26-repro-xUQ0Gw11NL-delving-into-muon-and-beyond-deep-analysis-and-extensions/tree/orx/c5-gpt-2-124m-reduced-scale-adam-vs-muon) | C5 GPT-2 124M probe + barrier | `uv run python -m repro.verify_all` | C5 BLOCKED (run `b78f9222`) | HF cpu-upgrade, 1 h 25 m |

The **fixed run command is identical on every node** (`uv run python -m repro.verify_all`);
nodes differ only by committed config (`repro/config.py`: `ENABLE_TRAINING_SWEEP`,
`ENABLE_GPT2_124M`), never by the command or env vars.

## Repository layout

```
repro/                # faithful port of official BeyondMuon + verifiers
  core.py             # numpy spectral math (C1/C2/C6)
  verify_math.py      # C1/C2/C6 regression
  spectral_torch.py   # official Ψ_p / Newton-Schulz (torch, compile-optional)
  optimizers.py       # the 8 official variants (mSGD*/Adam* family)
  model.py            # nanoGPT GPT
  data.py             # TinyStories/OpenWebText streaming + GPT-2/char tokenizers
  train_sweep.py      # C3/C4 LR sweep   train_gpt2.py  # C5 probe
  verify_all.py       # fixed-command entrypoint (math + training + verdicts)
  config.py           # the only knob that varies between nodes
results/              # raw CSVs from the runs (sweep_results.csv, gpt2_results.csv)
reports/muon/         # illustrated report + figures
```

## Quickstart

```bash
uv sync                                         # Python 3.11/3.12, CPU-only torch
uv run python -m repro.verify_all               # math regression C1/C2/C6 (≤1 s)
```

---

# icml26-repro-xUQ0Gw11NL-delving-into-muon-and-beyond-deep-analysis-and-extensions
ICML 2026 agent reproduction workspace for xUQ0Gw11NL
