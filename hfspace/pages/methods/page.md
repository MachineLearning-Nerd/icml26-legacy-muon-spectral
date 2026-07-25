# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_methods_v2", "created_at": "2026-07-25T17:36:00+00:00", "title": "Math core + faithful official-optimizer port + training harness"}
-->
**Math core** (`repro/core.py`, `repro/verify_math.py`): Ψ_p via SVD; `cond_number`;
coupled Newton-Schulz for X^{1/2}/X^{-1/2} (Algorithm 1, Appendix C) and Ψ_{1/2}, Ψ_{1/4}
(Appendix D) using only matrix multiplications — cross-checked against SVD to ≤1e-13 (C6).

**Official optimizer port** (`repro/spectral_torch.py`, `repro/optimizers.py`): a faithful
torch port of `marcotchen/BeyondMuon` (`optimizers/spectral_ns.py`, `adamw_ns.py`,
`sgdw_ns.py`, `optimizer_factory.py`) — all 8 variants (mSGD/mSGDS/mSGDQ/mSGDZ,
Adam/AdamS/AdamQ/AdamZ) with the exact spectral-transform dispatch (p=1 identity, p=0 Muon
quintic NS, p=½/¼ coupled NS). Only change: `torch.compile` is optional and attention uses
the explicit path (SDPA stalls on Linux CPU).

**Training harness** (`repro/train_sweep.py`, `repro/train_gpt2.py`, `repro/model.py`,
`repro/data.py`): nanoGPT model; streamed TinyStories corpus; per-variant matrix-LR sweeps
mirroring the paper's tuning protocol (Sec 4.2). Vector params always use AdamW @ 3e-4;
wd=0; β=(0.9,0.95); grad-clip 1.0; cosine schedule + warmup. Deterministic seed 1337.

**Fixed run command:** `uv run python -m repro.verify_all` — runs the math regression
always, and the C3/C4 sweep / C5 GPT-2 run when enabled by committed config
(`repro/config.py`). Each verifier exits nonzero on failure.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_methods_cpu", "created_at": "2026-07-25T17:36:00+00:00", "title": "CPU-feasibility notes (documented deviations)"}
-->
**Documented deviations for CPU feasibility (no GPU in this campaign):**
- **Tokenizer:** the C3/C4 sweep uses a compact char-level vocab (~97) so the vocab
  projection is not the training bottleneck; the optimizer-mechanism claims are
  tokenizer-independent. The C5 GPT-2 run uses the paper's exact GPT-2 BPE (50257).
- **Model scale:** C3/C4 use a 0.8M-param nanoGPT (vs the paper's 124M) — sufficient to
  expose LR-robustness and variant-similarity dynamics. C5 uses the real 124M.
- **Threads:** 1 torch intra-op thread (multi-threaded backward deadlocks on the HF Linux
  x86_64 CPU backend — verified: 8 threads stall after step 0; 1 thread progresses).
- **ns_iters:** 4 for the sweep (gram matrices are 128³); 15 (paper default) for C5.
