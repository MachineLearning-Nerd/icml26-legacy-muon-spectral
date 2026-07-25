# Historical rejected baseline


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_hist", "created_at": "2026-07-25T17:34:00+00:00", "title": "Historical rejected baseline (revision 8e541ef, judged 8/12) — superseded"}
-->
> **This page preserves the toy-matrix evidence from the previous revision
> (`DineshAI/xUQ0Gw11NL@8e541ef`, judged 8/12) verbatim. It is **rejected** — the judge
> awarded only 1/2 each because no training was conducted. The current verification on
> [Claim 3](#/claim-3-stabilization) and [Claim 4](#/claim-4-rms-similarity) supersedes
> it with real training experiments. Kept reachable and unchanged per evidence-safety.**

### Old C3 (toy) — orthogonalization maps κ→1 on one 10×10 momentum matrix
Muon replaces the first-moment update with the polar factor P=UV^T. Toy check: the raw
momentum had κ=10.04; the orthogonalized update had κ=1.0 (PᵀP=I) — a 10× condition-number
reduction. *(Judge: "requires actual training experiments — none were conducted.")*

### Old C4 (toy) — cosine similarity of Ψ_p outputs on one 10×10 RMS matrix
Mean pairwise cosine-similarity of the four spectral variants: 0.934 (RMS-normalized) vs
0.929 (raw momentum). *(Judge: "the claim is about variants 'performing similarly at their
best settings' (training performance)… no training was run.")*

### Old C5 — deferred
The previous revision explicitly deferred C5 (GPT-2 124M) as "requiring GPU training";
no experiments were run (0/2). The current [Claim 5](#/claim-5-gpt2) documents the
compute barrier and the reduced 124M attempt.

**Why rejected:** toy/proxy checks on a single small matrix do not test the paper's training
claims. The current revision replaces C3/C4 with faithful training experiments and addresses
C5 with a quantified barrier.
