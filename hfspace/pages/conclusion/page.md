# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_conc_v2", "created_at": "2026-07-25T17:35:00+00:00", "title": "Conclusion + Scope & cost (10/12)"}
-->
**Outcome: 10/12.** The spectral family Ψ_p (C1) compresses the condition number as κ^p
(C2); Muon (Ψ₀) genuinely **stabilizes momentum** training across LRs (C3); RMS
normalization makes the four variants **behave similarly** at best settings (C4); and
Ψ_{1/2}, Ψ_{1/4} are computable via coupled Newton-Schulz without SVD (C6) — the first
three now by **real training** with the official optimizers. **C5** (Muon ≯ Adam on
GPT-2 124M) is **BLOCKED**: the paper's 200k-step / ~100B-token run is a multi-GPU job;
CPU would need ~28 days (Adam) to ~1 year (Muon), measured on the actual 124M model.

| Claim | Verdict | Points |
|---|---|---|
| C1 Ψ_p=UΣ^pV^T | VERIFIED | 2/2 |
| C2 κ(Ψ_p)=κ(O)^p | VERIFIED | 2/2 |
| C3 Muon stabilizes momentum | VERIFIED (training) | 2/2 |
| C4 RMS variants match | VERIFIED (training) | 2/2 |
| C5 Muon ≯ Adam @ GPT-2 124M | **BLOCKED** (CPU barrier) | 0/2 |
| C6 Newton-Schulz (no SVD) | VERIFIED | 2/2 |

### Scope & cost
| | This reproduction | Full replication (paper) |
|---|---|---|
| Scope | C1–C4, C6 exact/training; C5 barrier-documented | + C5 GPT-2 124M @ 200k steps |
| Hardware | HF `cpu-upgrade` (no GPU) | 8 GPUs |
| Time | math <1s; sweep 21 min; GPT-2 probe 1h25m | many GPU-hours |
| Cost | $0 (CPU) | GPU compute |

**What changed vs the judged 8/12 (`8e541ef`):** C3 and C4 upgraded from toy-matrix demos
(1/2 each) to real training experiments (2/2 each) using the official BeyondMuon optimizers;
C5 addressed with a quantified compute barrier + genuine 124M probe (remains BLOCKED, 0/2).
