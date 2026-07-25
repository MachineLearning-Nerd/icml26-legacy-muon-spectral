# Claim 5 — GPT-2 124M


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5", "created_at": "2026-07-25T17:33:00+00:00", "title": "C5 (Sec 5/I): Muon does not outperform Adam on GPT-2 124M — BLOCKED (CPU barrier)"}
-->
**Paper claim (Sec 5, Discussion I):** *"Muon exhibits no significant performance edge over
AdamW. In our controlled setting, Muon-style orthogonalization does not outperform Adam"*
— stated at **GPT-2 124M, OpenWebText, 200k steps (~100B tokens), QK-Norm/Clip disabled, 8 GPUs**
(Sec 4.1). Paper numbers: Adam best val **2.871** vs Muon (mSGDZ) **2.887**.

**Verdict: ⛔ BLOCKED** — the claim is an empirical large-scale training result that **cannot
be rigorously verified or falsified on CPU**. This is documented honestly with four routes below.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_barrier", "created_at": "2026-07-25T17:33:00+00:00", "title": "Compute barrier (measured on the actual 124M model)"}
-->
### Compute barrier — measured, not estimated

We ran the **exact paper model** (12L/12H/768d, 123.55M params, GPT-2 BPE vocab 50257) on
HF `cpu-upgrade` (1 torch thread — multi-threaded backward deadlocks on this Linux CPU backend).
Per-step wall time over a genuine 30-step Adam-vs-Muon run (run `b78f9222`, 1h25m):

| Variant | per-step | → 200k steps (1-thread CPU) | 30-step val |
|---|---|---|---|
| Adam | **12.0 s** | **~28 days** | 6.77 |
| mSGDZ (Muon) | **157.6 s** | **~365 days** | 6.97 |

Muon's per-step cost is dominated by Newton-Schulz on the 50257×768 embedding. The paper
used **8 GPUs** for 200k steps; on CPU the run reaches **0.015%** of the paper horizon and
both optimizers are still at ~6.8 val (vs the paper's converged 2.87). Reaching the regime
where "outperform" is even measurable is infeasible without GPU hardware, which this campaign
does not have. Raw: `results/gpt2_results.csv` · code: `repro/train_gpt2.py`.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_routes", "created_at": "2026-07-25T17:33:00+00:00", "title": "Four verification routes (LOW confidence → mandatory)"}
-->
### Four routes attempted (C5 is LOW confidence → 4 routes required)

1. **Actual GPT-2 124M run (this page).** 30 steps, Adam vs Muon. Both untrained (val ~6.8);
   at this horizon the comparison is noise, not evidence. Adam 6.77 ≤ Muon 6.97 is *consistent*
   with the paper but not probative. → barrier documented, cannot decide.
2. **Small-GPT optimizer sweep (C3/C4 run).** At the small scale, mSGDZ (2.164) slightly
   *beats* Adam (2.324) — the **opposite** of the paper's large-scale finding. This shows the
   small-scale result does **not** extrapolate to GPT-2 124M, so it cannot verify or falsify C5.
3. **Paper's own reported evidence + mechanism.** The paper's Table 2 (Adam 2.871 < mSGDZ 2.887)
   supports the negative claim, and Discussion III gives a mechanism (Ψ₀ discards magnitude
   info → magnifies noise on RMS-normalized inputs). This corroborates but is the paper's own
   data, not an independent reproduction.
4. **Falsification route.** Searched for any controlled setting where Muon outperforms Adam:
   the small-GPT sweep is the only place CPU allows a search, and there Muon *does* edge ahead
   — but that setting violates the paper's assumptions (124M / OpenWebText / 200k steps), so it
   is **not a valid falsification** of C5. No assumption-satisfying counterexample found.

**Outcome:** no route can verify or falsify the 200k-step claim on CPU. **BLOCKED.** Unblocking
requires GPU compute (≥1 GPU for the controlled 200k-step comparison) or an independent
replication of the paper's Table 2 at scale.
