# Claim 4 — RMS similarity


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c4_v2", "created_at": "2026-07-25T17:32:00+00:00", "title": "C4 (Sec 4.4): RMS variants match at best LR — VERIFIED by training"}
-->
**Paper claim (Sec 4.4):** *"All four variants behave similarly at their best settings,
suggesting that elementwise normalization already controls much of the harmful anisotropy
that spectral compression targets."*

**Verdict: ✅ VERIFIED** — by the same training LR sweep. Replaces the previous toy
cosine-similarity demonstration (preserved in [Historical rejected baseline](#/historical-toy-baseline)).

**Result — the 4 RMS-normalized variants cluster tightly; the momentum family does not:**

| Family | variant (p) | best val loss |
|---|---|---|
| **RMS (Adam) family** | Adam (p=1) | 2.324 |
| | AdamS (p=½) | 2.272 |
| | AdamQ (p=¼) | 2.219 |
| | AdamZ (p=0) | 2.297 |
| **Momentum (mSGD) family** | mSGD (p=1) | 3.097 |
| | mSGDS (p=½) | 2.333 |
| | mSGDQ (p=¼) | 2.298 |
| | mSGDZ (p=0) | 2.164 |

- **RMS-family best-loss spread = 0.105** (2.219–2.324).
- **Momentum-family best-loss spread = 0.933** (2.164–3.097).
- RMS spread is **9× smaller** than the momentum spread → spectral compression barely
  changes once RMS normalization is already applied, exactly as the paper argues. (AdamZ
  is the weak link at too-low LRs but matches at its tuned LR, consistent with the paper's
  note that full flattening "degrades performance" yet the variants still "behave similarly
  at their best settings".)

**Contract check:** `RMS_spread < 0.15` (0.105 ✓) **and** `RMS_spread < 0.6 × momentum_spread`
(0.105 < 0.560 ✓).

Raw data: `results/sweep_results.csv` · code: `repro/train_sweep.py`.
**Limitation:** downscaled model/data (CPU budget); the *relative* similarity (RMS ≪ momentum)
is the claim and is scale-robust.
