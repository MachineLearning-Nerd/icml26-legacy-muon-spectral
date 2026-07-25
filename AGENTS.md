# AGENTS — icml26-repro-xUQ0Gw11NL (Delving into Muon and Beyond)

Reproduction of **Delving into Muon and Beyond: Deep Analysis and Extensions**
(arXiv 2602.04669, OpenReview xUQ0Gw11NL, ICML 2026 Spotlight). Official code:
`marcotchen/BeyondMuon` (after Karpathy's nanoGPT).

## Environment (fixed across every node)
- Manager: **uv** with one repo-level `.venv`, Python 3.11.
- CPU-only torch (https://download.pytorch.org/whl/cpu). No GPU in this campaign.
- Reproduce deps: `uv sync` (uses committed `pyproject.toml` + `uv.lock`).

## Fixed run command (identical on every node)
```
uv run python -m repro.verify_all
```
Always runs the math regression (C1/C2/C6). Training experiments are toggled by
**committed config** (`repro/config.py`: `ENABLE_TRAINING_SWEEP`, `ENABLE_GPT2_124M`),
never by the command. Baseline node = math only; children flip the flags.

## Compute policy (this campaign)
- Local CPU: only short tasks (≤1 core, ≤5 min), e.g. the math regression + smoke tests.
- Hugging Face `cpu-upgrade`: every training run (>1 core or >5 min or uncertain).

## Claim map
| Claim | Paper | Verifier |
|---|---|---|
| C1 | Sec 3.3 Ψ_p=UΣ^pV^T | `repro/verify_math.py` (numpy, ≤1e-10) |
| C2 | Sec 3.4 κ(Ψ_p)=κ(O)^p | `repro/verify_math.py` (rel err ≤1e-6) |
| C3 | Sec 4.3 Muon stabilizes momentum | `repro/train_sweep.py` LR-robustness |
| C4 | Sec 4.4 RMS variants match | `repro/train_sweep.py` best-loss spread |
| C5 | Sec 5 Muon ≯ Adam on GPT-2 124M | `repro/train_gpt2.py` + barrier doc |
| C6 | Sec 3.5 Newton-Schulz, no SVD | `repro/verify_math.py` (≤1e-6) |

## Cardinal rules honored
Never edit a populated baseline; one fixed command; vary committed code not knobs;
grow the tree downward; launch only via `orx exp run`; never merge/rebase a branch
with a completed run.
