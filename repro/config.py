"""Experiment configuration.

This is the ONLY knob that varies between experiment nodes (cardinal rule 3:
vary code/config, never the run command). The fixed run command
``uv run python -m repro.verify_all`` always runs the math regression (C1/C2/C6)
and then runs whichever training experiments are enabled below. The baseline
node disables all training; child nodes flip the relevant flags.

Paper reference setup (Sec 4.1): nanoGPT, GPT-2 124M (12L/12H/768d), OpenWebText,
seq 1024, global batch 480, 200k steps, wd=0, betas=(0.9,0.95), no QK-Norm/Clip,
vector lr=3e-4 shared, matrix lr tuned per optimizer. That full-scale run is an
8-GPU job (~100B tokens); it is infeasible on CPU, so the training children below
test the same *optimizer-comparison methodology* at a CPU-feasible scale and the
GPT-2 124M child documents the compute barrier for C5.
"""
from __future__ import annotations
import os

# --- which experiments the fixed run command executes ------------------------
ENABLE_MATH = True                 # C1/C2/C6 spectral-math regression (always on)
ENABLE_TRAINING_SWEEP = False      # C3/C4 (+ partial C5) small-GPT LR sweep
ENABLE_GPT2_124M = False           # C5 GPT-2 124M reduced-scale run

# Allow env override ONLY for local smoke tests (never used by orx runs, which
# vary committed code). orx runs use the committed values above.
ENABLE_TRAINING_SWEEP = os.environ.get("REPRO_SWEEP", "0") == "1" or ENABLE_TRAINING_SWEEP
ENABLE_GPT2_124M = os.environ.get("REPRO_GPT2", "0") == "1" or ENABLE_GPT2_124M

# --- shared optimizer settings (match paper Sec 4.1) -------------------------
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.95)
SGD_MOMENTUM = 0.9
GRAD_CLIP = 1.0
LR_VECTOR = 3e-4            # paper: shared AdamW vector learning rate
NS_ITERS = 15
SWEEP_NS_ITERS = 4   # smaller gram matrices (128^3); 4 iters is ample at this scale
SEED = 1337

# --- C3/C4 training sweep (small GPT, faithful optimizer comparison) ---------
# n_embd=128 keeps the Newton-Schulz gram matrices small (128^3) so the spectral
# variants are fast on 1-thread CPU. A real nanoGPT LM; optimizer-mechanism claims
# (C3/C4) are robust to model width.
SWEEP_MODEL = dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0, bias=False)
# micro=8, accum=1 => eff batch 8, seq 128 => 1024 tokens/step
SWEEP_MICRO_BATCH = 8
SWEEP_GRAD_ACCUM = 1
SWEEP_TRAIN_STEPS = 200
SWEEP_EVAL_INTERVAL = 100
SWEEP_EVAL_ITERS = 8
SWEEP_WARMUP = 20
# Ordered so the claim-critical variants run first (in case of timeout):
# C4 = Adam family (4), then C3 core = mSGD + mSGDZ, then C3 supporting.
SWEEP_GRIDS = {
    "Adam":   [3e-3, 4e-3, 6e-3, 1e-2],
    "AdamS":  [6e-3, 8e-3, 1e-2, 2e-2],
    "AdamQ":  [6e-3, 1e-2, 2e-2, 3e-2],
    "AdamZ":  [3e-3, 6e-3, 7e-3, 1e-2],
    "mSGD":   [1.0, 2.0, 5.0, 10.0],
    "mSGDZ":  [3e-3, 7e-3, 1e-2, 2e-2],
    "mSGDS":  [0.03, 0.1, 0.2, 0.3],
    "mSGDQ":  [0.03, 0.06, 0.1, 0.2],
}

# --- C5 GPT-2 124M reduced run (documents the compute barrier) ---------------
# The *exact* paper model (12L/12H/768d, vocab 50257). CPU-feasible token budget
# is ~3-4 orders of magnitude below the paper's 100B tokens; we run a genuine
# (Adam vs mSGDZ) comparison for as many steps as the CPU budget allows to expose
# early-training dynamics, and report C5 as the paper's negative finding.
GPT2_MODEL = dict(n_layer=12, n_head=12, n_embd=768, block_size=512, dropout=0.0, bias=False)
GPT2_MICRO_BATCH = 4
GPT2_GRAD_ACCUM = 2
GPT2_TRAIN_STEPS = 400          # << paper's 200k; documented barrier
GPT2_EVAL_INTERVAL = 50
GPT2_EVAL_ITERS = 20
GPT2_WARMUP = 20
GPT2_RUNS = {
    "Adam":  dict(lr_matrix=4e-3),
    "mSGDZ": dict(lr_matrix=7e-3),   # paper's best Muon lr (Sec 4.2)
}

# --- data ---------------------------------------------------------------------
# Faithful language-modeling task. Tokenizer choice:
#   DATA_VOCAB_MODE="char" : compact char vocab (~100) for the fast C3/C4 sweep
#                            (CPU-feasible; optimizer-mechanism claims are
#                            tokenizer-independent).
#   DATA_VOCAB_MODE="bpe"  : GPT-2 BPE (vocab 50257), the paper's exact tokenizer,
#                            used for the C5 GPT-2 124M run.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "owt_subset")
DATA_VOCAB_MODE = "char"
DATA_TARGET_TOKENS = 20_000_000   # chars (char mode) ~ real-text LM budget
