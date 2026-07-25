"""C5: GPT-2 124M (the paper's exact model) reduced-scale comparison.

Paper claim (Sec 5, Discussion I): in the controlled GPT-2 124M / OpenWebText
setup with QK-Norm/QK-Clip disabled, Muon-style orthogonalization does NOT
outperform Adam. That claim is stated at 200k steps (~100B tokens), an 8-GPU job.

This module runs the *exact* 124M model with Adam vs mSGDZ (Muon) for the
largest CPU-feasible token budget and reports the early-training comparison.
It cannot reach the paper's 100B-token horizon on CPU, so it is partial
evidence; verify_all documents the compute barrier and the BLOCKED verdict.
"""
from __future__ import annotations
import json
import os
import time

import numpy as np
import torch

from . import config as C
from .data import TokenData, prepare
from .model import GPT, GPTConfig
from . import optimizers as optz
from .train_sweep import train_one, _set_threads


def run(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    device = "cpu"
    _set_threads()
    dtype = torch.float32

    prep = prepare(C.DATA_DIR, C.DATA_TARGET_TOKENS, C.DATA_VOCAB_MODE)
    vocab_size = prep["vocab_size"]
    data = TokenData(C.DATA_DIR, C.GPT2_MODEL["block_size"], device=device)
    print(f"[gpt2-124M] vocab_mode={C.DATA_VOCAB_MODE} vocab_size={vocab_size} "
          f"train_tokens={prep.get('train_tokens'):,} source={prep.get('source')}")

    rows = []
    for variant, spec in C.GPT2_RUNS.items():
        print(f"[gpt2-124M] {variant} lr_matrix={spec['lr_matrix']}", flush=True)
        t0 = time.time()
        try:
            r = train_one(variant, spec["lr_matrix"], C.GPT2_MODEL, C.GPT2_TRAIN_STEPS,
                          data, C.GPT2_MICRO_BATCH, C.GPT2_GRAD_ACCUM, C.GPT2_WARMUP,
                          C.GPT2_EVAL_INTERVAL, C.GPT2_EVAL_ITERS, C.SEED,
                          C.LR_VECTOR, C.BETAS, C.WEIGHT_DECAY, C.SGD_MOMENTUM,
                          C.NS_ITERS, vocab_size, device=device, dtype=dtype)
        except Exception as e:
            r = {"variant": variant, "lr_matrix": spec["lr_matrix"], "diverged": True,
                 "error": repr(e), "best_val_loss": float("nan"),
                 "final_val_loss": float("nan"), "history": [], "train_seconds": 0.0}
        r["wall_seconds"] = round(time.time() - t0, 1)
        r["model_params"] = "GPT-2 124M (12L/12H/768d)"
        r["train_steps"] = C.GPT2_TRAIN_STEPS
        r["tokens_per_step"] = (C.GPT2_MICRO_BATCH * C.GPT2_GRAD_ACCUM
                                * C.GPT2_MODEL["block_size"])
        r["total_tokens"] = r["tokens_per_step"] * C.GPT2_TRAIN_STEPS
        rows.append(r)
        print(f"           -> val={r.get('final_val_loss', float('nan')):.4f} "
              f"({r['wall_seconds']}s)", flush=True)

    with open(os.path.join(out_dir, "gpt2_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 78)
    print("C5 GPT-2 124M (reduced) — Adam vs mSGDZ (Muon)")
    print("-" * 78)
    for r in rows:
        print(f"  {r['variant']:7s} lr={r.get('lr_matrix')}  "
              f"val={r.get('final_val_loss', float('nan')):.4f}  "
              f"best={r.get('best_val_loss', float('nan')):.4f}  "
              f"diverged={r.get('diverged')}")
    print(f"  paper horizon: 200,000 steps (~100B tokens); this run: "
          f"{C.GPT2_TRAIN_STEPS} steps (~{rows[0]['total_tokens']/1e6:.1f}M tokens) "
          f"= {C.GPT2_TRAIN_STEPS/200000*100:.3f}% of paper")
    print("=" * 78)
    return {"runs": rows, "data": prep}
