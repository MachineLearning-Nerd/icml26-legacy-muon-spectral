"""Shared training loop + C3/C4 LR sweep over the 8 official optimizer variants.

This is REAL training (not a matrix toy): a nanoGPT model trained on a real
language-modeling corpus with the exact official optimizers, across a per-variant
matrix-LR grid mirroring the paper's tuning protocol (Sec 4.2, Tables 1-2).

It yields direct evidence for:
  C3 (Sec 4.3): does Muon (mSGDZ) stabilize momentum vs mSGD across LRs?
  C4 (Sec 4.4): do the 4 RMS variants (Adam/AdamS/AdamQ/AdamZ) match at best LR?
  partial C5:   relative ordering of Adam vs mSGDZ (Muon) at this scale.
"""
from __future__ import annotations
import csv
import math
import os
import time

import numpy as np
import torch

from . import config as C
from .data import TokenData, prepare
from .model import GPT, GPTConfig
from . import optimizers as optz


def _effective_cpus():
    """Effective CPU count from the cgroup quota (NOT host cpus)."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().split()
        if parts and parts[0] != "max":
            return max(1, int(int(parts[0]) / int(parts[1])))
    except Exception:
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _set_threads():
    """Pin torch threads to the cgroup CPU quota (NOT the host CPU count).

    Containers report the host CPU count via ``os.cpu_count()``; spawning that many
    torch threads on a CPU-limited cgroup thrashes. We read the cgroup quota and
    cap threads to it.
    """
    eff = _effective_cpus()
    # Use a single torch intra-op thread: multi-threaded backward deadlocks on
    # the HF Linux x86_64 CPU backend (verified -- 8 threads stall after step 0;
    # 1 thread progresses at ~0.4s/step). REPRO_THREADS overrides for diagnostics.
    env_override = os.environ.get("REPRO_THREADS")
    nproc = max(1, int(env_override)) if env_override else 1
    torch.set_num_threads(nproc)
    try:
        torch.set_num_interop_threads(min(2, nproc))
    except RuntimeError:
        pass
    print(f"[threads] effective_cpus={eff} torch_threads={nproc} "
          f"host_cpu_count={os.cpu_count()}", flush=True)


def _cos_lr(it: int, max_lr: float, min_lr: float, warmup: int, decay_iters: int) -> float:
    if it < warmup:
        return max_lr * (it + 1) / (warmup + 1)
    if it > decay_iters:
        return min_lr
    dr = (it - warmup) / (decay_iters - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * dr)) * (max_lr - min_lr)


def train_one(variant: str, lr_matrix: float, model_cfg: dict, train_steps: int,
              data: TokenData, micro_batch: int, grad_accum: int, warmup: int,
              eval_interval: int, eval_iters: int, seed: int,
              lr_vector: float, betas, weight_decay, sgd_momentum, ns_iters: int,
              vocab_size: int, device: str = "cpu", dtype=torch.float32) -> dict:
    """Train one (variant, lr_matrix) config; return metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1)

    cfg = GPTConfig(vocab_size=vocab_size,
                    block_size=model_cfg["block_size"], n_layer=model_cfg["n_layer"],
                    n_head=model_cfg["n_head"], n_embd=model_cfg["n_embd"],
                    dropout=model_cfg["dropout"], bias=model_cfg["bias"])
    model = GPT(cfg).to(device=device, dtype=dtype)
    opts = optz.build_optimizers(model, variant, lr_matrix, lr_vector, betas,
                                 weight_decay, sgd_momentum, ns_iters)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = np.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = data.get_batch(split, micro_batch, rng)
                with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device != "cpu")):
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = float(losses.mean())
        model.train()
        return out

    decay_iters = max(train_steps, 10)
    min_lr = lr_matrix * 0.1
    min_lr_v = lr_vector * 0.1
    history = []
    best_val = float("inf")
    diverged = False
    final_train = float("nan")
    t0 = time.time()
    x, y = data.get_batch("train", micro_batch, rng)
    for it in range(train_steps):
        mlr = _cos_lr(it, lr_matrix, min_lr, warmup, decay_iters)
        vlr = _cos_lr(it, lr_vector, min_lr_v, warmup, decay_iters)
        optz.set_lrs(opts, mlr, vlr)
        total_loss = 0.0
        for _ in range(grad_accum):
            x, y = data.get_batch("train", micro_batch, rng)
            with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device != "cpu")):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        if C.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        total_loss += loss.item()
        if not math.isfinite(total_loss) or total_loss > 1e4:
            diverged = True
            break
        final_train = total_loss
        if it % 25 == 0 or it == train_steps - 1:
            print(f"          step {it:4d} train_loss={total_loss:.4f}", flush=True)
        if it % eval_interval == 0 or it == train_steps - 1:
            losses = estimate_loss()
            history.append({"step": it, "train_loss": losses["train"],
                            "val_loss": losses["val"], "lr": mlr})
            best_val = min(best_val, losses["val"])
    dt = time.time() - t0
    return {
        "variant": variant, "lr_matrix": lr_matrix,
        "family": optz.VARIANTS[variant.replace("_", "")][0],
        "exponent": optz.VARIANTS[variant.replace("_", "")][1],
        "final_train_loss": final_train, "best_val_loss": best_val,
        "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
        "diverged": diverged, "steps_done": len(history) * eval_interval,
        "train_seconds": round(dt, 1), "history": history,
    }


def run(out_dir: str) -> dict:
    from . import config as C
    os.makedirs(out_dir, exist_ok=True)
    device = "cpu"
    _set_threads()
    dtype = torch.float32

    prep = prepare(C.DATA_DIR, C.DATA_TARGET_TOKENS, C.DATA_VOCAB_MODE)
    vocab_size = prep["vocab_size"]
    data = TokenData(C.DATA_DIR, C.SWEEP_MODEL["block_size"], device=device)
    print(f"[sweep] vocab_mode={C.DATA_VOCAB_MODE} vocab_size={vocab_size} "
          f"train_tokens={prep.get('train_tokens'):,} source={prep.get('source')}")

    rows = []
    grid = [(v, lr) for v, lrs in C.SWEEP_GRIDS.items() for lr in lrs]
    total = len(grid)
    flat_keys = ["variant", "family", "exponent", "lr_matrix", "diverged",
                 "final_train_loss", "best_val_loss", "final_val_loss",
                 "steps_done", "train_seconds", "seed"]

    def _flush_csv():
        with open(os.path.join(out_dir, "sweep_results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flat_keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in flat_keys})
        import json
        with open(os.path.join(out_dir, "sweep_results.json"), "w") as f:
            json.dump(rows, f, indent=2, default=str)

    print(f"[sweep] {total} runs: {len(C.SWEEP_GRIDS)} variants x their LR grids")
    for i, (variant, lr) in enumerate(grid):
        print(f"[sweep] ({i+1}/{total}) {variant} lr_matrix={lr}", flush=True)
        try:
            r = train_one(variant, lr, C.SWEEP_MODEL, C.SWEEP_TRAIN_STEPS, data,
                          C.SWEEP_MICRO_BATCH, C.SWEEP_GRAD_ACCUM, C.SWEEP_WARMUP,
                          C.SWEEP_EVAL_INTERVAL, C.SWEEP_EVAL_ITERS, C.SEED,
                          C.LR_VECTOR, C.BETAS, C.WEIGHT_DECAY, C.SGD_MOMENTUM,
                          C.SWEEP_NS_ITERS, vocab_size, device=device, dtype=dtype)
        except Exception as e:
            r = {"variant": variant, "lr_matrix": lr, "diverged": True,
                 "final_train_loss": float("nan"), "best_val_loss": float("nan"),
                 "final_val_loss": float("nan"), "error": repr(e),
                 "family": optz.VARIANTS[variant.replace("_", "")][0],
                 "exponent": optz.VARIANTS[variant.replace("_", "")][1],
                 "steps_done": 0, "train_seconds": 0.0, "history": []}
        r["seed"] = C.SEED
        rows.append(r)
        _flush_csv()  # incremental: a timeout never loses completed runs
        status = "DIVERGED" if r.get("diverged") else f"val={r.get('final_val_loss', float('nan')):.4f}"
        print(f"          -> {status}  ({r.get('train_seconds', 0)}s)", flush=True)

    _print_summary(rows)
    return {"runs": rows, "data": prep}


def _print_summary(rows):
    print("\n" + "=" * 78)
    print("C3/C4 SWEEP SUMMARY (best val loss per variant)")
    print("-" * 78)
    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)
    for v in sorted(by_var):
        rs = [r for r in by_var[v] if not r.get("diverged")]
        fam, exp = optz.VARIANTS[v.replace("_", "")]
        if rs:
            best = min(rs, key=lambda r: r["best_val_loss"])
            n_stable = len(rs)
            n_total = len(by_var[v])
            print(f"  {v:7s} family={fam:5s} p={exp:.2f}  "
                  f"stable {n_stable}/{n_total} LRs  best_val={best['best_val_loss']:.4f} "
                  f"@ lr={best['lr_matrix']}")
        else:
            print(f"  {v:7s} family={fam:5s} p={exp:.2f}  "
                  f"ALL DIVERGED ({len(by_var[v])} LRs)")
    print("=" * 78)
