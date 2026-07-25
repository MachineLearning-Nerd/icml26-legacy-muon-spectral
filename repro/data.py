"""Data preparation and loader.

Tokenizer choice is set by ``DATA_VOCAB_MODE`` (config):
  - "char" : a compact character-level vocabulary (~100 tokens). Used for the
    C3/C4 small-GPT sweep so CPU training is fast; the optimizer-comparison
    conclusions are tokenizer-independent.
  - "bpe"  : GPT-2 BPE (tiktoken, vocab 50257), the paper's exact tokenizer.
    Used for the C5 GPT-2 124M run (paper-faithful).

Both stream TinyStories (real natural-language text; OpenWebText streaming is
attempted first) and write nanoGPT-format train.bin / val.bin (uint16) + meta.pkl.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import torch

BPE_VOCAB_SIZE = 50257
EOT_BPE = 50256  # GPT-2 <|endoftext|>


def _stream_text(target_chars: int, verbose: bool):
    """Stream text until ~target_chars collected. Returns (text_list, source)."""
    text_chunks: list[str] = []
    have = 0
    source = None

    def take(ds_iter, want):
        nonlocal have, source
        for ex in ds_iter:
            t = ex.get("text") or ""
            text_chunks.append(t)
            have += len(t)
            if verbose and have % 5_000_000 < len(t):
                print(f"[data] streamed ~{have:,} chars")
            if have >= want:
                break

    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
            take(iter(ds), target_chars)
            source = "openwebtext"
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"[data] openwebtext streaming failed ({e}); trying TinyStories")
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            take(iter(ds), target_chars)
            source = "tinystories"
    except Exception as e:  # pragma: no cover
        if verbose:
            print(f"[data] datasets unavailable ({e})")
    if have == 0:
        raise RuntimeError("Could not stream any text dataset for data prep")
    return text_chunks, source


def prepare(data_dir: str, target_tokens: int, vocab_mode: str = "char",
            verbose: bool = True) -> dict:
    """Tokenize streamed text and write train.bin/val.bin + meta.pkl.

    target_tokens is interpreted as target chars (char mode) or BPE tokens (bpe mode).
    Idempotent: reuses existing bins if large enough.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    meta_path = os.path.join(data_dir, "meta.pkl")

    if os.path.exists(train_path) and os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        n = np.memmap(train_path, dtype=np.uint16, mode="r").size
        if meta.get("vocab_mode") == vocab_mode and n >= target_tokens * 0.9:
            if verbose:
                print(f"[data] reuse {train_path} ({n:,} tokens, vocab_mode={vocab_mode})")
            return {"train_tokens": n, "vocab_size": meta["vocab_size"], "reused": True,
                    "source": meta.get("source"), "vocab_mode": vocab_mode}

    if vocab_mode == "bpe":
        return _prepare_bpe(data_dir, target_tokens, verbose)
    return _prepare_char(data_dir, target_tokens, verbose)


def _prepare_char(data_dir, target_chars, verbose):
    text_chunks, source = _stream_text(target_chars, verbose)
    full = "".join(text_chunks)
    chars = sorted(set(full))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    # char ids + a document boundary marker (0) between chunks
    ids = []
    for t in text_chunks:
        ids.extend(stoi[c] for c in t)
        ids.append(0)
    arr = np.array(ids, dtype=np.uint16)
    n_val = min(max(int(0.02 * arr.size), 20_000), 2_000_000)
    val, train = arr[:n_val], arr[n_val:]
    train.tofile(os.path.join(data_dir, "train.bin"))
    val.tofile(os.path.join(data_dir, "val.bin"))
    meta = {"vocab_size": vocab_size, "vocab_mode": "char", "source": source,
            "itos": chars}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    if verbose:
        print(f"[data] char source={source} vocab={vocab_size} train={train.size:,} "
              f"val={val.size:,} -> {data_dir}")
    return {"train_tokens": int(train.size), "val_tokens": int(val.size),
            "vocab_size": vocab_size, "source": source, "vocab_mode": "char"}


def _prepare_bpe(data_dir, target_tokens, verbose):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    text_chunks, source = _stream_text(target_tokens * 4, verbose)  # ~4 chars/token
    ids = []
    for t in text_chunks:
        ids.extend(enc.encode_ordinary(t))
        ids.append(EOT_BPE)
    arr = np.array(ids, dtype=np.uint16)
    n_val = min(max(int(0.02 * arr.size), 50_000), 2_000_000)
    val, train = arr[:n_val], arr[n_val:]
    train.tofile(os.path.join(data_dir, "train.bin"))
    val.tofile(os.path.join(data_dir, "val.bin"))
    meta = {"vocab_size": BPE_VOCAB_SIZE, "vocab_mode": "bpe", "source": source}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    if verbose:
        print(f"[data] bpe source={source} vocab={BPE_VOCAB_SIZE} train={train.size:,} "
              f"val={val.size:,} -> {data_dir}")
    return {"train_tokens": int(train.size), "val_tokens": int(val.size),
            "vocab_size": BPE_VOCAB_SIZE, "source": source, "vocab_mode": "bpe"}


def load_vocab_size(data_dir: str, default: int = 50257) -> int:
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            return int(pickle.load(f).get("vocab_size", default))
    return default


class TokenData:
    """nanoGPT-style memmap data loader (train.bin / val.bin as uint16 tokens)."""

    def __init__(self, data_dir: str, block_size: int, device: str = "cpu"):
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        val_path = os.path.join(data_dir, "val.bin")
        self.val = np.memmap(val_path, dtype=np.uint16, mode="r") if os.path.exists(val_path) else self.train
        self.block_size = block_size
        self.device = device

    def get_batch(self, split: str, batch_size: int, rng: np.random.Generator):
        data = self.val if split == "val" else self.train
        ix = rng.integers(0, len(data) - self.block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)
