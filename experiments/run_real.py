"""Train K SAE seeds on cached Pythia-70M activations, for several dictionary sizes.

    .venv/bin/python experiments/run_real.py --grid main --workers 2

Resumable: a run is skipped when results/real/<tag>/seed<s>.npz exists.
"""
import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fr.sae import train_sae      # noqa: E402

DATA = ROOT / "data/pythia"
BASE = dict(k=32, epochs=2, lr=1e-3, B=4096)


def grid(name):
    if name == "pilot":
        return [dict(BASE, m=2048, lr=lr) for lr in (1e-3, 3e-4)], range(1)
    if name == "main":
        return [dict(BASE, m=m) for m in (1024, 2048, 4096, 8192)], range(8)
    if name == "long":   # 5x more training tokens, one pass (training-length robustness)
        return [dict(BASE, m=2048, epochs=1, data="pythia_long")], range(8)
    raise ValueError(name)


def tag(c):
    t = f"m{c['m']}_k{c['k']}_e{c['epochs']}_lr{c['lr']:g}"
    return t + (f"_{c['data']}" if c.get("data", "pythia") != "pythia" else "")


def job(args):
    c, seed, threads = args
    out = ROOT / "results/real" / tag(c) / f"seed{seed}.npz"
    if out.exists():
        return str(out), 0.0
    torch.set_num_threads(threads)
    X = np.load(ROOT / "data" / c.get("data", "pythia") / "train.npy", mmap_mode="r")
    xe = torch.from_numpy(np.load(DATA / "eval.npy").astype(np.float32))
    B, nb = c["B"], len(X) // c["B"]
    batch = lambda s: torch.from_numpy(X[(s % nb) * B:(s % nb + 1) * B].astype(np.float32))
    t0 = time.time()
    run = train_sae(batch, c["epochs"] * nb, d=X.shape[1], m=c["m"], k=c["k"], seed=seed,
                    lr=c["lr"], x_eval=xe)
    run["W_dec"] = run["W_dec"].astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".tmp.npz"), **run, config=json.dumps(c), seconds=time.time() - t0)
    out.with_suffix(".tmp.npz").rename(out)
    return str(out), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfgs, seeds = grid(a.grid)
    # Largest dictionaries first so the slowest runs do not straggle at the end.
    jobs = [(c, s, max(1, 16 // a.workers)) for s in seeds
            for c in sorted(cfgs, key=lambda c: -c["m"])]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        for i, (path, sec) in enumerate(pool.imap_unordered(job, jobs), 1):
            if sec:
                print(f"[{i}/{len(jobs)}] {sec:6.1f}s {path}", flush=True)
    print(f"done {a.grid}: {len(jobs)} runs in {time.time() - t0:.0f}s")
