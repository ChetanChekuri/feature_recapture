"""Train K SAE seeds on synthetic data with a planted number of features.

    .venv/bin/python experiments/run_synth.py --grid main --workers 4

Resumable: a run is skipped when results/synth/<tag>/seed<s>.npz exists.
"""
import argparse
import itertools
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
from fr.synth import make_data    # noqa: E402

BASE = dict(d=128, k=8, alpha=0.7, mean_active=5.0, steps=1200, n_batches=400, lr=2e-3, data_seed=0)


def grid(name):
    """Each entry is one SAE family (a config); seeds are the capture occasions."""
    if name == "pilot":
        return [dict(BASE, N=1024, m=m, d=d) for m in (512, 1024) for d in (64, 128)], range(6)
    if name == "main":
        # Dictionary size m sweeps from 1/4 to 4x the planted count N.
        # (4N is run only where it costs m <= 4096 on the first data draw, for CPU time.)
        cfgs = [dict(BASE, N=N, m=int(N * r)) for N in (512, 1024, 2048)
                for r in (0.25, 0.5, 1.0, 2.0, 4.0) if N * r <= 4096]
        cfgs = [c for c in cfgs if not (c["N"] == 2048 and c["m"] == 4096)]
        cfgs += [dict(BASE, N=1024, m=int(1024 * r), data_seed=1) for r in (0.25, 0.5, 1.0, 2.0)]
        return cfgs, range(8)
    if name == "alpha":   # heavier / lighter tails of the feature-frequency law
        return [dict(BASE, N=1024, m=int(1024 * r), alpha=a) for a in (0.4, 1.0)
                for r in (0.5, 1.0, 2.0)], range(8)
    if name == "hier":    # features that split: 256 parents x 4 children = 1024 features
        return [dict(BASE, N=1024, m=m, hier=4, child_cos=0.6, data_seed=ds)
                for ds in (0, 1) for m in (128, 256, 512, 1024, 2048)], range(8)
    raise ValueError(name)


def tag(c):
    return f"N{c['N']}_d{c['d']}_m{c['m']}_k{c['k']}_a{c['alpha']}_mu{c['mean_active']}_ds{c['data_seed']}_s{c['steps']}" + hsuf(c)


def hsuf(c):
    return f"_h{c['hier']}c{c['child_cos']}" if c.get("hier") else ""


def dtag(c):
    return f"N{c['N']}_d{c['d']}_a{c['alpha']}_mu{c['mean_active']}_ds{c['data_seed']}_b{c['n_batches']}" + hsuf(c)


def out_dir(c):
    return ROOT / "results" / ("synth_hier" if c.get("hier") else "synth") / tag(c)


def job(args):
    c, seed, threads = args
    out = out_dir(c) / f"seed{seed}.npz"
    if out.exists():
        return str(out), 0.0
    torch.set_num_threads(threads)
    data = make_data(c)
    X = data.materialize(c["n_batches"], ROOT / "data/synth" / f"{dtag(c)}.npy")
    B = data.B
    nb = c["n_batches"]
    batch = lambda s: torch.from_numpy(np.ascontiguousarray(X[(s % nb) * B:(s % nb + 1) * B]))
    t0 = time.time()
    run = train_sae(batch, c["steps"], d=c["d"], m=c["m"], k=c["k"], seed=seed, lr=c["lr"],
                    x_eval=data.eval_set(16_384))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".tmp.npz"), **run, config=json.dumps(c), seconds=time.time() - t0)
    out.with_suffix(".tmp.npz").rename(out)
    return str(out), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="per worker (default: 16 / workers)")
    a = ap.parse_args()
    cfgs, seeds = grid(a.grid)
    # Materialise each dataset once, serially, before the parallel pool starts.
    for c in cfgs:
        make_data(c).materialize(c["n_batches"], ROOT / "data/synth" / f"{dtag(c)}.npy")
    jobs = [(c, s, a.threads or max(1, 16 // a.workers)) for c in cfgs for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:  # fork deadlocks OpenMP
        for i, (path, sec) in enumerate(pool.imap_unordered(job, jobs), 1):
            if sec:
                print(f"[{i}/{len(jobs)}] {sec:5.1f}s {path}", flush=True)
    print(f"done {a.grid}: {len(jobs)} runs in {time.time() - t0:.0f}s")
