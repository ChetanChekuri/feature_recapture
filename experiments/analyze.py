"""Compute capture histories and estimates for every finished SAE family.

    .venv/bin/python experiments/analyze.py synth|synth_hier|real

Writes results/summary_<which>.json. Cached per family in results/cache/ and
recomputed only when the family's run files change.
"""
import glob
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fr.analysis import family_summary   # noqa: E402
from fr.synth import make_data           # noqa: E402

MIN_SEEDS = 8
BOOT_N = 60   # bootstrap resamples per estimator (Chao-style)


def load(d):
    files = sorted(glob.glob(f"{d}/seed*.npz"), key=lambda f: int(Path(f).stem[4:]))
    return files, [dict(np.load(f)) for f in files]


def one(args):
    which, d = args
    files, runs = load(d)
    name = Path(d).name
    if len(runs) < MIN_SEEDS:
        return name, None
    stamp = sum(Path(f).stat().st_mtime_ns for f in files)
    cf = ROOT / "results/cache" / which / f"{name}.json"
    if cf.exists() and json.load(open(cf)).get("_stamp") == stamp:
        return name, json.load(open(cf))
    cfg = json.loads(str(runs[0]["config"]))
    F = None
    if which.startswith("synth"):
        F = make_data(cfg).refs.numpy()
    s = family_summary(runs, F=F, boot=BOOT_N)
    s.update(config=cfg, _stamp=stamp)
    json.dump(s, open(cf, "w"))
    return name, s


def main(which, workers):
    (ROOT / "results/cache" / which).mkdir(parents=True, exist_ok=True)
    dirs = sorted(glob.glob(str(ROOT / "results" / which / "*")))
    out = {}
    with get_context("spawn").Pool(workers) as pool:
        for name, s in pool.imap_unordered(one, [(which, d) for d in dirs]):
            if s is None:
                print("skip (incomplete)", name, flush=True)
                continue
            out[name] = s
            print("done", name, {k: s["blind"]["0.8"][k] for k in ("union", "chao_mh", "sf_aic")},
                  flush=True)
    json.dump(dict(sorted(out.items())), open(ROOT / f"results/summary_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
