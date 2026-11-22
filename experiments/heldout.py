"""Held-out check: fit on a subset of seeds, predict what the remaining seeds reveal.

For each family with K seeds, draw R random subsets of K/2 seeds. From the subset,
predict the number of reproducible species (caught by >= 2 seeds) that blind matching
finds across all K seeds. Prediction methods: no growth (count in the subset),
the equal-catchability model M_0, and the AIC-selected singleton-free model.

    .venv/bin/python experiments/heldout.py synth|real
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fr import estimators as E            # noqa: E402
from fr.match import blind_histories      # noqa: E402

TAU, R = 0.8, 10


def m0_predict(H, K_new):
    """Equal catchability (one class, no lower bound beyond p > 0)."""
    lp1 = E.SF_MODELS[1][0]
    f = lambda y, K, th: lp1(y, K, th, 0.0)
    N, th, _ = E._trunc_fit(H, f, E.SF_MODELS[1][1], jmin=2)
    lp = f(np.arange(K_new + 1), K_new, th)
    return float(N * np.exp(np.logaddexp.reduce(lp[2:])))


def main(which):
    out = {}
    for d in sorted(glob.glob(str(ROOT / "results" / which / "*"))):
        files = sorted(glob.glob(f"{d}/seed*.npz"), key=lambda f: int(Path(f).stem[4:]))
        if len(files) < 8:
            continue
        runs = [dict(np.load(f)) for f in files]
        K = len(runs)
        H_all, _, _ = blind_histories(runs, tau=TAU)
        target = int((H_all.sum(1) >= 2).sum())
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(R):
            idx = sorted(rng.choice(K, K // 2, replace=False))
            H, _, _ = blind_histories([runs[i] for i in idx], tau=TAU)
            rows.append(dict(subset=[int(i) for i in idx],
                             no_growth=int((H.sum(1) >= 2).sum()),
                             m0=m0_predict(H, K),
                             sf_aic=E.sf_predict(H, K)))
        out[Path(d).name] = dict(K=K, K_fit=K // 2, target=target, draws=rows)
        mean = {k: np.mean([r[k] for r in rows]) for k in ("no_growth", "m0", "sf_aic")}
        print(Path(d).name, "target", target, {k: round(v) for k, v in mean.items()})
    json.dump(out, open(ROOT / f"results/heldout_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
