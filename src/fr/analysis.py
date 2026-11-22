"""Summaries shared by the synthetic and real-model analyses."""
import numpy as np

from . import estimators as E
from .match import alive_rows, blind_histories, oracle_histories

MAIN = ("union", "chapman", "darroch_mt", "chao_mh", "jack1", "jack2", "mm",
        "betabin1", "betabin2", "mix2_2", "sf_aic")
BOOT = ("chao_mh", "jack2", "sf_aic")


def clean(v):
    v = float(v)
    return v if np.isfinite(v) and v < 1e9 else None   # None marks a diverged estimate


def estimate(H, boot=0, seed=0):
    out = {"Q": E.freq_counts(H)[1:].tolist(), "K": int(H.shape[1])}
    for name in MAIN:
        out[name] = clean(E.ESTIMATORS[name](H))
    out["sf_choice"] = E.singleton_free(H, return_choice=True)[1]
    if boot:
        for name in BOOT:
            if out[name] is not None:
                lo, hi = E.bootstrap_ci(H, E.ESTIMATORS[name], n_boot=boot, seed=seed)
                out[name + "_ci"] = [clean(lo), clean(hi)]
    return out


def junk_rates(runs, H, run_of, cluster, F, tau, min_freq=1e-5):
    """Fraction of blind species (clusters) with no member near any planted feature,
    split by how many runs caught them."""
    Ws = np.vstack([alive_rows(r, min_freq) for r in runs])
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    real = (Ws @ Fn.T).max(1) >= tau
    is_real = np.zeros(H.shape[0], bool)
    np.logical_or.at(is_real, cluster, real)
    y = H.sum(1)
    return {int(j): float(1 - is_real[y == j].mean()) for j in range(1, H.shape[1] + 1)
            if (y == j).any()}


def family_summary(runs, F=None, taus=(0.7, 0.8, 0.9), main_tau=0.8, boot=100,
                   subsample=(2, 4, 6)):
    m = runs[0]["W_dec"].shape[0]
    s = dict(m=int(m), K=len(runs), fvu=float(np.mean([r["fvu"] for r in runs])),
             alive=float(np.mean([(r["freq"] >= 1e-5).sum() for r in runs])), blind={}, oracle={})
    for tau in taus:
        Hb, run_of, cluster = blind_histories(runs, tau=tau)
        key = f"{tau:g}"
        s["blind"][key] = estimate(Hb, boot=boot if tau == main_tau else 0)
        if F is not None:
            Ho = oracle_histories(runs, F, tau=tau)
            s["oracle"][key] = estimate(Ho[Ho.sum(1) > 0], boot=boot if tau == main_tau else 0)
            s["blind"][key]["junk_by_count"] = junk_rates(runs, Hb, run_of, cluster, F, tau)
            if tau == main_tau:
                s["oracle"][key]["catch_rate"] = Ho.mean(1).tolist()   # per planted feature
    s["blind_sub"] = {}
    for k in subsample:
        if k < len(runs):
            Hb, _, _ = blind_histories(runs[:k], tau=main_tau)
            s["blind_sub"][str(k)] = estimate(Hb)
    return s
