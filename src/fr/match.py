"""Turn K independently trained SAEs into capture histories.

A capture history is a binary matrix H (species x occasions): H[s, r] = 1 when
SAE run r contains a latent for species s.

* oracle_histories: species are the planted features (synthetic data only).
* blind_histories:  species are clusters of latents linked across runs by mutual
  nearest neighbours with decoder cosine >= tau. No ground truth is used.
"""
import numpy as np


def alive_rows(run, min_freq):
    W = run["W_dec"][run["freq"] >= min_freq].astype(np.float32)
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def oracle_histories(runs, F, tau=0.9, min_freq=1e-5):
    """H[i, r] = 1 if run r has an alive latent with cosine >= tau to true feature i."""
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    H = np.zeros((len(F), len(runs)), dtype=np.int8)
    for r, run in enumerate(runs):
        H[:, r] = (F @ alive_rows(run, min_freq).T).max(1) >= tau
    return H


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def blind_histories(runs, tau=0.9, min_freq=1e-5):
    """Cluster alive latents of all runs; return (H, run_of_latent, cluster_of_latent)."""
    Ws = [alive_rows(r, min_freq) for r in runs]
    off = np.cumsum([0] + [len(W) for W in Ws])
    parent = np.arange(off[-1])
    for a in range(len(Ws)):
        for b in range(a + 1, len(Ws)):
            C = Ws[a] @ Ws[b].T
            ab, ba = C.argmax(1), C.argmax(0)
            ia = np.arange(len(ab))
            ok = (ba[ab] == ia) & (C[ia, ab] >= tau)
            for x, y in zip(ia[ok] + off[a], ab[ok] + off[b]):
                rx, ry = _find(parent, x), _find(parent, y)
                if rx != ry:
                    parent[rx] = ry
    roots = np.array([_find(parent, x) for x in range(off[-1])])
    _, cluster = np.unique(roots, return_inverse=True)
    run_of = np.repeat(np.arange(len(Ws)), [len(W) for W in Ws])
    H = np.zeros((cluster.max() + 1, len(Ws)), dtype=np.int8)
    H[cluster, run_of] = 1
    return H, run_of, cluster
