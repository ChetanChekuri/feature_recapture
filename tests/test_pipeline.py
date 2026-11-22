"""Correctness tests: plant a known answer and check each stage recovers it.

Run: .venv/bin/python tests/test_pipeline.py   (about 1 minute)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fr import estimators as E                      # noqa: E402
from fr.match import blind_histories, oracle_histories  # noqa: E402
from fr.sae import train_sae                         # noqa: E402
from fr.synth import HierSynthData, SynthData        # noqa: E402


def simulate(N, K, p, rng):
    H = (rng.random((N, K)) < np.asarray(p)[:, None]).astype(np.int8)
    return H[H.sum(1) > 0]


def test_homogeneous_unbiased():
    """Equal catchability: Lincoln-Petersen and M_t recover N; the CI covers N."""
    rng, N, K, cover = np.random.default_rng(1), 1000, 8, 0
    for rep in range(20):
        H = simulate(N, K, np.full(N, 0.3), rng)
        for name in ("chapman", "darroch_mt", "chao_mh"):
            assert abs(E.ESTIMATORS[name](H) - N) < 0.08 * N, (name, E.ESTIMATORS[name](H))
        lo, hi = E.bootstrap_ci(H, E.darroch_mt, n_boot=100, seed=rep)
        cover += lo <= N <= hi
    assert cover >= 16, cover


def test_heterogeneous_chao_beats_mt():
    """Unequal catchability: M_t and the union undercount; Chao's M_h bound is closer."""
    rng, N, K = np.random.default_rng(2), 2000, 8
    p = rng.beta(0.6, 2.5, N)
    H = simulate(N, K, p, rng)
    s, mt, ch = E.union(H), E.darroch_mt(H), E.chao_mh(H)
    assert s < mt < N and s <= ch
    assert abs(ch - N) < abs(mt - N), (s, mt, ch, N)


def test_singleton_free_ignores_junk():
    """Planted misidentification: adding 3000 spurious singletons (seed-specific junk)
    leaves the singleton-free estimate unchanged while Chao's bound explodes."""
    rng, N, K = np.random.default_rng(5), 1000, 8
    H = simulate(N, K, rng.beta(2.0, 2.0, N), rng)
    junk = np.eye(K, dtype=np.int8)[rng.integers(0, K, 3000)]
    Hj = np.vstack([H, junk])
    clean, dirty = E.singleton_free(H), E.singleton_free(Hj)
    assert abs(clean - N) < 0.1 * N and np.isclose(clean, dirty), (clean, dirty)
    assert E.chao_mh(Hj) > 3 * N
    assert E.singleton_free(np.ones((400, K), dtype=np.int8)) == 400  # all caught: N = S


def test_stray_recapture_does_not_diverge():
    """Over-complete dictionaries: nearly every real feature is caught by all seeds,
    thousands of junk singletons, one stray caught 5 times. Unbounded mixtures
    diverge here (Link 2003); the bounded singleton-free fit returns ~N."""
    K, N = 8, 1000
    H = np.vstack([np.ones((N, K), np.int8), np.eye(K, dtype=np.int8)[np.arange(6000) % K],
                   np.r_[np.ones(5), np.zeros(3)].astype(np.int8)[None]])
    assert abs(E.singleton_free(H) - N) < 0.02 * N, E.singleton_free(H)


def test_rarefaction_identities():
    rng = np.random.default_rng(3)
    H = simulate(500, 6, rng.uniform(0.05, 0.6, 500), rng)
    assert np.isclose(E.rarefy(H, 6), E.union(H))
    assert np.isclose(E.rarefy(H, 1), H.sum(0).mean())
    assert np.isclose(E.extrapolate(H, 0), E.union(H))
    assert E.extrapolate(H, 50) <= E.chao_mh(H) + 1e-6


def test_blind_matching_recovers_permutation():
    """Run b is a shuffled, slightly noisy copy of run a: every species is caught twice."""
    rng = np.random.default_rng(4)
    W = rng.standard_normal((300, 32)).astype(np.float32)
    Wb = W[rng.permutation(300)] + 0.02 * rng.standard_normal((300, 32)).astype(np.float32)
    runs = [dict(W_dec=W, freq=np.ones(300)), dict(W_dec=Wb, freq=np.ones(300))]
    H, _, _ = blind_histories(runs, tau=0.9)
    assert H.shape[0] == 300 and (H.sum(1) == 2).all()
    assert (oracle_histories(runs, W, tau=0.9).sum(0) == 300).all()


def test_sae_recovers_planted_features():
    """Easy regime (no crowding): one SAE finds nearly every planted feature."""
    torch.set_num_threads(4)
    data = SynthData(N=64, d=32, alpha=0.3, mean_active=3, data_seed=0, batch=2048)
    run = train_sae(data.batch, 600, d=32, m=128, k=4, seed=0, lr=3e-3, x_eval=data.eval_set(8192))
    frac = oracle_histories([run], data.F.numpy(), tau=0.9).mean()
    assert frac >= 0.9 and run["fvu"] < 0.2, (frac, run["fvu"])



def test_hierarchy_levels_are_separable():
    """Split features: at tau=0.8 a dictionary of parent directions matches only the
    parent references and one of child directions only the child references; a parent
    never activates two of its children at once."""
    data = HierSynthData(N=1024, d=128, n_children=4, child_cos=0.6, mean_active=5)
    R, N = data.refs.numpy(), data.N
    for W, level in ((data.P.numpy(), slice(N, None)), (data.F.numpy(), slice(0, N))):
        caught = oracle_histories([{"W_dec": W, "freq": np.ones(len(W))}], R, tau=0.8)[:, 0]
        assert caught[level].all() and caught.sum() == len(W), (caught[:N].sum(), caught[N:].sum())
    A = data.coeffs(0, B=4096)
    assert (A.reshape(4096, data.n_parents, 4) > 0).sum(-1).max() <= 1
    assert abs((A > 0).sum(1).float().mean() - 5) < 0.2

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all tests passed")
