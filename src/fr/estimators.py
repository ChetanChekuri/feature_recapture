"""Closed-population size estimators from capture histories.

H is a binary matrix (observed species x K occasions); species never caught are
absent from H, and estimating how many of them exist is the point.

References (verified in refs/VERIFIED.md):
  Chapman (1951) bias-corrected Lincoln-Petersen; Darroch (1958) model M_t;
  Chao (1987) lower bound for model M_h; Burnham & Overton (1978, 1979)
  jackknife for M_h; Colwell et al. (2012) incidence rarefaction/extrapolation;
  Raaijmakers (1987) Michaelis-Menten accumulation fit.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.special import gammaln


def freq_counts(H):
    """Q[j] = number of species caught on exactly j occasions (j = 0..K)."""
    return np.bincount(H.sum(1), minlength=H.shape[1] + 1)


def union(H):
    return float((H.sum(1) > 0).sum())


def chapman_pairs(H):
    """Mean over all occasion pairs of Chapman's Lincoln-Petersen estimate."""
    est = []
    for a, b in combinations(range(H.shape[1]), 2):
        n1, n2, m = H[:, a].sum(), H[:, b].sum(), (H[:, a] & H[:, b]).sum()
        est.append((n1 + 1) * (n2 + 1) / (m + 1) - 1)
    return float(np.mean(est))


def darroch_mt(H):
    """MLE for model M_t: solve prod_t (1 - n_t/N) = 1 - S/N for N > S."""
    S, n = union(H), H.sum(0).astype(float)
    if S == 0:
        return 0.0
    if n.sum() <= S:                          # no recaptures: N is unbounded
        return float("inf")
    if np.all(n == S):                        # every species caught every time
        return S
    # f -> +inf as N -> S and f -> 0- as N -> inf, with a single crossing.
    f = lambda N: np.sum(np.log1p(-n / N)) - np.log1p(-S / N)
    lo, hi = S * (1 + 1e-9) + 1e-9, 2 * S
    if f(lo) <= 0:                            # one occasion caught (almost) all of S
        return S
    while f(hi) > 0 and hi < 1e12:
        hi *= 2
    return float(brentq(f, lo, hi)) if f(hi) <= 0 else float("inf")


def chao_mh(H):
    """Chao (1987) lower bound for heterogeneous catchability (model M_h)."""
    K, Q = H.shape[1], freq_counts(H)
    q1, q2 = Q[1], Q[2]
    extra = q1 * q1 / (2 * q2) if q2 > 0 else q1 * (q1 - 1) / 2
    return float(union(H) + (K - 1) / K * extra)


def jackknife1(H):
    K, Q = H.shape[1], freq_counts(H)
    return float(union(H) + (K - 1) / K * Q[1])


def jackknife2(H):
    K, Q = H.shape[1], freq_counts(H)
    return float(union(H) + (2 * K - 3) / K * Q[1] - (K - 2) ** 2 / (K * (K - 1)) * Q[2])


def rarefy(H, t):
    """Expected number of species seen in t of the K occasions (Colwell et al. 2012)."""
    K, y = H.shape[1], H.sum(1)
    y = y[y > 0]
    lc = lambda n, r: gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1)
    miss = np.where(K - y >= t, np.exp(lc(K - y, t) - lc(K, t)), 0.0)
    return float(np.sum(1 - miss))


def extrapolate(H, t_more):
    """Expected species after K + t_more occasions (Colwell et al. 2012, Chao Q0)."""
    K, Q = H.shape[1], freq_counts(H)
    q1, q2 = Q[1], Q[2]
    q0 = (K - 1) / K * (q1 * q1 / (2 * q2) if q2 > 0 else q1 * (q1 - 1) / 2)
    if q0 == 0:
        return union(H)
    return float(union(H) + q0 * (1 - (1 - q1 / (K * q0 + q1)) ** t_more))


def mm_fit(H):
    """Asymptote of a Michaelis-Menten fit S(t) = a t / (b + t) to the rarefaction curve."""
    K = H.shape[1]
    t = np.arange(1, K + 1)
    s = np.array([rarefy(H, int(x)) for x in t])
    try:
        (a, b), _ = curve_fit(lambda x, a, b: a * x / (b + x), t, s, p0=[s[-1] * 1.5, 1.0],
                              maxfev=20000)
        return float(a), (float(a), float(b))
    except RuntimeError:
        return float("nan"), None


def mm_predict(params, t):
    a, b = params
    return a * t / (b + t)


def _trunc_fit(H, logpmf, x0, jmin):
    """ML fit of a catchability mixture to Q_jmin..Q_K only (singletons ignored when
    jmin = 2), then N = S_{>=jmin} / P(Y >= jmin)."""
    from scipy.optimize import minimize
    K, Q = H.shape[1], freq_counts(H)
    j = np.arange(jmin, K + 1)
    Sj = Q[jmin:].sum()
    if Sj == 0:
        return float("nan"), None, float("nan")

    def nll(th):
        lp = logpmf(np.arange(K + 1), K, th)
        tail = np.logaddexp.reduce(lp[jmin:])
        return -np.sum(Q[jmin:] * (lp[jmin:] - tail))

    best = min((minimize(nll, np.array(x), method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9)) for x in x0),
               key=lambda r: r.fun)
    lp = logpmf(np.arange(K + 1), K, best.x)
    p_tail = np.exp(np.logaddexp.reduce(lp[jmin:]))
    return float(Sj / p_tail), best.x, float(best.fun)


def _lchoose(n, r):
    return gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1)


def _betabin_logpmf(y, K, th):
    from scipy.special import betaln
    a, b = np.exp(th)
    return _lchoose(K, y) + betaln(y + a, K - y + b) - betaln(a, b)


BB_STARTS = [(0, 0), (-1, 0), (1, 1), (-2, -1)]


def _bin2_logpmf(y, K, th):
    """Two-class binomial mixture (Pledger 2000, model M_h2)."""
    from scipy.special import expit
    w, p1, p2 = expit(th)
    l1 = _lchoose(K, y) + y * np.log(p1) + (K - y) * np.log1p(-p1)
    l2 = _lchoose(K, y) + y * np.log(p2) + (K - y) * np.log1p(-p2)
    return np.logaddexp(np.log(w) + l1, np.log1p(-w) + l2)


def _mixc(c):
    """c-class binomial mixture with catch probabilities bounded below by p_min.
    Parameters: c-1 weight logits, then c probability logits."""
    from scipy.special import expit, softmax

    def logpmf(y, K, th, p_min):
        w = softmax(np.r_[0.0, th[:c - 1]])
        p = p_min + (1 - p_min) * expit(th[c - 1:]) * (1 - 1e-9)
        lps = [np.log(w[i]) + _lchoose(K, y) + y * np.log(p[i]) + (K - y) * np.log1p(-p[i])
               for i in range(c)]
        return np.logaddexp.reduce(np.array(lps), axis=0)
    return logpmf


SF_MODELS = {   # classes: (logpmf, starting points, number of parameters)
    1: (_mixc(1), [(0.0,), (3.0,), (-2.0,)], 1),
    2: (_mixc(2), [(0, -2, 3), (1, -3, 1), (-1, -1, 4), (2, -4, 4)], 3),
    3: (_mixc(3), [(0, 0, -3, 0, 3), (1, -1, -4, -1, 4), (-1, 1, -2, 1, 5)], 5),
}


def _sf_fit(H, p_min=None):
    """Fit 1-, 2- and 3-class mixtures to species caught >= 2 times; keep min AIC.

    Catch probabilities are bounded below by p_min (default 1/K: a feature counts if
    a seed finds it at least once in K seeds on average). Without the bound the
    one-truncated likelihood is flat in the mass placed near p = 0, so a handful of
    stray recaptures can send N to infinity (Link 2003); with it, N is identified."""
    K = H.shape[1]
    p_min = 1.0 / K if p_min is None else p_min
    best = None
    for c, (lp, x0, k) in SF_MODELS.items():
        f = lambda y, K_, th, lp=lp: lp(y, K_, th, p_min)
        N, th, nll = _trunc_fit(H, f, x0, jmin=2)
        aic = 2 * nll + 2 * k
        if np.isfinite(aic) and (best is None or aic < best[0]):
            best = (aic, N, th, f, c)
    return best


def singleton_free(H, return_choice=False, p_min=None):
    """Singleton-free estimate of the number of features with per-seed catch
    probability >= p_min (AIC-selected binomial mixture, Pledger 2000)."""
    best = _sf_fit(H, p_min)
    if best is None:
        return (float("nan"), None) if return_choice else float("nan")
    return (best[1], f"mix{best[4]}") if return_choice else best[1]


def sf_predict(H, K_new):
    """Predicted number of species caught >= 2 times if K_new seeds had been trained,
    from the singleton-free model fitted to H (a held-out check)."""
    best = _sf_fit(H)
    if best is None:
        return float("nan")
    _, N, th, f, _ = best
    lpk = f(np.arange(K_new + 1), K_new, th)
    return float(N * np.exp(np.logaddexp.reduce(lpk[2:])))


def betabin2(H):
    """Unbounded beta-binomial M_h fitted to species caught >= 2 times."""
    return _trunc_fit(H, _betabin_logpmf, BB_STARTS, jmin=2)[0]


def mix2_2(H):
    """Unbounded two-class binomial mixture fitted to species caught >= 2 times."""
    return _trunc_fit(H, _bin2_logpmf, [(0, -2, 2), (1, -3, 1), (-1, -1, 3)], jmin=2)[0]


def betabin1(H):
    """Beta-binomial M_h fitted to all observed species (uses singletons)."""
    return _trunc_fit(H, _betabin_logpmf, BB_STARTS, jmin=1)[0]


ESTIMATORS = {
    "union": union,
    "chapman": chapman_pairs,
    "darroch_mt": darroch_mt,
    "chao_mh": chao_mh,
    "jack1": jackknife1,
    "jack2": jackknife2,
    "mm": lambda H: mm_fit(H)[0],
    "betabin1": betabin1,
    "betabin2": betabin2,
    "mix2_2": mix2_2,
    "sf_aic": singleton_free,
}


def bootstrap_ci(H, est, n_boot=200, seed=0, level=0.95):
    """Chao-style bootstrap: pad H with the estimated unseen (all-zero) histories,
    resample species with replacement, re-estimate. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    N_hat = est(H)
    if not np.isfinite(N_hat):
        return float("nan"), float("nan")
    n0 = max(int(round(N_hat - len(H))), 0)
    P = np.vstack([H, np.zeros((n0, H.shape[1]), dtype=H.dtype)])
    vals = []
    for _ in range(n_boot):
        Hb = P[rng.integers(0, len(P), len(P))]
        vals.append(est(Hb[Hb.sum(1) > 0]))
    a = (1 - level) / 2
    return tuple(float(v) for v in np.quantile(vals, [a, 1 - a]))
