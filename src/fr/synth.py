"""Synthetic activations with a planted, known set of features.

x = sum_i a_i f_i, where feature i is active independently with probability p_i
(Zipf-like: p_i proportional to i^-alpha) and has magnitude a_i ~ U(lo, hi).
The number of true features N is the quantity the estimators must recover.
HierSynthData plants features that split: parents whose children share a direction.
"""
import numpy as np
import torch


class SynthData:
    def __init__(self, N, d, alpha=0.7, mean_active=8.0, data_seed=0, mag=(0.5, 1.5),
                 batch=4096):
        g = torch.Generator().manual_seed(10_007 * data_seed + 1)
        Fe = torch.randn(N, d, generator=g)
        self.F = Fe / Fe.norm(dim=1, keepdim=True)
        w = torch.arange(1, N + 1, dtype=torch.float64) ** (-alpha)
        self.p = (w / w.sum() * mean_active).clamp(max=1.0).float()
        self.N, self.d, self.mag, self.B, self.seed = N, d, mag, batch, data_seed

    @property
    def refs(self):
        """Directions oracle matching compares latents with."""
        return self.F

    def batch(self, step, stream=0, B=None):
        B = B or self.B
        g = torch.Generator().manual_seed(1_000_003 * self.seed + 7_919 * stream + step + 17)
        on = torch.rand(B, self.N, generator=g) < self.p
        lo, hi = self.mag
        a = on * (lo + (hi - lo) * torch.rand(B, self.N, generator=g))
        return a @ self.F

    def eval_set(self, n=32_768):
        return torch.cat([self.batch(s, stream=1) for s in range(n // self.B)])

    def materialize(self, n_batches, path):
        """Write the training stream to an .npy file once; every seed reuses it.
        (Sampling, not training, is the CPU bottleneck for these sizes.)"""
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            X = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+",
                                          dtype=np.float32, shape=(n_batches * self.B, self.d))
            for s in range(n_batches):
                X[s * self.B:(s + 1) * self.B] = self.batch(s).numpy()
            X.flush()
            del X
            path.with_suffix(".tmp.npy").rename(path)
        return np.load(path, mmap_mode="r")


class HierSynthData(SynthData):
    """Features that split: N/C parents, each with C children.

    Child j of parent i points along cos(theta) f_i + sin(theta) u_ij, where f_i and the
    u_ij are orthonormal, so a parent's children are equiangular (cosine cos^2 theta). Parent i is present with Zipf-like probability p_i, and
    when it is, exactly one of its children (chosen uniformly) is the active feature.
    A dictionary with one latent per parent can only learn the parent's average
    direction; one with a latent per child can learn the children. F holds the N child
    features, P the parent references (the normalised mean of each parent's children).
    """

    def __init__(self, N, d, n_children=4, child_cos=0.6, alpha=0.7, mean_active=8.0,
                 data_seed=0, mag=(0.5, 1.5), batch=4096):
        assert N % n_children == 0
        P = N // n_children
        g = torch.Generator().manual_seed(10_007 * data_seed + 2)
        Q = torch.linalg.qr(torch.randn(P, d, n_children + 1, generator=g)).Q
        f, u = Q[..., 0], Q[..., 1:].transpose(1, 2)
        self.F = (child_cos * f[:, None] + (1 - child_cos ** 2) ** 0.5 * u).reshape(N, d)
        mean = self.F.reshape(P, n_children, d).mean(1)
        self.P = mean / mean.norm(dim=1, keepdim=True)
        w = torch.arange(1, P + 1, dtype=torch.float64) ** (-alpha)
        self.p = (w / w.sum() * mean_active).clamp(max=1.0).float()   # per parent
        self.N, self.d, self.mag, self.B, self.seed = N, d, mag, batch, data_seed
        self.n_parents, self.C = P, n_children

    @property
    def refs(self):
        """Children first, then parents: oracle matching at both levels."""
        return torch.cat([self.F, self.P])

    def coeffs(self, step, stream=0, B=None):
        B = B or self.B
        g = torch.Generator().manual_seed(1_000_003 * self.seed + 7_919 * stream + step + 17)
        on = torch.rand(B, self.n_parents, generator=g) < self.p
        lo, hi = self.mag
        a = on * (lo + (hi - lo) * torch.rand(B, self.n_parents, generator=g))
        child = torch.randint(0, self.C, (B, self.n_parents), generator=g)
        idx = torch.arange(self.n_parents) * self.C + child
        return torch.zeros(B, self.N).scatter_(1, idx, a)

    def batch(self, step, stream=0, B=None):
        return self.coeffs(step, stream, B) @ self.F


def make_data(c):
    """The synthetic dataset a run config describes."""
    kw = dict(N=c["N"], d=c["d"], alpha=c["alpha"], mean_active=c["mean_active"],
              data_seed=c["data_seed"])
    if c.get("hier"):
        return HierSynthData(n_children=c["hier"], child_cos=c["child_cos"], **kw)
    return SynthData(**kw)
