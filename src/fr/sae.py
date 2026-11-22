"""TopK sparse autoencoder (Gao et al. 2024, arXiv:2406.04093) for CPU training.

Decoding uses embedding_bag, so the cost of the decoder scales with k rather than
with the dictionary size m.
"""
import numpy as np
import torch
import torch.nn.functional as F


class TopKSAE(torch.nn.Module):
    def __init__(self, d, m, k, seed, b_dec_init):
        super().__init__()
        # Offset keeps SAE-init streams disjoint from the synthetic-feature streams
        # (seed s would otherwise reproduce the planted features of data seed 0).
        g = torch.Generator().manual_seed(1_000_000_007 + int(seed))
        W = torch.randn(m, d, generator=g)
        W /= W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())                 # m x d, unit rows
        self.W_enc = torch.nn.Parameter(W.T.clone().contiguous())   # d x m, tied init
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(b_dec_init.clone())
        self.k = k

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def decode(self, v, i):
        return F.embedding_bag(i, self.W_dec, per_sample_weights=v, mode="sum") + self.b_dec

    def forward(self, x):
        pre = self.pre(x)
        v, i = pre.topk(self.k, dim=1)
        v = torch.relu(v)
        return self.decode(v, i), pre, v, i


def train_sae(batch_fn, n_steps, d, m, k, seed, lr=1e-3, x_eval=None,
              dead_tokens=200_000, k_aux=None, aux_coef=1 / 32):
    """Train one SAE on the stream batch_fn(step) -> [B, d].

    Every seed sees the same data in the same order; only the initialisation
    changes (the setting of Paulo & Belrose 2025). Returns a dict of numpy arrays.
    """
    x0 = batch_fn(0)
    B = x0.shape[0]
    sae = TopKSAE(d, m, k, seed, x0.mean(0))
    var_x = (x0 - x0.mean(0)).pow(2).sum(1).mean().item()
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    decay_from = int(0.8 * n_steps)
    since = torch.zeros(m)
    k_aux = k_aux or min(m // 2, 2 * k)
    for step in range(n_steps):
        x = x0 if step == 0 else batch_fn(step)
        xh, pre, v, i = sae(x)
        err = x - xh
        loss = err.pow(2).sum(1).mean() / var_x
        fired = torch.zeros(m, dtype=torch.bool)
        fired[i[v > 0]] = True
        since += B
        since[fired] = 0
        dead = since > dead_tokens
        n_dead = int(dead.sum())
        if n_dead > 0:
            idx = dead.nonzero().squeeze(1)
            vd, jd = pre[:, idx].topk(min(k_aux, n_dead), dim=1)
            eh = F.embedding_bag(idx[jd], sae.W_dec, per_sample_weights=torch.relu(vd), mode="sum")
            e = err.detach()
            loss = loss + aux_coef * (e - eh).pow(2).sum(1).mean() / e.pow(2).sum(1).mean().clamp_min(1e-12)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            g = sae.W_dec.grad
            g -= (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        if step >= decay_from:
            for pg in opt.param_groups:
                pg["lr"] = lr * (n_steps - step) / (n_steps - decay_from)
        opt.step()
        with torch.no_grad():
            sae.W_dec /= sae.W_dec.norm(dim=1, keepdim=True)
    return evaluate(sae, x_eval if x_eval is not None else x0)


@torch.no_grad()
def evaluate(sae, x_eval, chunk=8192):
    m = sae.W_dec.shape[0]
    counts = torch.zeros(m)
    sse, sst = 0.0, 0.0
    mu = x_eval.mean(0)
    for s in range(0, len(x_eval), chunk):
        x = x_eval[s:s + chunk]
        xh, _, v, i = sae(x)
        counts.index_add_(0, i[v > 0], torch.ones(int((v > 0).sum())))
        sse += (x - xh).pow(2).sum().item()
        sst += (x - mu).pow(2).sum().item()
    return dict(W_dec=sae.W_dec.detach().numpy().astype(np.float32),
                freq=(counts / len(x_eval)).numpy().astype(np.float32),
                fvu=np.float32(sse / sst))
