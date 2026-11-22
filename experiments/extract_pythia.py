"""Cache Pythia-70M residual-stream activations (after block LAYER) on Pile text.

    .venv/bin/python experiments/extract_pythia.py

Writes data/pythia/{train,eval}.npy (float16, scaled so E||x||^2 = d) and meta.json.
Train and eval come from disjoint documents. Train rows are shuffled once with a fixed seed. Position 0 of every context is
dropped (its norm is an outlier, as is standard in SAE training).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL, REV, LAYER, CTX = "EleutherAI/pythia-70m", "main", 3, 128
N_TRAIN, N_EVAL = 2_000_000, 65_536
OUT = ROOT / "data/pythia"


def contexts(tok, docs):
    buf = []
    for text in docs:
        buf.extend(tok(text)["input_ids"] + [tok.eos_token_id])
        while len(buf) >= CTX:
            yield buf[:CTX]
            buf = buf[CTX:]


@torch.no_grad()
def collect(model, tok, docs, n_tokens, bs=32):
    out, n, batch = np.empty((n_tokens, model.config.hidden_size), np.float16), 0, []
    for ctx in contexts(tok, docs):
        batch.append(ctx)
        if len(batch) == bs:
            hs = model(torch.tensor(batch), output_hidden_states=True).hidden_states[LAYER + 1]
            h = hs[:, 1:].reshape(-1, hs.shape[-1]).numpy()
            take = min(len(h), n_tokens - n)
            out[n:n + take] = h[:take]
            n += take
            batch = []
            if n >= n_tokens:
                return out
    raise RuntimeError(f"ran out of text at {n} tokens")


if __name__ == "__main__":
    if all((OUT / f).exists() for f in ("meta.json", "train.npy", "eval.npy")):
        print("cached:", OUT)
        sys.exit()
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32).eval()
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = ds["text"]
    ev = collect(model, tok, texts[:500], N_EVAL)          # eval: first 500 documents
    tr = collect(model, tok, texts[500:], N_TRAIN)         # train: the rest
    scale = float(np.sqrt((tr[:200_000].astype(np.float32) ** 2).sum(1).mean() / tr.shape[1]))
    # Shuffle tokens once (fixed order shared by all SAE seeds) so batches mix documents.
    tr = tr[np.random.default_rng(0).permutation(len(tr))]
    tr /= np.float16(scale)
    ev /= np.float16(scale)
    np.save(OUT / "train.npy", tr)
    np.save(OUT / "eval.npy", ev)
    json.dump(dict(model=MODEL, revision=REV, layer=LAYER, ctx=CTX, n_train=N_TRAIN,
                   n_eval=N_EVAL, scale=scale, dataset="NeelNanda/pile-10k",
                   eval_docs="0-499", train_docs="500-"), open(OUT / "meta.json", "w"), indent=1)
    print("saved", tr.shape, ev.shape, "scale", scale)
