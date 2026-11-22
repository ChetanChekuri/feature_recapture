"""Training-length robustness: cache 5x more Pythia-70M activations (10M tokens).

    .venv/bin/python experiments/extract_pythia_long.py

Same model, layer, context length, documents (500 onward) and scale constant as
extract_pythia.py, so only the amount of training data changes. Differences, made
for memory and speed: contexts (not tokens) are shuffled once with a fixed seed and
rows are streamed to disk; the forward pass stops after block LAYER (read with a
hook, which is checked to match the main cache bit-for-bit up to float16).
Writes data/pythia_long/train.npy and meta.json; evaluation reuses data/pythia/eval.npy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pythia import CTX, LAYER, MODEL, REV, contexts   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_TRAIN = 10_000_000
OUT = ROOT / "data/pythia_long"


@torch.no_grad()
def main():
    if (OUT / "meta.json").exists() and (OUT / "train.npy").exists():
        print("cached:", OUT)
        return
    torch.set_num_threads(16)
    OUT.mkdir(parents=True, exist_ok=True)
    main_meta = json.load(open(ROOT / "data/pythia/meta.json"))
    scale = np.float32(main_meta["scale"])
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32).eval()
    grab = {}
    model.gpt_neox.layers[LAYER].register_forward_hook(lambda m, i, o: grab.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    model.gpt_neox.layers = model.gpt_neox.layers[:LAYER + 1]

    def run(batch):
        model.gpt_neox(torch.tensor(batch))
        return grab["h"][:, 1:].reshape(-1, grab["h"].shape[-1]).numpy()

    texts = load_dataset("NeelNanda/pile-10k", split="train")["text"]
    # check the hook against the main cache (first evaluation batch, documents 0-499)
    ev = np.load(ROOT / "data/pythia/eval.npy", mmap_mode="r")
    first = [c for _, c in zip(range(32), contexts(tok, texts[:500]))]
    h = (run(first) / scale).astype(np.float16)
    a, b = h.astype(np.float32), ev[:len(h)].astype(np.float32)
    err = float((np.abs(a - b) / (np.abs(b) + 1)).max())    # float16 rounding on large coords
    print("hook vs main cache: max rel diff", err, "max abs", float(np.abs(a - b).max()),
          "max |x|", float(np.abs(b).max()))
    assert err < 0.01, err

    ctxs = list(contexts(tok, texts[500:]))
    per = CTX - 1
    need = -(-N_TRAIN // per)
    if len(ctxs) < need:
        raise RuntimeError(f"only {len(ctxs) * per} tokens available")
    order = np.random.default_rng(0).permutation(len(ctxs))[:need]
    X = np.lib.format.open_memmap(OUT / "train.tmp.npy", mode="w+", dtype=np.float16,
                                  shape=(N_TRAIN, model.config.hidden_size))
    n = 0
    for s in range(0, need, 64):
        h = run([ctxs[i] for i in order[s:s + 64]]) / scale
        take = min(len(h), N_TRAIN - n)
        X[n:n + take] = h[:take]
        n += take
    X.flush()
    del X
    (OUT / "train.tmp.npy").rename(OUT / "train.npy")
    json.dump(dict(main_meta, n_train=N_TRAIN, shuffle="contexts", hook_check_maxdiff=err,
                   available_tokens=len(ctxs) * per), open(OUT / "meta.json", "w"), indent=1)
    print("saved", N_TRAIN, "tokens")


if __name__ == "__main__":
    main()
