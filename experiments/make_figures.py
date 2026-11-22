"""Draw every figure from results/*.json (no training, no estimation here).

    .venv/bin/python experiments/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
TAU = "0.8"
# Categorical slots 1-4 of the validated default palette (light mode); identity is
# also carried by markers and direct labels, so colour is never the only cue.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK = "#8a8985", "#2b2b29"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.5, "lines.linewidth": 2, "lines.markersize": 4.5,
                     "legend.frameon": False, "savefig.dpi": 300})

SERIES = [  # key, label, colour, marker
    ("alive", "alive latents (one SAE)", GRAY, "s"),
    ("union", "distinct features over all seeds", ORANGE, "o"),
    ("chao_mh", "Chao lower bound", YELLOW, "^"),
    ("sf_aic", "singleton-free estimate (ours)", BLUE, "D"),
]


def load(name):
    p = ROOT / "results" / name
    return json.load(open(p)) if p.exists() else None


def value(fam, key, src="blind"):
    return fam["alive"] if key == "alive" else fam[src][TAU][key]


def synth_ratios(S, key, src="blind"):
    """{m/N: [N_hat/N per family]} over all synthetic families."""
    out = {}
    for fam in S.values():
        c = fam["config"]
        v = value(fam, key, src)
        out.setdefault(c["m"] / c["N"], []).append(np.nan if v is None else v / c["N"])
    return dict(sorted(out.items()))


def hero(S, R):
    fig, axes = plt.subplots(1, 2 if R else 1, figsize=(6.75 if R else 3.4, 2.7))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    for key, lab, col, mk in SERIES:
        d = synth_ratios(S, key)
        x = np.array(list(d))
        med = np.array([np.nanmedian(v) for v in d.values()])
        lo = np.array([np.nanmin(v) for v in d.values()])
        hi = np.array([np.nanmax(v) for v in d.values()])
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(x, med, color=col, marker=mk, label=lab)
    ax.axhline(1, color=INK, ls=(0, (3, 2)), lw=1)
    ax.text(0.26, 1.12, "planted features", color=INK, fontsize=7)
    ax.set(xscale="log", yscale="log", xlabel="dictionary size / planted features",
           ylabel="count / planted features", title="(a) synthetic: answer known")
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"])
    ax.minorticks_off()
    if R:
        ax = axes[1]
        fams = sorted(R.values(), key=lambda f: f["m"])
        m = np.array([f["m"] for f in fams])
        for key, lab, col, mk in SERIES:
            y = np.array([np.nan if value(f, key) is None else value(f, key) for f in fams], float)
            ax.plot(m, y, color=col, marker=mk, label=lab)
            if key == "sf_aic":
                ci = np.array([f["blind"][TAU].get("sf_aic_ci") or [np.nan, np.nan] for f in fams], float)
                ax.fill_between(m, ci[:, 0], ci[:, 1], color=col, alpha=0.2, lw=0)
        ax.set(xscale="log", yscale="log", xlabel="dictionary size $m$", ylabel="count",
               title="(b) Pythia-70M, layer 3: answer unknown")
        ax.set_xticks(m, [f"{v // 1024}k" for v in m])
        ax.minorticks_off()
    axes[0].legend(loc="upper left", fontsize=6.5, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_hero.pdf")
    fig.savefig(FIG / "fig1_hero.png")
    plt.close(fig)


def nm(r):
    return {0.5: "N/2", 1.0: "N", 2.0: "2N"}[r]


def mechanism(S):
    """(a) capture frequencies at m = N and m = 2N, split into planted vs junk species;
    (b) oracle catch rate against feature frequency rank (heterogeneous catchability)."""
    pick = {r: next(f for f in S.values() if f["config"]["N"] == 1024 and f["config"]["data_seed"] == 0
                    and f["config"]["m"] == int(1024 * r)) for r in (0.5, 1.0, 2.0)}
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4))
    ax = axes[0]
    w = 0.38
    for off, r, col in ((-w / 2, 1.0, BLUE), (w / 2, 2.0, ORANGE)):
        b = pick[r]["blind"][TAU]
        Q = np.array(b["Q"], float)
        j = np.arange(1, len(Q) + 1)
        junk = np.array([b["junk_by_count"].get(str(i), 0.0) for i in j])
        ax.bar(j + off, Q * (1 - junk), w, color=col, label=f"m = {nm(r)}: matches a planted feature")
        ax.bar(j + off, Q * junk, w, bottom=Q * (1 - junk), color=col, alpha=0.35, hatch="////",
               edgecolor="white", lw=0, label=f"m = {nm(r)}: junk")
    ax.set_ylim(0.8, 4e4)
    ax.set(yscale="log", xlabel="number of seeds (of 8) that found the feature",
           ylabel="number of features", title="(a) who gets counted once")
    ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.58, 1.0), handlelength=1.2)
    ax = axes[1]
    for r, col, mk in ((0.5, ORANGE, "o"), (1.0, BLUE, "D"), (2.0, AQUA, "^")):
        cr = np.array(pick[r]["oracle"][TAU]["catch_rate"])
        rank = np.arange(1, len(cr) + 1)
        bins = np.unique(np.geomspace(1, len(cr), 14).astype(int))
        xs = [np.sqrt(a * b) for a, b in zip(bins[:-1], bins[1:])]
        ys = [cr[a - 1:b - 1].mean() for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(xs, ys, color=col, marker=mk, label=f"m = {nm(r)}")
    ax.set(xscale="log", xlabel="planted feature, ranked by how often it fires",
           ylabel="fraction of seeds that catch it", title="(b) rare features are hard to catch", ylim=(-0.03, 1.03))
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_mechanism.pdf")
    plt.close(fig)


def heldout(HS, HR, S):
    """Relative error of predicting 8-seed reproducible-feature counts from 4 seeds."""
    groups = [("dictionary\n< features", {k: v for k, v in HS.items() if k in S
                                         and S[k]["config"]["m"] < S[k]["config"]["N"]}),
              ("dictionary\n$\\geq$ features", {k: v for k, v in HS.items() if k in S
                                                and S[k]["config"]["m"] >= S[k]["config"]["N"]})]
    if HR:
        groups.append(("Pythia-70M", HR))
    meth = (("no_growth", "no growth", GRAY, "s"), ("m0", "equal catchability", YELLOW, "^"),
            ("sf_aic", "singleton-free (ours)", BLUE, "D"))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    rng = np.random.default_rng(0)
    for gi, (_, H) in enumerate(groups):
        for mi, (key, lab, col, mk) in enumerate(meth):
            e = [100 * (np.nanmean([r[key] for r in v["draws"]]) / v["target"] - 1) for v in H.values()]
            x = gi + (mi - 1) * 0.25 + rng.uniform(-0.05, 0.05, len(e))
            ax.scatter(x, e, color=col, marker=mk, s=16, label=lab if gi == 0 else None, zorder=3,
                       edgecolor="white", lw=0.4)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set(ylabel="prediction error (%)", title="Predicting 4 held-out seeds")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_heldout.pdf")
    fig.savefig(FIG / "fig3_heldout.png")
    plt.close(fig)


def hier_levels(fam):
    """Planted features caught by at least one seed (oracle), as (children, parents)."""
    cr = np.array(fam["oracle"][TAU]["catch_rate"])
    N = fam["config"]["N"]
    return int((cr[:N] > 0).sum()), int((cr[N:] > 0).sum())


def hierarchy(SH):
    """Features that split: (a) census against dictionary size; (b) which level the seeds learn."""
    fams = list(SH.values())
    ms = np.array(sorted({f["m"] for f in fams}))
    N = fams[0]["config"]["N"]
    P = N // fams[0]["config"]["hier"]

    def per_m(fn):   # mean, min, max over data draws
        v = np.array([[fn(f) for f in fams if f["m"] == m] for m in ms], float)
        return v.mean(1), v.min(1), v.max(1)

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
    ax = axes[0]
    for key, lab, col, mk in SERIES:
        if key == "chao_mh":
            continue
        mean, lo, hi = per_m(lambda f: value(f, key))
        ax.plot(ms, mean, color=col, marker=mk, label=lab)
        if key == "sf_aic":
            lo = per_m(lambda f: f["blind"][TAU]["sf_aic_ci"][0])[1]
            hi = per_m(lambda f: f["blind"][TAU]["sf_aic_ci"][1])[2]
        ax.fill_between(ms, lo, hi, color=col, alpha=0.18, lw=0)
    for y in (N, P):
        ax.axhline(y, color=INK, ls=(0, (3, 2)), lw=1)
    ax.text(ms[0], N * 1.1, f"{N:,} child features", color=INK, fontsize=6.5)
    ax.text(ms[-1], P * 0.9, f"{P} parent features", color=INK, fontsize=6.5, ha="right", va="top")
    ax.set(xscale="log", yscale="log", xlabel="dictionary size $m$", ylabel="count",
           title="(a) the census when features split")
    ax.set_xticks(ms, [str(m) for m in ms])
    yt = [128, 256, 512, 1024, 2048, 4096]
    ax.set_yticks(yt, [f"{t:,}" for t in yt])
    ax.minorticks_off()
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.93), handlelength=1.6)
    ax = axes[1]
    for i, lab, col, mk in ((0, "children caught", AQUA, "o"), (1, "parents caught", YELLOW, "^")):
        mean, lo, hi = per_m(lambda f: hier_levels(f)[i])
        ax.plot(ms, mean, color=col, marker=mk, label=lab)
        ax.fill_between(ms, lo, hi, color=col, alpha=0.18, lw=0)
    mean, lo, hi = per_m(lambda f: sum(hier_levels(f)))
    ax.plot(ms, mean, color=INK, ls=(0, (1, 1.5)), lw=1.5, label="either level")
    for y in (N, P):
        ax.axhline(y, color=INK, ls=(0, (3, 2)), lw=1)
    ax.text(ms[0], N + 25, f"{N:,} children", color=INK, fontsize=6.5)
    ax.text(ms[-1], P + 25, f"{P} parents", color=INK, fontsize=6.5, ha="right")
    ax.set(xscale="log", xlabel="dictionary size $m$", ylabel="planted features\ncaught by any seed",
           title="(b) which level the seeds learn (oracle)")
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.minorticks_off()
    ax.legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_hierarchy.pdf")
    fig.savefig(FIG / "fig4_hierarchy.png")
    plt.close(fig)


if __name__ == "__main__":
    S, R = load("summary_synth.json"), load("summary_real.json")
    SH = load("summary_synth_hier.json")
    if SH:
        hierarchy(SH)
    if R:   # the main Pythia sweep; the 5x-data robustness family is reported in the text
        R = {k: v for k, v in R.items() if v["config"].get("data", "pythia") == "pythia"}
    hero(S, R)
    mechanism(S)
    HS, HR = load("heldout_synth.json"), load("heldout_real.json")
    if HR:
        HR = {k: v for k, v in HR.items() if "pythia_long" not in k}
    if HS:
        heldout(HS, HR, {**S, **(R or {})})
    print("figures written to", FIG)
