"""Write paper/numbers.tex (macros) and paper/tab_*.tex from results/*.json.

Every number in the paper comes from here; nothing is typed by hand.

    .venv/bin/python paper/make_numbers.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys                                  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import analyze                              # noqa: E402
import run_real                             # noqa: E402
import run_synth                            # noqa: E402

TAU = "0.8"
macros = {}


def load(name):
    p = ROOT / "results" / name
    return json.load(open(p)) if p.exists() else None


def put(name, val):
    macros[name] = val


def fmt_int(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


def times(x):
    if x >= 1000:
        return fmt_int(x) + "\\times"
    return f"{x:.0f}\\times" if x >= 10 else f"{x:.1f}\\times"


# ---------------------------------------------------------------- configuration
B = run_synth.BASE
put("MainTau", TAU)
put("TauLo", "0.7")   # the thresholds analysed in src/fr/analysis.py (family_summary taus)
put("TauHi", "0.9")
put("BootN", str(analyze.BOOT_N))
put("SynD", str(B["d"]))
put("SynAlpha", f"{B['alpha']:g}")
put("SynMeanActive", f"{B['mean_active']:g}")
put("SynK", str(B["k"]))
put("SynSeeds", str(len(run_synth.grid("main")[1])))
put("SynSteps", fmt_int(B["steps"]))
meta = json.load(open(ROOT / "data/pythia/meta.json")) if (ROOT / "data/pythia/meta.json").exists() else {}
put("RealLayer", str(meta.get("layer", 3)))
put("RealTrainTokens", fmt_int(meta.get("n_train", 0)))
put("RealEvalTokens", fmt_int(meta.get("n_eval", 0)))
put("RealK", str(run_real.BASE["k"]))
put("RealSeeds", str(len(run_real.grid("main")[1])))

# ---------------------------------------------------------------- synthetic
S = load("summary_synth.json")
S = {k: v for k, v in S.items() if v["config"]["d"] == B["d"]}
put("SynFamilies", str(len(S)))
put("SynRuns", fmt_int(sum(v["K"] for v in S.values())))
RATIOS = sorted({v["config"]["m"] / v["config"]["N"] for v in S.values()})


def ratios(key, r, src="blind"):
    out = []
    for v in S.values():
        if v["config"]["m"] / v["config"]["N"] != r:
            continue
        x = v["alive"] if key == "alive" else v[src][TAU][key]
        out.append(np.nan if x is None else x / v["config"]["N"])
    return np.array(out)


def ratio_cell(key, r, src="blind"):
    """Median of N_hat/N over settings; superscript = settings where it diverged."""
    x = ratios(key, r, src)
    if len(x) == 0:
        return "--"
    nd = int(np.isnan(x).sum())
    if nd == len(x):
        return "div."
    med = np.nanmedian(x)
    cell = f"{med:.2f}" if med < 10 else (f"{med:.0f}" if med < 1000 else fmt_int(med))
    return cell + (f"$^{{({nd})}}$" if nd else "")


ROWS = [("alive", "alive latents (one SAE)", "blind"), ("union", "distinct features, all seeds", "blind"),
        ("mm", "Michaelis--Menten asymptote", "blind"), ("chapman", "Lincoln--Petersen (Chapman)", "blind"),
        ("darroch_mt", "Darroch $M_t$", "blind"), ("chao_mh", "Chao $M_h$ bound", "blind"),
        ("jack2", "jackknife (2nd order)", "blind"), ("betabin1", "beta-binomial, all species", "blind"),
        ("betabin2", "\\quad singletons dropped, unbounded", "blind"),
        ("mix2_2", "two-class, singletons dropped, unbounded", "blind"),
        ("sf_aic", "\\textbf{singleton-free (ours)}", "blind"), ("sf_aic", "singleton-free, oracle IDs", "oracle"),
        ("union", "oracle: reachable (union, oracle IDs)", "oracle")]
put("NEstimators", str(sum(1 for _, _, src in ROWS if src == "blind")))
lab = {0.25: "$N/4$", 0.5: "$N/2$", 1.0: "$N$", 2.0: "$2N$", 4.0: "$4N$"}
lines = ["\\begin{tabular}{l" + "r" * len(RATIOS) + "}", "\\toprule",
         "estimator & \\multicolumn{%d}{c}{dictionary size $m$} \\\\" % len(RATIOS),
         " & " + " & ".join(lab[r] for r in RATIOS) + " \\\\", "\\midrule"]
for key, name, src in ROWS:
    lines.append(name + " & " + " & ".join(ratio_cell(key, r, src) for r in RATIOS) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper/tab_synth.tex").write_text("\n".join(lines) + "\n")

complete = [r for r in RATIOS if r >= 1]
sf_c = np.concatenate([ratios("sf_aic", r) for r in complete])
put("SfMaxErrComplete", pct(np.nanmax(np.abs(sf_c - 1)), 1))
put("SfDivComplete", str(int(np.isnan(sf_c).sum())))
put("NComplete", str(len(sf_c)))
put("UnionRatioEq", f"{np.nanmedian(ratios('union', 1.0)):.2f}")
put("UnionRatioMax", f"{np.nanmax(np.concatenate([ratios('union', r) for r in complete])):.1f}")
put("ChaoRatioEq", times(np.nanmedian(ratios("chao_mh", 1.0))))
put("ChaoRatioMax", times(np.nanmax(np.concatenate([ratios("chao_mh", r) for r in complete]))))
put("AliveRatioTwo", f"{np.nanmedian(ratios('alive', 2.0)):.2f}")
incomplete = [r for r in RATIOS if r < 1]
sf_i = np.concatenate([ratios("sf_aic", r) for r in incomplete])
put("NIncomplete", str(len(sf_i)))
put("SfDivIncomplete", str(int(np.isnan(sf_i).sum())))
put("OracleUnionHalf", pct(np.nanmedian(ratios("union", 0.5, "oracle"))))
put("OracleUnionQuarter", pct(np.nanmedian(ratios("union", 0.25, "oracle"))))

# reachable: in m < N, compare with the oracle union (planted features any seed caught)
reach = []
for v in S.values():
    if v["config"]["m"] < v["config"]["N"]:
        b, o = v["blind"][TAU]["sf_aic"], v["oracle"][TAU]["union"]
        reach.append(np.nan if b is None else b / o)
reach = np.array(reach)
put("SfReachMed", f"{np.nanmedian(reach):.2f}")
put("SfReachLo", f"{np.nanmin(reach):.2f}")
put("SfReachHi", f"{np.nanmax(reach):.2f}")
plateau = []
for v in S.values():
    c = v["config"]
    if c["m"] == 2 * c["N"]:
        base = next(w for w in S.values() if w["config"]["N"] == c["N"] and w["config"]["m"] == c["N"]
                    and w["config"]["data_seed"] == c["data_seed"])
        plateau.append(v["blind"][TAU]["sf_aic"] / base["blind"][TAU]["sf_aic"])
put("PlateauMed", f"{np.median(plateau):.2f}")


def H_from_Q(Q):
    K = len(Q)
    return np.array([np.r_[np.ones(j, np.int8), np.zeros(K - j, np.int8)]
                     for j, q in enumerate(Q, 1) for _ in range(q)])


from fr import estimators as E   # noqa: E402
rob = {}
for tau, nm in (("0.7", "TauLo"), ("0.9", "TauHi")):
    errs = [abs(v["blind"][tau]["sf_aic"] / v["config"]["N"] - 1) for v in S.values()
            if v["config"]["m"] >= v["config"]["N"] and v["blind"][tau]["sf_aic"] is not None]
    put(f"SfMaxErrComplete{nm}", pct(max(errs), 1))
errs = [abs(E.singleton_free(H_from_Q(v["blind"][TAU]["Q"]), p_min=0.5 / v["K"]) / v["config"]["N"] - 1)
        for v in S.values() if v["config"]["m"] >= v["config"]["N"]]
put("SfMaxErrCompleteHalfPmin", pct(max(errs), 1))

junk1 = {r: [v["blind"][TAU]["junk_by_count"].get("1", np.nan) for v in S.values()
             if v["config"]["m"] / v["config"]["N"] == r] for r in RATIOS}
put("JunkSingleEq", pct(np.nanmedian(junk1[1.0])))
put("JunkSingleOver", pct(np.nanmedian(sum((junk1[r] for r in RATIOS if r > 1), []))))
junk2 = [x for v in S.values() if v["config"]["m"] >= v["config"]["N"]
         for j, x in v["blind"][TAU]["junk_by_count"].items() if int(j) >= 2
         and v["blind"][TAU]["Q"][int(j) - 1] >= 20]
put("JunkMultiMax", pct(max(junk2)))

# ---------------------------------------------------------------- held-out
def heldout_err(H, key, keep=lambda name: True):
    e = []
    for name, v in H.items():
        if keep(name):
            p = np.nanmean([r[key] for r in v["draws"]])
            e.append(abs(p / v["target"] - 1))
    return np.array(e)


def regime(name):
    c = S[name]["config"]
    return "Small" if c["m"] < c["N"] else "Big"


HSyn = load("heldout_synth.json")
HSyn = {k: v for k, v in HSyn.items() if k in S}
for reg in ("Small", "Big"):
    for key, nm in (("no_growth", "NoGrowth"), ("m0", "MZero"), ("sf_aic", "Sf")):
        e = heldout_err(HSyn, key, lambda n: regime(n) == reg)
        put(f"HeldSyn{reg}{nm}Med", pct(np.nanmedian(e), 1))
        put(f"HeldSyn{reg}{nm}Max", pct(np.nanmax(e), 1))
    put(f"HeldSyn{reg}N", str(sum(regime(n) == reg for n in HSyn)))
HReal = load("heldout_real.json")
if HReal:
    HReal = {k: v for k, v in HReal.items() if "pythia_long" not in k}
    for key, nm in (("no_growth", "NoGrowth"), ("m0", "MZero"), ("sf_aic", "Sf")):
        e = heldout_err(HReal, key)
        put(f"HeldReal{nm}Med", pct(np.nanmedian(e), 1))
        put(f"HeldReal{nm}Max", pct(np.nanmax(e), 1))

# ---------------------------------------------------------------- features that split
SH = load("summary_synth_hier.json")
if SH:
    c0 = next(iter(SH.values()))["config"]
    Nh, C, cc = c0["N"], c0["hier"], c0["child_cos"]
    put("HierChildren", fmt_int(Nh))
    put("HierParents", fmt_int(Nh // C))
    put("HierC", str(C))
    put("HierSiblingCos", f"{cc ** 2:.2f}")
    put("HierParentCos", f"{np.sqrt(cc ** 2 + (1 - cc ** 2) / C):.2f}")
    put("HierDraws", str(len({v["config"]["data_seed"] for v in SH.values()})))
    put("HierRuns", str(sum(v["K"] for v in SH.values())))
    put("HierMMin", fmt_int(min(v["m"] for v in SH.values())))
    put("HierMMax", fmt_int(max(v["m"] for v in SH.values())))

    def levels(v):
        """Planted children and parents caught by at least one seed (oracle)."""
        cr = np.array(v["oracle"][TAU]["catch_rate"])
        return int((cr[:Nh] > 0).sum()), int((cr[Nh:] > 0).sum())

    def sf(v, tau=TAU):
        x = v["blind"][tau]["sf_aic"]
        return np.nan if x is None else x

    for m, nm in {Nh // 8: "Eighth", Nh // 4: "Quarter", Nh // 2: "Half", Nh: "Full",
                  2 * Nh: "Double"}.items():
        fs = [v for v in SH.values() if v["m"] == m]
        if fs:
            put(f"HierSf{nm}", fmt_int(np.mean([sf(v) for v in fs])))
            put(f"HierUnion{nm}", fmt_int(np.mean([v["blind"][TAU]["union"] for v in fs])))
            put(f"HierKids{nm}", fmt_int(np.mean([levels(v)[0] for v in fs])))
            put(f"HierPars{nm}", fmt_int(np.mean([levels(v)[1] for v in fs])))
            put(f"HierReach{nm}", fmt_int(np.mean([sum(levels(v)) for v in fs])))
    big = [v for v in SH.values() if v["m"] >= Nh]
    small = [v for v in SH.values() if v["m"] < Nh]
    full = [v for v in big if v["m"] == Nh]
    double = [v for v in big if v["m"] == 2 * Nh]
    put("HierSfErrFull", pct(max(abs(sf(v) / Nh - 1) for v in full), 1))
    put("HierSfErrDouble", pct(max(abs(sf(v, t) / Nh - 1) for v in double for t in v["blind"]), 1))
    for tau, nm in (("0.7", "TauLo"), ("0.9", "TauHi")):
        put(f"HierSfFull{nm}", fmt_int(np.mean([sf(v, tau) for v in full])))
    r = [sf(v) / sum(levels(v)) for v in small]
    put("HierSfReachMed", f"{np.nanmedian(r):.2f}")
    put("HierSfReachMin", f"{np.nanmin(r):.2f}")
    put("HierSfReachMax", f"{np.nanmax(r):.2f}")
    put("HierUnionRatioMax", f"{max(v['blind'][TAU]['union'] for v in SH.values()) / Nh:.1f}")
    put("HierJunkSingle", pct(np.nanmedian([v["blind"][TAU]["junk_by_count"].get("1", np.nan)
                                             for v in big])))
    put("HierDiverged", str(sum(v["blind"][TAU]["sf_aic"] is None for v in SH.values())))
    HH = load("heldout_synth_hier.json")
    if HH:
        for key, nm in (("no_growth", "NoGrowth"), ("sf_aic", "Sf")):
            for reg, keep in (("Small", lambda n: SH[n]["m"] < Nh), ("Big", lambda n: SH[n]["m"] >= Nh)):
                e = heldout_err(HH, key, keep)
                put(f"HeldHier{reg}{nm}Med", pct(np.nanmedian(e), 1))
                put(f"HeldHier{reg}{nm}Max", pct(np.nanmax(e), 1))

# ---------------------------------------------------------------- Pythia-70M
R = load("summary_real.json")
if R:
    LONG = [f for f in R.values() if f["config"].get("data", "pythia") != "pythia"]
    fams = sorted([f for f in R.values() if f["config"].get("data", "pythia") == "pythia"],
                  key=lambda f: f["m"])
    rows = ["\\begin{tabular}{rrrrrrrl}", "\\toprule",
            "$m$ & FVU & alive & distinct, all seeds & seen $\\ge 2$ & Chao $M_h$ & singleton-free & 95\\% CI \\\\",
            "\\midrule"]
    for f in fams:
        b = f["blind"][TAU]
        s2 = sum(b["Q"][1:])
        sf = b["sf_aic"]
        ci = b.get("sf_aic_ci") or [None, None]
        cis = "--" if None in ci else f"[{fmt_int(ci[0])}, {fmt_int(ci[1])}]"
        rows.append(f"{fmt_int(f['m'])} & {f['fvu']:.3f} & {fmt_int(f['alive'])} & {fmt_int(b['union'])} & "
                    f"{fmt_int(s2)} & {fmt_int(b['chao_mh']) if b['chao_mh'] else 'div.'} & "
                    f"{fmt_int(sf) if sf else 'div.'} & {cis} \\\\")
        k = f"{f['m'] // 1024}k"
        tagm = {"1k": "One", "2k": "Two", "4k": "Four", "8k": "Eight"}[k]
        put(f"RealSf{tagm}", fmt_int(sf) if sf else "\\text{div.}")
        put(f"RealUnion{tagm}", fmt_int(b["union"]))
        put(f"RealSeenTwo{tagm}", fmt_int(s2))
        put(f"RealFvu{tagm}", f"{f['fvu']:.3f}")
        put(f"RealChoice{tagm}", b["sf_choice"])
        if None not in ci:
            put(f"RealSfLo{tagm}", fmt_int(ci[0]))
            put(f"RealSfHi{tagm}", fmt_int(ci[1]))
    rows += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tab_real.tex").write_text("\n".join(rows) + "\n")
    trow = ["\\begin{tabular}{rrrrr}", "\\toprule",
            "$m$ & $\\tau=0.7$ & $\\tau=0.8$ & $\\tau=0.9$ & seen by all seeds ($\\tau=0.8$) \\\\", "\\midrule"]
    for f in fams:
        cells = [fmt_int(f["blind"][t]["sf_aic"]) if f["blind"][t]["sf_aic"] else "div." for t in ("0.7", "0.8", "0.9")]
        trow.append(f"{fmt_int(f['m'])} & " + " & ".join(cells) + f" & {fmt_int(f['blind'][TAU]['Q'][-1])} \\\\")
    trow += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tab_real_tau.tex").write_text("\n".join(trow) + "\n")
    sf = {f["m"]: f["blind"][TAU]["sf_aic"] for f in fams}
    for (a, b), nm in (((1024, 2048), "One"), ((2048, 4096), "Two"), ((4096, 8192), "Four")):
        put(f"RealGrowth{nm}", f"{sf[b] / sf[a]:.2f}")
    put("RealUnionGrowth", f"{fams[-1]['blind'][TAU]['union'] / fams[0]['blind'][TAU]['union']:.1f}")
    big = fams[-1]
    put("RealReproFrac", pct(big["blind"][TAU]["sf_aic"] / big["alive"]))
    put("RealSingleFrac", pct(big["blind"][TAU]["Q"][0] / big["blind"][TAU]["union"]))
    put("RealAllSeedsEight", fmt_int(big["blind"][TAU]["Q"][-1]))
    for tau, nm in (("0.7", "TauLo"), ("0.9", "TauHi")):
        for f in fams:
            tagm = {1024: "One", 2048: "Two", 4096: "Four", 8192: "Eight"}[f["m"]]
            v = f["blind"][tau]["sf_aic"]
            put(f"RealSf{tagm}{nm}", fmt_int(v) if v else "\\text{div.}")
    for k in ("4", "6"):
        v = big["blind_sub"][k]["sf_aic"]
        put(f"RealSfEightK{ {'4': 'Four', '6': 'Six'}[k] }".replace(" ", ""), fmt_int(v) if v else "\\text{div.}")
    if LONG:
        L = LONG[0]
        b = L["blind"][TAU]
        main2 = next(f for f in fams if f["m"] == L["m"])
        put("RealLongSf", fmt_int(b["sf_aic"]))
        ci = b.get("sf_aic_ci") or [None, None]
        if None not in ci:
            put("RealLongSfLo", fmt_int(ci[0]))
            put("RealLongSfHi", fmt_int(ci[1]))
        put("RealLongFvu", f"{L['fvu']:.3f}")
        put("RealLongSeenTwo", fmt_int(sum(b["Q"][1:])))
        put("RealLongAllSeeds", fmt_int(b["Q"][-1]))
        n_long = json.load(open(ROOT / "data/pythia_long/meta.json"))["n_train"]
        put("RealLongTokens", fmt_int(n_long))
        put("RealLongStepRatio", f"{n_long * L['config']['epochs'] / (meta['n_train'] * main2['config']['epochs']):g}")
        put("RealLongRatio", f"{b['sf_aic'] / main2['blind'][TAU]['sf_aic']:.2f}")
        put("RealAllSeedsTwo", fmt_int(main2["blind"][TAU]["Q"][-1]))
        for fam, nm in ((L, "Long"), (main2, "Two")):
            v = [fam["blind"][t]["sf_aic"] for t in ("0.7", "0.8", "0.9")]
            put(f"Real{nm}TauMin", fmt_int(min(v)))
            put(f"Real{nm}TauMax", fmt_int(max(v)))
        HRl = load("heldout_real.json")
        hl = next(v for k, v in HRl.items() if "pythia_long" in k)
        put("RealLongHeldErr", pct(abs(np.mean([r["sf_aic"] for r in hl["draws"]]) / hl["target"] - 1), 1))

with open(ROOT / "paper/numbers.tex", "w") as fh:
    fh.write("% Generated by paper/make_numbers.py from results/*.json. Do not edit.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros")
