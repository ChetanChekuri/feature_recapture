# Counting Fish in the Residual Stream

**How many features does a language model have?** Sparse autoencoders (SAEs) trained
with different random seeds find different features. This project treats each seed as a
net cast into a lake and uses **capture–recapture**, the method ecologists use to count
fish, to estimate how many features there are.

`paper/paper.pdf` is the manuscript. Everything below is how to check it.

```
bash build.sh          # tests -> analysis -> figures -> numbers -> lint -> PDF (minutes)
bash build.sh runs     # also retrain every SAE that is missing (~4-5 h on a laptop CPU)
bash build.sh test     # correctness tests only (~1 min)
```

CPU only. No GPU, no API keys. Pythia-70M and `NeelNanda/pile-10k` download from
Hugging Face on first run.

---

## The claims, and how to check each one

Every number in the PDF is a LaTeX macro written by `paper/make_numbers.py` from
`results/*.json`. `paper/check_macros.py` fails the build if a result-like number is
typed by hand. The macro names below are what to look up in `paper/numbers.tex`.

| # | Claim | Where | How to check |
|---|---|---|---|
| 1 | On synthetic data with a planted feature count N, naive counts overstate N once the dictionary is large enough: the union over seeds by up to `\UnionRatioMax`×, Chao's estimator by up to `\ChaoRatioMax`. | Table 1, Fig. 1a | `results/summary_synth.json` → `blind.0.8.union`, `chao_mh`; `paper/tab_synth.tex` |
| 2 | The cause is junk latents: most features caught by exactly one seed match no planted feature (`\JunkSingleEq` at m = N, `\JunkSingleOver` at m > N). | Fig. 2a | `blind.0.8.junk_by_count` in the synthetic summary |
| 3 | The singleton-free estimator is within `\SfMaxErrComplete` of N in all m ≥ N settings, and stays within `\SfMaxErrCompleteTauLo` / `\SfMaxErrCompleteTauHi` at τ = 0.7 / 0.9. | Table 1, §4.3 | `blind.<tau>.sf_aic`; robustness is recomputed in `make_numbers.py` |
| 4 | When m < N it tracks the *reachable* count (median ratio `\SfReachMed` to the oracle union), so its plateau across m is the feature count. | Fig. 1a, §4.3 | `oracle.0.8.union` vs `blind.0.8.sf_aic` |
| 5 | Unbounded mixtures diverge (Link 2003 non-identifiability). The bound p ≥ 1/K fixes it. | Table 1 ablation rows | `betabin2`, `mix2_2` columns; `tests/test_pipeline.py::test_stray_recapture_does_not_diverge` |
| 6 | Pythia-70M layer 3: the estimate grows `\RealSfOne` → `\RealSfEight` from 1k to 8k latents, with slowing growth and no plateau yet. | Table 2, Fig. 1b | `results/summary_real.json`, `paper/tab_real.tex` |
| 7 | Held-out seeds: fitted on 4 seeds, the estimator predicts 8-seed reproducible counts within `\HeldRealSfMax` on Pythia (no-growth baseline: `\HeldRealNoGrowthMed`). | Fig. 3, §4.6 | `results/heldout_real.json` |
| 8 | The real-model count is sensitive to the matching threshold τ. | Appendix table | `paper/tab_real_tau.tex` |
| 9 | Training on 5× more data (`\RealLongTokens` tokens) at m = 2048 makes seeds agree more and raises the estimate by `\RealLongRatio`×, so the Pythia counts are lower bounds. | §4.6, last paragraph | `results/real/m2048_k32_e1_lr0.001_pythia_long/`; `summary_real.json` |
| 10 | When planted features split (`\HierParents` parents × `\HierC` children), the census does not stop at the parent count. It rises with m, levels off at the child count at m = 2N (`\HierSfDouble`, within `\HierSfErrDouble`), and undercounts at m = N (`\HierSfFull`, `\HierSfErrFull` low). | Fig. 4, §4.5 | `results/summary_synth_hier.json`; `oracle.0.8.catch_rate` (children first, then parents); `tests/test_pipeline.py::test_hierarchy_levels_are_separable` |

## What the paper does *not* claim
- It does not claim to give the true total number of features in Pythia-70M. The estimate counts features that SAEs of a given size find with probability at least 1/K per seed. It is a lower bound while the curve is still rising.
- Synthetic success does not prove that real features form a fixed population. Feature splitting may mean they don't. The hierarchy test (§4.5) covers only one parent–child level with fixed geometry.
- The real-model SAEs are small: at most 8k latents, trained on 2M tokens (plus a 10M-token check).

## Layout
```
src/fr/          sae.py (TopK SAE), synth.py (planted data), match.py (capture histories),
                 estimators.py (all estimators), analysis.py (per-family summaries)
experiments/     run_synth.py, extract_pythia*.py, run_real.py, analyze.py, heldout.py,
                 make_figures.py
tests/           test_pipeline.py: planted-answer tests for every stage
paper/           paper.tex, sections/, make_numbers.py, check_macros.py, refs.bib
refs/VERIFIED.md citation checks
results/         per-run .npz (decoder, firing rates, FVU) and summary JSONs
data/            extraction metadata only; the 17 GB of cached activations and synthetic
                 data were deleted after training (`bash build.sh runs` re-creates them)
NOTES.md         development log, including the bugs found and fixed
REPRODUCE.md     plan for rebuilding the work from scratch, with a reading list and reviewer questions
```

## Known issues and decisions (see NOTES.md)
- An early bug let SAE seed 1 start from the planted dictionary, because it shared a random stream with the synthetic data. It was fixed, a guard was added, and every affected run was redone.
- The first singleton-free estimator (unbounded beta-binomial) diverged in over-complete settings. It is reported as an ablation, not hidden.
- `chapman1951some` is confirmed in a library catalog, but its end page (159 vs 160) is unresolved. See `refs/VERIFIED.md`.
