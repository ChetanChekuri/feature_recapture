# Working notes (for resuming)

Status log, newest last. Times are CDT.

- 2026-11-01 00:05 — Option 1 chosen. Env: uv, torch 2.15 CPU.
- Core code: src/fr/{sae,synth,match,estimators,analysis}.py; tests/test_pipeline.py (passes).
- Bug found & fixed: SAE init seed s reused the generator stream of synthetic data seed 0,
  so seed 1's init *was* the planted features. SAE init now offset by 1_000_000_007.
- Pilot finding: blind matching creates many seed-specific "junk" singletons → Chao/jackknife
  overshoot ~5-13x in the complete regime (m >= N). Singleton-free mixture fits (betabin2,
  mix2_2: fit only species caught >= 2 times) recover N within a few % there. In the incomplete
  regime (m < N), nothing recovers N (rare features have ~0 catch probability; Link 2003
  non-identifiability); betabin2 diverges, which itself flags incompleteness.
- Planned framing: estimate N-hat(m) across dictionary sizes; plateau at N = feature count.
  Held-out: predict discoveries of extra seeds. Ecology link for junk: sequencing-error
  singletons inflate Chao1 in microbial ecology (Chiu & Chao 2016).
- 00:45 launched: synthetic main grid (logs/synth_main.log, 152 runs, ~1 h),
  Pythia-70M extraction (logs/extract.log), chain.sh → real pilot → real main grid
  (logs/real_pilot.log, logs/real_main.log; 32 runs, ~1.5-2 h).
- Next: experiments/analyze.py synth|real → results/summary_*.json; then figures,
  macros (paper/numbers.tex), paper, build.sh, README, REPRODUCE, refs/VERIFIED.md.
- 00:40 synthetic grid trimmed (dropped N=2048,m=4096 and N=1024 ds1 m=4096) and restarted
  (resumable). 136 jobs. Extraction is slow because HF computes all 6 layers + LM head
  (fine, just slow; ~20-30 min).
- Paper drafted so far: paper/paper.tex, sections/method.tex, sections/related.tex,
  paper/refs.bib, refs/VERIFIED.md (all but chapman1951 verified; otis1978 via reviews).
  method.tex uses macros: \MainTau \BootN \SynD \SynAlpha \SynMeanActive \SynK \SynSeeds
  \SynSteps \RealLayer \RealTrainTokens \RealEvalTokens \RealK \RealSeeds — make_numbers.py
  must define them from configs/meta.json.
- On resume: (1) check logs/real_pilot.log FVU/dead for lr 1e-3 vs 3e-4; if 3e-4 clearly
  better, edit BASE lr in run_real.py BEFORE chain.sh launches the main grid (it starts when
  run_synth exits). (2) run `experiments/analyze.py synth` once synth is done.
  (3) write experiments/make_figures.py + paper/make_numbers.py, results section, intro,
  abstract, discussion, appendix, build.sh, README, REPRODUCE.
- 00:50 Added AIC-selected singleton-free estimator `sf_aic` (M0 / beta-binomial / 2-class
  mixture fitted to Q>=2, min AIC; Pledger 2000). Fixes mix2 inflating N when every species
  is caught by all seeds. New planted test: 3000 junk singletons leave sf_aic unchanged,
  Chao explodes. Early synthetic families (N=512): blind sf ≈ N at m=N; union and Chao blow
  up at m=2N (junk latents ~ m-N per seed are singletons).
- build.sh written; still missing: experiments/make_figures.py, paper/make_numbers.py,
  paper/check_macros.py (verify every \Macro used in sections is defined in numbers.tex).
- Note: analysis caches per family; delete results/cache/ after changing estimators.
- 02:10 KEY CHANGE: unbounded singleton-free fits (betabin, mix2) DIVERGE in the over-complete
  regime (one stray recapture next to Q_K=1024 → U-shaped fit, N→∞) and mostly in m<N.
  New main estimator `sf_aic` = 1–3-class binomial mixture with p >= p_min = 1/K, AIC, fitted
  to Q>=2. From Q counts: blind sf/N = 0.97–1.02 for all m>=N families (τ=0.8), ≈ oracle
  reachable count for m<N (0.30 vs 0.28–0.35 at N/4; 0.60–0.69 at N/2). Plateau story holds.
  Tests: added test_stray_recapture_does_not_diverge. Method/appendix text updated.
- Relaunched: analyze.py synth (cache cleared) then heldout.py synth (logs/analyze_synth.log,
  logs/heldout_synth.log). Real main grid running (logs/real_main.log).
- 01:15 CST (DST ended; clocks went back 1h). Synthetic analysis + heldout done; Table 1,
  Figs 1a/2/3 generated; make_numbers/check_macros pass; paper compiles (paper/paper.pdf).
  Written: intro, related, method, results (synthetic), discussion+limitations+impact,
  AI-assistance placeholder, appendix (estimators, details). Pending: results_real.tex,
  abstract.tex, README.md, final build.sh run.
- Background waiter (task) runs analyze real → heldout real → numbers → figures when
  logs/real_main.log prints "done main". The lr=3e-4 pilot dir is moved to results/real_pilot/.
- 02:25 CST Real main grid done (32 runs, 85 min). Pythia census: sf 870 → 1616 → 2520 → 3172
  (m=1k..8k), no plateau; held-out 4→8 seeds within 4.3% (no-growth misses 37%). τ-sensitive
  (8k: 4254 at τ=0.7, 1238 at τ=0.9). Robustness family launched: 10M tokens (data/pythia_long,
  context-shuffled, hook-checked against main cache), m=2048, 1 epoch, 8 seeds
  (logs/real_long.log). Waiter task re-runs analysis/heldout/numbers/figures when done.
- Written: abstract, results_real, README, REPRODUCE. Pending: results_long.tex
  paragraph (macros RealLong* exist), final `bash build.sh`.
- Total compute so far ≈ 4.3 h (synth 65m, extract 37m, pilot 15m, real 85m, analyses ~30m,
  long extraction ~18m); long training ≈ 40m more → ≈ 5 h.
- 03:05 CST DONE. Long-data family: m=2048, 10M tokens → FVU 0.072 (vs 0.082), all-8-seed
  features 384 (vs 167), sf 2054 [1868, 2472] vs 1616 (×1.27), τ-range 1732–2079 (vs 1123–1616),
  held-out −0.4%. Written into results_long.tex, intro, abstract, discussion, README claim 9.
  `bash build.sh` passes end to end (2 min without `runs`); paper 12 pages; 80 macros, lint ok.
- If resumed again: NO new experiments. Optional proofreading pass only (check PDF wording vs
  numbers, overfull boxes in intro.tex:13-15 and appendix_estimators.tex:20). Data cache in
  data/ (~12 GB) is regenerable; ask the user before deleting.
- 06:52 CST resume (proofreading only, no new experiments). Fixed: overfull boxes
  (\emergencystretch), parenthetical numeric citations in method/appendix, wording of the p_min
  definition, jackknife sentence, "union grows almost in step" (it grows faster), τ=0.9 trend
  ("falls" between 4k and 8k, not "stalls"), "median" on no-growth held-out numbers, compute
  paragraph (now lists caching, analysis, ~5 h total), Sharkey bib title case, new macro
  RealLongStepRatio (10M tokens × 1 pass = 2.5× the training steps of the main runs; disclosed in
  results_long). Build passes, 12 pages, 81 macros.
- 10:46 CST hierarchy test added at the user's request ("Wait actually add it"). HierSynthData:
  256 parents x 4 children (child = 0.6 g + 0.8 u_j, QR-orthonormal, sibling cos 0.36, parent
  cos 0.72), one child fires per active parent. Grid `hier`: m in {128..2048} x 2 draws x 8 seeds
  = 80 SAEs, 4891 s with --workers 4 --threads 1 (the CPU was shared with the user's grokking
  job, load ~30; the first attempt was cancelled at 8/80 and resumed). Results: census 154/308/615
  at m=N/8..N/2 (no stop at 256 parents; seeds learn mixed levels, oracle panel b); m=N 909
  (-11.3%, tau-sensitive 828-1197); m=2N 1030 (within 1.5% at all tau). Union 6.7x, junk
  singletons 93%. Held-out sf is worse here (median +13.6% small, +7.4% big; no-growth 6.7%/3.5%),
  reported. New: results_hier.tex (§4.5), fig4_hierarchy, appendix settings + test bullet,
  abstract/intro/discussion sentences, README claim 10, REPRODUCE step 6b. Fig 3 tick labels
  overlapped and its legend covered Pythia points; fixed. build.sh passes; 14 pages, main
  text still ends on p. 9.
- 2026-11-04 cleanup: deleted the 10 cached .npy files in data/ (17 GB: Pythia activations 2 GB +
  10M-token activations 9.6 GB + synthetic streams 5.1 GB). Kept data/pythia*/meta.json (read by
  make_numbers.py). Extractors now skip only when the .npy files exist too, so `build.sh runs`
  regenerates them if needed. .venv (1.4 GB) kept: it is hardlinked from the uv cache, so deleting
  it frees almost nothing, and build.sh needs it.
