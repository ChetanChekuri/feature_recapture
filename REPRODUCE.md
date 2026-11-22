# Rebuilding this paper yourself

This is a plan for redoing the work from scratch, not just re-running it. Do the
steps in order. Each step names what to build, how to check it, and what you
should understand before moving on. Budget about two weeks of evenings.

Hardware: any 8-core laptop. No GPU is needed. Full compute is about 6.5 hours, plus about 17 GB of disk for cached activations and synthetic data in `data/`. Only training reads `data/`; once the SAEs in `results/` exist you can delete the `.npy` files there, and `bash build.sh runs` re-creates them if a run is missing.

---

## Week 1: the statistics, with no neural networks

**Step 1. Capture–recapture by hand (1 evening).**
Read Chao (1987) and the first chapter of Otis et al. (1978), the closed-population
models M_0, M_t and M_h. Simulate a population of N=1000 fish caught on 8 occasions
with equal catch probability p=0.3. Code Lincoln–Petersen, Chapman's correction and
Darroch's M_t from the formulas. Check that each recovers N.
*Check:* `tests/test_pipeline.py::test_homogeneous_unbiased` is the same test.

**Step 2. Heterogeneity (1 evening).** Draw each fish's catch probability from
Beta(0.6, 2.5). Watch M_t undercount. Code Chao's lower bound
`S + (K-1)/K * Q1^2 / (2 Q2)` and both jackknives. Read Link (2003). Then write a
paragraph, in your own words, on why no estimator can find fish whose catch
probability is near zero.

**Step 3. Misidentification (1 evening).** Add 3000 fake fish caught exactly once
("tag-reading errors"). Watch Chao's bound explode. Read Chiu & Chao (2016) on
sequencing-error singletons. Then code the singleton-free fit: maximise the
likelihood of Q_2..Q_K under a beta-binomial, a single binomial and a two-class
mixture, choose by AIC, and set N = S_{>=2} / P(Y >= 2).
*Check:* `test_singleton_free_ignores_junk`.

## Week 2: sparse autoencoders

**Step 4. A TopK SAE (2 evenings).** Read Gao et al. (2024), sections 2–3. Write a
TopK SAE with unit-norm decoder rows, tied initialisation and the AuxK loss.
Train it on 64 planted features in 32 dimensions and check that it recovers
nearly all of them (`test_sae_recovers_planted_features`).
**Watch out:** seed the SAE initialisation from a different random stream than the
synthetic features. We hit this bug ourselves: with the same stream, seed 1's
initialisation *was* the planted dictionary (see NOTES.md).

**Step 5. Seeds as captures (1 evening).** Train 8 seeds on the same data and match
latents across seeds by mutual nearest neighbour with cosine >= 0.8. Read Paulo &
Belrose (2025) first. Build the capture-history matrix and look at its frequency
counts Q_j before running any estimator. You should see a spike at j=1, made of
junk, and a spike at j=8.

**Step 6. The synthetic sweep (run overnight).** Run
`experiments/run_synth.py --grid main` and then `experiments/analyze.py synth`.
Compare your Table 1 with ours.

**Step 6b. Features that split (about 1 hour).** Change the synthetic data so that each of
256 parents splits into 4 children, and exactly one child fires when its parent is active
(`HierSynthData` in `src/fr/synth.py`). Before running anything, work out the cosines
between siblings and between a child and its parent, and check that τ = 0.8 separates the
two levels (`test_hierarchy_levels_are_separable`). Then run `run_synth.py --grid hier` and
`analyze.py synth_hier`. Predict first: at m = 256 do the seeds learn the parents?

**Step 7. Pythia-70M (run overnight).** Run `experiments/extract_pythia.py` and then
`experiments/run_real.py --grid main`. Then run the training-budget check:
`experiments/extract_pythia_long.py` (10M tokens, about 10 GB on disk) and
`run_real.py --grid long`. Before you look at our numbers, write down
what you expect: will the estimate level off as the dictionary grows?

**Step 8. Write it up.** Rebuild with `bash build.sh`. Every number in the PDF
comes from `paper/make_numbers.py`. If you change an estimator, delete
`results/cache/` first.

---

## Reading list (in order)

1. Chao, A. (1987). Estimating the population size for capture–recapture data with unequal catchability. *Biometrics*.
2. Otis, Burnham, White & Anderson (1978). *Statistical inference from capture data on closed animal populations*. Wildlife Monographs 62. Read chapter 1 and the M_h section.
3. Link, W. A. (2003). Nonidentifiability of population size from capture–recapture data with heterogeneous detection probabilities. *Biometrics*.
4. Chiu, C.-H. & Chao, A. (2016). Estimating and comparing microbial diversity in the presence of sequencing errors. *PeerJ*.
5. Colwell et al. (2012). Models and estimators linking individual-based and sample-based rarefaction... *J. Plant Ecology*.
6. Elhage et al. (2022). Toy Models of Superposition. arXiv:2209.10652.
7. Bricken et al. (2023). Towards Monosemanticity. transformer-circuits.pub.
8. Gao et al. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093.
9. Paulo & Belrose (2025). Sparse Autoencoders Trained on the Same Data Learn Different Features. arXiv:2501.16615.
10. Leask et al. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. arXiv:2502.04878.
11. Engels, Riggs & Tegmark (2024). Decomposing the Dark Matter of Sparse Autoencoders. arXiv:2410.14670.

## Questions a reviewer will ask (have an answer ready)

1. **"Features aren't fish. Is there a fixed population at all?"** In synthetic data,
   yes, by construction. In a real model, maybe not: feature splitting means a bigger
   dictionary can find finer features. The paper's answer is that the estimate
   targets the features a dictionary of a given size can reach. Whether it levels
   off across sizes is the empirical question. Know exactly what Figure 1b shows.
   Then point to the splitting test (Figure 4): with planted parents and children, the
   census does not stop at the parents. It levels off at the children only once they all fit,
   and undercounts by about a tenth when the dictionary is just large enough.
2. **"Why do seeds count as independent captures?"** They share data and data order.
   Only the initialisation differs. That makes the captures dependent in a
   structured way, mostly through feature frequency, which is the heterogeneity
   M_h models. Be ready to say what kind of dependence would break the estimators,
   such as two seeds sharing a junk direction.
3. **"Your matching threshold is arbitrary."** The appendix reports τ = 0.7 and 0.9.
   Know how much the numbers move.
4. **"Isn't dropping singletons throwing away data?"** Yes. It trades variance for
   robustness to misidentification. In the synthetic runs, the singletons were
   mostly junk whenever the dictionary was at least as large as the feature count.
   Know the junk fractions.
5. **"What happens when the dictionary is too small?"** The singleton-free fit often
   diverges or undercounts, and the paper reports this, not hides it. Be able to
   explain it with Link (2003).
6. **"Is 8 seeds enough?"** See the held-out-seed experiment: fit on 4, predict 8.
7. **"Your SAEs are undertrained (2M tokens)."** Yes, and the paper tests this. With
   5× more data, seeds agree more and the estimate rises about 1.3×. Say plainly that the
   Pythia counts are lower bounds for this training budget.
8. **"Does this scale to real SAEs with 1M+ latents?"** The cost is K training runs.
   Be honest: we only tested up to 8k latents on Pythia-70M.
9. **"How is this different from Paulo & Belrose?"** They measured overlap. We turn
   overlap into a population estimate, show the naive versions fail, and validate
   against planted ground truth.

## What you should be able to explain without notes

- Why Chao's estimator is a *lower bound* (Cauchy–Schwarz on the catch-probability moments).
- Why junk latents are counted once, while real but rare features are also often caught only once.
- What AIC is doing when it picks M_0 over a two-class mixture.
- Why "number of alive latents" is not a feature count.
