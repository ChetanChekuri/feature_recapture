# Citation checks

Each entry in `paper/refs.bib` was checked against the page listed. Checks ran on
2026-10-31 and 2026-11-01. "Crossref" means `https://api.crossref.org/works/<DOI>`,
which returns the publisher's own metadata.

| key | status | checked against | notes |
|---|---|---|---|
| paulo2025sparse | verified | https://arxiv.org/abs/2501.16615 | Paulo & Belrose; title matches. ICLR 2026 version reports 42% shared vs 30% in v1 |
| gao2024scaling | verified | https://arxiv.org/abs/2406.04093 | Leo Gao et al. |
| engels2024decomposing | verified | https://arxiv.org/abs/2410.14670 | Engels, Riggs, Tegmark; comments say TMLR |
| leask2025sparse | verified | https://arxiv.org/abs/2502.04878 | Leask et al. |
| michaud2023quantization | verified | https://arxiv.org/abs/2303.13506 | Michaud et al. |
| vanderwal2025polypythias | verified | https://arxiv.org/abs/2503.09543 | comments say ICLR 2025 (not cited in the paper unless the PolyPythia extension runs) |
| cunningham2023sparse | verified | https://arxiv.org/abs/2309.08600 | venue not shown on arXiv, so cited as an arXiv preprint |
| elhage2022toy | verified | https://arxiv.org/abs/2209.10652 | |
| biderman2023pythia | verified | https://arxiv.org/abs/2304.01373 | cited as an arXiv preprint |
| gao2020pile | verified | https://arxiv.org/abs/2101.00027 | |
| nasr2023scalable | verified | https://arxiv.org/abs/2311.17035 and its HTML full text | Good–Turing extrapolation of total memorisation is in Sec. 3.4 ("sequential Good-Turing") |
| lan2024sparse | verified | https://arxiv.org/abs/2410.06981 | exact title and six authors |
| bricken2023monosemanticity | verified | page fetched with curl; `<title>` matches; Bricken, Templeton, Conerly, Olah appear; dated Oct 2023 | |
| templeton2024scaling | verified | page fetched with curl; `<title>` matches; Templeton, Conerly appear; "May 21, 2024" | |
| sharkey2022interim | verified | https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj | byline shows "Lee Sharkey, Dan Braun, beren"; "beren" is Beren Millidge's handle |
| pile10k | verified | https://huggingface.co/datasets/NeelNanda/pile-10k | card: "The first 10K elements of The Pile" |
| darroch1958multiple | verified | Crossref 10.1093/biomet/45.3-4.343 | Biometrika 45(3-4):343–359 |
| fienberg1972multiple | verified | Crossref 10.1093/biomet/59.3.591 | Biometrika 59(3):591–603 |
| chao1987estimating | verified | Crossref 10.2307/2531532 | Biometrics 43(4), first page 783. Crossref gives no end page; 791 is from the standard citation |
| burnham1978estimation | verified | Crossref 10.1093/biomet/65.3.625 | Biometrika 65(3):625–633 |
| pledger2000unified | verified | Crossref 10.1111/j.0006-341X.2000.00434.x | Biometrics 56(2):434–442 |
| dorazio2003mixture | verified | Crossref 10.1111/1541-0420.00042 | Biometrics 59(2):351–364. The DOI ending .00041 is a different paper; corrected |
| link2003nonidentifiability | verified | Crossref 10.1111/j.0006-341X.2003.00129.x | Biometrics 59(4):1123–1130 |
| colwell2012models | verified | Crossref 10.1093/jpe/rtr044 | J. Plant Ecol. 5(1):3–21, seven authors |
| chiu2016estimating | verified | Crossref 10.7717/peerj.1634 | PeerJ 4:e1634 |
| good1953population | verified | Crossref 10.1093/biomet/40.3-4.237 | Biometrika 40(3-4):237–264 |
| raaijmakers1987statistical | verified | Crossref 10.2307/2531533 | Biometrics 43(4), first page 793. Crossref gives no end page |
| otis1978statistical | verified (2026-11-17) | https://www.usgs.gov/software/capture | USGS page confirms authors, title, Wildlife Monographs 62, 1978, 135 pp. Page range 3–135 could not be confirmed, so the bib now gives the monograph number only |
| chapman1951some | verified (2026-11-17) | https://openlibrary.org/books/OL200768M | Open Library confirms author, title, UC Publications in Statistics v.1 no.7, UC Press, 1951, starting page 131. End page differs: catalog says 131–159, the common citation (e.g. arXiv:1205.1150) says 131–160; bib keeps 131–160. Check the HathiTrust scan (catalog.hathitrust.org/Record/005761134) to settle it |

## To do before submission
- [x] `otis1978statistical` checked (pages dropped; unconfirmed).
- [x] `chapman1951some` checked in a library catalog (end page 159 vs 160 unresolved).
- [ ] Re-run this table for any citation added later.
