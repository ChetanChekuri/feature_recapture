"""Lint: every result macro used in the paper is defined in numbers.tex, and no
result-like number is typed by hand in the abstract, results or discussion.

    .venv/bin/python paper/check_macros.py
"""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (P / "numbers.tex").read_text()))
texts = {f.name: f.read_text() for f in [P / "paper.tex", *sorted((P / "sections").glob("*.tex"))]}
LATEX = {"Big", "Bigg", "Delta", "Gamma", "Lambda", "Omega", "Phi", "Pi", "Psi", "Sigma", "Theta"}
used = {m for t in texts.values() for m in re.findall(r"(?<!\\)\\([A-Z][A-Za-z]+)\b", t)} - LATEX
missing = sorted(used - defined)

# Hand-typed numbers: strip comments, citations, refs, labels and model names, then
# flag decimals and percentages, which in this paper are always results.
bad = []
for name in ("abstract.tex", "results.tex", "results_hier.tex", "discussion.tex", "intro.tex"):
    t = texts.get(name, "")
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\(cite\w*|ref|label|eqref|url|href|includegraphics|input)(\[[^\]]*\])?\{[^}]*\}", "", t)
    t = t.replace("95\\%", "")   # confidence level, not a result
    for hit in re.findall(r"(?<![\w.])\d+\.\d+(?![\w])|\d+\s*\\?%", t):
        bad.append(f"{name}: {hit}")

if missing:
    print("UNDEFINED macros:", ", ".join(missing))
if bad:
    print("hand-typed result-like numbers:\n  " + "\n  ".join(bad))
if missing or bad:
    sys.exit(1)
print(f"macro check ok: {len(used & defined)} macros used, all defined; no hand-typed results")
