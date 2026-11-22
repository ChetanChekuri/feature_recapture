#!/usr/bin/env bash
# Rebuild everything: tests -> (runs) -> analysis -> figures -> numbers -> PDF.
#
#   bash build.sh          tests, analysis, figures, numbers, PDF (uses existing runs)
#   bash build.sh runs     also train any missing SAEs and cache activations
#                          (about 6.5 h on an 8-core laptop CPU; resumable; ~17 GB disk in data/,
#                          which only training needs and which can be deleted afterwards)
#   bash build.sh test     correctness tests only (about 1 min)
#
# Every training run writes results/<synth|synth_hier|real>/<family>/seed<s>.npz and is skipped
# if that file exists, so `runs` picks up where an interrupted build stopped.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || uv sync
PY=.venv/bin/python
TECTONIC=${TECTONIC:-tectonic}
hr() { printf '=%.0s' {1..72}; echo; }

hr; echo "tests"; hr
$PY tests/test_pipeline.py
[ "${1:-}" = "test" ] && exit 0

if [ "${1:-}" = "runs" ]; then
  hr; echo "training runs"; hr
  $PY experiments/run_synth.py --grid main --workers 4
  $PY experiments/run_synth.py --grid hier --workers 4
  $PY experiments/extract_pythia.py 8
  $PY experiments/run_real.py --grid main --workers 2
  $PY experiments/extract_pythia_long.py
  $PY experiments/run_real.py --grid long --workers 2
fi

hr; echo "analysis"; hr
$PY experiments/analyze.py synth 4
$PY experiments/analyze.py synth_hier 4
$PY experiments/analyze.py real 4
$PY experiments/heldout.py synth
$PY experiments/heldout.py synth_hier
$PY experiments/heldout.py real

hr; echo "figures and numbers"; hr
$PY experiments/make_figures.py
$PY paper/make_numbers.py
$PY paper/check_macros.py

hr; echo "paper"; hr
(cd paper && $TECTONIC -X compile paper.tex > ../logs/tectonic.log 2>&1) \
  || { tail -30 logs/tectonic.log; exit 1; }
echo "paper/paper.pdf built"
