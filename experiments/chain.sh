#!/usr/bin/env bash
# Background queue used during development: real-model pilot after extraction,
# then the real-model main grid once the synthetic grid has finished.
cd "$(dirname "$0")/.."
while [ ! -f data/pythia/meta.json ]; do sleep 30; done
.venv/bin/python experiments/run_real.py --grid pilot --workers 2 > logs/real_pilot.log 2>&1
while pgrep -f "run_synth.py --grid main" > /dev/null; do sleep 60; done
.venv/bin/python experiments/run_real.py --grid main --workers 2 > logs/real_main.log 2>&1
