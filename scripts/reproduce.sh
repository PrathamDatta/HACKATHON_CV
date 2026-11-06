#!/usr/bin/env bash
# Reproduce every number in eval_P4_*.md from scratch (about 2 hours on 4 CPU cores).
#   NYU Depth V2 (1449 labelled frames) must be unzipped in data/nyu/nyu_depth_train_float32_1449 (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
TEAM="${TEAM:-T01}"; USNS="${USNS:-USN1,USN2,USN3,USN4}"
python scripts/prepare_data.py
# equal step budget for both models (6,200 steps, about 35-45 min each when run side by side on 4 cores)
python -m depthup.train --kind dl_only --steps 6200 --minutes 120 --workers 1 --threads 2 --val-every 500 &
python -m depthup.train --kind hybrid  --steps 6200 --minutes 120 --workers 1 --threads 2 --val-every 500 &
wait
# weights/hybrid_convex.pt (superseded v1 arbiter, kept as an evaluation arm) is committed and not retrained
python -m depthup.evaluate --protocol clean       --n 449
python -m depthup.evaluate --protocol noisy       --n 150
python -m depthup.evaluate --protocol noisy+holes --n 150
python -m depthup.ablate --n 80
python -m depthup.benchmark
python -m depthup.failures
python -m depthup.report --team "$TEAM" --usns "$USNS"
python scripts/make_docs.py
python -m pytest -q tests
