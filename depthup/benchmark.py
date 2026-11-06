"""Efficiency benchmark: latency (median/p95) and peak memory per method, one fresh process each.

    python -m depthup.benchmark   ->  results/efficiency.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD = r"""
import sys, time, json, resource
sys.path.insert(0, %r)
import numpy as np, torch
torch.set_num_threads(%d)
from depthup import data, degrade, pipeline
method, scale = sys.argv[1], int(sys.argv[2])
rgb, dep = data.get_pair(1005)
lr, valid = degrade.degrade(dep, degrade.Degradation(scale=scale), np.random.default_rng(0))
if method in ('hybrid', 'dl_only', 'auto'):
    pipeline.load_model('dl_only'); pipeline.load_model('hybrid')
pipeline.METHODS[method](lr, valid, rgb, scale)          # warm-up
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
ts = []
for _ in range(12):
    t = time.perf_counter(); pipeline.METHODS[method](lr, valid, rgb, scale); ts.append((time.perf_counter() - t) * 1000)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
ts = sorted(ts)
print(json.dumps(dict(median_ms=ts[len(ts)//2], p95_ms=ts[-2], peak_rss_mb=peak, warm_rss_mb=base)))
"""


def main(threads: int = 4):
    from .model import DLOnlyNet, HybridNet, n_params
    out = {"params": {"hybrid": n_params(HybridNet()), "dl_only": n_params(DLOnlyNet())},
           "weights_kb": {k: (ROOT / "weights" / f"{k}.pt").stat().st_size / 1024
                          for k in ("hybrid", "dl_only") if (ROOT / "weights" / f"{k}.pt").exists()},
           "image": "480x640", "threads": threads, "timing": {}}
    for m in ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical", "dl_only", "hybrid", "auto"]:
        for s in (2, 4, 8):
            r = subprocess.run([sys.executable, "-I", "-c", CHILD % (str(ROOT), threads), m, str(s)], capture_output=True, text=True)
            if r.returncode:
                print(m, s, r.stderr[-300:])
                continue
            out["timing"][f"{m}|{s}"] = json.loads(r.stdout.strip().splitlines()[-1])
            print(m, s, out["timing"][f"{m}|{s}"], flush=True)
    (ROOT / "results" / "efficiency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
