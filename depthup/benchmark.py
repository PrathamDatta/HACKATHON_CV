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
import sys, time, json
sys.path.insert(0, %r)
def peak_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:   # Windows: peak working set via psapi
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL('kernel32', use_last_error=True); k32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.WinDLL('psapi').GetProcessMemoryInfo(wintypes.HANDLE(k32.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
import numpy as np, torch
torch.set_num_threads(%d)
from depthup import data, degrade, pipeline
method, scale = sys.argv[1], int(sys.argv[2])
rgb, dep = data.get_pair(1005)
lr, valid = degrade.degrade(dep, degrade.Degradation(scale=scale), np.random.default_rng(0))
if method in ('hybrid', 'dl_only', 'auto'):
    pipeline.load_model('dl_only'); pipeline.load_model('hybrid')
if method == 'dl_only_L':
    pipeline.load_model('dl_only_L')
if method in ('hybrid_L', 'auto_L'):
    pipeline.load_model('dl_only_L'); pipeline.load_model('hybrid_L')
pipeline.METHODS[method](lr, valid, rgb, scale)          # warm-up
base = peak_mb()
ts = []
for _ in range(12):
    t = time.perf_counter(); pipeline.METHODS[method](lr, valid, rgb, scale); ts.append((time.perf_counter() - t) * 1000)
peak = peak_mb()
ts = sorted(ts)
print(json.dumps(dict(median_ms=ts[len(ts)//2], p95_ms=ts[-2], peak_rss_mb=peak, warm_rss_mb=base)))
"""


def main(threads: int = 4, only=None):
    from .model import DLOnlyNet, HybridNet, n_params
    out = {"params": {"hybrid": n_params(HybridNet()), "dl_only": n_params(DLOnlyNet())},
           "weights_kb": {k: (ROOT / "weights" / f"{k}.pt").stat().st_size / 1024
                          for k in ("hybrid", "dl_only") if (ROOT / "weights" / f"{k}.pt").exists()},
           "image": "480x640", "threads": threads, "timing": {}}
    arms = ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical", "dl_only", "hybrid", "auto"]
    arms += [m for m in ("dl_only_L", "hybrid_L", "auto_L") if (ROOT / "weights" / f"{m.replace('auto', 'hybrid')}.pt").exists()]
    if only:
        arms = [m for m in arms if m in only]
    for m in arms:
        for s in (2, 4, 8):
            r = subprocess.run([sys.executable, "-I", "-c", CHILD % (str(ROOT), threads), m, str(s)], capture_output=True, text=True, env={**__import__('os').environ, 'DEPTHUP_DEVICE': 'cpu'})
            if r.returncode:
                print(m, s, r.stderr[-300:])
                continue
            out["timing"][f"{m}|{s}"] = json.loads(r.stdout.strip().splitlines()[-1])
            print(m, s, out["timing"][f"{m}|{s}"], flush=True)
    (ROOT / "results" / "efficiency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", help="benchmark only these arms (e.g. bicubic dl_only dl_only_L)")
    main(only=ap.parse_args().arms)
