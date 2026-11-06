"""Benchmark every method on the held-out NYU test split.

    python -m depthup.evaluate --protocol clean --scales 2 4 8 --n 449
    python -m depthup.evaluate --protocol noisy+holes --scales 2 4 8 --n 150

Writes results/per_image_<protocol>.jsonl (one row per image/scale/method) and prints a summary.
The test split (NYU frames 1000-1448) was never used for training, validation or tuning.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ALL_METHODS = ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical", "dl_only", "hybrid_convex", "hybrid", "auto"]


def _seed(idx, scale, protocol):
    return zlib.crc32(f"{idx}-{scale}-{protocol}".encode())


def _work(args):
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    from . import data, degrade, metrics, pipeline
    idx, scales, protocol, methods = args
    rgb, dep = data.get_pair(idx)
    rows = []
    for s in scales:
        d = degrade.Degradation(scale=s, mode="bicubic", **degrade.PROTOCOLS[protocol])
        lr, valid = degrade.degrade(dep, d, np.random.default_rng(_seed(idx, s, protocol)))
        for m in methods:
            pred, ms = pipeline.timed(m, lr, valid, rgb, s)
            row = dict(idx=idx, scale=s, protocol=protocol, method=m, latency_ms=ms)
            row.update(metrics.evaluate(np.asarray(pred, np.float32), dep, rgb))
            rows.append(row)
    return rows


def run(protocol, scales, n, methods, procs, offset=0, append=False):
    from . import data
    test = data.split_indices("test")
    step = max(1, len(test) // n)
    idxs = test[offset::step][:n]
    out = RES / f"per_image_{protocol}.jsonl"
    RES.mkdir(exist_ok=True)
    with mp.get_context("spawn").Pool(procs) as pool, open(out, "a" if append else "w") as f:
        for i, rows in enumerate(pool.imap_unordered(_work, [(i, scales, protocol, methods) for i in idxs])):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            if i % 20 == 0:
                print(f"{protocol}: {i + 1}/{len(idxs)}", flush=True)
    return out


def load_rows(protocol):
    return [json.loads(l) for l in open(RES / f"per_image_{protocol}.jsonl")]


def summarize(rows):
    """Mean of every metric per (protocol, scale, method)."""
    uniq = {(r["idx"], r["scale"], r["protocol"], r["method"]): r for r in rows}      # re-runs never double-count
    agg: dict = {}
    for r in uniq.values():
        agg.setdefault((r["protocol"], r["scale"], r["method"]), []).append(r)
    out = {}
    for k, v in agg.items():
        keys = [c for c in v[0] if c not in ("idx", "scale", "protocol", "method")]
        out["|".join(map(str, k))] = {c: float(np.nanmean([r[c] for r in v])) for c in keys} | {"n": len(v)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="clean")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--n", type=int, default=449)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu | cuda (GPU inference for the learned arms)")
    ap.add_argument("--append", action="store_true", help="append rows (use after relabelling an old method)")
    a = ap.parse_args()
    os.environ["DEPTHUP_DEVICE"] = a.device
    run(a.protocol, a.scales, a.n, a.methods, a.procs, append=a.append)
