"""Ablation study: contribution of each pipeline component.

Classical chain (no training needed) and learned-arbiter ablations (inference-time re-weighting of the
trained hybrid's experts). Results -> results/ablation.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CLASSICAL_VARIANTS = {
    "bicubic": None,
    "A  plain JBU (no gate, no depth term, no clean-up)": dict(use_gate=False, use_depth_term=False, use_clean=False),
    "B  + depth-consistency term": dict(use_gate=False, use_depth_term=True, use_clean=False),
    "C  + discontinuity gate (texture-copy suppression)": dict(use_gate=True, use_depth_term=True, use_clean=False),
    "D  + noise-adaptive LR clean-up  (= Classical, ours)": dict(use_gate=True, use_depth_term=True, use_clean=True),
    "E  D + hard colour-plateau snap": dict(use_gate=True, use_depth_term=True, use_clean=True, use_snap=True),
}
HYBRID_VARIANTS = ["E0 only (bicubic)", "E1 only (classical gated JBU)", "learned gate, residual delta switched off",
                   "residual delta only (gate switched off)", "full hybrid: E0 + g(E1-E0) + delta  (ours)"]


def _work(args):
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    from . import classical as C, data, degrade, metrics, pipeline
    idx, scale, protocol = args
    rgb, dep = data.get_pair(idx)
    d = degrade.Degradation(scale=scale, **degrade.PROTOCOLS[protocol])
    lr, valid = degrade.degrade(dep, d, np.random.default_rng(zlib.crc32(f"{idx}-{scale}-{protocol}".encode())))
    rows = []

    def add(group, name, pred):
        r = dict(idx=idx, scale=scale, protocol=protocol, group=group, variant=name)
        r.update(metrics.evaluate(np.asarray(pred, np.float32), dep, rgb))
        rows.append(r)

    for name, kw in CLASSICAL_VARIANTS.items():
        if kw is None:
            add("classical", name, C.bicubic_up(C.naive_fill(lr, valid), scale))
        else:
            add("classical", name, C.gate_pipeline(lr, valid, rgb, scale, **kw).depth)
    if (pipeline.WEIGHTS / "hybrid.pt").exists():
        depth, ex = pipeline.run_hybrid(lr, valid, rgb, scale, return_all=True)
        d_bic, d_gate, d_dl, g = ex["classical"].base, ex["classical"].depth, ex["dl_branch"], ex["weights"][1]
        for name, pred in zip(HYBRID_VARIANTS, [d_bic, d_gate, d_bic + g * (d_gate - d_bic), d_dl, depth]):
            add("hybrid", name, pred)
        if (pipeline.WEIGHTS / "hybrid_convex.pt").exists():
            add("hybrid", "v1 softmax-convex arbiter (superseded)", pipeline.run_hybrid(lr, valid, rgb, scale, kind="hybrid_convex"))
        if (pipeline.WEIGHTS / "dl_only.pt").exists():
            add("hybrid", "deep learning only (no classical features)", pipeline.run_dl_only(lr, valid, rgb, scale))
    return rows


def summarize(rows):
    agg: dict = {}
    for r in rows:
        agg.setdefault((r["protocol"], r["scale"], r["group"], r["variant"]), []).append(r)
    out = {}
    for k, v in agg.items():
        out["|".join(map(str, k))] = {m: float(np.nanmean([r[m] for r in v])) for m in
                                      ("rmse_cm", "mae_cm", "boundary_rmse_cm", "edge_f1", "texture_copy_rmse_cm", "ssim")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    from . import data
    test = data.split_indices("test")
    idxs = test[::max(1, len(test) // a.n)][:a.n]
    jobs = [(i, s, p) for i in idxs for s in (4, 8) for p in ("clean", "noisy+holes")]
    rows = []
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs)):
            rows += r
            if i % 40 == 0:
                print(f"ablation {i + 1}/{len(jobs)}", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ablation.json").write_text(json.dumps(summarize(rows), indent=1))
    print("saved results/ablation.json")
