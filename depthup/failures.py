"""Failure-case mining: which test frames does the hybrid get wrong, and what do they have in common?

    python -m depthup.failures   ->  results/failure_cases.json, docs/figures/failure_*.png

Run after evaluate.py (needs results/per_image_*.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from . import data, degrade, metrics, pipeline, viz

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def image_stats(rgb, dep):
    """Scene descriptors used to explain failures."""
    e = metrics.gt_edges(dep)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    band = metrics.boundary_band(e, 2)
    inner = cv2.erode(e.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    thin = e.sum() > 0 and (e & ~cv2.erode(cv2.dilate(e.astype(np.uint8), np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8)).astype(bool)).sum() / e.sum()
    return dict(edge_density=float(e.mean()), luminance=float(gray.mean()), mean_depth=float(dep.mean()),
                depth_range=float(dep.max() - dep.min()), edge_colour_contrast=float(mag[band].mean()) if band.any() else 0.0,
                texture_level=float(np.percentile(mag, 90)), thin_edge_ratio=float(thin or 0))


def load(protocol):
    return [json.loads(l) for l in open(ROOT / "results" / f"per_image_{protocol}.jsonl")]


def main(kind: str = "hybrid"):
    FIG.mkdir(parents=True, exist_ok=True)
    out = {"correlations": {}, "cases": []}
    for prot, scale in (("clean", 8), ("noisy+holes", 4)):
        try:
            rows = load(prot)
        except FileNotFoundError:
            continue
        hy = {r["idx"]: r for r in rows if r["method"] == kind and r["scale"] == scale}
        bi = {r["idx"]: r for r in rows if r["method"] == "bicubic" and r["scale"] == scale}
        idxs = sorted(set(hy) & set(bi))
        stats = {i: image_stats(*data.get_pair(i)) for i in idxs}
        y = np.array([hy[i]["rmse_cm"] for i in idxs])
        gain = np.array([bi[i]["rmse_cm"] - hy[i]["rmse_cm"] for i in idxs])
        out["correlations"][f"{prot}@{scale}x"] = {
            k: {"spearman_vs_hybrid_rmse": float(spearmanr([stats[i][k] for i in idxs], y)[0]),
                "spearman_vs_gain_over_bicubic": float(spearmanr([stats[i][k] for i in idxs], gain)[0])}
            for k in next(iter(stats.values()))}
        out["correlations"][f"{prot}@{scale}x"]["_summary"] = dict(
            n=len(idxs), frac_hybrid_better_than_bicubic=float((gain > 0).mean()), median_gain_cm=float(np.median(gain)),
            worst_regression_cm=float(gain.min()))
        picks = {"highest hybrid RMSE": int(idxs[int(np.argmax(y))]),
                 "smallest gain over bicubic": int(idxs[int(np.argmin(gain))]),
                 "highest boundary RMSE": int(max(idxs, key=lambda i: hy[i]["boundary_rmse_cm"])),
                 "lowest edge F1": int(min(idxs, key=lambda i: hy[i]["edge_f1"]))}
        for why, idx in dict.fromkeys((k, v) for k, v in picks.items()):
            pass
        seen = set()
        for why, idx in picks.items():
            if idx in seen:
                continue
            seen.add(idx)
            rgb, dep = data.get_pair(idx)
            lr, valid = degrade.degrade(dep, degrade.Degradation(scale=scale, **degrade.PROTOCOLS[prot]),
                                        np.random.default_rng(__import__("zlib").crc32(f"{idx}-{scale}-{prot}".encode())))
            pb = pipeline.METHODS["bicubic"](lr, valid, rgb, scale)
            ph, ex = pipeline.run_hybrid(lr, valid, rgb, scale, return_all=True, kind=kind)
            lo, hi = float(dep.min()), float(dep.max())
            fig, ax = plt.subplots(2, 4, figsize=(13, 7))
            panels = [("RGB", rgb), ("GT depth", viz.colorize(dep, lo, hi)), ("bicubic", viz.colorize(pb, lo, hi)), ("hybrid", viz.colorize(ph, lo, hi)),
                      ("GT discontinuities", viz.overlay_edges(rgb, metrics.gt_edges(dep))), ("bicubic |err|", viz.error_map(pb, dep)),
                      ("hybrid |err|", viz.error_map(ph, dep)), ("learned blend β (trust in classical)", viz.gray(ex["weights"][1]))]
            for a, (t, im) in zip(ax.ravel(), panels):
                a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
            m, mb = hy[idx], bi[idx]
            fig.suptitle(f"{why}: frame {idx} · {prot} {scale}× · hybrid RMSE {m['rmse_cm']:.1f} cm (bicubic {mb['rmse_cm']:.1f}) · "
                         f"boundary {m['boundary_rmse_cm']:.1f} (bicubic {mb['boundary_rmse_cm']:.1f})", fontsize=10)
            fig.tight_layout(); fig.savefig(FIG / f"failure_{prot.replace('+', '_')}_{idx}.png", dpi=110); plt.close(fig)
            out["cases"].append(dict(protocol=prot, scale=scale, reason=why, idx=idx, hybrid_rmse=m["rmse_cm"], bicubic_rmse=mb["rmse_cm"],
                                     hybrid_boundary=m["boundary_rmse_cm"], bicubic_boundary=mb["boundary_rmse_cm"], **stats[idx]))
    (ROOT / "results" / "failure_cases.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["correlations"], indent=1)[:3000])


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="hybrid", help="hybrid or hybrid_L")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.environ["DEPTHUP_DEVICE"] = a.device
    main(a.kind)
