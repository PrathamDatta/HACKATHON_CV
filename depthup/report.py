"""Aggregate results -> summary.json, eval_P4_<team>.md, figures.

    python -m depthup.report --team T19 --usns "1RUA24CSE0324,1RAU24CSE0353,1RUA24CSE0354"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data, degrade, metrics, pipeline, viz
from .evaluate import summarize

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "docs" / "figures"
ORDER = ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical", "dl_only", "hybrid_convex", "hybrid", "auto",
         "dl_only_L", "hybrid_L", "auto_L", "auto_tta", "auto_L_tta", "auto_E_tta"]
# colour-blind safe (Okabe-Ito); ours highlighted
COL = {"nearest": "#999999", "bilinear": "#bbbbbb", "bicubic": "#555555", "jbu": "#56B4E9", "guided": "#009E73",
       "classical": "#E69F00", "dl_only": "#CC79A7", "hybrid_convex": "#8c564b", "hybrid": "#D55E00", "auto": "#B22222",
       "dl_only_L": "#9467bd", "hybrid_L": "#e377c2", "auto_L": "#17becf", "auto_tta": "#7f7f7f", "auto_L_tta": "#000000", "auto_E_tta": "#d62728"}
PROTOCOLS = ["clean", "noisy", "noisy+holes"]
HYB = "hybrid_L" if (pipeline.WEIGHTS / "hybrid_L.pt").exists() else "hybrid"      # model used for the figures


def load_all():
    rows = []
    for p in PROTOCOLS:
        f = RES / f"per_image_{p}.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in open(f)]
    return rows


def bootstrap_ci(a, b, n=2000, seed=0):
    """Paired bootstrap CI of mean(a - b)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_table(rows, a_name="hybrid", b_name="bicubic"):
    out = []
    for prot in PROTOCOLS:
        for s in (2, 4, 8):
            sel = {m: {r["idx"]: r for r in rows if r["protocol"] == prot and r["scale"] == s and r["method"] == m}
                   for m in (b_name, a_name)}
            common = sorted(set(sel[b_name]) & set(sel[a_name]))
            if not common:
                continue
            for met in ("rmse_cm", "boundary_rmse_cm"):
                mean, lo, hi = bootstrap_ci([sel[a_name][i][met] for i in common], [sel[b_name][i][met] for i in common])
                out.append((prot, s, met, mean, lo, hi, len(common)))
    return out


def write_eval_md(S, eff, team, usns, rows):
    L = [usns, "", f"# eval_P4_{team}: Edge-Preserving Depth Map Upsampling and Refinement", "",
         "Dataset: NYU Depth V2 (1,449 labelled frames). Split (contiguous scene blocks): train 0-799, val 800-999, "
         "**test 1000-1448 (never used for training, validation or tuning)**. Depth assumed = raw x 10 m. "
         "Metrics are means over test frames. RMSE/MAE in cm. PSNR peak = per-image GT max depth. "
         "Boundary band = +/-3 px around GT discontinuities (|grad| > 0.12 m/px). Edge F1 uses 2 px tolerance.", ""]
    names = {m: pipeline.LABELS[m] for m in ORDER}
    for prot in PROTOCOLS:
        ns = sorted({v["n"] for k, v in S.items() if k.startswith(prot + "|")})
        if not ns:
            continue
        L += [f"## Protocol `{prot}`  (n = {', '.join(map(str, ns))} test frames)", ""]
        sections = [("1. RMSE and MAE (cm, lower is better)", [("rmse_cm", "RMSE"), ("mae_cm", "MAE")]),
                    ("2. PSNR (dB) and SSIM (higher is better)", [("psnr_db", "PSNR"), ("ssim", "SSIM")]),
                    ("3. Edge preservation and boundary accuracy", [("boundary_rmse_cm", "Boundary RMSE cm"), ("edge_f1", "Edge F1"),
                                                                     ("texture_copy_rmse_cm", "Texture-copy RMSE cm")])]
        for title, cols in sections:
            L += [f"### {title}", "", "| Scale | Method | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
            for s in (2, 4, 8):
                for m in ORDER:
                    v = S.get(f"{prot}|{s}|{m}")
                    if v:
                        L.append(f"| {s}x | {names[m]} | " + " | ".join(f"{v[c[0]]:.3f}" if "ssim" in c[0] or "f1" in c[0] else f"{v[c[0]]:.2f}" for c in cols) + " |")
            L.append("")
    L += ["## 4. Performance across upsampling factors (RMSE cm, clean protocol)", "",
          "| Method | 2x | 4x | 8x |", "|---|---|---|---|"]
    for m in ORDER:
        vs = [S.get(f"clean|{s}|{m}") for s in (2, 4, 8)]
        if all(vs):
            L.append(f"| {names[m]} | " + " | ".join(f"{v['rmse_cm']:.2f}" for v in vs) + " |")
    L += ["", "## 5. Processing latency and memory (480x640 input, 4 CPU threads, fresh process per method)", ""]
    if eff:
        L += [f"Parameters: hybrid {eff['params']['hybrid']:,}, deep-only {eff['params']['dl_only']:,}. "
              f"Weight files: " + ", ".join(f"{k} {v:.0f} KB" for k, v in eff["weights_kb"].items()) + ".", "",
              "| Method | Scale | Median ms | p95 ms | Peak RSS MB | RSS growth MB |", "|---|---|---|---|---|---|"]
        for m in ORDER:
            for s in (2, 4, 8):
                t = eff["timing"].get(f"{m}|{s}")
                if t:
                    L.append(f"| {names[m]} | {s}x | {t['median_ms']:.0f} | {t['p95_ms']:.0f} | {t['peak_rss_mb']:.0f} | {t['peak_rss_mb'] - t['warm_rss_mb']:.0f} |")
    for a_name, b_name in (("hybrid", "bicubic"), ("hybrid", "dl_only"), ("hybrid_L", "dl_only_L"), ("auto_L_tta", "auto_L"), ("auto_L_tta", "auto")):
        pt = paired_table(rows, a_name, b_name)
        if pt:
            L += ["", f"## 6. Statistical check: {a_name} minus {b_name} "
                  "(paired bootstrap, 95% CI; negative = hybrid better; a CI that contains 0 means no reliable difference)", "",
                  "| Protocol | Scale | Metric | Mean diff | 95% CI | n |", "|---|---|---|---|---|---|"]
            for prot, s, met, mean, lo, hi, n in pt:
                L.append(f"| {prot} | {s}x | {met} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | {n} |")
    (ROOT / f"eval_P4_{team}.md").write_text("\n".join(L) + "\n")


def fig_metrics(S):
    FIG.mkdir(parents=True, exist_ok=True)
    for prot in PROTOCOLS:
        if not any(k.startswith(prot + "|") for k in S):
            continue
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for a, (met, ttl) in zip(ax, [("rmse_cm", "RMSE (cm) ↓"), ("boundary_rmse_cm", "Boundary RMSE (cm) ↓"), ("edge_f1", "Edge F1 ↑")]):
            for m in ORDER:
                ys = [S.get(f"{prot}|{s}|{m}", {}).get(met) for s in (2, 4, 8)]
                if None in ys:
                    continue
                a.plot([2, 4, 8], ys, "-o", color=COL[m], lw=3 if m == "hybrid" else 1.6, label=pipeline.LABELS[m],
                       alpha=1 if m in ("hybrid", "bicubic", "classical", "dl_only") else 0.55)
            a.set_xscale("log", base=2); a.set_xticks([2, 4, 8]); a.set_xticklabels(["2×", "4×", "8×"])
            a.set_title(ttl, fontsize=11); a.grid(alpha=0.25)
        ax[0].legend(fontsize=7.5, frameon=False)
        fig.suptitle(f"NYU Depth V2 test · protocol: {prot}", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / f"metrics_{prot.replace('+', '_')}.png", dpi=130)
        plt.close(fig)


def fig_pareto(S, eff):
    if not eff:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ORDER:
        v, t = S.get(f"clean|4|{m}"), eff["timing"].get(f"{m}|4")
        if v and t:
            ax.scatter(t["median_ms"], v["rmse_cm"], s=90, color=COL[m]); ax.annotate(m, (t["median_ms"], v["rmse_cm"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("median latency, ms (480×640, 4×)"); ax.set_ylabel("RMSE cm (4×, clean)"); ax.grid(alpha=0.25)
    ax.set_title("Accuracy vs. cost")
    fig.tight_layout(); fig.savefig(FIG / "pareto.png", dpi=130); plt.close(fig)


def _prep(idx, scale, protocol):
    rgb, dep = data.get_pair(idx)
    lr, valid = degrade.degrade(dep, degrade.Degradation(scale=scale, **degrade.PROTOCOLS[protocol]), np.random.default_rng(1))
    return rgb, dep, lr, valid


def fig_qualitative(idxs=(1005, 1160, 1330), scale=8, protocol="clean", name="qualitative_8x"):
    cols = ["bicubic", "jbu", "classical", "dl_only", "hybrid"] + ([HYB, "dl_only_L"] if HYB == "hybrid_L" else [])
    fig, ax = plt.subplots(2 * len(idxs), len(cols) + 3, figsize=(2.2 * (len(cols) + 3), 3.4 * 2 * len(idxs) / 2))
    for r, idx in enumerate(idxs):
        rgb, dep, lr, valid = _prep(idx, scale, protocol)
        lo, hi = float(dep.min()), float(dep.max())
        ax[2 * r, 0].imshow(rgb); ax[2 * r, 0].set_title("RGB", fontsize=8)
        lrv = viz.colorize(np.repeat(np.repeat(lr, scale, 0), scale, 1), lo, hi); lrv[np.repeat(np.repeat(~valid, scale, 0), scale, 1)] = 0
        ax[2 * r, 1].imshow(lrv); ax[2 * r, 1].set_title(f"input ({scale}× LR)", fontsize=8)
        outs = {c: pipeline.METHODS[c](lr, valid, rgb, scale) for c in cols}
        for k, c in enumerate(cols):
            ax[2 * r, k + 2].imshow(viz.colorize(outs[c], lo, hi)); ax[2 * r, k + 2].set_title(pipeline.LABELS[c], fontsize=8)
            m = metrics.evaluate(outs[c].astype(np.float32), dep, rgb)
            ax[2 * r + 1, k + 2].imshow(viz.error_map(outs[c], dep)); ax[2 * r + 1, k + 2].set_title(f"RMSE {m['rmse_cm']:.1f} · bdry {m['boundary_rmse_cm']:.1f} cm", fontsize=7)
        ax[2 * r, -1].imshow(viz.colorize(dep, lo, hi)); ax[2 * r, -1].set_title("ground truth", fontsize=8)
        ax[2 * r + 1, 0].imshow(viz.overlay_edges(rgb, metrics.gt_edges(dep))); ax[2 * r + 1, 0].set_title("GT discontinuities", fontsize=8)
        ax[2 * r + 1, 1].axis("off"); ax[2 * r + 1, -1].axis("off")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{scale}× upsampling, protocol '{protocol}' (error maps 0–30 cm)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=120); plt.close(fig)


def fig_stages(idx=1160, scale=4, protocol="noisy+holes"):
    rgb, dep, lr, valid = _prep(idx, scale, protocol)
    depth, ex = pipeline.run_hybrid(lr, valid, rgb, scale, return_all=True, kind=HYB)
    o, w = ex["classical"], ex["weights"]
    lo, hi = float(dep.min()), float(dep.max())
    lrv = viz.colorize(np.repeat(np.repeat(lr, scale, 0), scale, 1), lo, hi); lrv[np.repeat(np.repeat(~valid, scale, 0), scale, 1)] = 0
    panels = [("RGB", rgb), ("noisy LR input (black = missing)", lrv), ("cleaned bicubic base", viz.colorize(o.base, lo, hi)),
              ("depth discontinuities", viz.gray(o.extras["soft"])), ("discontinuity gate G", viz.gray(o.extras["gate"])),
              ("RGB edges", viz.gray(o.extras["rgb_edge"])), ("RGB edges blocked by gate", viz.gray(np.clip(o.extras["rgb_edge"] * (1 - o.extras["gate"]), 0, 1))),
              ("classical gated JBU", viz.colorize(o.depth, lo, hi)), ("colour-plateau snap cand.", viz.colorize(o.extras["snapped"], lo, hi)),
              ("E0 + δ (residual only)", viz.colorize(ex["dl_branch"], lo, hi)), ("1−β (trust in bicubic)", viz.gray(w[0])),
              ("learned blend β (trust in classical)", viz.gray(w[1])), ("learned |δ| (0–20 cm)", viz.gray(w[2])), ("HYBRID output", viz.colorize(depth, lo, hi)),
              ("ground truth", viz.colorize(dep, lo, hi)), ("hybrid |error| 0–30cm", viz.error_map(depth, dep))]
    fig, ax = plt.subplots(4, 4, figsize=(12, 9.4))
    for a, (t, im) in zip(ax.ravel(), panels):
        a.imshow(im); a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Intermediate stages · test frame {idx}, {scale}×, noisy+holes", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "stages.png", dpi=120); plt.close(fig)


def fig_boundary(idx=1330, scale=8):
    rgb, dep, lr, valid = _prep(idx, scale, "clean")
    e = metrics.gt_edges(dep)
    # choose the 64x64 window with the strongest depth discontinuity
    cnt = np.add.reduceat(np.add.reduceat(e.astype(int), np.arange(0, e.shape[0] - 63, 32), 0), np.arange(0, e.shape[1] - 63, 32), 1)
    cy, cx = [int(v) * 32 for v in np.unravel_index(np.argmax(cnt), cnt.shape)]
    names = ["bicubic", "jbu", "classical", "dl_only", "hybrid"] + ([HYB] if HYB == "hybrid_L" else [])
    outs = {n: pipeline.METHODS[n](lr, valid, rgb, scale) for n in names}
    lo, hi = float(dep.min()), float(dep.max())
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(2, 8, height_ratios=[1, 1])
    crops = [("RGB", viz.crop(rgb, cy, cx, 64))] + [(n, viz.crop(viz.colorize(outs[n], lo, hi), cy, cx, 64)) for n in names] + [("GT", viz.crop(viz.colorize(dep, lo, hi), cy, cx, 64))]
    for k, (t, im) in enumerate(crops):
        a = fig.add_subplot(gs[0, k]); a.imshow(viz.zoom(im, 3), interpolation="nearest"); a.set_title(t, fontsize=8); a.axis("off")
    ay, ax_ = np.nonzero(viz.crop(e, cy, cx, 64))
    row = int(np.median(ay)) + cy
    a = fig.add_subplot(gs[1, :])
    xs = np.arange(cx, cx + 64)
    a.plot(xs, dep[row, cx:cx + 64], "k-", lw=3, label="ground truth")
    for n in names:
        a.plot(xs, outs[n][row, cx:cx + 64], color=COL[n], lw=2 if n == "hybrid" else 1.3, label=n)
    a.set_title(f"depth profile along image row {row} (frame {idx}, {scale}×)", fontsize=9); a.set_ylabel("metres"); a.legend(ncol=6, fontsize=8, frameon=False); a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "boundary_profile.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="T19")
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    rows = load_all()
    S = summarize(rows) if rows else {}
    (RES / "summary.json").write_text(json.dumps(S, indent=1))
    eff = json.loads((RES / "efficiency.json").read_text()) if (RES / "efficiency.json").exists() else None
    write_eval_md(S, eff, a.team, a.usns, rows)
    fig_metrics(S); fig_pareto(S, eff)
    if not a.no_figs and (pipeline.WEIGHTS / "hybrid.pt").exists():
        fig_qualitative(); fig_qualitative((1005, 1160, 1330), 4, "noisy+holes", "qualitative_4x_noisy_holes")
        fig_stages(); fig_boundary()
    print("report done")
