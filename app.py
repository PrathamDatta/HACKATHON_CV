"""Streamlit app: edge-preserving depth map upsampling & refinement (P4).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from depthup import degrade, io as dio, metrics, pipeline, viz

ROOT = Path(__file__).parent
SAMPLES = sorted(ROOT.glob("samples/*_rgb.jpg"))
SHOW = ["bicubic", "jbu", "guided", "classical", "dl_only", "hybrid", "auto", "dl_only_L", "hybrid_L", "auto_L"]

st.set_page_config(page_title="EdgeGate Depth Upsampler", page_icon="🧭", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.2rem;max-width:1300px}
div[data-testid="stMetricValue"]{font-size:1.4rem}</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def warm():
    ok = {}
    for k in ("hybrid", "dl_only"):
        try:
            pipeline.load_model(k)
            ok[k] = True
        except FileNotFoundError:
            ok[k] = False
    for k in ("hybrid_L", "dl_only_L"):          # large models: load at startup so the first Reconstruct is not slow
        try:
            pipeline.load_model(k)
            ok[k] = True
        except FileNotFoundError:
            ok[k] = False
        except Exception:                        # e.g. out of memory on a tiny host: the app still works with the small models
            ok[k] = False
    ok["auto"] = ok["hybrid"] and ok["dl_only"]
    ok["auto_L"] = ok["hybrid_L"] and ok["dl_only_L"]
    return ok


def run_methods(lr, valid, rgb, scale, names):
    outs, lat = {}, {}
    for n in names:
        outs[n], lat[n] = pipeline.timed(n, lr, valid, rgb, scale)
    return outs, lat


def sample_pair(path: Path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    dep = np.asarray(Image.open(str(path).replace("_rgb.jpg", "_depth.png"))).astype(np.float32) / 1000.0
    return rgb, dep


st.title("EdgeGate — edge-preserving depth upsampling")
st.caption("Sparse / low-resolution depth → dense high-resolution depth. Classical gated bilateral filtering "
           "+ a small learned blend-and-residual network. Pick a depth-quality regime and compare against interpolation, classical and deep-only baselines.")

models = warm()
if not (models["hybrid"] and models["dl_only"]):
    st.warning("Trained weights not found — learned methods are disabled. Run `python -m depthup.train`.")
names_available = [n for n in SHOW if (n not in models or models[n])]
# the large model takes ~2.5 s per image on 4 CPU threads, so the default set is kept small enough for a free 2-vCPU host
DEFAULT_METHODS = [n for n in ("bicubic", "classical", "auto", "dl_only_L") if n in names_available]

quick = st.button("▶ Reconstruct  (uses the sidebar settings; a sample at 4× by default)", type="primary", use_container_width=True)
msg = st.container()     # errors appear here, above the tabs, so they are never off-screen
tab_run, tab_stages, tab_bench, tab_about = st.tabs(["Reconstruct", "Intermediate stages", "Benchmark results", "About"])

with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["Simulate sensor (RGB + dense depth)", "Upsample my depth map (RGB + low-res depth)"])
    scale = st.select_slider("Upsampling factor", options=[2, 4, 8], value=4)
    if mode.startswith("Simulate"):
        src = st.radio("Source", ["Sample image", "Upload my own"], horizontal=True)
        if src == "Sample image":
            pick = st.selectbox("Sample", SAMPLES, format_func=lambda p: p.stem.replace("_rgb", ""))
            up_rgb = up_dep = None
        else:
            up_rgb = st.file_uploader("RGB image (jpg/png)", type=["jpg", "jpeg", "png"])
            up_dep = st.file_uploader("Ground-truth depth (.npy / 16-bit png)", type=["npy", "npz", "png", "tif", "tiff"])
            unit = st.selectbox("Depth unit", ["auto", "metres", "millimetres", "relative"])
        st.subheader("Sensor degradation")
        noise = st.slider("Noise σ (% of depth)", 0.0, 5.0, 0.0, 0.5) / 100
        outl = st.slider("Spike outliers (%)", 0.0, 2.0, 0.0, 0.1) / 100
        holes = st.slider("Missing values (%)", 0, 50, 0, 5) / 100
        sample_mode = st.radio("LR sampling", ["bicubic", "nearest"], horizontal=True)
    else:
        up_rgb = st.file_uploader("RGB guidance image", type=["jpg", "jpeg", "png"])
        up_dep = st.file_uploader("Low-res depth (.npy / 16-bit png)", type=["npy", "npz", "png", "tif", "tiff"])
        unit = st.selectbox("Depth unit", ["auto", "metres", "millimetres", "relative"])
    methods = st.multiselect("Methods to compare", names_available, default=DEFAULT_METHODS)
    go = st.button("Reconstruct", type="primary", use_container_width=True) or quick

state = st.session_state


def prepare():
    """Return dict(rgb, lr, valid, gt|None, scale, note). Raises dio.InputError on bad input."""
    if mode.startswith("Simulate"):
        if src == "Sample image":
            rgb, gt = sample_pair(pick)
            note = "NYU Depth V2 test frame (never used for training or tuning)"
        else:
            if up_rgb is None or up_dep is None:
                raise dio.InputError("Upload both an RGB image and its dense depth map, or pick a sample.")
            rgb = dio.read_rgb(up_rgb.getvalue())
            gt, note = dio.read_depth(up_dep.getvalue(), up_dep.name, unit)
            if gt.shape != rgb.shape[:2]:
                raise dio.InputError(f"RGB is {rgb.shape[1]}x{rgb.shape[0]} but depth is {gt.shape[1]}x{gt.shape[0]}; they must match.")
            gt = np.where(np.isfinite(gt), gt, 0)
            gt = np.where(gt > 0.01, gt, np.nanmedian(gt[gt > 0.01]) if (gt > 0.01).any() else 1.0)
        rgb, gt = dio.fit_to_scale(rgb, scale, gt.astype(np.float32))
        lr, valid = degrade.degrade(gt, degrade.Degradation(scale, sample_mode, noise, outl, holes), np.random.default_rng(0))
        return dict(rgb=rgb, lr=lr, valid=valid, gt=gt, scale=scale, note=note)
    if up_rgb is None or up_dep is None:
        raise dio.InputError("Upload an RGB guidance image and a low-resolution depth map.")
    rgb = dio.read_rgb(up_rgb.getvalue())
    lr, note = dio.read_depth(up_dep.getvalue(), up_dep.name, unit)
    lr, valid = dio.to_valid(lr)
    rgb, lr, s = dio.match_depth_to_rgb(rgb, lr)
    valid = dio.to_valid(lr)[1]
    return dict(rgb=rgb, lr=lr, valid=valid, gt=None, scale=s, note=note + f"; scale inferred as {s}x")


if go:
    if not methods:
        msg.error("Select at least one method.")
    else:
        try:
            with st.spinner("Reconstructing…"):
                job = prepare()
                outs, lat = run_methods(job["lr"], job["valid"], job["rgb"], job["scale"], methods)
            state["job"], state["outs"], state["lat"], state["extra"] = job, outs, lat, None   # stages are computed on demand
        except dio.InputError as e:
            msg.error(str(e))
        except Exception as e:  # never show a stack trace to a judge
            msg.error(f"Something went wrong while reconstructing: {e}")

job = state.get("job")
with tab_run:
    if not job:
        st.info("Press **Reconstruct** above to run the default sample, or open the sidebar (the **»** arrow, top-left on a phone) "
                "to pick another sample, upload your own RGB + depth, or change the degradation.")
    else:
        rgb, gt, outs, lat = job["rgb"], job["gt"], state["outs"], state["lat"]
        st.caption(f"{job['note']} · LR {job['lr'].shape[1]}×{job['lr'].shape[0]} → HR {rgb.shape[1]}×{rgb.shape[0]} "
                   f"({job['scale']}×) · {(~job['valid']).mean() * 100:.0f}% missing")
        allv = [job["lr"][job["valid"]]] + ([gt] if gt is not None else [])
        lo, hi = float(min(a.min() for a in allv)), float(max(a.max() for a in allv))
        if gt is not None:
            rows = []
            for n, p in outs.items():
                m = metrics.evaluate(p.astype(np.float32), gt, rgb)
                rows.append({"Method": pipeline.LABELS[n], "RMSE cm": m["rmse_cm"], "MAE cm": m["mae_cm"], "PSNR dB": m["psnr_db"],
                             "SSIM": m["ssim"], "Boundary RMSE cm": m["boundary_rmse_cm"], "Edge F1": m["edge_f1"],
                             "Latency ms": lat[n]})
            st.dataframe(rows, hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.NumberColumn(format="%.3f" if k in ("SSIM", "Edge F1") else "%.2f")
                                        for k in rows[0] if k != "Method"})
        cols = st.columns(2 + (gt is not None))
        cols[0].image(rgb, caption="RGB guidance", use_container_width=True)
        lr_vis = viz.colorize(np.where(job["valid"], job["lr"], np.nan_to_num(job["lr"])), lo, hi)
        lr_vis[~job["valid"]] = (0, 0, 0)
        cols[1].image(viz.zoom(lr_vis, job["scale"]), caption="Input depth (black = missing)", use_container_width=True)
        if gt is not None:
            cols[2].image(viz.colorize(gt, lo, hi), caption="Ground truth", use_container_width=True)
        st.subheader("Reconstructions")
        grid = st.columns(3)
        for i, (n, p) in enumerate(outs.items()):
            with grid[i % 3]:
                st.image(viz.colorize(p, lo, hi), caption=f"{pipeline.LABELS[n]} · {lat[n]:.0f} ms", use_container_width=True)
                if gt is not None:
                    st.image(viz.error_map(p, gt), caption="|error| (0–30 cm)", use_container_width=True)
        if gt is not None:
            st.subheader("Boundary zoom")
            e = metrics.gt_edges(gt)
            ys, xs = np.nonzero(e)
            if len(ys):
                c = st.columns(2)
                cy = c[0].slider("Zoom y", 0, gt.shape[0] - 64, int(np.clip(np.median(ys) - 32, 0, gt.shape[0] - 64)))
                cx = c[1].slider("Zoom x", 0, gt.shape[1] - 64, int(np.clip(np.median(xs) - 32, 0, gt.shape[1] - 64)))
                zc = st.columns(len(outs) + 2)
                zc[0].image(viz.zoom(viz.crop(rgb, cy, cx, 64), 4), caption="RGB", use_container_width=True)
                for k, (n, p) in enumerate(outs.items()):
                    zc[k + 1].image(viz.zoom(viz.crop(viz.colorize(p, lo, hi), cy, cx, 64), 4), caption=n, use_container_width=True)
                zc[-1].image(viz.zoom(viz.crop(viz.colorize(gt, lo, hi), cy, cx, 64), 4), caption="GT", use_container_width=True)
        import io as _io
        buf = _io.BytesIO()
        best = outs.get("hybrid", next(iter(outs.values())))
        np.save(buf, best.astype(np.float32))
        st.download_button("Download result (.npy, metres)", buf.getvalue(), "depth_hr.npy")

with tab_stages:
    ex = state.get("extra")
    if not job:
        st.info("Run a reconstruction first, then come back here to see the intermediate stages.")
    elif ex is None:
        st.write("The intermediate stages (discontinuity map, gate, learned blend and residual) take a couple of extra seconds to compute.")
        if st.button("Compute intermediate stages", type="primary"):
            hk = "hybrid_L" if models.get("hybrid_L") else "hybrid"
            if not models.get(hk):
                st.error("The hybrid model weights are not available.")
            else:
                with st.spinner("Computing stages…"):
                    _, state["extra"] = pipeline.run_hybrid(job["lr"], job["valid"], job["rgb"], job["scale"], return_all=True, kind=hk)
                st.rerun()
    else:
        o, w, rgb = ex["classical"], ex["weights"], job["rgb"]
        st.markdown("**Pipeline:** clean LR → bicubic base → depth-discontinuity detection → RGB gate → "
                    "gated tri-lateral JBU → learned blend β + residual δ.")
        items = [("Missing-value mask (LR)", viz.zoom(np.where(job["valid"], 255, 40).astype(np.uint8)[..., None].repeat(3, -1), job["scale"])),
                 ("Bicubic base (cleaned)", viz.colorize(o.base)),
                 ("Depth discontinuities (LR step detector)", viz.gray(o.extras["soft"])),
                 ("Discontinuity gate G (where RGB may influence depth)", viz.gray(o.extras["gate"])),
                 ("RGB edges", viz.gray(o.extras["rgb_edge"])),
                 ("Edges RGB ∧ ¬depth (texture the gate blocks)", viz.gray(np.clip(o.extras["rgb_edge"] * (1 - o.extras["gate"]), 0, 1))),
                 ("Classical gated-JBU output", viz.colorize(o.depth)),
                 ("Colour-plateau snap candidate", viz.colorize(o.extras["snapped"])),
                 ("E0 + δ (residual only)", viz.colorize(ex["dl_branch"])),
                 ("Learned blend 1−β (trust in bicubic)", viz.gray(w[0])),
                 ("Learned blend β (trust in classical sharpening)", viz.gray(w[1])),
                 ("Learned residual |δ| (0–20 cm)", viz.gray(w[2]))]
        cols = st.columns(3)
        for i, (t, img) in enumerate(items):
            cols[i % 3].image(img, caption=t, use_container_width=True)
        st.caption(f"Estimated sensor noise σ ≈ {o.extras['sigma_est'] * 100:.2f} cm · spikes removed: {o.extras['spikes']}")

with tab_bench:
    p = ROOT / "results" / "summary.json"
    if not p.exists():
        st.info("Run `python -m depthup.report` to generate benchmark results.")
    else:
        S = json.loads(p.read_text())
        for prot in ("clean", "noisy", "noisy+holes"):
            rows = []
            for key, v in S.items():
                pr, sc, m = key.split("|")
                if pr == prot:
                    rows.append({"Scale": f"{sc}×", "Method": pipeline.LABELS.get(m, m), "RMSE cm": v["rmse_cm"], "MAE cm": v["mae_cm"],
                                 "PSNR": v["psnr_db"], "SSIM": v["ssim"], "Boundary RMSE": v["boundary_rmse_cm"],
                                 "Edge F1": v["edge_f1"], "Latency ms": v["latency_ms"], "n": v["n"]})
            if rows:
                st.markdown(f"**Protocol: {prot}**")
                st.dataframe(sorted(rows, key=lambda r: (r["Scale"], r["RMSE cm"])), hide_index=True, use_container_width=True)
        for f in sorted((ROOT / "docs" / "figures").glob("*.png"))[:6]:
            st.image(str(f), use_container_width=True)

with tab_about:
    st.markdown((ROOT / "docs" / "ABOUT_APP.md").read_text() if (ROOT / "docs" / "ABOUT_APP.md").exists() else "See README.md")
