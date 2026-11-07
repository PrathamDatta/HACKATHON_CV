"""Generate docs/ABLATION.md, docs/FAILURES.md and README.md from results/*.json (no hand-copied numbers in tables)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/summary.json").read_text())
A = json.loads((ROOT / "results/ablation.json").read_text())
F = json.loads((ROOT / "results/failure_cases.json").read_text())
E = json.loads((ROOT / "results/efficiency.json").read_text())
EXTRA = [m for m in ("dl_only_L", "hybrid_L", "auto_L", "auto_L_tta", "auto_E_tta") if "clean|2|" + m in S]
LAB = {"bicubic": "Bicubic", "jbu": "JBU", "guided": "Guided filter", "classical": "Classical gated JBU (ours)",
       "dl_only": "Deep learning only", "hybrid_convex": "Hybrid v1 (softmax arbiter)", "hybrid": "Hybrid v2 (ours)",
       "auto": "EdgeGate-Auto (ours)"}


def ablation_md():
    L = ["# Ablation study", "",
         "80 evenly-spaced **test** frames (subset of the 449; numbers differ slightly from the full-test table in `eval_P4_*.md`). "
         "All classical and hybrid rows use the **small CPU-trained models**; the large deep-only model (`dl_only_L`) was trained afterwards and appears only in the main comparison (`eval_P4_*.md`). Same degradations and seeds for every variant. RMSE and boundary RMSE in cm (lower is better), edge F1 higher is better, "
         "texture-copy RMSE (error where depth is flat but RGB textured) lower is better.", ""]
    for prot in ("clean", "noisy+holes"):
        for sc in (4, 8):
            L += [f"## {prot}, {sc}x", "", "| Group | Variant | RMSE | Boundary RMSE | Edge F1 | Texture-copy RMSE |", "|---|---|---|---|---|---|"]
            for k, v in A.items():
                p, s, g, name = k.split("|")
                if p == prot and int(s) == sc:
                    L.append(f"| {g} | {name} | {v['rmse_cm']:.2f} | {v['boundary_rmse_cm']:.1f} | {v['edge_f1']:.2f} | {v['texture_copy_rmse_cm']:.2f} |")
            L.append("")
    g = lambda p, s, name, m: next(v[m] for k, v in A.items() if k.startswith(f"{p}|{s}|") and k.split("|")[3].startswith(name))  # noqa: E731
    L += ["## What each component contributes (8x; see tables for 4x)", "",
          f"1. **Learning is the biggest RMSE lever.** Adding the learned residual to bicubic (`residual δ only`) cuts RMSE from "
          f"{g('clean', 8, 'E0 only', 'rmse_cm'):.2f} to {g('clean', 8, 'residual δ only', 'rmse_cm'):.2f} cm on clean input and from "
          f"{g('noisy+holes', 8, 'E0 only', 'rmse_cm'):.2f} to {g('noisy+holes', 8, 'residual δ only', 'rmse_cm'):.2f} cm on noisy+holes.",
          f"2. **Edge sharpness comes from the classical stage.** Plain JBU raises edge F1 from {g('clean', 8, 'bicubic', 'edge_f1'):.2f} to "
          f"{g('clean', 8, 'A ', 'edge_f1'):.2f} (clean, 8x) without hurting RMSE.",
          f"3. **The discontinuity gate does what it was designed to, modestly.** Plain JBU slightly *increases* texture-copy error over bicubic "
          f"({g('clean', 8, 'A ', 'texture_copy_rmse_cm'):.2f} vs {g('clean', 8, 'bicubic', 'texture_copy_rmse_cm'):.2f} cm, clean 8x); adding the depth term and gate brings it to "
          f"{g('clean', 8, 'C ', 'texture_copy_rmse_cm'):.2f} cm, below bicubic, with RMSE unchanged. Under noisy+holes the gate has no measurable effect "
          f"({g('noisy+holes', 8, 'B ', 'texture_copy_rmse_cm'):.2f} to {g('noisy+holes', 8, 'C ', 'texture_copy_rmse_cm'):.2f}): noise dominates there.",
          f"4. **LR clean-up matters only when the input is degraded** (as designed: it is adaptive). Noisy+holes 8x: RMSE "
          f"{g('noisy+holes', 8, 'C ', 'rmse_cm'):.2f} to {g('noisy+holes', 8, 'D ', 'rmse_cm'):.2f} cm and texture-copy {g('noisy+holes', 8, 'C ', 'texture_copy_rmse_cm'):.2f} to "
          f"{g('noisy+holes', 8, 'D ', 'texture_copy_rmse_cm'):.2f}; on clean input it changes nothing.",
          f"5. **Negative result: the hard colour-plateau snap hurts** ({g('clean', 8, 'D ', 'rmse_cm'):.2f} to {g('clean', 8, 'E ', 'rmse_cm'):.2f} cm clean 8x). "
          "NYU's inpainted ground truth has soft boundaries, so a wrong plateau assignment costs the full depth jump. It is kept only as an input feature.",
          f"6. **The learned blend β adds a small further gain** on top of the residual: {g('clean', 8, 'residual δ only', 'rmse_cm'):.2f} to "
          f"{g('clean', 8, 'full hybrid', 'rmse_cm'):.2f} (clean) and {g('noisy+holes', 8, 'residual δ only', 'rmse_cm'):.2f} to {g('noisy+holes', 8, 'full hybrid', 'rmse_cm'):.2f} (noisy+holes), i.e. 2-3%.",
          f"7. **Structure of the arbiter matters.** The v1 softmax-convex arbiter is clearly worse than the v2 residual form "
          f"({g('clean', 8, 'v1', 'rmse_cm'):.2f} vs {g('clean', 8, 'full hybrid', 'rmse_cm'):.2f} clean; {g('noisy+holes', 8, 'v1', 'rmse_cm'):.2f} vs {g('noisy+holes', 8, 'full hybrid', 'rmse_cm'):.2f} noisy+holes).",
          f"8. **Deep-only vs hybrid depends on input quality.** On pristine input deep-only is slightly better "
          f"({g('clean', 8, 'deep learning only', 'rmse_cm'):.2f} vs {g('clean', 8, 'full hybrid', 'rmse_cm'):.2f}); with noise and holes the hybrid is clearly better "
          f"({g('noisy+holes', 8, 'full hybrid', 'rmse_cm'):.2f} vs {g('noisy+holes', 8, 'deep learning only', 'rmse_cm'):.2f}). This motivates the Auto selector.", "",
          "**Not ablated by retraining** (time): removing the snap/edge/gate *input channels* of the network would need retraining; the ablation above "
          "switches components off at inference or compares separately trained models (deep-only = no classical features at all)."]
    return "\n".join(L) + "\n"


def failures_md():
    c = F["correlations"]
    L = ["# Failure-case analysis", "",
         "Method: for every test frame we computed scene descriptors (GT edge density, depth range, mean depth, luminance, edge colour contrast, texture level) "
         "and correlated them (Spearman) with the **small hybrid's** RMSE (the failure analysis was run before the large model existed; the same scene properties are expected to be hard for any model, but this was not re-checked); we then inspected the worst frames (`results/failure_cases.json`, figures in `docs/figures/failure_*.png`).", "",
         "## What predicts failure", "", "| Descriptor | ρ vs hybrid RMSE (clean 8x, n=449) | ρ vs hybrid RMSE (noisy+holes 4x, n=150) |", "|---|---|---|"]
    a, b = c["clean@8x"], c["noisy+holes@4x"]
    for k in ("edge_density", "depth_range", "mean_depth", "texture_level", "luminance", "edge_colour_contrast"):
        L.append(f"| {k} | {a[k]['spearman_vs_hybrid_rmse']:.2f} | {b[k]['spearman_vs_hybrid_rmse']:.2f} |")
    sa, sb = a["_summary"], b["_summary"]
    L += ["", f"Error is driven by **scene geometry, not appearance**: frames with many depth discontinuities and a large depth range are hard (ρ ≈ 0.9), while brightness and the local colour contrast "
          f"at edges are unrelated (|ρ| ≤ 0.13). Our initial hypothesis that low-contrast edges would dominate failures was **not supported**. "
          f"The hybrid beats bicubic on {sa['frac_hybrid_better_than_bicubic'] * 100:.0f}% of frames (clean 8x; median gain {sa['median_gain_cm']:.1f} cm) and on "
          f"{sb['frac_hybrid_better_than_bicubic'] * 100:.0f}% (noisy+holes 4x; median gain {sb['median_gain_cm']:.1f} cm): there is no frame on which it is worse. "
          "The lists below are therefore the *hardest* frames, not regressions.", "",
          "| Protocol | Selection | Frame | Hybrid RMSE | Bicubic RMSE | Hybrid boundary | Bicubic boundary | GT edge density | Depth range m |", "|---|---|---|---|---|---|---|---|---|"]
    for k in F["cases"]:
        L.append(f"| {k['protocol']} {k['scale']}x | {k['reason'].replace('worst regression vs bicubic', 'smallest gain over bicubic')} | {k['idx']} | {k['hybrid_rmse']:.1f} | {k['bicubic_rmse']:.1f} | {k['hybrid_boundary']:.1f} | "
                 f"{k['bicubic_boundary']:.1f} | {k['edge_density']:.3f} | {k['depth_range']:.1f} |")
    L += ["", "## Failure case 1: thin structures and glass (frame 1015, clean 8x, `failure_clean_1015.png`)",
          "Worst frame of the test set (RMSE about 21 cm, versus a test mean of about 5 cm). A dark corridor with a mirrored/glass door. "
          "**Causes (visible in the error maps):** (a) the door frames and mirror edges are narrower than one low-resolution cell (8 px) while the depth jump across them is metres, "
          "so the structure is aliased away in the input and cannot be recovered by any method (bicubic and hybrid errors are almost identical there); "
          "(b) the glass/reflection region has depth that bears no relation to the RGB structure, so RGB guidance carries no information. "
          "The hybrid gains almost nothing here (21.6 vs 22.2 cm). **Mitigation not implemented:** multi-scale or higher-resolution LR evidence, or a reflective-surface cue.", "",
          "## Failure case 2: holes on boundaries and distance-proportional noise (frame 1240, noisy+holes 4x, `failure_noisy_holes_1240.png`)",
          "Large depth jumps across a 9 m range. **Causes:** (a) when missing samples fall exactly on a boundary there are no samples of one of the two surfaces nearby, so no method can place the edge precisely; "
          "the hybrid still halves bicubic's boundary error (25 vs 45 cm) but remains the worst case; (b) the corruption noise is multiplicative (2% of depth), so far surfaces (about 4 m) carry about 8 cm of noise, "
          "visible as residual speckle on the far wall that the learned residual only partly removes.", "",
          "## Failure case 3: false discontinuities from surviving spikes (frame 1160, `docs/figures/stages.png`)",
          "In the 'depth discontinuities' panel, isolated dots appear away from any real edge: noise spikes that survive clean-up trigger the second-difference step detector, "
          "which opens the gate locally and lets RGB colour influence the classical estimate (a small texture-copy risk). The learned blend β mostly suppresses these "
          "(its trust map is low around them) but not completely. **Mitigation:** a connected-component size filter on the discontinuity map (a real step is a line, a spike is a point).", "",
          "## Failure case 4: boundary F1 is unstable on simple scenes (frames 1032, 1072)",
          "Frames chosen by 'lowest edge F1' have only about 0.4% edge pixels; a handful of mislocated edge pixels moves F1 a lot while RMSE is small (about 2 cm). "
          "This is a metric artefact, not a reconstruction failure, and the reason we report boundary-band RMSE alongside F1.", "",
          "## Other limitations",
          "* NYU ground truth is Kinect depth with inpainting: boundaries are soft, so our boundary metrics under-reward sharpness beyond the GT's own precision. "
          "A hard plateau snap, which is sharp by construction, therefore *loses* on these metrics (see ablation).",
          "* The synthetic noise model (multiplicative Gaussian + spikes + holes) is not a real Kinect noise model; transfer to real low-resolution time-of-flight data is untested.",
          "* Trained and tested on indoor scenes only; no cross-dataset (KITTI/Middlebury) test was run."]
    return "\n".join(L) + "\n"


NAMES = {"bicubic": "Bicubic", "jbu": "Joint bilateral", "classical": "Classical (ours)", "dl_only": "Deep only (small)",
         "hybrid": "Hybrid v2 (small)", "auto": "Auto (small)", "dl_only_L": "Deep only-L (GPU)", "hybrid_L": "Hybrid-L (GPU)",
         "auto_L": "Auto-L (GPU)", "auto_L_tta": "Auto-L + flip TTA (GPU)", "auto_E_tta": "**Auto-L ensemble + TTA (GPU)**"}
ARMS = ["bicubic", "jbu", "classical", "dl_only", "hybrid", "auto"] + EXTRA


def best_arms():
    """One generated sentence per protocol: which arm has the lowest RMSE in how many of the 3 scales."""
    out = []
    for prot in ("clean", "noisy", "noisy+holes"):
        wins = {}
        for s in (2, 4, 8):
            m = min((a for a in ARMS if f"{prot}|{s}|{a}" in S), key=lambda a: S[f"{prot}|{s}|{a}"]["rmse_cm"])
            wins[m] = wins.get(m, 0) + 1
        out.append(f"`{prot}`: " + ", ".join(f"{NAMES[m]} best at {n}/3 scales" for m, n in wins.items()))
    return "; ".join(out)


def headline():
    rows = ["| Condition | " + " | ".join(NAMES[a] for a in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for prot, n in (("clean", 449), ("noisy", 150), ("noisy+holes", 150)):
        for s in (2, 4, 8):
            v = [S[f"{prot}|{s}|{m}"]["rmse_cm"] for m in ARMS]
            b = min(v)
            rows.append(f"| {prot}, {s}x (n={n}) | " + " | ".join((f"**{x:.2f}**" if x == b else f"{x:.2f}") for x in v) + " |")
    return "\n".join(rows)


def readme():
    t = E["timing"]
    lat = lambda m, s: f"{t[f'{m}|{s}']['median_ms']:.0f}"  # noqa: E731
    gain = lambda p, s: 100 * (1 - S[f"{p}|{s}|dl_only_L"]["rmse_cm"] / S[f"{p}|{s}|dl_only"]["rmse_cm"]) if "dl_only_L" in EXTRA else 0.0  # noqa: E731
    L_NOTE = ("* **Large GPU-trained deep-only model (`dl_only_L`, about 9 M parameters, 12,350 steps, batch 16, 256 px crops, bf16 + EMA, trained on frames 0-949):** it has the lowest RMSE and boundary RMSE in every protocol and scale, "
              f"e.g. {gain('clean', 8):.0f}% lower RMSE than the small deep-only model at clean 8x and {gain('noisy+holes', 4):.0f}% lower at noisy+holes 4x. "
              "**A large hybrid (`hybrid_L`) was not trained for lack of time**, so whether the classical features help a *large* model is untested; the only evidence on that question is at small-model scale (next bullet).") if "dl_only_L" in EXTRA else ""
    return f"""---
title: EdgeGate Depth Upsampler
sdk: docker
app_port: 8501
---
# EdgeGate: edge-preserving depth map upsampling (Computer Vision Hackathon 2026, problem P4)

Turns sparse or low-resolution depth (2x / 4x / 8x, with noise and missing values) into dense high-resolution depth, using RGB guidance **without copying RGB texture into depth**.

**Approach in one paragraph.** A classical stage cleans the depth (hole filling, spike removal, noise-adaptive smoothing), detects *depth discontinuities* with a step detector, and lets RGB influence the result **only near those discontinuities** (a gated, tri-lateral joint bilateral upsampling). A small U-Net (0.22 M parameters) then predicts a residual and a per-pixel blend β between the safe bicubic estimate and the sharp classical one: `out = E0 + β (E1 − E0) + δ`. Because β = 0 recovers the deep-only model, the hybrid cannot be structurally worse than it. An **Auto** selector uses the deep-only model for pristine input and the hybrid for noisy or incomplete input.

## Results (NYU Depth V2, held-out test frames; RMSE in cm, lower is better)

{headline()}

* All numbers come from `results/` and are reproduced by `scripts/reproduce.sh`; full metrics (MAE, PSNR, SSIM, boundary RMSE, edge F1, texture-copy RMSE, latency, memory, paired bootstrap CIs) are in **`eval_P4_T01.md`**.
* **Best arm per protocol (generated from the table):** {best_arms()}.
{L_NOTE}
* **Small CPU-trained models (0.22 M parameters):** every learned arm is far better than interpolation. On pristine input the deep-only model is marginally better than the hybrid at 4x and 8x; with noise and missing values the hybrid is clearly better (by 8 to 14% RMSE). The classical stage alone sharpens boundaries (edge F1 0.76 vs 0.45 at 8x) but does not reduce RMSE on clean input. Details and negative results: `docs/ABLATION.md`, `docs/FAILURES.md`.
* Latency, 480x640, 4x, 4 CPU threads, median (measured on the GPU laptop's CPU; a free 2-vCPU host will be slower): bicubic {lat('bicubic', 4)} ms, classical {lat('classical', 4)} ms, deep-only (small) {lat('dl_only', 4)} ms, hybrid (small) {lat('hybrid', 4)} ms{(', **deep-only-L ' + lat('dl_only_L', 4) + ' ms**') if 'dl_only_L|4' in t else ''}. Model files: small about 0.86 MB each, large about 36 MB. Peak-memory figures in `eval_P4_*.md` come from a Windows counter and are not comparable with earlier Linux runs.
* Depth units are *assumed* (raw x 10 m); only relative comparisons between methods are meaningful in absolute terms.

![qualitative](docs/figures/qualitative_4x_noisy_holes.png)

## Run it

```bash
pip install -r requirements.txt            # CPU is enough
streamlit run app.py                       # http://localhost:8501  (weights and sample frames are in the repo)
```
or `docker build -t edgegate . && docker run -p 8501:8501 edgegate`.

The app has two modes: **simulate** (RGB + dense depth: it degrades the depth and compares every method against ground truth) and **upsample my depth** (RGB + low-resolution depth, no ground truth). Accepted depth: `.npy`/`.npz` (metres), 16-bit PNG (mm), 8-bit PNG (relative, rescaled to 0.5-5 m). Bad files give a readable error, not a stack trace. A depth map is required: depth is not estimated from RGB alone.

## Reproduce the numbers
1. Download the NYU Depth V2 labelled set used here (`xcll/nyu_depth_train_float32_1449` on Hugging Face), unzip to `data/nyu/`.
2. `bash scripts/reproduce.sh` (about 2 h on 4 CPU cores): builds the cache, trains both models (35 min each), evaluates every method, runs ablations, efficiency benchmark and failure mining, regenerates `eval_P4_*.md` and figures. Set `TEAM` and `USNS` environment variables for the eval file header.
3. `python -m pytest -q tests` runs 17 fast unit tests (no dataset needed).

## Repository map
| Path | Purpose |
|---|---|
| `depthup/classical.py` | interpolation, hole filling, spike/noise handling, JBU, guided filter, discontinuity detector, gate, colour-plateau snap |
| `depthup/model.py`, `pipeline.py` | networks and the comparison arms (`METHODS`) |
| `depthup/train.py`, `evaluate.py`, `ablate.py`, `benchmark.py`, `failures.py`, `report.py` | training and every experiment |
| `depthup/io.py` | validated loading of user uploads |
| `app.py`, `Dockerfile`, `.streamlit/` | web app and deployment |
| `docs/architecture.png`, `DESIGN.md`, `ABLATION.md`, `FAILURES.md` | architecture and design documentation |
| `eval_P4_T01.md` | comparative evaluation (rename with your team id) |

## Data, ethics and limitations
Data split: the small models trained on frames 0-799 and validated on 800-999; the large model trained on 0-949 and validated on 950-999; **frames 1000-1448 (test) were never used for training, validation or tuning by any model**. NYU Depth V2 is licensed for research use; the six sample frames in `samples/` are test-split frames included for the demo. The model was trained on indoor Kinect scenes only and is untested on outdoor, LiDAR or real low-resolution ToF data; the corruption model is synthetic. Depth upsampling output must not be treated as a measurement in safety-critical settings: reconstructed boundaries are estimates, and thin structures narrower than one low-resolution cell are not recoverable (see `docs/FAILURES.md`).
"""


if __name__ == "__main__":
    (ROOT / "docs/ABLATION.md").write_text(ablation_md())
    (ROOT / "docs/FAILURES.md").write_text(failures_md())
    (ROOT / "README.md").write_text(readme())
    print("docs written")
