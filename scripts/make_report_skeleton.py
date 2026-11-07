"""Create report/Hackathon_Solution_Report.md in the organisers' template structure.

Everything numeric (tables) is generated from results/*.json so it matches the repository. The explanatory prose is
deliberately NOT written here: each section carries 'TEAM TO WRITE' markers plus a short list of facts to cover, which the
team must replace with their own sentences (the report's AI-written share is limited to 30% by the hackathon AI policy).

    python scripts/make_report_skeleton.py     # run ONCE, then edit the markdown by hand
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/summary.json").read_text())
A = json.loads((ROOT / "results/ablation.json").read_text())
E = json.loads((ROOT / "results/efficiency.json").read_text())
F = json.loads((ROOT / "results/failure_cases.json").read_text())
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
ARMS = [("bicubic", "Interpolation baseline (bicubic)"), ("classical", "**Classical (ours): gated joint bilateral**"),
        ("dl_only", "Deep learning only (small, 0.22 M params)"), ("hybrid", "**Proposed hybrid (small)**"),
        ("dl_only_L", "**Deep learning only, large (9 M params, GPU-trained)**")]


def t(m, s=4):
    return E["timing"].get(f"{m}|{s}", {}).get("median_ms")


def quant(prot, s):
    rows = ["| Approach / model variant | RMSE (cm) ↓ | MAE (cm) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (cm) ↓ | Edge F1 ↑ | CPU latency (ms) |",
            "| :---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for m, label in ARMS:
        v = S[f"{prot}|{s}|{m}"]
        lat = t(m, s)
        rows.append(f"| {label} | {v['rmse_cm']:.2f} | {v['mae_cm']:.2f} | {v['psnr_db']:.2f} | {v['ssim']:.3f} | {v['boundary_rmse_cm']:.1f} | {v['edge_f1']:.3f} | {lat:.0f} |" if lat else
                    f"| {label} | {v['rmse_cm']:.2f} | {v['mae_cm']:.2f} | {v['psnr_db']:.2f} | {v['ssim']:.3f} | {v['boundary_rmse_cm']:.1f} | {v['edge_f1']:.3f} | n/a |")
    return "\n".join(rows)


def scale_table():
    rows = ["| Protocol | Scale | " + " | ".join(l.replace("**", "") for _, l in ARMS) + " |", "| :---- | :---: |" + " :---: |" * len(ARMS)]
    for prot in ("clean", "noisy", "noisy+holes"):
        for s in (2, 4, 8):
            rows.append(f"| {prot} | {s}x | " + " | ".join(f"{S[f'{prot}|{s}|{m}']['rmse_cm']:.2f}" for m, _ in ARMS) + " |")
    return "\n".join(rows)


def ablation(prot, sc):
    rows = ["| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture-copy RMSE (cm) |", "| :---- | :---: | :---: | :---: | :---: |"]
    for k, v in A.items():
        p, s, g, name = k.split("|")
        if p == prot and int(s) == sc:
            rows.append(f"| {name} | {v['rmse_cm']:.2f} | {v['boundary_rmse_cm']:.1f} | {v['edge_f1']:.2f} | {v['texture_copy_rmse_cm']:.2f} |")
    return "\n".join(rows)


corr = F["correlations"]["clean@8x"]
doc = f"""# Hackathon Technical Solution Report

| Field | Details |
| :---- | :---- |
| **Problem ID** | P4 |
| **Problem Title** | Edge-Preserving Depth Map Upsampling and Refinement |
| **Team Name** | `[TEAM TO FILL: registered team name]` |
| **Team Members** | `[TEAM TO FILL: full names and USNs]` |
| **GitHub Repository** | https://github.com/PrathamDatta/HACKATHON_CV (branch `claude/practical-pasteur-0j1ipi`), commit `{commit}` `[UPDATE to the final commit hash after your last push]` |
| **Live Demo / Deployment** | https://hackathoncv-s8pekuxnusulgddjwfqg8d.streamlit.app |

---

## 1. Problem Understanding & Solution Strategy

* **Core Technical Challenge:** `[TEAM TO WRITE, in your own words, 3-4 sentences.]`
  *Facts you may use (delete this list when done):* depth maps from sensors are low-resolution, noisy and have holes; upsampling by 2x/4x/8x blurs object boundaries; RGB-guided filters can copy colour texture into flat depth ("texture copying"); constraints: dense output, boundary preservation, handle missing/noisy values, report latency and memory.
* **Proposed Methodology:** `[TEAM TO WRITE, 2-3 sentences.]`
  *Facts:* classical stage cleans the depth and lets RGB influence the result only near detected depth discontinuities (a gate); a learned network predicts a correction on top; the best measured model on the test set is a larger deep-only network (the hybrid was evaluated at small scale); a selector (Auto) chooses between models by measured noise/holes.

---

## 2. Technical Architecture & Implementation

> *Reference repository modules; no raw code in this report. Architecture diagram: `docs/architecture.png`.*

![Architecture diagram](../docs/architecture.png)

### 2.1 Classical Algorithmic Baseline

* **Algorithms Utilized:** `[TEAM TO WRITE]`
  *Facts (module `depthup/classical.py`):* nearest/bilinear/bicubic interpolation; normalised-convolution hole filling; robust noise estimate (Immerkaer kernel + MAD); spike removal; joint bilateral upsampling; guided filter; second-difference step detector for depth discontinuities; gate map; colour-plateau snap (tried; hurt accuracy).
* **Adaptive Selection Logic:** `[TEAM TO WRITE]`
  *Facts:* smoothing only when estimated noise > 1 cm; detector threshold grows with the scale factor; colour tolerance sigma_c is tight near discontinuities and effectively infinite elsewhere; sample confidence lowers the weight of filled values.

### 2.2 Deep Learning Pipeline

* **Model Architecture:** `[TEAM TO WRITE]`
  *Facts (`depthup/model.py`):* 3-level U-Net with GroupNorm + GELU and dilated bottleneck; small version 0.22 M parameters (width 16), large version about 9 M (width 64, 3 blocks per stage); deep-only inputs RGB + bicubic depth + scale; hybrid adds classical outputs.
* **Loss Functions & Training:** `[TEAM TO WRITE]`
  *Facts (`depthup/train.py`):* L1 weighted 7x on ground-truth depth discontinuities + 0.5 x gradient-matching L1; on-the-fly degradation (clean / noisy / noisy+holes, scales 2/4/8); NYU Depth V2, train frames 0-799 (small) or 0-949 (large), validation the remaining frames before 1000, **test frames 1000-1448 never used**; AdamW, cosine schedule; large model: 12,350 steps, batch 16, 256 px crops, bf16, EMA weights, GPU.

### 2.3 Hybrid Integration & Adaptive Mechanism

* **Fusion Strategy:** `[TEAM TO WRITE]`
  *Facts:* hybrid output = E0 + beta (E1 - E0) + delta, where E0 = bicubic, E1 = classical gated estimate, beta = learned per-pixel blend, delta = learned residual; beta = 0 reduces to the deep-only model; an earlier softmax-mixing design lost and was replaced.
* **Decision Logic:** `[TEAM TO WRITE]`
  *Facts:* Auto selector uses deep-only for pristine input and hybrid for noisy/incomplete input (fixed rule, never looks at ground truth); the learned blend beta turned out only weakly selective.

---

## 3. Experimental Benchmarks & Evaluation

*Dataset: NYU Depth V2, held-out test frames 1000-1448 (449 frames for `clean`, 150 for each noisy protocol). Depth units assumed (raw x 10 m). CPU latency measured at 480x640, 4 threads, on the team's laptop. Tables below are generated from `results/*.json` by `scripts/make_report_skeleton.py`.*

### 3.1 Quantitative Benchmarks

**Clean input, 4x upsampling**

{quant('clean', 4)}

**Noisy input with 20% missing values, 4x upsampling**

{quant('noisy+holes', 4)}

**RMSE (cm) across protocols and scales**

{scale_table()}

* **Analysis:** `[TEAM TO WRITE, 2-3 sentences on the trade-off between accuracy, boundary quality and speed.]`
  *Facts:* the large model is lowest in every cell but takes about {t('dl_only_L'):.0f} ms on CPU versus about {t('dl_only'):.0f} ms for the small one; the classical stage improves edge F1 but not RMSE on clean input; the small hybrid beats small deep-only only under noise and holes.

### 3.2 Visual Outputs & Intermediate Representations

![Qualitative comparison, 8x clean](../docs/figures/qualitative_8x.png)

![Qualitative comparison, 4x noisy with holes](../docs/figures/qualitative_4x_noisy_holes.png)

![Intermediate stages: discontinuity map, gate, learned blend, residual](../docs/figures/stages.png)

![Boundary profile](../docs/figures/boundary_profile.png)

`[TEAM TO WRITE: one or two sentences per figure saying what the reader should notice.]`

### 3.3 Ablation Study

*Small CPU-trained models, 80 test frames, 8x upsampling.*

**Clean input**

{ablation('clean', 8)}

**Noisy input with missing values**

{ablation('noisy+holes', 8)}

* **Interpretation:** `[TEAM TO WRITE]`
  *Facts:* learned residual gives the largest RMSE drop; joint bilateral filtering gives most of the edge-F1 gain; the gate lowers texture-copy error slightly on clean input; noise-adaptive clean-up helps only on degraded input; the hard snap hurts; the v1 softmax arbiter is worse than the final blend. The large hybrid and an ensemble were not trained (deadline), and ablations were not repeated on the large model.

---

## 4. Failure Modes & Limitations

* **Observed Failure Case:** `[TEAM TO WRITE]`
  *Facts (`docs/FAILURES.md`):* worst frame is {F['cases'][0]['idx']} (clean 8x, RMSE about {F['cases'][0]['hybrid_rmse']:.0f} cm for the small hybrid), a dark corridor with a glass/mirror door; error correlates with GT edge density (Spearman {corr['edge_density']['spearman_vs_hybrid_rmse']:.2f}) and depth range ({corr['depth_range']['spearman_vs_hybrid_rmse']:.2f}) but not with brightness; isolated noise spikes can trigger the step detector.
* **Root Cause & Mitigation:** `[TEAM TO WRITE]`
  *Facts:* thin structures narrower than one low-resolution cell are aliased away; glass/reflections have depth unrelated to RGB structure; mitigation ideas: connected-component filtering of the discontinuity map, multi-scale evidence, cross-dataset validation. Other limitations: synthetic degradation, one indoor dataset, assumed depth units, large hybrid not trained.

---

## 5. Deployment & Quickstart Guide

### Quickstart Commands

```sh
git clone https://github.com/PrathamDatta/HACKATHON_CV
cd HACKATHON_CV
git checkout claude/practical-pasteur-0j1ipi
pip install -r requirements.txt
streamlit run app.py
```

### Deployment Format

* Streamlit dashboard deployed on Streamlit Community Cloud (CPU only); a `Dockerfile` is also provided for container deployment. `[TEAM TO WRITE one sentence on how you deployed it and any limits you observed.]`

---

## 6. AI Assistance Declaration

`[TEAM: review and correct this section so it is exactly true. The policy requires clear declaration.]`

* **AI tool used:** Claude Code (Anthropic) was used throughout this project.
* **What the AI did:** designed and wrote the code in `depthup/` (classical methods, networks, training, evaluation, app), ran the experiments and the ablation and failure analysis on a cloud CPU environment, generated the tables and figures from the result files, drafted the repository documentation (`README.md`, `docs/*.md`), and prepared the skeleton of this report (tables, figure references, fact lists).
* **What the team did:** `[TEAM TO WRITE truthfully: e.g. chose the problem, ran GPU training on a laptop, deployed the app, wrote the prose of this report, prepared for the viva.]`
* **Authorship of this report's prose:** `[TEAM TO FILL: state which sections were written by the team in their own words.]`
* **Estimated AI share of this report's text:** `[TEAM TO FILL honestly after rewriting; the policy limit is 30%.]`

---

### Student Submission Checklist

- [ ] Working GitHub repository link included with an updated `README.md`.
- [ ] No raw code pasted into this report.
- [ ] All table metrics match test runs reproducible via the repository.
- [ ] All team member details accurately recorded.
"""
out = ROOT / "report" / "Hackathon_Solution_Report.md"
out.parent.mkdir(exist_ok=True)
out.write_text(doc)
print("wrote", out, f"({len(doc.splitlines())} lines)")
