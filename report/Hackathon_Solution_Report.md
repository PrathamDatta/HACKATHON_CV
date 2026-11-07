# Hackathon Technical Solution Report

| Field | Details |
| :---- | :---- |
| **Problem ID** | P4 |
| **Problem Title** | Edge-Preserving Depth Map Upsampling and Refinement |
| **Team Name** | `[TEAM TO FILL: registered team name]` |
| **Team Members** | `[TEAM TO FILL: full names and USNs]` |
| **GitHub Repository** | https://github.com/PrathamDatta/HACKATHON_CV (branch `claude/practical-pasteur-0j1ipi`), commit `2c97d4e` `[UPDATE to the final commit hash after your last push]` |
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

| Approach / model variant | RMSE (cm) ↓ | MAE (cm) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (cm) ↓ | Edge F1 ↑ | CPU latency (ms) |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Interpolation baseline (bicubic) | 4.61 | 1.54 | 42.10 | 0.984 | 13.6 | 0.740 | 1 |
| **Classical (ours): gated joint bilateral** | 4.77 | 1.43 | 41.79 | 0.985 | 14.2 | 0.909 | 323 |
| Deep learning only (small, 0.22 M params) | 2.77 | 1.06 | 46.37 | 0.993 | 7.8 | 0.950 | 99 |
| **Proposed hybrid (small)** | 2.92 | 1.09 | 45.90 | 0.992 | 8.3 | 0.945 | 430 |
| **Deep learning only, large (9 M params, GPU-trained)** | 1.69 | 0.65 | 50.56 | 0.997 | 4.7 | 0.989 | 2585 |

**Noisy input with 20% missing values, 4x upsampling**

| Approach / model variant | RMSE (cm) ↓ | MAE (cm) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (cm) ↓ | Edge F1 ↑ | CPU latency (ms) |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Interpolation baseline (bicubic) | 8.90 | 4.50 | 34.47 | 0.912 | 19.3 | 0.560 | 1 |
| **Classical (ours): gated joint bilateral** | 6.37 | 3.17 | 37.76 | 0.958 | 17.1 | 0.686 | 323 |
| Deep learning only (small, 0.22 M params) | 5.48 | 2.53 | 39.07 | 0.974 | 15.0 | 0.861 | 99 |
| **Proposed hybrid (small)** | 4.72 | 2.34 | 40.39 | 0.979 | 12.8 | 0.863 | 430 |
| **Deep learning only, large (9 M params, GPU-trained)** | 3.82 | 1.88 | 42.23 | 0.986 | 10.4 | 0.928 | 2585 |

**RMSE (cm) across protocols and scales**

| Protocol | Scale | Interpolation baseline (bicubic) | Classical (ours): gated joint bilateral | Deep learning only (small, 0.22 M params) | Proposed hybrid (small) | Deep learning only, large (9 M params, GPU-trained) |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
| clean | 2x | 2.56 | 2.86 | 1.76 | 1.75 | 0.95 |
| clean | 4x | 4.61 | 4.77 | 2.77 | 2.92 | 1.69 |
| clean | 8x | 7.69 | 7.69 | 4.98 | 5.17 | 3.43 |
| noisy | 2x | 6.71 | 4.06 | 2.51 | 2.54 | 1.88 |
| noisy | 4x | 7.53 | 5.33 | 3.71 | 3.64 | 2.77 |
| noisy | 8x | 9.20 | 7.57 | 5.70 | 5.73 | 4.35 |
| noisy+holes | 2x | 7.52 | 4.65 | 3.61 | 3.18 | 2.48 |
| noisy+holes | 4x | 8.90 | 6.37 | 5.48 | 4.72 | 3.82 |
| noisy+holes | 8x | 11.31 | 9.13 | 8.21 | 7.58 | 6.11 |

* **Analysis:** `[TEAM TO WRITE, 2-3 sentences on the trade-off between accuracy, boundary quality and speed.]`
  *Facts:* the large model is lowest in every cell but takes about 2585 ms on CPU versus about 99 ms for the small one; the classical stage improves edge F1 but not RMSE on clean input; the small hybrid beats small deep-only only under noise and holes.

### 3.2 Visual Outputs & Intermediate Representations

![Qualitative comparison, 8x clean](../docs/figures/qualitative_8x.png)

![Qualitative comparison, 4x noisy with holes](../docs/figures/qualitative_4x_noisy_holes.png)

![Intermediate stages: discontinuity map, gate, learned blend, residual](../docs/figures/stages.png)

![Boundary profile](../docs/figures/boundary_profile.png)

`[TEAM TO WRITE: one or two sentences per figure saying what the reader should notice.]`

### 3.3 Ablation Study

*Small CPU-trained models, 80 test frames, 8x upsampling.*

**Clean input**

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture-copy RMSE (cm) |
| :---- | :---: | :---: | :---: | :---: |
| bicubic | 7.93 | 23.2 | 0.46 | 2.68 |
| A  plain JBU (no gate, no depth term, no clean-up) | 7.83 | 22.6 | 0.76 | 2.73 |
| B  + depth-consistency term | 7.98 | 23.1 | 0.79 | 2.62 |
| C  + discontinuity gate (texture-copy suppression) | 7.94 | 23.2 | 0.76 | 2.57 |
| D  + noise-adaptive LR clean-up  (= Classical, ours) | 7.98 | 23.3 | 0.76 | 2.57 |
| E  D + hard colour-plateau snap | 13.42 | 36.6 | 0.71 | 5.17 |
| E0 only (bicubic) | 7.98 | 23.3 | 0.46 | 2.68 |
| E1 only (classical gated JBU) | 7.98 | 23.3 | 0.76 | 2.57 |
| learned blend β only (residual δ off) | 7.55 | 22.1 | 0.62 | 2.57 |
| residual δ only (β = 0) | 5.58 | 15.8 | 0.81 | 2.40 |
| full hybrid: E0 + β(E1−E0) + δ  (ours) | 5.41 | 15.3 | 0.84 | 2.32 |
| v1 softmax-convex arbiter (superseded) | 5.86 | 16.6 | 0.82 | 2.44 |
| deep learning only (no classical features) | 5.25 | 14.8 | 0.89 | 2.21 |

**Noisy input with missing values**

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture-copy RMSE (cm) |
| :---- | :---: | :---: | :---: | :---: |
| bicubic | 13.28 | 31.2 | 0.43 | 7.95 |
| A  plain JBU (no gate, no depth term, no clean-up) | 12.24 | 29.5 | 0.66 | 7.30 |
| B  + depth-consistency term | 12.58 | 30.4 | 0.65 | 7.35 |
| C  + discontinuity gate (texture-copy suppression) | 12.62 | 30.4 | 0.63 | 7.37 |
| D  + noise-adaptive LR clean-up  (= Classical, ours) | 11.02 | 28.6 | 0.57 | 5.44 |
| E  D + hard colour-plateau snap | 13.91 | 35.8 | 0.56 | 6.72 |
| E0 only (bicubic) | 11.29 | 29.0 | 0.42 | 5.67 |
| E1 only (classical gated JBU) | 11.02 | 28.6 | 0.57 | 5.44 |
| learned blend β only (residual δ off) | 10.92 | 28.1 | 0.51 | 5.53 |
| residual δ only (β = 0) | 9.25 | 23.0 | 0.72 | 5.22 |
| full hybrid: E0 + β(E1−E0) + δ  (ours) | 9.09 | 22.7 | 0.73 | 5.13 |
| v1 softmax-convex arbiter (superseded) | 9.79 | 24.2 | 0.71 | 5.53 |
| deep learning only (no classical features) | 9.90 | 25.2 | 0.74 | 5.19 |

* **Interpretation:** `[TEAM TO WRITE]`
  *Facts:* learned residual gives the largest RMSE drop; joint bilateral filtering gives most of the edge-F1 gain; the gate lowers texture-copy error slightly on clean input; noise-adaptive clean-up helps only on degraded input; the hard snap hurts; the v1 softmax arbiter is worse than the final blend. The large hybrid and an ensemble were not trained (deadline), and ablations were not repeated on the large model.

---

## 4. Failure Modes & Limitations

* **Observed Failure Case:** `[TEAM TO WRITE]`
  *Facts (`docs/FAILURES.md`):* worst frame is 1015 (clean 8x, RMSE about 22 cm for the small hybrid), a dark corridor with a glass/mirror door; error correlates with GT edge density (Spearman 0.94) and depth range (0.89) but not with brightness; isolated noise spikes can trigger the step detector.
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
