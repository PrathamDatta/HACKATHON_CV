# Ablation study

80 evenly-spaced **test** frames (subset of the 449; numbers differ slightly from the full-test table in `eval_P4_*.md`). All classical and hybrid rows use the **small CPU-trained models**; the large deep-only model (`dl_only_L`) was trained afterwards and appears only in the main comparison (`eval_P4_*.md`). Same degradations and seeds for every variant. RMSE and boundary RMSE in cm (lower is better), edge F1 higher is better, texture-copy RMSE (error where depth is flat but RGB textured) lower is better.

## clean, 4x

| Group | Variant | RMSE | Boundary RMSE | Edge F1 | Texture-copy RMSE |
|---|---|---|---|---|---|
| classical | bicubic | 4.73 | 14.0 | 0.74 | 1.55 |
| classical | A  plain JBU (no gate, no depth term, no clean-up) | 4.64 | 13.8 | 0.88 | 1.50 |
| classical | B  + depth-consistency term | 4.92 | 14.7 | 0.91 | 1.54 |
| classical | C  + discontinuity gate (texture-copy suppression) | 4.91 | 14.7 | 0.91 | 1.57 |
| classical | D  + noise-adaptive LR clean-up  (= Classical, ours) | 4.92 | 14.7 | 0.91 | 1.57 |
| classical | E  D + hard colour-plateau snap | 8.39 | 24.6 | 0.88 | 1.91 |
| hybrid | E0 only (bicubic) | 4.75 | 14.1 | 0.74 | 1.55 |
| hybrid | E1 only (classical gated JBU) | 4.92 | 14.7 | 0.91 | 1.57 |
| hybrid | learned blend β only (residual δ off) | 4.47 | 13.3 | 0.81 | 1.54 |
| hybrid | residual δ only (β = 0) | 3.20 | 9.1 | 0.93 | 1.35 |
| hybrid | full hybrid: E0 + β(E1−E0) + δ  (ours) | 3.03 | 8.6 | 0.94 | 1.35 |
| hybrid | v1 softmax-convex arbiter (superseded) | 3.16 | 9.0 | 0.93 | 1.41 |
| hybrid | deep learning only (no classical features) | 2.87 | 8.2 | 0.95 | 1.21 |

## clean, 8x

| Group | Variant | RMSE | Boundary RMSE | Edge F1 | Texture-copy RMSE |
|---|---|---|---|---|---|
| classical | bicubic | 7.93 | 23.2 | 0.46 | 2.68 |
| classical | A  plain JBU (no gate, no depth term, no clean-up) | 7.83 | 22.6 | 0.76 | 2.73 |
| classical | B  + depth-consistency term | 7.98 | 23.1 | 0.79 | 2.62 |
| classical | C  + discontinuity gate (texture-copy suppression) | 7.94 | 23.2 | 0.76 | 2.57 |
| classical | D  + noise-adaptive LR clean-up  (= Classical, ours) | 7.98 | 23.3 | 0.76 | 2.57 |
| classical | E  D + hard colour-plateau snap | 13.42 | 36.6 | 0.71 | 5.17 |
| hybrid | E0 only (bicubic) | 7.98 | 23.3 | 0.46 | 2.68 |
| hybrid | E1 only (classical gated JBU) | 7.98 | 23.3 | 0.76 | 2.57 |
| hybrid | learned blend β only (residual δ off) | 7.55 | 22.1 | 0.62 | 2.57 |
| hybrid | residual δ only (β = 0) | 5.58 | 15.8 | 0.81 | 2.40 |
| hybrid | full hybrid: E0 + β(E1−E0) + δ  (ours) | 5.41 | 15.3 | 0.84 | 2.32 |
| hybrid | v1 softmax-convex arbiter (superseded) | 5.86 | 16.6 | 0.82 | 2.44 |
| hybrid | deep learning only (no classical features) | 5.25 | 14.8 | 0.89 | 2.21 |

## noisy+holes, 4x

| Group | Variant | RMSE | Boundary RMSE | Edge F1 | Texture-copy RMSE |
|---|---|---|---|---|---|
| classical | bicubic | 10.18 | 22.2 | 0.61 | 7.05 |
| classical | A  plain JBU (no gate, no depth term, no clean-up) | 9.02 | 20.8 | 0.70 | 6.16 |
| classical | B  + depth-consistency term | 9.39 | 21.7 | 0.71 | 6.30 |
| classical | C  + discontinuity gate (texture-copy suppression) | 9.40 | 21.7 | 0.71 | 6.32 |
| classical | D  + noise-adaptive LR clean-up  (= Classical, ours) | 7.64 | 19.6 | 0.72 | 4.14 |
| classical | E  D + hard colour-plateau snap | 9.22 | 24.3 | 0.70 | 4.50 |
| hybrid | E0 only (bicubic) | 7.81 | 19.9 | 0.69 | 4.21 |
| hybrid | E1 only (classical gated JBU) | 7.64 | 19.6 | 0.72 | 4.14 |
| hybrid | learned blend β only (residual δ off) | 7.58 | 19.3 | 0.71 | 4.17 |
| hybrid | residual δ only (β = 0) | 5.78 | 14.9 | 0.87 | 3.19 |
| hybrid | full hybrid: E0 + β(E1−E0) + δ  (ours) | 5.69 | 14.6 | 0.87 | 3.18 |
| hybrid | v1 softmax-convex arbiter (superseded) | 6.27 | 15.9 | 0.86 | 3.60 |
| hybrid | deep learning only (no classical features) | 6.68 | 17.4 | 0.87 | 3.53 |

## noisy+holes, 8x

| Group | Variant | RMSE | Boundary RMSE | Edge F1 | Texture-copy RMSE |
|---|---|---|---|---|---|
| classical | bicubic | 13.28 | 31.2 | 0.43 | 7.95 |
| classical | A  plain JBU (no gate, no depth term, no clean-up) | 12.24 | 29.5 | 0.66 | 7.30 |
| classical | B  + depth-consistency term | 12.58 | 30.4 | 0.65 | 7.35 |
| classical | C  + discontinuity gate (texture-copy suppression) | 12.62 | 30.4 | 0.63 | 7.37 |
| classical | D  + noise-adaptive LR clean-up  (= Classical, ours) | 11.02 | 28.6 | 0.57 | 5.44 |
| classical | E  D + hard colour-plateau snap | 13.91 | 35.8 | 0.56 | 6.72 |
| hybrid | E0 only (bicubic) | 11.29 | 29.0 | 0.42 | 5.67 |
| hybrid | E1 only (classical gated JBU) | 11.02 | 28.6 | 0.57 | 5.44 |
| hybrid | learned blend β only (residual δ off) | 10.92 | 28.1 | 0.51 | 5.53 |
| hybrid | residual δ only (β = 0) | 9.25 | 23.0 | 0.72 | 5.22 |
| hybrid | full hybrid: E0 + β(E1−E0) + δ  (ours) | 9.09 | 22.7 | 0.73 | 5.13 |
| hybrid | v1 softmax-convex arbiter (superseded) | 9.79 | 24.2 | 0.71 | 5.53 |
| hybrid | deep learning only (no classical features) | 9.90 | 25.2 | 0.74 | 5.19 |

## What each component contributes (8x; see tables for 4x)

1. **Learning is the biggest RMSE lever.** Adding the learned residual to bicubic (`residual δ only`) cuts RMSE from 7.98 to 5.58 cm on clean input and from 11.29 to 9.25 cm on noisy+holes.
2. **Edge sharpness comes from the classical stage.** Plain JBU raises edge F1 from 0.46 to 0.76 (clean, 8x) without hurting RMSE.
3. **The discontinuity gate does what it was designed to, modestly.** Plain JBU slightly *increases* texture-copy error over bicubic (2.73 vs 2.68 cm, clean 8x); adding the depth term and gate brings it to 2.57 cm, below bicubic, with RMSE unchanged. Under noisy+holes the gate has no measurable effect (7.35 to 7.37): noise dominates there.
4. **LR clean-up matters only when the input is degraded** (as designed: it is adaptive). Noisy+holes 8x: RMSE 12.62 to 11.02 cm and texture-copy 7.37 to 5.44; on clean input it changes nothing.
5. **Negative result: the hard colour-plateau snap hurts** (7.98 to 13.42 cm clean 8x). NYU's inpainted ground truth has soft boundaries, so a wrong plateau assignment costs the full depth jump. It is kept only as an input feature.
6. **The learned blend β adds a small further gain** on top of the residual: 5.58 to 5.41 (clean) and 9.25 to 9.09 (noisy+holes), i.e. 2-3%.
7. **Structure of the arbiter matters.** The v1 softmax-convex arbiter is clearly worse than the v2 residual form (5.86 vs 5.41 clean; 9.79 vs 9.09 noisy+holes).
8. **Deep-only vs hybrid depends on input quality.** On pristine input deep-only is slightly better (5.25 vs 5.41); with noise and holes the hybrid is clearly better (9.09 vs 9.90). This motivates the Auto selector.

**Not ablated by retraining** (time): removing the snap/edge/gate *input channels* of the network would need retraining; the ablation above switches components off at inference or compares separately trained models (deep-only = no classical features at all).
