# Failure-case analysis

Method: for every test frame we computed scene descriptors (GT edge density, depth range, mean depth, luminance, edge colour contrast, texture level) and correlated them (Spearman) with the **small hybrid's** RMSE (the failure analysis was run before the large model existed; the same scene properties are expected to be hard for any model, but this was not re-checked); we then inspected the worst frames (`results/failure_cases.json`, figures in `docs/figures/failure_*.png`).

## What predicts failure

| Descriptor | ρ vs hybrid RMSE (clean 8x, n=449) | ρ vs hybrid RMSE (noisy+holes 4x, n=150) |
|---|---|---|
| edge_density | 0.94 | 0.92 |
| depth_range | 0.89 | 0.90 |
| mean_depth | 0.68 | 0.76 |
| texture_level | 0.35 | 0.25 |
| luminance | 0.13 | 0.12 |
| edge_colour_contrast | -0.01 | 0.03 |

Error is driven by **scene geometry, not appearance**: frames with many depth discontinuities and a large depth range are hard (ρ ≈ 0.9), while brightness and the local colour contrast at edges are unrelated (|ρ| ≤ 0.13). Our initial hypothesis that low-contrast edges would dominate failures was **not supported**. The hybrid beats bicubic on 100% of frames (clean 8x; median gain 2.1 cm) and on 100% (noisy+holes 4x; median gain 4.1 cm): there is no frame on which it is worse. The lists below are therefore the *hardest* frames, not regressions.

| Protocol | Selection | Frame | Hybrid RMSE | Bicubic RMSE | Hybrid boundary | Bicubic boundary | GT edge density | Depth range m |
|---|---|---|---|---|---|---|---|---|
| clean 8x | highest hybrid RMSE | 1015 | 21.6 | 22.2 | 51.3 | 52.4 | 0.063 | 6.8 |
| clean 8x | smallest gain over bicubic | 1032 | 1.7 | 2.0 | 7.6 | 9.4 | 0.004 | 1.9 |
| clean 8x | lowest edge F1 | 1072 | 2.0 | 2.4 | 7.4 | 9.4 | 0.004 | 2.3 |
| noisy+holes 4x | highest hybrid RMSE | 1240 | 12.2 | 21.9 | 25.2 | 44.6 | 0.059 | 9.2 |
| noisy+holes 4x | smallest gain over bicubic | 1118 | 3.0 | 5.8 | 8.5 | 11.7 | 0.018 | 2.2 |
| noisy+holes 4x | highest boundary RMSE | 1272 | 9.1 | 14.4 | 26.9 | 39.7 | 0.024 | 8.4 |
| noisy+holes 4x | lowest edge F1 | 1170 | 3.0 | 7.0 | 9.4 | 13.6 | 0.008 | 2.4 |

## Failure case 1: thin structures and glass (frame 1015, clean 8x, `failure_clean_1015.png`)
Worst frame of the test set (RMSE about 21 cm, versus a test mean of about 5 cm). A dark corridor with a mirrored/glass door. **Causes (visible in the error maps):** (a) the door frames and mirror edges are narrower than one low-resolution cell (8 px) while the depth jump across them is metres, so the structure is aliased away in the input and cannot be recovered by any method (bicubic and hybrid errors are almost identical there); (b) the glass/reflection region has depth that bears no relation to the RGB structure, so RGB guidance carries no information. The hybrid gains almost nothing here (21.6 vs 22.2 cm). **Mitigation not implemented:** multi-scale or higher-resolution LR evidence, or a reflective-surface cue.

## Failure case 2: holes on boundaries and distance-proportional noise (frame 1240, noisy+holes 4x, `failure_noisy_holes_1240.png`)
Large depth jumps across a 9 m range. **Causes:** (a) when missing samples fall exactly on a boundary there are no samples of one of the two surfaces nearby, so no method can place the edge precisely; the hybrid still halves bicubic's boundary error (25 vs 45 cm) but remains the worst case; (b) the corruption noise is multiplicative (2% of depth), so far surfaces (about 4 m) carry about 8 cm of noise, visible as residual speckle on the far wall that the learned residual only partly removes.

## Failure case 3: false discontinuities from surviving spikes (frame 1160, `docs/figures/stages.png`)
In the 'depth discontinuities' panel, isolated dots appear away from any real edge: noise spikes that survive clean-up trigger the second-difference step detector, which opens the gate locally and lets RGB colour influence the classical estimate (a small texture-copy risk). The learned blend β mostly suppresses these (its trust map is low around them) but not completely. **Mitigation:** a connected-component size filter on the discontinuity map (a real step is a line, a spike is a point).

## Failure case 4: boundary F1 is unstable on simple scenes (frames 1032, 1072)
Frames chosen by 'lowest edge F1' have only about 0.4% edge pixels; a handful of mislocated edge pixels moves F1 a lot while RMSE is small (about 2 cm). This is a metric artefact, not a reconstruction failure, and the reason we report boundary-band RMSE alongside F1.

## Other limitations
* NYU ground truth is Kinect depth with inpainting: boundaries are soft, so our boundary metrics under-reward sharpness beyond the GT's own precision. A hard plateau snap, which is sharp by construction, therefore *loses* on these metrics (see ablation).
* The synthetic noise model (multiplicative Gaussian + spikes + holes) is not a real Kinect noise model; transfer to real low-resolution time-of-flight data is untested.
* Trained and tested on indoor scenes only; no cross-dataset (KITTI/Middlebury) test was run.
