# Project report: EdgeGate, edge-preserving depth map upsampling (Problem P4)

*A study document. Read sections 1-3 first for the big picture, 4-6 for how it works, 7-9 for what we found, 10-12 for viva preparation.*
*Status: all code, the small CPU-trained models, the large GPU-trained deep-only model (`dl_only_L`) and their evaluation are complete. Sections 8.1-8.3 describe the small models; section 8.4 adds the large model. The large hybrid and the seed ensemble were NOT trained (deadline).*

---

## 1. The problem in plain words
A depth sensor (like the Kinect, or a phone's time-of-flight sensor) measures how far each pixel is, but its output is often **low-resolution, noisy and full of holes**, while the colour camera next to it gives a sharp, high-resolution RGB image. Problem P4 asks: **take the low-resolution/sparse depth map and produce a dense, high-resolution one, using the RGB image as a guide**, while keeping object boundaries sharp and not painting RGB textures into the depth map.

Key terms:
* **Upsampling factor** (2x, 4x, 8x): the depth is 2, 4 or 8 times smaller than the target in each direction. 8x means each low-res pixel must be expanded into 64 pixels.
* **Depth discontinuity / boundary**: where depth jumps (e.g. the edge of a table against the floor). Blurring there creates "flying pixels" and wrong 3-D shapes.
* **Texture copying**: a flat wall with a painted pattern has *constant depth*, but a naive RGB-guided filter lets the pattern leak into the depth map, producing fake bumps. This is the central artefact the problem statement wants avoided.
* **Guidance**: using the aligned RGB image to decide where depth edges probably are.

## 2. What the judges score (rubric, 100 points)
Evaluation metrics 25 (ranked against other teams) | Demo and viva 25 | Implementation 20 | UI/deployment 15 | Documentation 15. A missing live URL caps deployment at 0, a missing GitHub link caps implementation at 0, no diagram/design doc caps documentation at 5. The PDF text and table disagree on weights ("each of the other five carries 15" vs 25/20/15/15/25); we followed the table. Required deliverables: live URL, GitHub repo with code, architecture diagram, design document, `eval_P4_<TeamID>.md` (first line = comma-separated USNs, metrics in the PDF's order), ablation study and failure analysis.

P4's required technical content: (A) classical: nearest/bilinear/bicubic, joint bilateral filtering, guided filtering, edge/boundary refinement, discontinuity detection, noise removal and hole filling; (B) deep learning: a depth upsampling/completion network using RGB guidance; (C) hybrid combining both with boundary preservation; advanced: 2x/4x/8x, missing and noisy depth, compare interpolation/classical/deep/hybrid, unseen data, boundary behaviour. Metrics: RMSE, MAE, PSNR, SSIM, edge/boundary accuracy, per-factor performance, latency and memory.

## 3. The solution at a glance
**EdgeGate** is a two-stage hybrid:
1. A **classical stage** cleans the depth, detects *where depth really jumps* (a step detector), and lets RGB influence the result **only near those jumps** ("gate"). Everywhere else it uses plain smooth interpolation, so RGB texture cannot leak in.
2. A **small U-Net (0.22 M parameters)** sees RGB plus the classical outputs and predicts a correction: `out = E0 + beta*(E1 - E0) + delta`, where E0 = bicubic, E1 = the classical sharp estimate, beta = a learned per-pixel trust in the classical estimate, delta = a learned residual.
3. An **Auto selector** picks the deep-only network for pristine input and the hybrid for noisy/incomplete input.

Architecture diagram: `docs/architecture.png`. Every box is a real function name.

## 4. Data and experimental protocol
* **Dataset:** NYU Depth V2, 1,449 labelled indoor RGB-D frames (640x480), dense inpainted Kinect depth. We assume `metres = raw x 10` (stored normalised); absolute centimetres are therefore approximate, but comparisons between methods are unaffected.
* **Split** (contiguous blocks because frames are ordered by scene, which limits train/test leakage): train 0-799, validation 800-999, **test 1000-1448 (449 frames, never trained on)**.
* **Degradation** (we create the "sensor" input ourselves from the clean GT): downsample by 2/4/8 with anti-aliased bicubic, then protocol-dependent corruption:
  * `clean` - nothing else;
  * `noisy` - 2% multiplicative Gaussian noise (Kinect noise grows with distance) plus 0.5% spike outliers ("flying pixels");
  * `noisy+holes` - as noisy plus 20% missing pixels (half random, half blobs).
* **Caveats to state honestly:** the GT is itself inpainted Kinect depth (soft boundaries); degradation is synthetic; indoor only.

## 5. The classical toolbox (what each piece does and why)
All in `depthup/classical.py`.

| Piece | Idea | Why it is there |
|---|---|---|
| Nearest / bilinear / bicubic | Interpolate between low-res samples. Bicubic uses a 4x4 neighbourhood with a smooth cubic kernel. | The mandatory baselines; also our safe fallback E0. |
| **Hole filling** (`normalized_fill`) | *Normalised convolution*: blur (value x valid) / blur (valid) at growing scales, filling inwards. | Missing pixels cannot be interpolated directly. Dividing by the blurred mask avoids the "holes pull values towards zero" bias. |
| **Noise estimate** (`estimate_noise`) | Convolve with the Immerkaer kernel [[1,-2,1],[-2,4,-2],[1,-2,1]] (removes smooth structure) and take the median absolute value / 6 x 1.4826 (robust to edges). | Makes the pipeline *adaptive*: it only smooths when estimated noise exceeds 1 cm, so clean input is not blurred. |
| **Spike removal** | A pixel that is a local extremum of its 3x3 neighbourhood *and* far from the median is replaced by the median. | Removes flying pixels but keeps real steps (a step edge pixel is not an extremum). |
| **Joint bilateral upsampling** (`jbu`) | Each output pixel is a weighted average of nearby low-res samples. Weight = spatial closeness x **colour similarity in the RGB guide** x depth consistency x sample confidence. | The classic RGB-guided method: samples that look like the pixel (same colour) probably have the same depth. Also needed: **log-domain softmax** for the weights (the first version underflowed to 0 and produced garbage). |
| **Guided filter** (`guided_filter`) | He et al.: fit a local linear model of the output as a function of the guide image, using box filters (fast). | Required by P4; used as a baseline. |
| **Step detector** (`lr_discontinuity`) | Second difference (L(x+1) - 2L(x) + L(x-1)) in four directions on the low-res depth. A **step** gives a big response, a **sloped plane** gives ~0. Threshold grows with scale. | Our first detector measured depth *range* in a window and fired on every slanted floor/wall, creating fake steps. The second-difference version flags steps, not ramps. |
| **Gate** (`gate_map`) | Spread detector evidence over the band where upsampling blur occurs. Colour tolerance sigma_c = exp((1-G) ln 5 + G ln 0.15): tight where depth jumps, effectively infinite elsewhere. | **This is the anti-texture-copy mechanism**: RGB can only vote where depth already shows a discontinuity. |
| **Colour-plateau snap** | For each pixel near an edge, pick the nearer of the two depth levels by colour similarity. | A "sharp edge" idea. It **hurt** accuracy on NYU (see 8.3), so it is only an input feature. |

## 6. The learned stage
* **Network** (`depthup/model.py`): a 3-level U-Net with GroupNorm + GELU and a dilated bottleneck (dilations 2, 4, 8 to see large context, important at 8x). 0.22 M parameters, 0.86 MB file.
* **Hybrid inputs (10 channels):** RGB (3), E0, E1, snap candidate, gate G, RGB edge magnitude, confidence, scale/8. **Output:** residual delta and blend logit.
* **Formula:** `out = E0 + beta*(E1 - E0) + delta`. With beta = 0 the output has the same form as the deep-only model (bicubic plus a learned residual), so the hybrid can represent that solution; training does not guarantee it matches it (on clean input the small hybrid ended slightly worse).
* **Deep-only baseline:** same backbone, inputs RGB + bicubic depth + scale only (5 channels), output = bicubic + residual. It is the fair test of whether classical features add anything.
* **Training** (`depthup/train.py`): random 128x128 crops, random scale in {2,4,8}, on-the-fly degradation (50% clean, 25% noisy, 25% noisy+holes, random magnitudes), flips. **Loss** = L1 weighted 7x on GT discontinuities + 0.5 x gradient-matching L1 (so edges are learned, not just averaged). AdamW, cosine learning rate. Both models got equal step budgets (6,200 vs 6,206) after we noticed an unfair comparison earlier.
* **Auto** (`pipeline.run_auto`): no holes and estimated noise < 1.2 cm -> deep-only; else hybrid.

## 7. Metrics (what each number means)
* **RMSE / MAE (cm):** typical error; RMSE punishes large errors (boundaries) more.
* **PSNR / SSIM:** signal-to-noise and structural similarity on depth (peak = per-image max depth).
* **Boundary RMSE:** error only within 3 px of GT discontinuities (about 4% of pixels) - where upsampling fails.
* **Edge F1 (2 px tolerance):** do predicted depth edges coincide with GT edges? Caution: unstable on simple scenes with almost no edges.
* **Texture-copy RMSE:** error where depth is flat but RGB is textured - the direct measure of texture copying (our own metric).
* **Latency / memory:** median time per 480x640 image (fresh process, 4 CPU threads) and peak RSS.
* **Paired bootstrap CI:** resample test frames to get a 95% confidence interval for the *difference* between two methods; if it excludes 0 the difference is reliable.

## 8. Results (small CPU-trained models; RMSE in cm, lower is better)
Held-out test frames: 449 for `clean`, 150 for the noisy protocols.

| Condition | Bicubic | Joint bilateral | Classical (ours) | Deep only | Hybrid v2 (ours) | **Auto (ours)** |
|---|---|---|---|---|---|---|
| clean, 2x (n=449) | 2.56 | 2.60 | 2.86 | 1.76 | **1.75** | 1.76 |
| clean, 4x (n=449) | 4.61 | 4.47 | 4.77 | **2.77** | 2.92 | 2.77 |
| clean, 8x (n=449) | 7.69 | 7.49 | 7.69 | **4.98** | 5.17 | 5.00 |
| noisy, 2x (n=150) | 6.71 | 5.50 | 4.06 | **2.51** | 2.54 | 2.54 |
| noisy, 4x (n=150) | 7.53 | 6.41 | 5.33 | 3.71 | **3.64** | 3.64 |
| noisy, 8x (n=150) | 9.20 | 8.30 | 7.57 | **5.70** | 5.73 | 5.73 |
| noisy+holes, 2x (n=150) | 7.52 | 6.30 | 4.65 | 3.61 | **3.18** | **3.18** |
| noisy+holes, 4x (n=150) | 8.90 | 7.75 | 6.37 | 5.48 | **4.72** | **4.72** |
| noisy+holes, 8x (n=150) | 11.31 | 10.23 | 9.13 | 8.21 | **7.58** | 7.58 |

Edge F1 (higher is better) and RMSE side by side:

| Condition | RMSE bicubic | RMSE classical | RMSE deep-only | RMSE hybrid | F1 bicubic | F1 classical | F1 deep-only | F1 hybrid |
|---|---|---|---|---|---|---|---|---|
| clean, 4x | 4.61 | 4.77 | 2.77 | 2.92 | 0.74 | 0.91 | 0.95 | 0.95 |
| clean, 8x | 7.69 | 7.69 | 4.98 | 5.17 | 0.45 | 0.76 | 0.90 | 0.84 |
| noisy+holes, 4x | 8.90 | 6.37 | 5.48 | 4.72 | 0.56 | 0.69 | 0.86 | 0.86 |
| noisy+holes, 8x | 11.31 | 9.13 | 8.21 | 7.58 | 0.36 | 0.53 | 0.73 | 0.71 |

Speed (480x640, 4 threads, median): bicubic 3 ms, classical 398 ms, deep-only 162 ms, hybrid 527 ms, Auto 164 ms.

### 8.1 Main findings
1. **Learning dominates RMSE.** Every learned arm is far better than interpolation (e.g. clean 8x: bicubic 7.69 cm vs deep-only 4.98 cm).
2. **The classical stage mainly buys sharpness, not RMSE.** It lifts edge F1 at 8x from 0.45 to 0.76, and beats bicubic under noise, but on clean input its RMSE is slightly *worse* than bicubic.
3. **Deep-only vs hybrid depends on input quality.** On pristine input deep-only is marginally better (hybrid minus deep-only = +0.15 to +0.19 cm at 4x/8x, CI excludes 0). With noise and holes the hybrid is clearly better (-0.44 to -0.76 cm RMSE, about -2 cm boundary RMSE). That is why Auto exists.
4. **The hybrid never loses to bicubic:** it was better on 100% of test frames in both settings checked.

### 8.2 Ablation (80 test frames, 8x; see `docs/ABLATION.md` for all tables)
| Step | clean RMSE | noisy+holes RMSE |
|---|---|---|
| bicubic (E0) | 7.98 | 11.29 |
| classical gated JBU (E1) | 7.98 | 11.02 |
| + learned residual only (beta = 0) | 5.58 | 9.25 |
| full hybrid (beta and delta) | **5.41** | **9.09** |
| deep-only | 5.25 | 9.90 |
| v1 softmax arbiter (superseded) | 5.86 | 9.79 |
| hard colour-plateau snap added to classical | 13.42 | 13.91 |

Reading it: the residual does most of the work; the learned blend adds a small extra 2-3%; the gate reduces texture-copy error slightly (it brings plain JBU's excess back below bicubic on clean input); LR clean-up matters only when input is degraded (12.62 -> 11.02 cm); the hard snap is a **negative result**.

### 8.3 Why the snap failed
NYU ground truth is Kinect depth with inpainting, so its boundaries are soft. A hard decision "this pixel belongs to the near surface or the far surface" is perfectly sharp but, when wrong, costs the full depth jump (often more than a metre), and RMSE punishes that heavily.


### 8.4 The large GPU-trained model (`dl_only_L`)
RMSE in cm on the same held-out test frames as above (449 clean, 150 per noisy protocol):

| Condition | Bicubic | Small deep-only | Small hybrid | **Large deep-only (dl_only_L)** |
|---|---|---|---|---|
| clean, 2x | 2.56 | 1.76 | 1.75 | **0.95** |
| clean, 4x | 4.61 | 2.77 | 2.92 | **1.69** |
| clean, 8x | 7.69 | 4.98 | 5.17 | **3.43** |
| noisy, 2x | 6.71 | 2.51 | 2.54 | **1.88** |
| noisy, 4x | 7.53 | 3.71 | 3.64 | **2.77** |
| noisy, 8x | 9.20 | 5.70 | 5.73 | **4.35** |
| noisy+holes, 2x | 7.52 | 3.61 | 3.18 | **2.48** |
| noisy+holes, 4x | 8.90 | 5.48 | 4.72 | **3.82** |
| noisy+holes, 8x | 11.31 | 8.21 | 7.58 | **6.11** |

* It wins every cell on RMSE and boundary RMSE, typically 25-45% below the small deep-only model (e.g. clean 8x: 3.43 vs 4.98 cm).
* Training: 12,350 steps, batch 16, 256 px crops, bf16, EMA, frames 0-949, best validation RMSE 1.686 cm; about 9 M parameters, 36 MB, roughly 2.6 s per image on 4 laptop-CPU threads.
* **Important honesty point for the viva:** the best model has *no classical features*. The hybrid idea helped only the small models under noise and holes. We did not train a large hybrid, so we cannot say whether classical features would still help at large scale. The ablations (section 8.2) and failure analysis (section 9) were done on the small models.
* Why bigger won: 40x more parameters, 50-100x more training signal (larger crops and batches, GPU, longer schedule), EMA weights and photometric augmentation, on 150 more training frames.

## 9. Failure analysis (`docs/FAILURES.md`)
* **What predicts failure:** error correlates with **GT edge density (Spearman 0.94) and depth range (0.89)** - cluttered scenes with many large depth jumps - and is essentially unrelated to brightness or local colour contrast (|rho| <= 0.13). Our own hypothesis (low-contrast edges cause failures) was *not supported*.
* **Worst frame 1015 (clean 8x, RMSE ~21 cm):** a dark corridor with a glass/mirror door. Door frames are narrower than one low-res cell (information is aliased away at 8x) and the glass region has depth unrelated to RGB structure. Hybrid ~ bicubic here.
* **Frame 1240 (noisy+holes 4x, worst for that setting):** large depth jumps; missing samples fall exactly on boundaries (nothing to learn the edge from); noise is multiplicative so far surfaces carry ~8 cm noise.
* **Observed weakness:** isolated surviving noise spikes trigger the step detector, opening the gate wrongly (visible in the stage figure); a connected-component size filter would fix it.
* **Metric artefact:** edge F1 is unstable on simple scenes (~0.4% edge pixels).

## 10. Engineering story: mistakes and fixes (great for the viva)
1. **JBU returned garbage** -> tight colour weights underflowed `exp()` to 0 -> fixed with a log-domain softmax.
2. **Gated pipeline worse than bicubic** -> the detector measured depth *range* in a window, so sloped planes looked like edges -> replaced by a second-difference *step* detector with a scale-dependent threshold.
3. **Hard snap hurt** -> kept only as a learned input feature; reported as a negative result.
4. **Hybrid v1 lost to deep-only** -> a softmax over experts can never give one expert weight 1, so blurry estimates leaked in; it also had fewer training steps in the same wall-clock time -> redesigned as `E0 + beta(E1-E0) + delta` and equal step budgets. *Disclosure:* this design change was made after seeing test results.
5. **I mis-attributed two log files** and briefly reported the validation scores swapped; the test results exposed it and it was corrected. Lesson: verify which log is which.
6. **Visualisation naming clash:** "gate" meant both the classical discontinuity gate G and the learned blend beta -> renamed in figures and docs.
7. **UI problems found by browser testing:** errors rendered off-screen below the tabs; sidebar hidden on phones -> errors moved above tabs, added a main-area Reconstruct button.
8. **Observation:** the learned beta is only weakly selective (mostly 0.4-0.5); most gain is from delta. We report that rather than claim sharp learned arbitration.

## 11. Likely viva questions with model answers
1. *Why does RGB guidance cause texture copying and how do you stop it?* Colour edges that are not depth edges (patterns, shadows) get treated as depth edges by a colour-weighted filter. We only allow colour weighting where an independent depth-based step detector says a real discontinuity exists (the gate).
2. *Why a second difference for edge detection?* A step gives a large second difference; a linear ramp (sloped floor) gives zero. A first-derivative or range test fires on slopes.
3. *Why normalised convolution for holes?* Plain blurring treats missing values as 0, dragging estimates down; dividing by the blurred validity mask corrects it.
4. *What does the joint bilateral filter do?* Averages low-res samples weighted by spatial distance and by similarity of the *high-res guide colours*, so averaging does not cross colour (hence probable depth) boundaries.
5. *Why is the hybrid not better than deep-only on clean data?* The classical expert adds little when input is clean and the network already learns edge placement; it helps when input is noisy/incomplete because the classical stage cleans and the blend can trust it. The paired confidence intervals show both effects.
6. *Why did you add Auto, is that cheating?* No: the rule uses only measurable properties of the input (holes, estimated noise), is fixed a priori (threshold 1.2 cm) and never looks at ground truth.
7. *Why RMSE and boundary RMSE both?* RMSE is dominated by flat regions; boundary RMSE isolates where upsampling actually fails.
8. *What is the texture-copy RMSE?* Error in regions where depth is flat but RGB is textured - if texture leaks in, error rises there.
9. *What is the train/val/test split and why contiguous?* 0-799 / 800-999 / 1000-1448, contiguous because frames are ordered by scene, so scenes do not leak between sets.
10. *Biggest limitations?* Synthetic degradation, one indoor dataset, assumed depth units, small model, no cross-dataset test, thin structures below one low-res cell are unrecoverable.
11. *What would you do with more time?* Cross-dataset test, reimplement a published baseline (e.g. DKN) under our protocol, a pretrained depth prior, connected-component filtering of the discontinuity map.
12. *What does the 0.22 M parameter count mean for deployment?* 0.86 MB file, ~160 ms per frame on CPU; the large GPU models trade size for accuracy.

## 12. Related work we consulted
DKN (deformable kernel networks, arXiv 1903.11286): texture copy from regular sampling; Depth Edge Guided CNN (arXiv 2003.10138): depth edges as guidance; Robust Guided Image Filtering (arXiv 1703.09379); FDSR (CVPR 2021), SGNet (arXiv 2312.05799), DORNet (arXiv 2410.11666): frequency/gradient guidance and lightweight designs; Qiao et al. (arXiv 2303.09307) and JIIF (arXiv 2107.08717): frequency decoupling and implicit functions (future work). We did **not** compare against published numbers because protocols differ.

## 13. Where the project stands and what is left
**Done:** code (`depthup/`), small trained models, full evaluation, ablation, failure analysis, architecture diagram, design doc, README, Streamlit app (tested on desktop, phone viewport and bad-upload paths), Dockerfile, 17 unit tests, pushed to branch `claude/practical-pasteur-0j1ipi`.

**GPU run (done, partially):** `dl_only_L` was trained and evaluated on the laptop GPU. The large hybrid and the ensemble were cancelled because of the deadline; `scripts/gpu_pipeline.py` can still train them (see `docs/GPU_TRAINING.md`).

**Pending (team):** live deployment (note: ordinary shared cPanel hosting often cannot run PyTorch/Streamlit; Hugging Face Spaces with the Docker file is the safer route; the large model may be slow on a free CPU host), USNs and team ID in `eval_P4_<TeamID>.md`, the organisers' design-doc template, viva preparation.

## 14. Repository map
| Path | What it is |
|---|---|
| `depthup/classical.py` | all classical methods (interpolation, fill, noise, JBU, guided filter, detector, gate, snap) |
| `depthup/degrade.py` | synthetic sensor degradation and protocols |
| `depthup/model.py` | HybridNet, DLOnlyNet, v1 convex arbiter |
| `depthup/pipeline.py` | feature building, all comparison arms (`METHODS`), Auto, TTA, ensemble |
| `depthup/train.py` | trainer (CPU/GPU, bf16, EMA, resume, probe) |
| `depthup/evaluate.py`, `ablate.py`, `benchmark.py`, `failures.py`, `report.py` | experiments and reporting |
| `depthup/metrics.py`, `io.py`, `viz.py`, `data.py` | metrics, safe file loading, plots, dataset handling |
| `app.py`, `Dockerfile`, `.streamlit/` | web app and deployment |
| `scripts/` | data prep, `reproduce.sh`, `gpu_pipeline.py`, doc generator, diagram |
| `docs/` | `DESIGN.md`, `ABLATION.md`, `FAILURES.md`, `architecture.png`, `figures/`, GPU guide, handoff prompt |
| `eval_P4_T19.md` | comparative evaluation file  |
| `results/` | raw per-image results, summaries, logs |

## 15. Glossary
**Bicubic**: smooth cubic interpolation over 4x4 pixels. **JBU**: joint bilateral upsampling. **Guided filter**: fast edge-preserving filter using a guide image. **Normalised convolution**: blur of value x mask divided by blur of mask; fills holes without bias. **Second difference**: discrete second derivative, zero on ramps. **Gate**: spatial map saying where RGB may influence depth. **Residual**: a correction added to an existing estimate. **EMA**: exponential moving average of weights, usually a more accurate final model. **TTA**: test-time augmentation (here: average with the prediction on a horizontally flipped input). **Ensemble**: average of independently trained models. **Spearman rho**: rank correlation. **Bootstrap CI**: confidence interval from resampling test frames. **Pareto**: trade-off between accuracy and cost.
