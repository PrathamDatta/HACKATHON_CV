# Design document: EdgeGate (P4, edge-preserving depth map upsampling and refinement)

## 1. Problem and requirement coverage

| P4 requirement | Where it is met |
|---|---|
| Nearest / bilinear / bicubic baselines | `classical.nearest_up/bilinear_up/bicubic_up` |
| Joint bilateral filtering | `classical.jbu` (plain JBU arm and the gated variant) |
| Guided filtering | `classical.guided_filter`, `classical.guided_up` |
| Edge detection, boundary-aware refinement, depth discontinuity detection | `classical.lr_discontinuity`, `gate_map`, `plateau_snap`, `rgb_edges` |
| Noise removal and hole filling | `classical.normalized_fill`, `remove_spikes`, `clean_lr` |
| Deep depth completion / upsampling network, RGB guidance | `model.DLOnlyNet`, `model.HybridNet` (both take RGB) |
| Fine-tune on depth datasets | `train.py` on NYU Depth V2 train split |
| Hybrid: interpolation + edge-aware filtering + deep learning, boundary-preserving refinement | `pipeline.run_hybrid`: E0 + β(E1 − E0) + δ |
| Prevent oversmoothing and texture-copying | the gate (Section 3.2); measured by the *texture-copy RMSE* metric |
| 2x / 4x / 8x | all evaluated, one model handles all three (scale is an input channel) |
| Missing and noisy depth | `degrade.PROTOCOLS`: `clean`, `noisy`, `noisy+holes` |
| Compare interpolation / classical / deep / hybrid | `evaluate.py`, `eval_P4_<team>.md` |
| Unseen depth maps | held-out NYU test frames 1000-1448, never used for training, validation or tuning |
| Behaviour around boundaries | boundary-band RMSE, edge F1, boundary-profile figure |
| RMSE, MAE, PSNR, SSIM, edge/boundary accuracy, per-factor results, latency, memory | `metrics.py`, `evaluate.py`, `benchmark.py` |
| Ablation and failure analysis (instruction sheet) | `docs/ABLATION.md`, `docs/FAILURES.md` |

## 2. Data and protocol

* **NYU Depth V2**, 1,449 labelled RGB-D frames (640x480). Depth is dense (Kinect + inpainting), stored normalised; we assume `metres = raw x 10`. Every absolute error is therefore in *assumed* metres. Relative comparisons between methods are unaffected.
* **Split** (contiguous blocks, since frames are ordered by capture scene, which limits scene leakage): train 0-799, validation 800-999, **test 1000-1448**. Hyper-parameters of the classical stage were tuned on an 8-frame subset of validation; model selection used validation only. The test split was used by `evaluate.py`, `ablate.py`, `failures.py`, and **one design decision (the v1 to v2 arbiter change) was made after seeing v1's test results**, which we state openly: v2's test numbers are therefore not a fully blind estimate of a pre-registered design, although the change was motivated by structure (softmax cannot saturate) and the same step budget.
* **Degradation** (`degrade.py`): anti-aliased bicubic down-sampling by 2/4/8 (training also uses raw sub-sampling 20% of the time), plus protocol-dependent corruption: `clean`; `noisy` (2% multiplicative Gaussian noise, 0.5% spike outliers); `noisy+holes` (as noisy plus 20% missing LR pixels, half salt, half blobs).
* **Caveat:** the ground truth is itself inpainted Kinect depth, so its object boundaries are softer and less precise than LiDAR or synthetic depth. Boundary metrics reward sharpness only up to the GT's own accuracy.

## 3. Method

### 3.1 Pipeline (see `docs/architecture.png`; names match the code)
1. **LR clean-up** (`clean_lr`): multi-scale normalised-convolution hole filling; robust noise estimate (Immerkaer kernel + MAD); removal of isolated extrema (flying pixels); bilateral smoothing **only when estimated noise exceeds 1 cm** (adaptive: clean inputs are untouched).
2. **Bicubic base D0** = expert **E0**.
3. **Step detector + gate g** (`lr_discontinuity`, `gate_map`): second differences in four directions on the cleaned LR depth. A step gives a large response; a sloped plane gives about zero, so floors and walls are not flagged. The tolerance grows with the scale (`0.025 m x scale`) because a wall/floor crease produces a second difference of the order of slope x scale. Evidence is spread over the band where upsampling blur occurs. Known weakness: surviving noise spikes can trigger it (docs/FAILURES.md).
4. **Gated tri-lateral JBU** (`jbu`) = expert **E1**. Taps are the 4x4 nearest LR samples. Weights = spatial Gaussian x colour kernel x depth-consistency kernel x sample confidence, evaluated in the log domain (a plain implementation underflowed and produced garbage). The colour tolerance is **spatially varying**: `sigma_c = exp((1-g) ln 5 + g ln 0.15)`: tight near depth discontinuities, effectively infinite elsewhere. **RGB can only vote where the depth already shows a discontinuity, which is what blocks texture-copying.** Outside the band the output falls back to D0.
5. **HybridNet** (`model.py`, 0.22 M parameters, 3-level U-Net with dilated bottleneck, GroupNorm): inputs RGB, E0, E1, colour-plateau-snap candidate, discontinuity gate G, RGB edge magnitude, confidence, scale. Outputs a residual δ and a blend logit.
6. **Learned blend**: `out = E0 + β (E1 − E0) + δ`, with β = sigmoid(logit) per pixel. β is the learned trust in the classical sharpening; δ is the learned correction. With β = 0 the output has the same form as the deep-only model (bicubic plus a learned residual), so the hybrid can represent that solution; training does not guarantee it matches it (on clean input the small hybrid ended slightly worse). (Not to be confused with G, the classical discontinuity gate in step 3.)
7. **Auto selector** (`pipeline.run_auto`): measured degradation chooses the model. No missing values and estimated noise below 1.2 cm (threshold fixed a priori; clean inputs estimate 0.2-0.5 cm, 2% noise gives above 2 cm) uses the deep-only network; otherwise the hybrid. This reflects the measured trade-off (deep-only slightly better on pristine depth, hybrid clearly better on degraded depth).

### 3.2 Why this design
* The literature identifies texture-copying as the failure of regularly-sampled guided filters when RGB and depth structure disagree (DKN, Kim et al.), and addresses it with depth-edge guidance (EGCNN) or learned deformable kernels. Our gate is a cheap, interpretable counterpart: depth-edge evidence decides where colour is allowed to matter.
* A hard "colour-plateau snap" (assign each pixel to the nearer depth plateau by colour) was tried first as a classical post-process. On NYU's soft ground truth it **hurt** (RMSE 11 vs 5.5 cm at 4x): a wrong assignment costs the full depth jump. We therefore kept it only as an *input feature* the learned arbiter may exploit, and report the hard version in the ablation as a negative result.
* **A first arbiter design failed and was replaced.** v1 mixed the experts with a softmax (`HybridNetConvex`, kept as an evaluation arm). A softmax can never give one expert weight 1, so it leaked blurry estimates into the output and lost to deep-only on clean and noisy input; it also received fewer training steps in the same wall-clock time (4,501 vs 6,206). v2 uses the residual form above and equal step counts (6,200 vs 6,206).
* β and δ can be visualised (app tab "Intermediate stages"). In practice β is only weakly spatially selective (mostly 0.4-0.5 with dips near some edges); most of the gain comes from δ (see `docs/ABLATION.md`), and we report that rather than claim a sharp learned arbitration.

### 3.3 Training (`train.py`)
On-the-fly degradation of random 128x128 HR crops (flip augmentation), scale in {2,4,8} (one model for all), protocol mix 50% clean / 25% noisy / 25% noisy+holes with random magnitudes. Loss = boundary-weighted L1 (GT gradient > 0.1 m/px weighted x7) + 0.5 x gradient-matching L1, plus the same loss on the DL branch alone so E2 stays a useful stand-alone expert. AdamW, lr 2e-3 with cosine decay, batch 8, **6,200 steps for the hybrid and 6,206 for deep-only** (equal budget); best-validation checkpoint kept. Loss is applied to the final output only (the v1 auxiliary branch loss was dropped).

### 3.4 Comparison arms
Nearest, bilinear, bicubic (interpolation); JBU, guided filter, **classical gated JBU (ours)** (classical); **DLOnlyNet** (same backbone, inputs RGB + bicubic only, no classical features); Hybrid v1 (softmax arbiter, superseded); **Hybrid v2 (ours)**; **EdgeGate-Auto (ours)**.

### 3.5 Metrics (`metrics.py`)
RMSE, MAE (cm); PSNR (peak = per-image GT max) and SSIM; **boundary RMSE** in a +/-3 px band around GT discontinuities (|grad| > 0.12 m/px, ~4% of pixels); **edge F1** (2 px tolerance); **texture-copy RMSE** (error where depth is flat but RGB is textured, the direct measurable of texture-copying); delta<1.25; latency (median / p95, fresh process); peak RSS.

## 4. Related work consulted
* Kim, Ponce, Ham. *Deformable kernel networks for guided depth map upsampling* (DKN), [arXiv 1903.11286](https://www.alphaxiv.org/abs/1903.11286): motivation for texture-copy and adaptive neighbourhoods.
* Liu et al. *Depth Edge Guided CNNs for sparse depth upsampling*, [arXiv 2003.10138](https://ar5iv.labs.arxiv.org/html/2003.10138): depth edges as guidance.
* *Robust Guided Image Filtering*, [arXiv 1703.09379](https://arxiv.org/pdf/1703.09379): bilateral guidance weights avoid texture-copy.
* He et al. *Towards fast and accurate real-world depth super-resolution* (FDSR), [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.pdf); SGNet [arXiv 2312.05799](https://arxiv.org/pdf/2312.05799); DORNet [arXiv 2410.11666](https://arxiv.org/pdf/2410.11666): gradient/high-frequency guidance and lightweight models (about 0.5 M parameters).
* Qiao et al. [arXiv 2303.09307](https://arxiv.org/pdf/2303.09307); Tang et al. JIIF [arXiv 2107.08717](https://arxiv.org/pdf/2107.08717): frequency decoupling and implicit interpolation (future work).
* Published RMSEs are **not** compared against ours: they depend on training set, protocol and scale. We compare only methods we ran ourselves on the same split.

## 5. Final models and a deviation from the plan (added at submission time)
* **Large deep-only model `dl_only_L`** (`depthup/model.py` `DLOnlyNet`, width 64, 3 blocks per stage, 3 dilated-block repeats, about 9 M parameters). Trained on a GPU for 12,350 steps (batch 16, 256 px crops, bf16 autocast, EMA weights, photometric augmentation, learning rate 1e-3 with cosine decay) on frames 0-949; validation on frames 950-999 (best validation RMSE 1.686 cm). The small CPU-trained models used frames 0-799 and 800-999. **Test frames 1000-1448 were never used by any model.**
* **It beats every other arm on RMSE and boundary RMSE in all nine protocol/scale cells** (see `eval_P4_T19.md`; e.g. clean 8x: 3.43 cm vs 4.98 cm for the small deep-only model and 7.69 cm for bicubic).
* **What was not done:** the large hybrid (`hybrid_L`) and the seed ensemble were planned but not trained, because the submission deadline did not leave time for them. The evidence that classical features help (the small hybrid winning under noise and holes) therefore exists only at small-model scale; whether they would help a large model is untested, and the documented ablations and failure analysis refer to the small models.
* **Cost of the large model:** about 36 MB and roughly 2.6 s per 480x640 image on 4 CPU threads (laptop CPU), against 0.9 MB and about 0.1 s for the small deep-only model. The live app therefore shows bicubic, our classical stage, the small Auto selector and `dl_only_L` by default.
