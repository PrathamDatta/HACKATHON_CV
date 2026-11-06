USN1,USN2,USN3,USN4

# eval_P4_T01: Edge-Preserving Depth Map Upsampling and Refinement

Dataset: NYU Depth V2 (1,449 labelled frames). Split (contiguous scene blocks): train 0-799, val 800-999, **test 1000-1448 (never used for training, validation or tuning)**. Depth assumed = raw x 10 m. Metrics are means over test frames. RMSE/MAE in cm. PSNR peak = per-image GT max depth. Boundary band = +/-3 px around GT discontinuities (|grad| > 0.12 m/px). Edge F1 uses 2 px tolerance.

## Protocol `clean`  (n = 449 test frames)

### 1. RMSE and MAE (cm, lower is better)

| Scale | Method | RMSE | MAE |
|---|---|---|---|
| 2x | Nearest | 3.59 | 0.98 |
| 2x | Bilinear | 2.94 | 0.84 |
| 2x | Bicubic | 2.56 | 0.72 |
| 2x | Joint bilateral (JBU) | 2.60 | 0.73 |
| 2x | Guided filter | 3.06 | 1.04 |
| 2x | Classical (ours: gated JBU) | 2.86 | 0.69 |
| 2x | Deep learning only | 1.76 | 0.64 |
| 2x | Hybrid (ours) | 1.95 | 0.68 |
| 4x | Nearest | 6.55 | 2.09 |
| 4x | Bilinear | 5.11 | 1.70 |
| 4x | Bicubic | 4.61 | 1.54 |
| 4x | Joint bilateral (JBU) | 4.47 | 1.46 |
| 4x | Guided filter | 5.17 | 1.97 |
| 4x | Classical (ours: gated JBU) | 4.77 | 1.43 |
| 4x | Deep learning only | 2.77 | 1.06 |
| 4x | Hybrid (ours) | 3.04 | 1.13 |
| 8x | Nearest | 10.40 | 3.96 |
| 8x | Bilinear | 8.39 | 3.32 |
| 8x | Bicubic | 7.69 | 3.05 |
| 8x | Joint bilateral (JBU) | 7.49 | 2.85 |
| 8x | Guided filter | 8.62 | 3.77 |
| 8x | Classical (ours: gated JBU) | 7.69 | 2.81 |
| 8x | Deep learning only | 4.98 | 2.04 |
| 8x | Hybrid (ours) | 5.60 | 2.30 |

### 2. PSNR (dB) and SSIM (higher is better)

| Scale | Method | PSNR | SSIM |
|---|---|---|---|
| 2x | Nearest | 44.23 | 0.991 |
| 2x | Bilinear | 46.01 | 0.994 |
| 2x | Bicubic | 47.24 | 0.995 |
| 2x | Joint bilateral (JBU) | 47.03 | 0.995 |
| 2x | Guided filter | 45.51 | 0.993 |
| 2x | Classical (ours: gated JBU) | 46.35 | 0.995 |
| 2x | Deep learning only | 50.18 | 0.997 |
| 2x | Hybrid (ours) | 49.36 | 0.997 |
| 4x | Nearest | 38.97 | 0.971 |
| 4x | Bilinear | 41.19 | 0.982 |
| 4x | Bicubic | 42.10 | 0.984 |
| 4x | Joint bilateral (JBU) | 42.31 | 0.987 |
| 4x | Guided filter | 41.01 | 0.982 |
| 4x | Classical (ours: gated JBU) | 41.79 | 0.985 |
| 4x | Deep learning only | 46.37 | 0.993 |
| 4x | Hybrid (ours) | 45.59 | 0.992 |
| 8x | Nearest | 34.95 | 0.932 |
| 8x | Bilinear | 36.87 | 0.962 |
| 8x | Bicubic | 37.63 | 0.964 |
| 8x | Joint bilateral (JBU) | 37.83 | 0.971 |
| 8x | Guided filter | 36.61 | 0.966 |
| 8x | Classical (ours: gated JBU) | 37.64 | 0.967 |
| 8x | Deep learning only | 41.23 | 0.981 |
| 8x | Hybrid (ours) | 40.23 | 0.977 |

### 3. Edge preservation and boundary accuracy

| Scale | Method | Boundary RMSE cm | Edge F1 | Texture-copy RMSE cm |
|---|---|---|---|---|
| 2x | Nearest | 10.60 | 0.940 | 1.25 |
| 2x | Bilinear | 8.70 | 0.879 | 1.03 |
| 2x | Bicubic | 7.55 | 0.924 | 0.91 |
| 2x | Joint bilateral (JBU) | 7.71 | 0.943 | 0.84 |
| 2x | Guided filter | 9.02 | 0.901 | 0.98 |
| 2x | Classical (ours: gated JBU) | 8.50 | 0.961 | 0.86 |
| 2x | Deep learning only | 5.01 | 0.985 | 0.84 |
| 2x | Hybrid (ours) | 5.55 | 0.984 | 0.94 |
| 4x | Nearest | 19.30 | 0.931 | 2.31 |
| 4x | Bilinear | 15.16 | 0.675 | 1.70 |
| 4x | Bicubic | 13.57 | 0.740 | 1.54 |
| 4x | Joint bilateral (JBU) | 13.25 | 0.867 | 1.44 |
| 4x | Guided filter | 15.10 | 0.794 | 1.69 |
| 4x | Classical (ours: gated JBU) | 14.18 | 0.909 | 1.55 |
| 4x | Deep learning only | 7.84 | 0.950 | 1.19 |
| 4x | Hybrid (ours) | 8.63 | 0.935 | 1.39 |
| 8x | Nearest | 29.92 | 0.715 | 3.51 |
| 8x | Bilinear | 24.39 | 0.367 | 2.60 |
| 8x | Bicubic | 22.37 | 0.447 | 2.64 |
| 8x | Joint bilateral (JBU) | 21.56 | 0.738 | 2.51 |
| 8x | Guided filter | 23.80 | 0.644 | 3.16 |
| 8x | Classical (ours: gated JBU) | 22.35 | 0.762 | 2.56 |
| 8x | Deep learning only | 14.00 | 0.896 | 2.16 |
| 8x | Hybrid (ours) | 15.84 | 0.816 | 2.41 |

## Protocol `noisy`  (n = 150 test frames)

### 1. RMSE and MAE (cm, lower is better)

| Scale | Method | RMSE | MAE |
|---|---|---|---|
| 2x | Nearest | 8.42 | 4.39 |
| 2x | Bilinear | 5.52 | 2.98 |
| 2x | Bicubic | 6.71 | 3.67 |
| 2x | Joint bilateral (JBU) | 5.50 | 2.94 |
| 2x | Guided filter | 3.64 | 1.93 |
| 2x | Classical (ours: gated JBU) | 4.06 | 2.33 |
| 2x | Deep learning only | 2.51 | 1.47 |
| 2x | Hybrid (ours) | 2.79 | 1.54 |
| 4x | Nearest | 9.67 | 4.93 |
| 4x | Bilinear | 6.77 | 3.53 |
| 4x | Bicubic | 7.53 | 4.05 |
| 4x | Joint bilateral (JBU) | 6.41 | 3.32 |
| 4x | Guided filter | 5.22 | 2.64 |
| 4x | Classical (ours: gated JBU) | 5.33 | 2.82 |
| 4x | Deep learning only | 3.71 | 2.10 |
| 4x | Hybrid (ours) | 4.07 | 2.28 |
| 8x | Nearest | 11.91 | 5.99 |
| 8x | Bilinear | 8.85 | 4.57 |
| 8x | Bicubic | 9.20 | 4.92 |
| 8x | Joint bilateral (JBU) | 8.30 | 4.20 |
| 8x | Guided filter | 7.82 | 3.97 |
| 8x | Classical (ours: gated JBU) | 7.57 | 3.85 |
| 8x | Deep learning only | 5.70 | 3.10 |
| 8x | Hybrid (ours) | 6.41 | 3.47 |

### 2. PSNR (dB) and SSIM (higher is better)

| Scale | Method | PSNR | SSIM |
|---|---|---|---|
| 2x | Nearest | 34.74 | 0.857 |
| 2x | Bilinear | 38.44 | 0.927 |
| 2x | Bicubic | 36.70 | 0.895 |
| 2x | Joint bilateral (JBU) | 38.46 | 0.926 |
| 2x | Guided filter | 42.30 | 0.976 |
| 2x | Classical (ours: gated JBU) | 41.49 | 0.961 |
| 2x | Deep learning only | 45.70 | 0.988 |
| 2x | Hybrid (ours) | 44.75 | 0.986 |
| 4x | Nearest | 33.66 | 0.875 |
| 4x | Bilinear | 36.82 | 0.938 |
| 4x | Bicubic | 35.80 | 0.918 |
| 4x | Joint bilateral (JBU) | 37.26 | 0.944 |
| 4x | Guided filter | 39.39 | 0.972 |
| 4x | Classical (ours: gated JBU) | 39.25 | 0.963 |
| 4x | Deep learning only | 42.34 | 0.980 |
| 4x | Hybrid (ours) | 41.50 | 0.977 |
| 8x | Nearest | 32.05 | 0.879 |
| 8x | Bilinear | 34.73 | 0.942 |
| 8x | Bicubic | 34.25 | 0.933 |
| 8x | Joint bilateral (JBU) | 35.25 | 0.950 |
| 8x | Guided filter | 36.11 | 0.962 |
| 8x | Classical (ours: gated JBU) | 36.37 | 0.956 |
| 8x | Deep learning only | 38.58 | 0.969 |
| 8x | Hybrid (ours) | 37.63 | 0.964 |

### 3. Edge preservation and boundary accuracy

| Scale | Method | Boundary RMSE cm | Edge F1 | Texture-copy RMSE cm |
|---|---|---|---|---|
| 2x | Nearest | 12.64 | 0.572 | 7.74 |
| 2x | Bilinear | 9.37 | 0.611 | 4.88 |
| 2x | Bicubic | 9.52 | 0.597 | 6.25 |
| 2x | Joint bilateral (JBU) | 8.83 | 0.641 | 5.09 |
| 2x | Guided filter | 8.72 | 0.888 | 2.88 |
| 2x | Classical (ours: gated JBU) | 8.40 | 0.816 | 3.26 |
| 2x | Deep learning only | 6.04 | 0.960 | 1.99 |
| 2x | Hybrid (ours) | 6.56 | 0.936 | 2.22 |
| 4x | Nearest | 19.27 | 0.655 | 8.00 |
| 4x | Bilinear | 14.65 | 0.529 | 5.30 |
| 4x | Bicubic | 13.91 | 0.554 | 6.42 |
| 4x | Joint bilateral (JBU) | 13.33 | 0.675 | 5.38 |
| 4x | Guided filter | 14.02 | 0.775 | 3.42 |
| 4x | Classical (ours: gated JBU) | 13.55 | 0.711 | 3.64 |
| 4x | Deep learning only | 9.13 | 0.911 | 2.76 |
| 4x | Hybrid (ours) | 9.75 | 0.880 | 3.13 |
| 8x | Nearest | 28.29 | 0.557 | 8.40 |
| 8x | Bilinear | 22.59 | 0.308 | 5.70 |
| 8x | Bicubic | 21.24 | 0.365 | 6.72 |
| 8x | Joint bilateral (JBU) | 20.34 | 0.666 | 5.86 |
| 8x | Guided filter | 21.47 | 0.618 | 4.33 |
| 8x | Classical (ours: gated JBU) | 21.03 | 0.556 | 4.39 |
| 8x | Deep learning only | 14.68 | 0.804 | 3.88 |
| 8x | Hybrid (ours) | 16.22 | 0.736 | 4.50 |

## Protocol `noisy+holes`  (n = 150 test frames)

### 1. RMSE and MAE (cm, lower is better)

| Scale | Method | RMSE | MAE |
|---|---|---|---|
| 2x | Nearest | 9.05 | 4.56 |
| 2x | Bilinear | 6.35 | 3.25 |
| 2x | Bicubic | 7.52 | 3.92 |
| 2x | Joint bilateral (JBU) | 6.30 | 3.21 |
| 2x | Guided filter | 4.37 | 2.20 |
| 2x | Classical (ours: gated JBU) | 4.65 | 2.49 |
| 2x | Deep learning only | 3.61 | 1.76 |
| 2x | Hybrid (ours) | 3.45 | 1.73 |
| 4x | Nearest | 10.83 | 5.30 |
| 4x | Bilinear | 8.04 | 3.97 |
| 4x | Bicubic | 8.90 | 4.50 |
| 4x | Joint bilateral (JBU) | 7.75 | 3.78 |
| 4x | Guided filter | 6.26 | 3.04 |
| 4x | Classical (ours: gated JBU) | 6.37 | 3.17 |
| 4x | Deep learning only | 5.48 | 2.53 |
| 4x | Hybrid (ours) | 5.20 | 2.62 |
| 8x | Nearest | 13.79 | 6.69 |
| 8x | Bilinear | 10.67 | 5.26 |
| 8x | Bicubic | 11.31 | 5.69 |
| 8x | Joint bilateral (JBU) | 10.23 | 4.94 |
| 8x | Guided filter | 9.16 | 4.55 |
| 8x | Classical (ours: gated JBU) | 9.13 | 4.45 |
| 8x | Deep learning only | 8.21 | 3.87 |
| 8x | Hybrid (ours) | 8.20 | 4.09 |

### 2. PSNR (dB) and SSIM (higher is better)

| Scale | Method | PSNR | SSIM |
|---|---|---|---|
| 2x | Nearest | 34.18 | 0.858 |
| 2x | Bilinear | 37.32 | 0.923 |
| 2x | Bicubic | 35.79 | 0.892 |
| 2x | Joint bilateral (JBU) | 37.36 | 0.922 |
| 2x | Guided filter | 40.76 | 0.971 |
| 2x | Classical (ours: gated JBU) | 40.38 | 0.959 |
| 2x | Deep learning only | 42.75 | 0.984 |
| 2x | Hybrid (ours) | 43.10 | 0.984 |
| 4x | Nearest | 32.78 | 0.871 |
| 4x | Bilinear | 35.44 | 0.932 |
| 4x | Bicubic | 34.47 | 0.912 |
| 4x | Joint bilateral (JBU) | 35.74 | 0.937 |
| 4x | Guided filter | 37.83 | 0.968 |
| 4x | Classical (ours: gated JBU) | 37.76 | 0.958 |
| 4x | Deep learning only | 39.07 | 0.974 |
| 4x | Hybrid (ours) | 39.51 | 0.974 |
| 8x | Nearest | 30.90 | 0.873 |
| 8x | Bilinear | 33.22 | 0.935 |
| 8x | Bicubic | 32.62 | 0.925 |
| 8x | Joint bilateral (JBU) | 33.57 | 0.943 |
| 8x | Guided filter | 34.77 | 0.958 |
| 8x | Classical (ours: gated JBU) | 34.80 | 0.952 |
| 8x | Deep learning only | 35.68 | 0.962 |
| 8x | Hybrid (ours) | 35.65 | 0.959 |

### 3. Edge preservation and boundary accuracy

| Scale | Method | Boundary RMSE cm | Edge F1 | Texture-copy RMSE cm |
|---|---|---|---|---|
| 2x | Nearest | 16.10 | 0.589 | 7.86 |
| 2x | Bilinear | 12.38 | 0.621 | 5.27 |
| 2x | Bicubic | 13.21 | 0.611 | 6.55 |
| 2x | Joint bilateral (JBU) | 11.87 | 0.649 | 5.45 |
| 2x | Guided filter | 10.71 | 0.861 | 3.33 |
| 2x | Classical (ours: gated JBU) | 11.02 | 0.817 | 3.36 |
| 2x | Deep learning only | 9.77 | 0.943 | 2.36 |
| 2x | Hybrid (ours) | 9.10 | 0.932 | 2.36 |
| 4x | Nearest | 24.03 | 0.633 | 8.23 |
| 4x | Bilinear | 18.95 | 0.531 | 5.77 |
| 4x | Bicubic | 19.29 | 0.560 | 6.83 |
| 4x | Joint bilateral (JBU) | 17.99 | 0.656 | 5.83 |
| 4x | Guided filter | 16.87 | 0.741 | 3.95 |
| 4x | Classical (ours: gated JBU) | 17.07 | 0.686 | 3.88 |
| 4x | Deep learning only | 14.98 | 0.861 | 3.35 |
| 4x | Hybrid (ours) | 13.77 | 0.844 | 3.36 |
| 8x | Nearest | 33.06 | 0.506 | 9.22 |
| 8x | Bilinear | 26.94 | 0.311 | 6.64 |
| 8x | Bicubic | 26.82 | 0.363 | 7.65 |
| 8x | Joint bilateral (JBU) | 25.14 | 0.620 | 6.83 |
| 8x | Guided filter | 24.36 | 0.584 | 5.25 |
| 8x | Classical (ours: gated JBU) | 24.70 | 0.527 | 5.11 |
| 8x | Deep learning only | 21.57 | 0.725 | 4.93 |
| 8x | Hybrid (ours) | 20.98 | 0.690 | 5.17 |

## 4. Performance across upsampling factors (RMSE cm, clean protocol)

| Method | 2x | 4x | 8x |
|---|---|---|---|
| Nearest | 3.59 | 6.55 | 10.40 |
| Bilinear | 2.94 | 5.11 | 8.39 |
| Bicubic | 2.56 | 4.61 | 7.69 |
| Joint bilateral (JBU) | 2.60 | 4.47 | 7.49 |
| Guided filter | 3.06 | 5.17 | 8.62 |
| Classical (ours: gated JBU) | 2.86 | 4.77 | 7.69 |
| Deep learning only | 1.76 | 2.77 | 4.98 |
| Hybrid (ours) | 1.95 | 3.04 | 5.60 |

## 5. Processing latency and memory (480x640 input, 4 CPU threads, fresh process per method)

Parameters: hybrid 219,012, deep-only 218,241. Weight files: hybrid 866 KB, dl_only 864 KB.

| Method | Scale | Median ms | p95 ms | Peak RSS MB | RSS growth MB |
|---|---|---|---|---|---|
| Nearest | 2x | 1 | 1 | 266 | 0 |
| Nearest | 4x | 1 | 1 | 266 | 0 |
| Nearest | 8x | 1 | 1 | 265 | 0 |
| Bilinear | 2x | 2 | 3 | 267 | 0 |
| Bilinear | 4x | 2 | 3 | 266 | 0 |
| Bilinear | 8x | 2 | 2 | 266 | 0 |
| Bicubic | 2x | 3 | 5 | 267 | 0 |
| Bicubic | 4x | 3 | 4 | 266 | 0 |
| Bicubic | 8x | 3 | 3 | 266 | 0 |
| Joint bilateral (JBU) | 2x | 286 | 296 | 391 | 0 |
| Joint bilateral (JBU) | 4x | 292 | 319 | 389 | 0 |
| Joint bilateral (JBU) | 8x | 291 | 306 | 384 | 0 |
| Guided filter | 2x | 121 | 125 | 337 | 1 |
| Guided filter | 4x | 118 | 123 | 336 | 0 |
| Guided filter | 8x | 131 | 172 | 339 | 0 |
| Classical (ours: gated JBU) | 2x | 397 | 417 | 411 | 2 |
| Classical (ours: gated JBU) | 4x | 353 | 357 | 409 | 0 |
| Classical (ours: gated JBU) | 8x | 359 | 391 | 409 | 2 |
| Deep learning only | 2x | 132 | 159 | 547 | 57 |
| Deep learning only | 4x | 130 | 146 | 565 | 57 |
| Deep learning only | 8x | 129 | 155 | 572 | 73 |
| Hybrid (ours) | 2x | 522 | 576 | 591 | 81 |
| Hybrid (ours) | 4x | 537 | 583 | 571 | 37 |
| Hybrid (ours) | 8x | 478 | 506 | 572 | 48 |

## 6. Statistical check: Hybrid minus Bicubic (paired bootstrap, 95% CI; negative = Hybrid better)

| Protocol | Scale | Metric | Mean diff | 95% CI | n |
|---|---|---|---|---|---|
| clean | 2x | rmse_cm | -0.607 | [-0.660, -0.561] | 449 |
| clean | 2x | boundary_rmse_cm | -2.000 | [-2.125, -1.881] | 449 |
| clean | 4x | rmse_cm | -1.574 | [-1.677, -1.483] | 449 |
| clean | 4x | boundary_rmse_cm | -4.938 | [-5.165, -4.728] | 449 |
| clean | 8x | rmse_cm | -2.085 | [-2.238, -1.951] | 449 |
| clean | 8x | boundary_rmse_cm | -6.530 | [-6.874, -6.213] | 449 |
| noisy | 2x | rmse_cm | -3.913 | [-3.969, -3.857] | 150 |
| noisy | 2x | boundary_rmse_cm | -2.967 | [-3.122, -2.824] | 150 |
| noisy | 4x | rmse_cm | -3.456 | [-3.537, -3.389] | 150 |
| noisy | 4x | boundary_rmse_cm | -4.167 | [-4.464, -3.899] | 150 |
| noisy | 8x | rmse_cm | -2.799 | [-2.938, -2.678] | 150 |
| noisy | 8x | boundary_rmse_cm | -5.013 | [-5.460, -4.587] | 150 |
| noisy+holes | 2x | rmse_cm | -4.067 | [-4.134, -4.003] | 150 |
| noisy+holes | 2x | boundary_rmse_cm | -4.111 | [-4.311, -3.907] | 150 |
| noisy+holes | 4x | rmse_cm | -3.697 | [-3.799, -3.604] | 150 |
| noisy+holes | 4x | boundary_rmse_cm | -5.522 | [-5.873, -5.173] | 150 |
| noisy+holes | 8x | rmse_cm | -3.113 | [-3.269, -2.963] | 150 |
| noisy+holes | 8x | boundary_rmse_cm | -5.838 | [-6.343, -5.347] | 150 |
