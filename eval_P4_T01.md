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
| 2x | Hybrid v1 (softmax arbiter) | 1.95 | 0.68 |
| 2x | Hybrid (ours) | 1.75 | 0.60 |
| 2x | EdgeGate-Auto (ours) | 1.76 | 0.64 |
| 2x | Deep-only-L (GPU-trained) | 0.95 | 0.34 |
| 4x | Nearest | 6.55 | 2.09 |
| 4x | Bilinear | 5.11 | 1.70 |
| 4x | Bicubic | 4.61 | 1.54 |
| 4x | Joint bilateral (JBU) | 4.47 | 1.46 |
| 4x | Guided filter | 5.17 | 1.97 |
| 4x | Classical (ours: gated JBU) | 4.77 | 1.43 |
| 4x | Deep learning only | 2.77 | 1.06 |
| 4x | Hybrid v1 (softmax arbiter) | 3.04 | 1.13 |
| 4x | Hybrid (ours) | 2.92 | 1.09 |
| 4x | EdgeGate-Auto (ours) | 2.77 | 1.06 |
| 4x | Deep-only-L (GPU-trained) | 1.69 | 0.65 |
| 8x | Nearest | 10.40 | 3.96 |
| 8x | Bilinear | 8.39 | 3.32 |
| 8x | Bicubic | 7.69 | 3.05 |
| 8x | Joint bilateral (JBU) | 7.49 | 2.85 |
| 8x | Guided filter | 8.62 | 3.77 |
| 8x | Classical (ours: gated JBU) | 7.69 | 2.81 |
| 8x | Deep learning only | 4.98 | 2.04 |
| 8x | Hybrid v1 (softmax arbiter) | 5.60 | 2.30 |
| 8x | Hybrid (ours) | 5.17 | 2.12 |
| 8x | EdgeGate-Auto (ours) | 5.00 | 2.05 |
| 8x | Deep-only-L (GPU-trained) | 3.43 | 1.40 |

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
| 2x | Hybrid v1 (softmax arbiter) | 49.36 | 0.997 |
| 2x | Hybrid (ours) | 50.27 | 0.997 |
| 2x | EdgeGate-Auto (ours) | 50.18 | 0.997 |
| 2x | Deep-only-L (GPU-trained) | 55.58 | 0.999 |
| 4x | Nearest | 38.97 | 0.971 |
| 4x | Bilinear | 41.19 | 0.982 |
| 4x | Bicubic | 42.10 | 0.984 |
| 4x | Joint bilateral (JBU) | 42.31 | 0.987 |
| 4x | Guided filter | 41.01 | 0.982 |
| 4x | Classical (ours: gated JBU) | 41.79 | 0.985 |
| 4x | Deep learning only | 46.37 | 0.993 |
| 4x | Hybrid v1 (softmax arbiter) | 45.59 | 0.992 |
| 4x | Hybrid (ours) | 45.90 | 0.992 |
| 4x | EdgeGate-Auto (ours) | 46.37 | 0.993 |
| 4x | Deep-only-L (GPU-trained) | 50.56 | 0.997 |
| 8x | Nearest | 34.95 | 0.932 |
| 8x | Bilinear | 36.87 | 0.962 |
| 8x | Bicubic | 37.63 | 0.964 |
| 8x | Joint bilateral (JBU) | 37.83 | 0.971 |
| 8x | Guided filter | 36.61 | 0.966 |
| 8x | Classical (ours: gated JBU) | 37.64 | 0.967 |
| 8x | Deep learning only | 41.23 | 0.981 |
| 8x | Hybrid v1 (softmax arbiter) | 40.23 | 0.977 |
| 8x | Hybrid (ours) | 40.94 | 0.980 |
| 8x | EdgeGate-Auto (ours) | 41.21 | 0.981 |
| 8x | Deep-only-L (GPU-trained) | 44.45 | 0.989 |

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
| 2x | Hybrid v1 (softmax arbiter) | 5.55 | 0.984 | 0.94 |
| 2x | Hybrid (ours) | 5.00 | 0.984 | 0.82 |
| 2x | EdgeGate-Auto (ours) | 5.01 | 0.985 | 0.84 |
| 2x | Deep-only-L (GPU-trained) | 2.65 | 0.997 | 0.43 |
| 4x | Nearest | 19.30 | 0.931 | 2.31 |
| 4x | Bilinear | 15.16 | 0.675 | 1.70 |
| 4x | Bicubic | 13.57 | 0.740 | 1.54 |
| 4x | Joint bilateral (JBU) | 13.25 | 0.867 | 1.44 |
| 4x | Guided filter | 15.10 | 0.794 | 1.69 |
| 4x | Classical (ours: gated JBU) | 14.18 | 0.909 | 1.55 |
| 4x | Deep learning only | 7.84 | 0.950 | 1.19 |
| 4x | Hybrid v1 (softmax arbiter) | 8.63 | 0.935 | 1.39 |
| 4x | Hybrid (ours) | 8.28 | 0.945 | 1.34 |
| 4x | EdgeGate-Auto (ours) | 7.84 | 0.950 | 1.19 |
| 4x | Deep-only-L (GPU-trained) | 4.70 | 0.989 | 0.80 |
| 8x | Nearest | 29.92 | 0.715 | 3.51 |
| 8x | Bilinear | 24.39 | 0.367 | 2.60 |
| 8x | Bicubic | 22.37 | 0.447 | 2.64 |
| 8x | Joint bilateral (JBU) | 21.56 | 0.738 | 2.51 |
| 8x | Guided filter | 23.80 | 0.644 | 3.16 |
| 8x | Classical (ours: gated JBU) | 22.35 | 0.762 | 2.56 |
| 8x | Deep learning only | 14.00 | 0.896 | 2.16 |
| 8x | Hybrid v1 (softmax arbiter) | 15.84 | 0.816 | 2.41 |
| 8x | Hybrid (ours) | 14.57 | 0.836 | 2.30 |
| 8x | EdgeGate-Auto (ours) | 14.05 | 0.895 | 2.17 |
| 8x | Deep-only-L (GPU-trained) | 9.46 | 0.946 | 1.66 |

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
| 2x | Hybrid v1 (softmax arbiter) | 2.79 | 1.54 |
| 2x | Hybrid (ours) | 2.54 | 1.41 |
| 2x | EdgeGate-Auto (ours) | 2.54 | 1.41 |
| 2x | Deep-only-L (GPU-trained) | 1.88 | 1.13 |
| 4x | Nearest | 9.67 | 4.93 |
| 4x | Bilinear | 6.77 | 3.53 |
| 4x | Bicubic | 7.53 | 4.05 |
| 4x | Joint bilateral (JBU) | 6.41 | 3.32 |
| 4x | Guided filter | 5.22 | 2.64 |
| 4x | Classical (ours: gated JBU) | 5.33 | 2.82 |
| 4x | Deep learning only | 3.71 | 2.10 |
| 4x | Hybrid v1 (softmax arbiter) | 4.07 | 2.28 |
| 4x | Hybrid (ours) | 3.64 | 1.98 |
| 4x | EdgeGate-Auto (ours) | 3.64 | 1.98 |
| 4x | Deep-only-L (GPU-trained) | 2.77 | 1.57 |
| 8x | Nearest | 11.91 | 5.99 |
| 8x | Bilinear | 8.85 | 4.57 |
| 8x | Bicubic | 9.20 | 4.92 |
| 8x | Joint bilateral (JBU) | 8.30 | 4.20 |
| 8x | Guided filter | 7.82 | 3.97 |
| 8x | Classical (ours: gated JBU) | 7.57 | 3.85 |
| 8x | Deep learning only | 5.70 | 3.10 |
| 8x | Hybrid v1 (softmax arbiter) | 6.41 | 3.47 |
| 8x | Hybrid (ours) | 5.73 | 3.06 |
| 8x | EdgeGate-Auto (ours) | 5.73 | 3.06 |
| 8x | Deep-only-L (GPU-trained) | 4.35 | 2.38 |

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
| 2x | Hybrid v1 (softmax arbiter) | 44.75 | 0.986 |
| 2x | Hybrid (ours) | 45.61 | 0.989 |
| 2x | EdgeGate-Auto (ours) | 45.62 | 0.989 |
| 2x | Deep-only-L (GPU-trained) | 48.24 | 0.993 |
| 4x | Nearest | 33.66 | 0.875 |
| 4x | Bilinear | 36.82 | 0.938 |
| 4x | Bicubic | 35.80 | 0.918 |
| 4x | Joint bilateral (JBU) | 37.26 | 0.944 |
| 4x | Guided filter | 39.39 | 0.972 |
| 4x | Classical (ours: gated JBU) | 39.25 | 0.963 |
| 4x | Deep learning only | 42.34 | 0.980 |
| 4x | Hybrid v1 (softmax arbiter) | 41.50 | 0.977 |
| 4x | Hybrid (ours) | 42.54 | 0.983 |
| 4x | EdgeGate-Auto (ours) | 42.53 | 0.983 |
| 4x | Deep-only-L (GPU-trained) | 44.93 | 0.989 |
| 8x | Nearest | 32.05 | 0.879 |
| 8x | Bilinear | 34.73 | 0.942 |
| 8x | Bicubic | 34.25 | 0.933 |
| 8x | Joint bilateral (JBU) | 35.25 | 0.950 |
| 8x | Guided filter | 36.11 | 0.962 |
| 8x | Classical (ours: gated JBU) | 36.37 | 0.956 |
| 8x | Deep learning only | 38.58 | 0.969 |
| 8x | Hybrid v1 (softmax arbiter) | 37.63 | 0.964 |
| 8x | Hybrid (ours) | 38.62 | 0.972 |
| 8x | EdgeGate-Auto (ours) | 38.62 | 0.972 |
| 8x | Deep-only-L (GPU-trained) | 41.04 | 0.981 |

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
| 2x | Hybrid v1 (softmax arbiter) | 6.56 | 0.936 | 2.22 |
| 2x | Hybrid (ours) | 6.43 | 0.956 | 1.91 |
| 2x | EdgeGate-Auto (ours) | 6.44 | 0.956 | 1.90 |
| 2x | Deep-only-L (GPU-trained) | 4.31 | 0.980 | 1.54 |
| 4x | Nearest | 19.27 | 0.655 | 8.00 |
| 4x | Bilinear | 14.65 | 0.529 | 5.30 |
| 4x | Bicubic | 13.91 | 0.554 | 6.42 |
| 4x | Joint bilateral (JBU) | 13.33 | 0.675 | 5.38 |
| 4x | Guided filter | 14.02 | 0.775 | 3.42 |
| 4x | Classical (ours: gated JBU) | 13.55 | 0.711 | 3.64 |
| 4x | Deep learning only | 9.13 | 0.911 | 2.76 |
| 4x | Hybrid v1 (softmax arbiter) | 9.75 | 0.880 | 3.13 |
| 4x | Hybrid (ours) | 9.44 | 0.902 | 2.62 |
| 4x | EdgeGate-Auto (ours) | 9.45 | 0.902 | 2.62 |
| 4x | Deep-only-L (GPU-trained) | 6.89 | 0.953 | 2.11 |
| 8x | Nearest | 28.29 | 0.557 | 8.40 |
| 8x | Bilinear | 22.59 | 0.308 | 5.70 |
| 8x | Bicubic | 21.24 | 0.365 | 6.72 |
| 8x | Joint bilateral (JBU) | 20.34 | 0.666 | 5.86 |
| 8x | Guided filter | 21.47 | 0.618 | 4.33 |
| 8x | Classical (ours: gated JBU) | 21.03 | 0.556 | 4.39 |
| 8x | Deep learning only | 14.68 | 0.804 | 3.88 |
| 8x | Hybrid v1 (softmax arbiter) | 16.22 | 0.736 | 4.50 |
| 8x | Hybrid (ours) | 14.93 | 0.793 | 3.99 |
| 8x | EdgeGate-Auto (ours) | 14.93 | 0.793 | 3.99 |
| 8x | Deep-only-L (GPU-trained) | 11.48 | 0.891 | 3.03 |

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
| 2x | Hybrid v1 (softmax arbiter) | 3.45 | 1.73 |
| 2x | Hybrid (ours) | 3.18 | 1.61 |
| 2x | EdgeGate-Auto (ours) | 3.18 | 1.61 |
| 2x | Deep-only-L (GPU-trained) | 2.48 | 1.30 |
| 4x | Nearest | 10.83 | 5.30 |
| 4x | Bilinear | 8.04 | 3.97 |
| 4x | Bicubic | 8.90 | 4.50 |
| 4x | Joint bilateral (JBU) | 7.75 | 3.78 |
| 4x | Guided filter | 6.26 | 3.04 |
| 4x | Classical (ours: gated JBU) | 6.37 | 3.17 |
| 4x | Deep learning only | 5.48 | 2.53 |
| 4x | Hybrid v1 (softmax arbiter) | 5.20 | 2.62 |
| 4x | Hybrid (ours) | 4.72 | 2.34 |
| 4x | EdgeGate-Auto (ours) | 4.72 | 2.34 |
| 4x | Deep-only-L (GPU-trained) | 3.82 | 1.88 |
| 8x | Nearest | 13.79 | 6.69 |
| 8x | Bilinear | 10.67 | 5.26 |
| 8x | Bicubic | 11.31 | 5.69 |
| 8x | Joint bilateral (JBU) | 10.23 | 4.94 |
| 8x | Guided filter | 9.16 | 4.55 |
| 8x | Classical (ours: gated JBU) | 9.13 | 4.45 |
| 8x | Deep learning only | 8.21 | 3.87 |
| 8x | Hybrid v1 (softmax arbiter) | 8.20 | 4.09 |
| 8x | Hybrid (ours) | 7.58 | 3.75 |
| 8x | EdgeGate-Auto (ours) | 7.58 | 3.75 |
| 8x | Deep-only-L (GPU-trained) | 6.11 | 2.96 |

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
| 2x | Hybrid v1 (softmax arbiter) | 43.10 | 0.984 |
| 2x | Hybrid (ours) | 43.84 | 0.987 |
| 2x | EdgeGate-Auto (ours) | 43.84 | 0.987 |
| 2x | Deep-only-L (GPU-trained) | 45.95 | 0.991 |
| 4x | Nearest | 32.78 | 0.871 |
| 4x | Bilinear | 35.44 | 0.932 |
| 4x | Bicubic | 34.47 | 0.912 |
| 4x | Joint bilateral (JBU) | 35.74 | 0.937 |
| 4x | Guided filter | 37.83 | 0.968 |
| 4x | Classical (ours: gated JBU) | 37.76 | 0.958 |
| 4x | Deep learning only | 39.07 | 0.974 |
| 4x | Hybrid v1 (softmax arbiter) | 39.51 | 0.974 |
| 4x | Hybrid (ours) | 40.39 | 0.979 |
| 4x | EdgeGate-Auto (ours) | 40.39 | 0.979 |
| 4x | Deep-only-L (GPU-trained) | 42.23 | 0.986 |
| 8x | Nearest | 30.90 | 0.873 |
| 8x | Bilinear | 33.22 | 0.935 |
| 8x | Bicubic | 32.62 | 0.925 |
| 8x | Joint bilateral (JBU) | 33.57 | 0.943 |
| 8x | Guided filter | 34.77 | 0.958 |
| 8x | Classical (ours: gated JBU) | 34.80 | 0.952 |
| 8x | Deep learning only | 35.68 | 0.962 |
| 8x | Hybrid v1 (softmax arbiter) | 35.65 | 0.959 |
| 8x | Hybrid (ours) | 36.37 | 0.965 |
| 8x | EdgeGate-Auto (ours) | 36.37 | 0.965 |
| 8x | Deep-only-L (GPU-trained) | 38.25 | 0.976 |

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
| 2x | Hybrid v1 (softmax arbiter) | 9.10 | 0.932 | 2.36 |
| 2x | Hybrid (ours) | 8.55 | 0.944 | 2.10 |
| 2x | EdgeGate-Auto (ours) | 8.55 | 0.944 | 2.10 |
| 2x | Deep-only-L (GPU-trained) | 6.42 | 0.969 | 1.76 |
| 4x | Nearest | 24.03 | 0.633 | 8.23 |
| 4x | Bilinear | 18.95 | 0.531 | 5.77 |
| 4x | Bicubic | 19.29 | 0.560 | 6.83 |
| 4x | Joint bilateral (JBU) | 17.99 | 0.656 | 5.83 |
| 4x | Guided filter | 16.87 | 0.741 | 3.95 |
| 4x | Classical (ours: gated JBU) | 17.07 | 0.686 | 3.88 |
| 4x | Deep learning only | 14.98 | 0.861 | 3.35 |
| 4x | Hybrid v1 (softmax arbiter) | 13.77 | 0.844 | 3.36 |
| 4x | Hybrid (ours) | 12.79 | 0.863 | 2.98 |
| 4x | EdgeGate-Auto (ours) | 12.79 | 0.863 | 2.98 |
| 4x | Deep-only-L (GPU-trained) | 10.37 | 0.928 | 2.49 |
| 8x | Nearest | 33.06 | 0.506 | 9.22 |
| 8x | Bilinear | 26.94 | 0.311 | 6.64 |
| 8x | Bicubic | 26.82 | 0.363 | 7.65 |
| 8x | Joint bilateral (JBU) | 25.14 | 0.620 | 6.83 |
| 8x | Guided filter | 24.36 | 0.584 | 5.25 |
| 8x | Classical (ours: gated JBU) | 24.70 | 0.527 | 5.11 |
| 8x | Deep learning only | 21.57 | 0.725 | 4.93 |
| 8x | Hybrid v1 (softmax arbiter) | 20.98 | 0.690 | 5.17 |
| 8x | Hybrid (ours) | 19.61 | 0.709 | 4.77 |
| 8x | EdgeGate-Auto (ours) | 19.61 | 0.709 | 4.77 |
| 8x | Deep-only-L (GPU-trained) | 16.14 | 0.838 | 3.90 |

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
| Hybrid v1 (softmax arbiter) | 1.95 | 3.04 | 5.60 |
| Hybrid (ours) | 1.75 | 2.92 | 5.17 |
| EdgeGate-Auto (ours) | 1.76 | 2.77 | 5.00 |
| Deep-only-L (GPU-trained) | 0.95 | 1.69 | 3.43 |

## 5. Processing latency and memory (480x640 input, 4 CPU threads, fresh process per method)

Parameters: hybrid 218,978, deep-only 218,241. Weight files: hybrid 865 KB, dl_only 864 KB.

| Method | Scale | Median ms | p95 ms | Peak RSS MB | RSS growth MB |
|---|---|---|---|---|---|
| Nearest | 2x | 1 | 1 | 527 | 0 |
| Nearest | 4x | 1 | 1 | 526 | 0 |
| Nearest | 8x | 1 | 1 | 526 | 0 |
| Bilinear | 2x | 1 | 1 | 528 | 0 |
| Bilinear | 4x | 1 | 1 | 527 | 0 |
| Bilinear | 8x | 1 | 1 | 527 | 0 |
| Bicubic | 2x | 1 | 1 | 528 | 0 |
| Bicubic | 4x | 1 | 1 | 527 | 0 |
| Bicubic | 8x | 1 | 1 | 528 | 0 |
| Joint bilateral (JBU) | 2x | 228 | 239 | 631 | 0 |
| Joint bilateral (JBU) | 4x | 239 | 246 | 631 | 0 |
| Joint bilateral (JBU) | 8x | 227 | 249 | 630 | 0 |
| Guided filter | 2x | 94 | 94 | 592 | 0 |
| Guided filter | 4x | 93 | 94 | 592 | 0 |
| Guided filter | 8x | 93 | 94 | 593 | 0 |
| Classical (ours: gated JBU) | 2x | 313 | 340 | 657 | 7 |
| Classical (ours: gated JBU) | 4x | 323 | 334 | 651 | 3 |
| Classical (ours: gated JBU) | 8x | 320 | 336 | 649 | 1 |
| Deep learning only | 2x | 100 | 123 | 762 | 12 |
| Deep learning only | 4x | 99 | 102 | 761 | 11 |
| Deep learning only | 8x | 101 | 111 | 761 | 11 |
| Hybrid (ours) | 2x | 453 | 477 | 881 | 102 |
| Hybrid (ours) | 4x | 430 | 456 | 871 | 97 |
| Hybrid (ours) | 8x | 433 | 467 | 869 | 96 |
| EdgeGate-Auto (ours) | 2x | 104 | 108 | 763 | 11 |
| EdgeGate-Auto (ours) | 4x | 105 | 108 | 761 | 12 |
| EdgeGate-Auto (ours) | 8x | 103 | 110 | 762 | 11 |
| Deep-only-L (GPU-trained) | 2x | 2337 | 2570 | 1491 | 36 |
| Deep-only-L (GPU-trained) | 4x | 2585 | 3154 | 1490 | 35 |
| Deep-only-L (GPU-trained) | 8x | 2258 | 2407 | 1489 | 35 |

## 6. Statistical check: hybrid minus bicubic (paired bootstrap, 95% CI; negative = hybrid better; a CI that contains 0 means no reliable difference)

| Protocol | Scale | Metric | Mean diff | 95% CI | n |
|---|---|---|---|---|---|
| clean | 2x | rmse_cm | -0.808 | [-0.868, -0.753] | 449 |
| clean | 2x | boundary_rmse_cm | -2.555 | [-2.695, -2.426] | 449 |
| clean | 4x | rmse_cm | -1.689 | [-1.800, -1.593] | 449 |
| clean | 4x | boundary_rmse_cm | -5.286 | [-5.531, -5.057] | 449 |
| clean | 8x | rmse_cm | -2.519 | [-2.691, -2.368] | 449 |
| clean | 8x | boundary_rmse_cm | -7.795 | [-8.189, -7.444] | 449 |
| noisy | 2x | rmse_cm | -4.163 | [-4.225, -4.102] | 150 |
| noisy | 2x | boundary_rmse_cm | -3.090 | [-3.260, -2.935] | 150 |
| noisy | 4x | rmse_cm | -3.883 | [-3.978, -3.802] | 150 |
| noisy | 4x | boundary_rmse_cm | -4.470 | [-4.795, -4.177] | 150 |
| noisy | 8x | rmse_cm | -3.471 | [-3.647, -3.323] | 150 |
| noisy | 8x | boundary_rmse_cm | -6.307 | [-6.826, -5.841] | 150 |
| noisy+holes | 2x | rmse_cm | -4.344 | [-4.419, -4.272] | 150 |
| noisy+holes | 2x | boundary_rmse_cm | -4.655 | [-4.891, -4.411] | 150 |
| noisy+holes | 4x | rmse_cm | -4.181 | [-4.317, -4.062] | 150 |
| noisy+holes | 4x | boundary_rmse_cm | -6.500 | [-6.927, -6.082] | 150 |
| noisy+holes | 8x | rmse_cm | -3.733 | [-3.923, -3.552] | 150 |
| noisy+holes | 8x | boundary_rmse_cm | -7.218 | [-7.770, -6.662] | 150 |

## 6. Statistical check: hybrid minus dl_only (paired bootstrap, 95% CI; negative = hybrid better; a CI that contains 0 means no reliable difference)

| Protocol | Scale | Metric | Mean diff | 95% CI | n |
|---|---|---|---|---|---|
| clean | 2x | rmse_cm | -0.012 | [-0.021, -0.003] | 449 |
| clean | 2x | boundary_rmse_cm | -0.019 | [-0.046, 0.009] | 449 |
| clean | 4x | rmse_cm | 0.155 | [0.138, 0.171] | 449 |
| clean | 4x | boundary_rmse_cm | 0.438 | [0.396, 0.480] | 449 |
| clean | 8x | rmse_cm | 0.193 | [0.159, 0.231] | 449 |
| clean | 8x | boundary_rmse_cm | 0.570 | [0.467, 0.675] | 449 |
| noisy | 2x | rmse_cm | 0.031 | [0.016, 0.046] | 150 |
| noisy | 2x | boundary_rmse_cm | 0.391 | [0.339, 0.441] | 150 |
| noisy | 4x | rmse_cm | -0.067 | [-0.087, -0.047] | 150 |
| noisy | 4x | boundary_rmse_cm | 0.316 | [0.251, 0.381] | 150 |
| noisy | 8x | rmse_cm | 0.033 | [-0.018, 0.082] | 150 |
| noisy | 8x | boundary_rmse_cm | 0.251 | [0.120, 0.389] | 150 |
| noisy+holes | 2x | rmse_cm | -0.436 | [-0.479, -0.395] | 150 |
| noisy+holes | 2x | boundary_rmse_cm | -1.218 | [-1.341, -1.088] | 150 |
| noisy+holes | 4x | rmse_cm | -0.757 | [-0.833, -0.686] | 150 |
| noisy+holes | 4x | boundary_rmse_cm | -2.193 | [-2.421, -1.972] | 150 |
| noisy+holes | 8x | rmse_cm | -0.629 | [-0.739, -0.520] | 150 |
| noisy+holes | 8x | boundary_rmse_cm | -1.966 | [-2.294, -1.625] | 150 |
