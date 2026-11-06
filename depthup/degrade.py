"""Synthetic sensor degradation: low resolution, Kinect-like noise/outliers, missing values."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Degradation:
    scale: int = 4
    mode: str = "bicubic"      # 'bicubic' (anti-aliased) or 'nearest' (raw sub-sampling)
    noise: float = 0.0         # relative Gaussian noise sigma (fraction of depth)
    outliers: float = 0.0      # fraction of LR pixels replaced by spikes (flying pixels)
    holes: float = 0.0         # fraction of LR pixels that are missing


PROTOCOLS = {
    "clean": dict(noise=0.0, outliers=0.0, holes=0.0),
    "noisy": dict(noise=0.02, outliers=0.005, holes=0.0),
    "noisy+holes": dict(noise=0.02, outliers=0.005, holes=0.20),
}


def _pil_resize(a: np.ndarray, size_wh: tuple[int, int], method=Image.BICUBIC) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(size_wh, method))


def downsample(depth: np.ndarray, scale: int, mode: str = "bicubic") -> np.ndarray:
    h, w = depth.shape
    if h % scale or w % scale:
        raise ValueError(f"depth size {w}x{h} is not divisible by scale {scale}")
    if mode == "nearest":
        return depth[scale // 2::scale, scale // 2::scale].copy()
    return _pil_resize(depth, (w // scale, h // scale))


def blob_mask(shape: tuple[int, int], frac: float, rng: np.random.Generator) -> np.ndarray:
    """Random missing-value mask: half salt dropout, half spatially coherent blobs."""
    if frac <= 0:
        return np.zeros(shape, bool)
    drop = rng.random(shape) < frac * 0.5
    field = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 1.5)
    thr = np.quantile(field, frac * 0.5)
    return drop | (field < thr)


def degrade(depth_hr: np.ndarray, d: Degradation, rng: np.random.Generator | None = None):
    """Return (lr depth, valid mask) at low resolution. Invalid pixels hold 0."""
    rng = rng or np.random.default_rng(0)
    lr = downsample(depth_hr, d.scale, d.mode).astype(np.float32)
    if d.noise > 0:
        lr = lr * (1 + d.noise * rng.standard_normal(lr.shape).astype(np.float32))
    if d.outliers > 0:
        spike = rng.random(lr.shape) < d.outliers
        lr = np.where(spike, lr + rng.choice([-1, 1], lr.shape) * rng.uniform(0.4, 1.2, lr.shape), lr)
    lr = np.clip(lr, 0.05, None).astype(np.float32)
    valid = ~blob_mask(lr.shape, d.holes, rng)
    return np.where(valid, lr, 0).astype(np.float32), valid
