"""Depth metrics: accuracy, structural, boundary-aware and texture-copy specific."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

EDGE_GRAD_THR = 0.12      # metres per pixel; GT gradient above this counts as a depth discontinuity
BAND_RADIUS = 3           # pixels around a GT discontinuity forming the "boundary band"


def gt_edges(gt: np.ndarray) -> np.ndarray:
    """Binary ground-truth depth-discontinuity map."""
    gx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy) > EDGE_GRAD_THR


def boundary_band(edges: np.ndarray, r: int = BAND_RADIUS) -> np.ndarray:
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(edges.astype(np.uint8), k).astype(bool)


def texture_flat_mask(gt: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Pixels where depth is smooth but RGB is textured: the places texture-copying shows up."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    rgb_mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    textured = rgb_mag > np.percentile(rgb_mag, 70)
    far_from_edges = ~boundary_band(gt_edges(gt), 6)
    return textured & far_from_edges


def edge_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> tuple[float, float, float]:
    """Boundary precision/recall/F1 with a pixel tolerance."""
    pe, ge = gt_edges(pred), gt_edges(gt)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    ge_d, pe_d = cv2.dilate(ge.astype(np.uint8), k).astype(bool), cv2.dilate(pe.astype(np.uint8), k).astype(bool)
    prec = (pe & ge_d).sum() / max(pe.sum(), 1)
    rec = (ge & pe_d).sum() / max(ge.sum(), 1)
    return float(prec), float(rec), float(2 * prec * rec / max(prec + rec, 1e-9))


def evaluate(pred: np.ndarray, gt: np.ndarray, rgb: np.ndarray) -> dict:
    """All per-image metrics. Depth in metres; RMSE/MAE reported in centimetres."""
    err = pred.astype(np.float64) - gt.astype(np.float64)
    peak = float(gt.max())
    mse = float((err ** 2).mean())
    edges = gt_edges(gt)
    band = boundary_band(edges)
    flat = texture_flat_mask(gt, rgb)
    p, r, f1 = edge_f1(pred, gt)
    out = {
        "rmse_cm": 100 * np.sqrt(mse),
        "mae_cm": 100 * float(np.abs(err).mean()),
        "psnr_db": float(10 * np.log10(peak ** 2 / max(mse, 1e-12))),
        "ssim": float(structural_similarity(gt, pred.astype(np.float32), data_range=peak)),
        "boundary_rmse_cm": 100 * float(np.sqrt((err[band] ** 2).mean())) if band.any() else np.nan,
        "boundary_mae_cm": 100 * float(np.abs(err[band]).mean()) if band.any() else np.nan,
        "edge_precision": p, "edge_recall": r, "edge_f1": f1,
        "texture_copy_rmse_cm": 100 * float(np.sqrt((err[flat] ** 2).mean())) if flat.any() else np.nan,
        "delta1": float((np.maximum(pred / np.maximum(gt, 1e-3), gt / np.maximum(pred, 1e-3)) < 1.25).mean()),
    }
    return {k: float(v) for k, v in out.items()}
