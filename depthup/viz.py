"""Visual helpers shared by the app and the figure scripts."""
from __future__ import annotations

import cv2
import numpy as np


def colorize(d: np.ndarray, lo: float | None = None, hi: float | None = None, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    lo = float(np.nanmin(d)) if lo is None else lo
    hi = float(np.nanmax(d)) if hi is None else hi
    u8 = np.clip((d - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cmap), cv2.COLOR_BGR2RGB)


def error_map(pred: np.ndarray, gt: np.ndarray, vmax: float = 0.3) -> np.ndarray:
    return colorize(np.abs(pred - gt), 0, vmax, cv2.COLORMAP_MAGMA)


def gray(a: np.ndarray) -> np.ndarray:
    return colorize(a, 0, 1, cv2.COLORMAP_VIRIDIS)


def overlay_edges(rgb: np.ndarray, edges: np.ndarray, color=(255, 40, 40)) -> np.ndarray:
    out = rgb.copy()
    out[edges] = color
    return out


def crop(a: np.ndarray, y: int, x: int, size: int = 80) -> np.ndarray:
    y, x = max(0, min(y, a.shape[0] - size)), max(0, min(x, a.shape[1] - size))
    return a[y:y + size, x:x + size]


def zoom(a: np.ndarray, f: int = 4) -> np.ndarray:
    return cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
