"""Robust loading/validation of user supplied RGB and depth files (bad input -> clear errors)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_SIDE = 960          # keep latency under ~10 s on a small CPU instance
SCALES = (2, 4, 8)


class InputError(ValueError):
    """Raised with a message that is safe to show directly to the user."""


def read_rgb(raw: bytes) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InputError("Could not read the RGB file. Upload a JPG or PNG image.") from e
    im = im.convert("RGB")
    if min(im.size) < 32:
        raise InputError(f"RGB image is too small ({im.size[0]}x{im.size[1]}); need at least 32 px per side.")
    return np.asarray(im, np.uint8)


def read_depth(raw: bytes, filename: str = "", unit: str = "auto") -> tuple[np.ndarray, str]:
    """Return (depth float32 in metres, human readable note).

    unit: 'auto' | 'metres' | 'millimetres' | 'relative'. 8-bit images are 'relative' depth and are
    rescaled to 0.5-5 m so metric thresholds stay meaningful (the note says so).
    """
    name = filename.lower()
    note = ""
    try:
        if name.endswith(".npy"):
            a = np.load(io.BytesIO(raw), allow_pickle=False)
        elif name.endswith(".npz"):
            z = np.load(io.BytesIO(raw), allow_pickle=False)
            a = z[z.files[0]]
        else:
            im = Image.open(io.BytesIO(raw))
            im.load()
            if im.mode in ("RGB", "RGBA"):
                im = im.convert("L")
            a = np.asarray(im)
    except (UnidentifiedImageError, OSError, ValueError, KeyError) as e:
        raise InputError("Could not read the depth file. Use .npy, .npz, or a 8/16-bit PNG/TIFF.") from e
    if a.ndim == 3:
        a = a[..., 0]
    if a.ndim != 2:
        raise InputError(f"Depth must be a 2-D array, got shape {a.shape}.")
    integer = np.issubdtype(a.dtype, np.integer)
    a = a.astype(np.float32)
    if not np.isfinite(a).any():
        raise InputError("Depth map contains no finite values.")
    if unit == "auto":
        if integer and a.max() > 255:
            unit = "millimetres"
        elif integer:
            unit = "relative"
        else:
            unit = "metres"
    if unit == "millimetres":
        a, note = a / 1000.0, "interpreted as millimetres -> metres"
    elif unit == "relative":
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = 0.5 + 4.5 * (a - lo) / max(hi - lo, 1e-6)
        note = "relative depth rescaled to 0.5-5 m (metric errors are therefore indicative only)"
    else:
        note = "interpreted as metres"
    return a, note


def to_valid(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mark non-finite or non-positive samples as missing."""
    valid = np.isfinite(depth) & (depth > 0.01)
    return np.where(valid, depth, 0).astype(np.float32), valid


def fit_to_scale(rgb: np.ndarray, scale: int, depth: np.ndarray | None = None):
    """Shrink (if huge) and crop so both H and W are divisible by `scale`."""
    h, w = rgb.shape[:2]
    f = min(1.0, MAX_SIDE / max(h, w))
    if f < 1.0:
        size = (max(scale, int(w * f)), max(scale, int(h * f)))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.BICUBIC))
        if depth is not None:
            depth = np.asarray(Image.fromarray(depth, mode="F").resize(size, Image.NEAREST))
    h, w = (rgb.shape[0] // scale) * scale, (rgb.shape[1] // scale) * scale
    rgb = rgb[:h, :w]
    return (rgb, depth[:h, :w]) if depth is not None else rgb


def match_depth_to_rgb(rgb: np.ndarray, lr_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """For 'upsample my depth': pick an allowed integer scale from the size ratio, crop/resize RGB to match."""
    h, w = lr_depth.shape
    H, W = rgb.shape[:2]
    ratio = W / w
    scale = min(SCALES, key=lambda s: abs(s - ratio))
    if abs(ratio - scale) / scale > 0.25 or abs((H / h) - ratio) / ratio > 0.25:
        raise InputError(
            f"RGB ({W}x{H}) and depth ({w}x{h}) don't have a compatible size ratio. "
            f"The depth map must be roughly 1/2, 1/4 or 1/8 of the RGB size with the same aspect ratio.")
    rgb = np.asarray(Image.fromarray(rgb).resize((w * scale, h * scale), Image.BICUBIC))
    return rgb, lr_depth, scale
