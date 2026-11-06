"""NYU Depth V2 loading, fixed splits and a memory-mapped cache.

The 1449 labelled NYU frames are stored as normalised float32 depth (3 identical channels)
plus an RGB jpg. We assume depth_metres = raw * 10 (NYU's sensor range is 10 m); every
reported error is therefore in *assumed* metres/centimetres. Document this wherever numbers
are quoted.

Split (contiguous blocks, because the labelled set is ordered by capture scene, so contiguous
blocks keep scenes from leaking between train and test):
    train 0-799, val 800-999, test 1000-1448  (449 test frames, as in the common DKN protocol)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("NYU_DIR", ROOT / "data/nyu/nyu_depth_train_float32_1449"))
CACHE_DIR = Path(os.environ.get("NYU_CACHE", ROOT / "data/cache"))
DEPTH_SCALE_M = 10.0
H, W = 480, 640
N_FRAMES = 1449
SPLITS = {"train": range(0, 800), "val": range(800, 1000), "test": range(1000, 1449)}


def build_cache() -> None:
    """Convert raw h5/jpg files into two .npy files that can be memory-mapped."""
    import h5py
    from PIL import Image

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rgb = np.lib.format.open_memmap(CACHE_DIR / "rgb.npy", "w+", np.uint8, (N_FRAMES, H, W, 3))
    dep = np.lib.format.open_memmap(CACHE_DIR / "depth.npy", "w+", np.float32, (N_FRAMES, H, W))
    for i in range(N_FRAMES):
        with h5py.File(RAW_DIR / "images" / f"{i}.h5", "r") as f:
            dep[i] = f["depth"][..., 0] * DEPTH_SCALE_M
        rgb[i] = np.asarray(Image.open(RAW_DIR / "conditions" / f"{i}_condition.jpg").convert("RGB"))
    rgb.flush()
    dep.flush()


_cache: dict = {}


def _arrays():
    if not _cache:
        if not (CACHE_DIR / "rgb.npy").exists():
            raise FileNotFoundError(
                f"NYU cache not found in {CACHE_DIR}. Run `python scripts/prepare_data.py` first.")
        _cache["rgb"] = np.load(CACHE_DIR / "rgb.npy", mmap_mode="r")
        _cache["depth"] = np.load(CACHE_DIR / "depth.npy", mmap_mode="r")
    return _cache["rgb"], _cache["depth"]


def get_pair(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 HxWx3, depth float32 HxW in assumed metres)."""
    rgb, dep = _arrays()
    return np.array(rgb[idx]), np.array(dep[idx])


def split_indices(name: str) -> list[int]:
    return list(SPLITS[name])
