"""Feature construction and the four reconstruction systems used throughout the project.

    interpolation   : nearest / bilinear / bicubic
    classical       : JBU, guided filter, and our gated pipeline ("gate")
    deep            : DLOnlyNet (RGB + bicubic depth -> residual)
    hybrid          : classical features + HybridNet arbiter
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import classical as C
from .model import DLOnlyNet, HybridNet

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
_models: dict = {}


def hybrid_features(lr, valid, rgb, scale):
    """Classical pass -> stacked float32 channels (10,H,W) plus raw experts and diagnostics."""
    o = C.gate_pipeline(lr, valid, rgb, scale)
    ex = o.extras
    m = float(o.base.mean())
    conf_hr = C.bilinear_up(ex["conf"], scale)
    rgbf = rgb.astype(np.float32) / 255.0
    chans = [rgbf[..., 0], rgbf[..., 1], rgbf[..., 2], o.base - m, o.depth - m, ex["snapped"] - m,
             ex["gate"], ex["rgb_edge"], conf_hr, np.full_like(o.base, scale / 8.0)]
    x = np.stack(chans).astype(np.float32)
    return x, o.base.astype(np.float32), o.depth.astype(np.float32), m, o


def dl_features(lr, valid, rgb, scale):
    d_bic = C.bicubic_up(C.naive_fill(lr, valid), scale)
    m = float(d_bic.mean())
    rgbf = rgb.astype(np.float32) / 255.0
    x = np.stack([rgbf[..., 0], rgbf[..., 1], rgbf[..., 2], d_bic - m, np.full_like(d_bic, scale / 8.0)])
    return x.astype(np.float32), d_bic.astype(np.float32), m


def load_model(kind: str):
    if kind not in _models:
        net = HybridNet() if kind == "hybrid" else DLOnlyNet()
        path = WEIGHTS / f"{kind}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing trained weights: {path}")
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        _models[kind] = net
    return _models[kind]


@torch.no_grad()
def run_hybrid(lr, valid, rgb, scale, return_all=False):
    x, d_bic, d_gate, m, o = hybrid_features(lr, valid, rgb, scale)
    xt = torch.from_numpy(x)[None]
    out, d_dl, w = load_model("hybrid")(xt, torch.from_numpy(d_bic - m)[None, None], torch.from_numpy(d_gate - m)[None, None])
    depth = out[0, 0].numpy() + m
    if not return_all:
        return depth
    return depth, dict(classical=o, dl_branch=d_dl[0, 0].numpy() + m, weights=w[0].numpy())


@torch.no_grad()
def run_dl_only(lr, valid, rgb, scale):
    x, d_bic, m = dl_features(lr, valid, rgb, scale)
    out = load_model("dl_only")(torch.from_numpy(x)[None], torch.from_numpy(d_bic - m)[None, None])
    return out[0, 0].numpy() + m


def _fill(lr, valid):
    return C.naive_fill(lr, valid)


METHODS = {
    "nearest": lambda lr, v, rgb, s: C.nearest_up(_fill(lr, v), s),
    "bilinear": lambda lr, v, rgb, s: C.bilinear_up(_fill(lr, v), s),
    "bicubic": lambda lr, v, rgb, s: C.bicubic_up(_fill(lr, v), s),
    "jbu": lambda lr, v, rgb, s: C.jbu(_fill(lr, v), np.ones_like(lr), rgb.astype(np.float32) / 255, s,
                                       sigma_s=0.45 * s, sigma_c=0.2),
    "guided": lambda lr, v, rgb, s: C.guided_up(_fill(lr, v), rgb.astype(np.float32) / 255, s),
    "classical": lambda lr, v, rgb, s: C.gate_pipeline(lr, v, rgb, s).depth,
    "dl_only": run_dl_only,
    "hybrid": run_hybrid,
}
LABELS = {"nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic", "jbu": "Joint bilateral (JBU)",
          "guided": "Guided filter", "classical": "Classical (ours: gated JBU)", "dl_only": "Deep learning only",
          "hybrid": "Hybrid (ours)"}


def timed(name, lr, valid, rgb, scale):
    t = time.perf_counter()
    out = METHODS[name](lr, valid, rgb, scale)
    return out, 1000 * (time.perf_counter() - t)
