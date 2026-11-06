"""Feature construction and the four reconstruction systems used throughout the project.

    interpolation   : nearest / bilinear / bicubic
    classical       : JBU, guided filter, and our gated pipeline ("gate")
    deep            : DLOnlyNet (RGB + bicubic depth -> residual)
    hybrid          : classical features + HybridNet arbiter
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from . import classical as C
from .model import DLOnlyNet, HybridNet, HybridNetConvex

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
_models: dict = {}
DEVICE = torch.device(os.environ.get("DEPTHUP_DEVICE", "cpu"))   # set DEPTHUP_DEVICE=cuda for GPU inference


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
    """kind = 'hybrid' | 'dl_only' | 'hybrid_convex' (+ optional size tag, e.g. 'hybrid_L'). Reads cfg from the checkpoint."""
    if kind not in _models:
        base = kind.split("_")[0] if not kind.startswith(("dl_only", "hybrid_convex")) else ("dl_only" if kind.startswith("dl_only") else "hybrid_convex")
        path = WEIGHTS / f"{kind}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing trained weights: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ck.get("cfg", {}) if "state" in ck else {}
        state = ck["state"] if "state" in ck else ck
        net = {"hybrid": HybridNet, "hybrid_convex": HybridNetConvex, "dl_only": DLOnlyNet}[base](**cfg)
        net.load_state_dict(state)
        net.eval().to(DEVICE)
        _models[kind] = net
    return _models[kind]


@torch.no_grad()
def run_hybrid(lr, valid, rgb, scale, return_all=False, kind="hybrid"):
    x, d_bic, d_gate, m, o = hybrid_features(lr, valid, rgb, scale)
    t = lambda a: torch.from_numpy(a)[None].to(DEVICE)  # noqa: E731
    out, d_dl, w = load_model(kind)(t(x), t(d_bic - m)[None], t(d_gate - m)[None])
    depth = out[0, 0].cpu().numpy() + m
    if not return_all:
        return depth
    return depth, dict(classical=o, dl_branch=d_dl[0, 0].cpu().numpy() + m, weights=w[0].cpu().numpy())


@torch.no_grad()
def run_dl_only(lr, valid, rgb, scale, kind="dl_only"):
    x, d_bic, m = dl_features(lr, valid, rgb, scale)
    out = load_model(kind)(torch.from_numpy(x)[None].to(DEVICE), torch.from_numpy(d_bic - m)[None, None].to(DEVICE))
    return out[0, 0].cpu().numpy() + m


AUTO_NOISE_THR = 0.012   # metres; fixed a priori (clean LR noise estimate is ~0.002-0.005, 2% noise gives >0.02)


def auto_choice(lr, valid):
    """'dl_only' for pristine input (no holes, low estimated noise), else 'hybrid'."""
    if valid.all() and C.estimate_noise(lr) < AUTO_NOISE_THR:
        return "dl_only"
    return "hybrid"


def run_auto(lr, valid, rgb, scale, tag=""):
    """Adaptive selector: the deep-only net is best on pristine depth, the hybrid on degraded depth."""
    sfx = f"_{tag}" if tag else ""
    if auto_choice(lr, valid) == "dl_only":
        return run_dl_only(lr, valid, rgb, scale, kind="dl_only" + sfx)
    return run_hybrid(lr, valid, rgb, scale, kind="hybrid" + sfx)


def run_auto_ens(lr, valid, rgb, scale, tags=("L", "L2")):
    """Auto selector over an ensemble of independently trained models (outputs averaged)."""
    pristine = auto_choice(lr, valid) == "dl_only"
    outs = [run_dl_only(lr, valid, rgb, scale, kind=f"dl_only_{t}") if pristine else run_hybrid(lr, valid, rgb, scale, kind=f"hybrid_{t}")
            for t in tags]
    return np.mean(outs, 0)


def with_tta(fn):
    """Horizontal-flip test-time augmentation: average the prediction with the flipped-input prediction."""
    def g(lr, valid, rgb, scale):
        a = fn(lr, valid, rgb, scale)
        b = fn(np.ascontiguousarray(lr[:, ::-1]), np.ascontiguousarray(valid[:, ::-1]), np.ascontiguousarray(rgb[:, ::-1]), scale)
        return 0.5 * (a + b[:, ::-1])
    return g


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
    "auto": run_auto,
    "auto_tta": None,           # filled below
    "hybrid_L": lambda lr, v, rgb, s: run_hybrid(lr, v, rgb, s, kind="hybrid_L"),
    "dl_only_L": lambda lr, v, rgb, s: run_dl_only(lr, v, rgb, s, kind="dl_only_L"),
    "auto_L": lambda lr, v, rgb, s: run_auto(lr, v, rgb, s, tag="L"),
    "auto_L_tta": None,
    "hybrid_convex": lambda lr, v, rgb, s: run_hybrid(lr, v, rgb, s, kind="hybrid_convex"),
}
METHODS["auto_tta"] = with_tta(run_auto)
METHODS["auto_E_tta"] = with_tta(run_auto_ens)
METHODS["auto_L_tta"] = with_tta(lambda lr, v, rgb, s: run_auto(lr, v, rgb, s, tag="L"))
LABELS = {"auto_E_tta": "EdgeGate-Auto-L ensemble + flip TTA", "hybrid_L": "Hybrid-L (GPU-trained)", "dl_only_L": "Deep-only-L (GPU-trained)", "auto_L": "EdgeGate-Auto-L (GPU-trained)",
          "auto_tta": "EdgeGate-Auto + flip TTA", "auto_L_tta": "EdgeGate-Auto-L + flip TTA", "nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic", "jbu": "Joint bilateral (JBU)",
          "guided": "Guided filter", "classical": "Classical (ours: gated JBU)", "dl_only": "Deep learning only",
          "hybrid": "Hybrid (ours)", "hybrid_convex": "Hybrid v1 (softmax arbiter)", "auto": "EdgeGate-Auto (ours)"}


def timed(name, lr, valid, rgb, scale):
    t = time.perf_counter()
    out = METHODS[name](lr, valid, rgb, scale)
    return out, 1000 * (time.perf_counter() - t)
