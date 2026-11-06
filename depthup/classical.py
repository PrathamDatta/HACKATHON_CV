"""Classical depth upsampling & refinement.

Contents
--------
* interpolation baselines: nearest / bilinear / bicubic
* hole filling (multi-scale normalised convolution), spike removal, noise-adaptive denoising
* joint bilateral upsampling (JBU, Kopf et al.) and guided-filter upsampling (He et al.)
* depth discontinuity detection (local jump map) and RGB edge map
* OUR classical pipeline  ("GATE"): discontinuity-gated tri-lateral JBU
  + colour-plateau snapping (CPS) with adaptive parameters.

The key idea against texture-copying: RGB only gets a vote where the *depth itself* shows a
discontinuity (the gate g). In depth-smooth areas the RGB range kernel is flattened, so painted
textures cannot leak into the depth map. Where a discontinuity exists, CPS re-assigns each
pixel to the nearer of the two depth plateaus using colour similarity, which recreates a sharp
step aligned with the real colour boundary instead of a blurred ramp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------- interpolation


def _resize(a: np.ndarray, scale: int, method) -> np.ndarray:
    h, w = a.shape
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((w * scale, h * scale), method))


def nearest_up(lr, scale):
    return np.repeat(np.repeat(lr, scale, 0), scale, 1).astype(np.float32)


def bilinear_up(lr, scale):
    return _resize(lr, scale, Image.BILINEAR)


def bicubic_up(lr, scale):
    return _resize(lr, scale, Image.BICUBIC)


# ------------------------------------------------------------------------- hole filling etc.


def naive_fill(lr, valid):
    """Nearest-valid fill. Used to give plain interpolation baselines a fair input."""
    if valid.all():
        return lr
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return lr[tuple(idx)]


def normalized_fill(lr, valid):
    """Multi-scale normalised-convolution hole filling (fills inwards from valid samples)."""
    if valid.all():
        return lr.copy()
    out = np.where(valid, lr, 0).astype(np.float32)
    known = valid.astype(np.float32)
    for sigma in (1, 2, 4, 8, 16, 32, 64):
        if known.min() > 0:
            break
        num = cv2.GaussianBlur(out * known, (0, 0), sigma)
        den = cv2.GaussianBlur(known, (0, 0), sigma)
        new = (known == 0) & (den > 0.05)
        out[new] = num[new] / den[new]
        known[new] = 1.0
    out[known == 0] = out[known > 0].mean() if (known > 0).any() else 1.0
    return out


def estimate_noise(lr):
    """Robust sigma estimate (Immerkaer kernel + MAD); edges barely influence the median."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(lr.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    return float(1.4826 * np.median(np.abs(r)) / 6.0)


def remove_spikes(lr, sigma):
    """Replace isolated local extrema (flying pixels) with the 3x3 median."""
    med = cv2.medianBlur(lr.astype(np.float32), 3)
    mx = ndi.maximum_filter(lr, 3, mode="nearest")
    mn = ndi.minimum_filter(lr, 3, mode="nearest")
    # an isolated spike is the extremum of its neighbourhood *and* far from the median
    second_hi = ndi.rank_filter(lr, -2, size=3, mode="nearest")
    second_lo = ndi.rank_filter(lr, 1, size=3, mode="nearest")
    thr = 6 * sigma + 0.30
    spike = ((lr >= mx) & (lr - second_hi > thr)) | ((lr <= mn) & (second_lo - lr > thr))
    return np.where(spike, med, lr), spike


def clean_lr(lr, valid):
    """Adaptive LR clean-up: hole fill -> noise estimate -> spike removal -> bilateral smoothing."""
    info = {}
    filled = normalized_fill(lr, valid)
    sigma = estimate_noise(filled)
    info["sigma_est"] = sigma
    cleaned, spikes = remove_spikes(filled, sigma)
    info["spikes"] = int(spikes.sum())
    if sigma > 0.01:  # noise-adaptive: leave clean inputs untouched
        cleaned = cv2.bilateralFilter(cleaned.astype(np.float32), 3, float(max(2.5 * sigma, 0.03)), 1.0)
    return cleaned.astype(np.float32), info


def confidence_map(valid):
    """LR confidence: 1 where measured, ~0.15 where filled."""
    return np.where(valid, 1.0, 0.15).astype(np.float32)


# ------------------------------------------------------------------- guided / joint bilateral


def box(a, r):
    return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(p, guide, r, eps):
    """Guided filter (He et al.) with a colour guide (HxWx3) and a scalar input p."""
    I = guide.astype(np.float32)
    p = p.astype(np.float32)
    mean_I = np.stack([box(I[..., c], r) for c in range(3)], -1)
    mean_p = box(p, r)
    corr_Ip = np.stack([box(I[..., c] * p, r) for c in range(3)], -1)
    cov_Ip = corr_Ip - mean_I * mean_p[..., None]
    var = np.empty((*p.shape, 3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = box(I[..., i] * I[..., j], r) - mean_I[..., i] * mean_I[..., j]
            var[..., i, j] = var[..., j, i] = v
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]
    b = mean_p - (a * mean_I).sum(-1)
    return (box(a[..., 0], r) * I[..., 0] + box(a[..., 1], r) * I[..., 1]
            + box(a[..., 2], r) * I[..., 2] + box(b, r)).astype(np.float32)


def guided_up(lr, rgb, scale, r=None, eps=1e-3):
    """Bicubic upsample then refine with an RGB guided filter (baseline)."""
    return guided_filter(bicubic_up(lr, scale), rgb, r or max(2, scale), eps)


def _shifted_lr_samples(lr, conf, rgb, scale, radius=2):
    """Yield (value, confidence, guide colour, HR-distance^2) for each LR neighbour tap."""
    h, w = lr.shape
    H, W = h * scale, w * scale
    cy = (np.arange(H) // scale).astype(int)
    cx = (np.arange(W) // scale).astype(int)
    # HR coordinates of the LR sample centres
    py = (np.arange(h) * scale + (scale - 1) / 2.0)
    px = (np.arange(w) * scale + (scale - 1) / 2.0)
    yy, xx = np.arange(H)[:, None], np.arange(W)[None, :]
    # nearest-centre offset so the 5x5 window is centred on the pixel
    oy = np.floor((yy - (scale - 1) / 2.0) / scale).astype(int)
    ox = np.floor((xx - (scale - 1) / 2.0) / scale).astype(int)
    rgb_f = rgb.astype(np.float32)
    for dy in range(-radius + 1, radius + 1):
        for dx in range(-radius + 1, radius + 1):
            ny = np.clip(oy + dy, 0, h - 1)
            nx = np.clip(ox + dx, 0, w - 1)
            v = lr[ny, nx]
            c = conf[ny, nx]
            gy = np.clip(np.round(py[ny]).astype(int), 0, H - 1)
            gx = np.clip(np.round(px[nx]).astype(int), 0, W - 1)
            g = rgb_f[gy, gx]
            d2 = (yy - py[ny]) ** 2 + (xx - px[nx]) ** 2
            yield v, c, g, d2


def jbu(lr, conf, rgb, scale, sigma_s=None, sigma_c=0.1, sigma_d=None, ref=None, radius=2):
    """Joint bilateral upsampling.

    sigma_c may be a scalar or an HxW map (spatially varying colour tolerance -> the gate).
    sigma_d/ref add a depth-consistency term (tri-lateral): taps whose depth differs from the
    reference estimate `ref` by much more than sigma_d are down-weighted.
    """
    sigma_s = sigma_s or 0.45 * scale
    rgb = rgb.astype(np.float32)
    inv_c = 1.0 / (2 * np.asarray(sigma_c, np.float32) ** 2)
    logw, vals = [], []
    for v, c, g, d2 in _shifted_lr_samples(lr, conf, rgb, scale, radius):
        lw = -d2 / (2 * sigma_s ** 2) - ((rgb - g) ** 2).sum(-1) * inv_c + np.log(np.maximum(c, 1e-6))
        if sigma_d is not None and ref is not None:
            lw = lw - ((v - ref) ** 2) / (2 * np.asarray(sigma_d) ** 2)
        logw.append(lw.astype(np.float32))
        vals.append(v)
    logw = np.stack(logw)
    logw -= logw.max(0, keepdims=True)           # log-domain softmax: never underflows
    w = np.exp(logw)
    return ((w * np.stack(vals)).sum(0) / w.sum(0)).astype(np.float32)


# ------------------------------------------------------------------ discontinuities and edges

TAU_PER_SCALE = 0.025   # metres of depth jump per unit of upsampling scale tolerated as "surface"


def lr_discontinuity(clean, scale, sigma_est, tau_scale=TAU_PER_SCALE):
    """Step-edge detector on the LR depth.

    A *step* gives a large second difference, a sloped plane gives ~0, so floors/walls are not
    flagged. The tolerance grows with `scale`: a sloped surface changes by slope*scale between LR
    samples and a crease (wall/floor) produces a second difference of that order.
    Returns (soft map in [0,1] at LR resolution, raw jump magnitude, tolerance map).
    """
    L = clean.astype(np.float32)
    kernels = []
    k = np.zeros((3, 3), np.float32)
    k[1, :] = [1, -2, 1]; kernels.append(k.copy())
    k = np.zeros((3, 3), np.float32)
    k[:, 1] = [1, -2, 1]; kernels.append(k.copy())
    k = np.zeros((3, 3), np.float32); k[0, 0] = 1; k[1, 1] = -2; k[2, 2] = 1; kernels.append(k)
    k = np.zeros((3, 3), np.float32); k[0, 2] = 1; k[1, 1] = -2; k[2, 0] = 1; kernels.append(k)
    c = np.max([np.abs(cv2.filter2D(L, -1, kk, borderType=cv2.BORDER_REFLECT)) for kk in kernels], 0)
    tau = np.maximum.reduce([np.full_like(L, 0.05), 0.03 * L, np.full_like(L, 5 * sigma_est * 2.45),
                             np.full_like(L, tau_scale * scale)])
    soft = np.clip((c - tau) / tau, 0, 1).astype(np.float32)
    return soft, c, tau


def depth_discontinuity(clean, scale, sigma_est, tau_scale=TAU_PER_SCALE):
    """HR-resolution discontinuity evidence: (soft map, jump map) both HxW."""
    soft_lr, c, _ = lr_discontinuity(clean, scale, sigma_est, tau_scale)
    soft = nearest_up(soft_lr, scale)
    jump = nearest_up(c, scale)
    return soft, jump


def rgb_edges(rgb):
    """Normalised RGB gradient magnitude in [0,1]."""
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0, (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1).astype(np.float32)


def gate_map(soft, scale):
    """Spread the discontinuity evidence over the band where upsampling blur would occur."""
    k = scale + 1 if scale % 2 == 0 else scale
    g = cv2.dilate(soft, np.ones((k, k), np.uint8))
    return cv2.GaussianBlur(g, (0, 0), max(1.0, scale / 3)).clip(0, 1).astype(np.float32)


# ------------------------------------------------------------------ colour-plateau snapping


def plateau_snap(clean, rgb, scale, temp=0.01):
    """Colour-plateau snapping (CPS), computed at LR resolution (cheap) and applied at HR.

    For every LR cell take the lowest/highest depth in its 3x3 LR neighbourhood (the two plateaus
    meeting at an edge), estimate each plateau's colour from the HR pixels of the cells that
    belong to it, and give every HR pixel the plateau whose colour matches it best.
    Returns (snapped HR depth, snap confidence HR in [0,1]).
    """
    L = clean.astype(np.float32)
    h, w = L.shape
    rgbf = rgb.astype(np.float32) / 255.0
    cell = rgbf.reshape(h, scale, w, scale, 3).mean((1, 3))                 # mean colour per LR cell
    lo = ndi.minimum_filter(L, 3, mode="nearest")
    hi = ndi.maximum_filter(L, 3, mode="nearest")
    m_lo = (np.abs(L - lo) <= np.abs(L - hi)).astype(np.float32)
    m_hi = 1.0 - m_lo

    def side(m):
        num = np.stack([cv2.boxFilter(cell[..., c] * m, -1, (3, 3), normalize=False,
                                      borderType=cv2.BORDER_REFLECT) for c in range(3)], -1)
        den = cv2.boxFilter(m, -1, (3, 3), normalize=False, borderType=cv2.BORDER_REFLECT)
        return num / np.maximum(den, 1e-6)[..., None], den

    c_lo, n_lo = side(m_lo)
    c_hi, n_hi = side(m_hi)
    sep = np.sqrt(((c_lo - c_hi) ** 2).sum(-1))
    conf_lr = np.clip((sep - 0.04) / 0.10, 0, 1) * ((n_lo > 0.5) & (n_hi > 0.5))

    up = lambda a: np.repeat(np.repeat(a, scale, 0), scale, 1)               # noqa: E731
    lo_h, hi_h, conf = up(lo), up(hi), up(conf_lr.astype(np.float32))
    clo, chi = [np.repeat(np.repeat(c, scale, 0), scale, 1) for c in (c_lo, c_hi)]
    d_lo = ((rgbf - clo) ** 2).sum(-1)
    d_hi = ((rgbf - chi) ** 2).sum(-1)
    p_hi = 1.0 / (1.0 + np.exp(np.clip((d_hi - d_lo) / temp, -30, 30)))
    snapped = lo_h + (hi_h - lo_h) * p_hi
    return snapped.astype(np.float32), conf.astype(np.float32)


# ------------------------------------------------------------------------- full pipelines


@dataclass
class ClassicalOutput:
    depth: np.ndarray
    base: np.ndarray                      # bicubic of cleaned LR (D0)
    jbu_plain: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


def gate_pipeline(lr, valid, rgb, scale, use_gate=True, use_snap=False, use_clean=True,
                  use_depth_term=True, tau_scale=TAU_PER_SCALE, sigma_c_edge=0.15,
                  snap_gain=1.0) -> ClassicalOutput:
    """Our classical pipeline. Flags allow component-wise ablation.

    `use_snap` blends the hard colour-plateau snap into the output (off by default: on NYU's soft
    ground truth a hard snap is too risky to apply unconditionally). The *candidate* snapped map is
    always returned in `extras['snapped']` so the learned arbiter can choose it per pixel.
    """
    if use_clean:
        clean, info = clean_lr(lr, valid)
        conf = confidence_map(valid)
    else:
        clean, info = naive_fill(lr, valid), {"sigma_est": 0.0, "spikes": 0}
        conf = np.ones_like(lr)
    sigma = info["sigma_est"]
    d0 = bicubic_up(clean, scale)
    soft, jump = depth_discontinuity(clean, scale, sigma, tau_scale)
    g = gate_map(soft, scale)
    e_c = rgb_edges(rgb)
    rgbf = rgb.astype(np.float32) / 255.0

    # colour tolerance: tight at discontinuities, effectively disabled (huge) in depth-smooth areas
    if use_gate:
        sigma_c = np.exp(np.log(5.0) * (1 - g) + np.log(sigma_c_edge) * g)
    else:
        sigma_c = sigma_c_edge
    sigma_d = np.maximum(0.5 * jump, 0.04) if use_depth_term else None
    j = jbu(clean, conf, rgbf, scale, sigma_c=sigma_c, sigma_d=sigma_d, ref=d0 if use_depth_term else None)
    snapped, snap_conf = plateau_snap(clean, rgb, scale)
    cand_snap = g * snapped + (1 - g) * d0                      # snapped candidate for the arbiter
    out = j
    if use_snap:
        w = np.clip(g * snap_conf * snap_gain, 0, 1)
        out = (1 - w) * j + w * snapped
    if use_gate:  # outside the band the plain bicubic surface is the safest estimate
        out = g * out + (1 - g) * d0
    extras = dict(soft=soft, jump=jump, gate=g, rgb_edge=e_c, snap_conf=snap_conf, conf=conf,
                  sigma_est=sigma, spikes=info["spikes"], snapped=cand_snap.astype(np.float32))
    return ClassicalOutput(out.astype(np.float32), d0.astype(np.float32), j, extras)
