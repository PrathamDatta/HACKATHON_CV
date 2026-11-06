"""Learned stage: a compact U-Net that *arbitrates* between classical and learned experts.

Experts (all full resolution):
    E0 = clean bicubic          (safe, smooth)
    E1 = gated JBU (classical)  (sharp at depth discontinuities, RGB-gated)
    E2 = E0 + delta             (the network's own residual correction, RGB guided)
The network predicts delta and a per-pixel softmax over {E0, E1, E2}. Output = sum_k w_k * E_k.
`DLOnlyNet` is the same backbone with no classical inputs/experts (the "deep learning only" arm).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(i, o, k=3, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, padding=d * (k // 2), dilation=d, bias=False),
                         nn.GroupNorm(8, o), nn.GELU())


class Backbone(nn.Module):
    def __init__(self, c_in: int, c_out: int, w: int = 32):
        super().__init__()
        self.stem = nn.Sequential(conv(c_in, w), conv(w, w))
        self.d1 = nn.Sequential(conv(w, 2 * w, s=2), conv(2 * w, 2 * w))
        self.d2 = nn.Sequential(conv(2 * w, 4 * w, s=2), conv(4 * w, 4 * w))
        self.mid = nn.Sequential(conv(4 * w, 4 * w, d=2), conv(4 * w, 4 * w, d=4), conv(4 * w, 4 * w, d=8))
        self.u1 = conv(4 * w + 2 * w, 2 * w)
        self.u0 = conv(2 * w + w, w)
        self.head = nn.Conv2d(w, c_out, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.mid(self.d2(s1))
        u1 = self.u1(torch.cat([F.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
        u0 = self.u0(torch.cat([F.interpolate(u1, size=s0.shape[-2:], mode="bilinear", align_corners=False), s0], 1))
        return self.head(u0)


class HybridNet(nn.Module):
    """inputs: rgb(3) d_bic d_gate d_snap gate rgb_edge conf scale  -> 10 channels."""
    C_IN = 10

    def __init__(self, w: int = 16):
        super().__init__()
        self.net = Backbone(self.C_IN, 4, w)
        nn.init.zeros_(self.net.head.weight)
        with torch.no_grad():
            self.net.head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))

    def forward(self, x, d_bic, d_gate):
        o = self.net(x)
        delta = o[:, :1]
        d_dl = d_bic + delta
        logits = o[:, 1:]
        wts = torch.softmax(logits, 1)
        out = wts[:, 0:1] * d_bic + wts[:, 1:2] * d_gate + wts[:, 2:3] * d_dl
        return out, d_dl, wts


class DLOnlyNet(nn.Module):
    """inputs: rgb(3) d_bic scale -> 5 channels; output = d_bic + residual."""
    C_IN = 5

    def __init__(self, w: int = 16):
        super().__init__()
        self.net = Backbone(self.C_IN, 1, w)
        nn.init.zeros_(self.net.head.weight)
        nn.init.zeros_(self.net.head.bias)

    def forward(self, x, d_bic):
        return d_bic + self.net(x)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
