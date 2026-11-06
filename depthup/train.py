"""Train HybridNet / DLOnlyNet on NYU train split with on-the-fly degradation.

    python -m depthup.train --kind hybrid --minutes 25
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from . import data, degrade, metrics, pipeline
from .model import DLOnlyNet, HybridNet, n_params

cv2.setNumThreads(1)
PATCH = 128


def sample_degradation(rng: np.random.Generator) -> degrade.Degradation:
    scale = int(rng.choice([2, 4, 8]))
    r = rng.random()
    if r < 0.5:
        kw = dict(noise=0.0, outliers=0.0, holes=0.0)
    elif r < 0.75:
        kw = dict(noise=rng.uniform(0.005, 0.035), outliers=rng.uniform(0, 0.01), holes=0.0)
    else:
        kw = dict(noise=rng.uniform(0.0, 0.035), outliers=rng.uniform(0, 0.01), holes=rng.uniform(0.05, 0.35))
    mode = "bicubic" if rng.random() < 0.8 else "nearest"
    return degrade.Degradation(scale=scale, mode=mode, **kw)


class PatchStream(IterableDataset):
    def __init__(self, kind: str, split: str = "train", seed: int = 0):
        self.kind, self.split, self.seed = kind, split, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 1000 + int(time.time()) % 1000)
        idxs = data.split_indices(self.split)
        while True:
            rgb, dep = data.get_pair(int(rng.choice(idxs)))
            y, x = rng.integers(0, (data.H - PATCH) // 8 + 1) * 8, rng.integers(0, (data.W - PATCH) // 8 + 1) * 8
            rgb, dep = rgb[y:y + PATCH, x:x + PATCH], dep[y:y + PATCH, x:x + PATCH]
            if rng.random() < 0.5:
                rgb, dep = rgb[:, ::-1], dep[:, ::-1]
            rgb, dep = np.ascontiguousarray(rgb), np.ascontiguousarray(dep)
            d = sample_degradation(rng)
            lr, valid = degrade.degrade(dep, d, rng)
            if self.kind == "hybrid":
                xin, d_bic, d_gate, m, _ = pipeline.hybrid_features(lr, valid, rgb, d.scale)
                yield (torch.from_numpy(xin), torch.from_numpy(d_bic - m)[None], torch.from_numpy(d_gate - m)[None],
                       torch.from_numpy(dep - m)[None], torch.tensor(float(d.scale)))
            else:
                xin, d_bic, m = pipeline.dl_features(lr, valid, rgb, d.scale)
                z = torch.zeros(1, PATCH, PATCH)
                yield (torch.from_numpy(xin), torch.from_numpy(d_bic - m)[None], z,
                       torch.from_numpy(dep - m)[None], torch.tensor(float(d.scale)))


def grad(a):
    return a[..., :, 1:] - a[..., :, :-1], a[..., 1:, :] - a[..., :-1, :]


def loss_fn(pred, gt, bic):
    """L1 + gradient-matching + boundary-weighted L1 (weights from GT depth gradients)."""
    gx, gy = grad(gt)
    mag = F.pad(gx.abs(), (0, 1, 0, 0)) + F.pad(gy.abs(), (0, 0, 0, 1))
    w = 1.0 + 6.0 * (mag > 0.1).float()
    l1 = (w * (pred - gt).abs()).mean()
    px, py = grad(pred)
    lg = (px - gx).abs().mean() + (py - gy).abs().mean()
    return l1 + 0.5 * lg


@torch.no_grad()
def validate(net, kind, n=24):
    net.eval()
    rng = np.random.default_rng(123)
    errs = []
    for i in data.split_indices("val")[::200 // n][:n]:
        rgb, dep = data.get_pair(i)
        for s in (2, 4, 8):
            lr, valid = degrade.degrade(dep, degrade.Degradation(scale=s), rng)
            pred = pipeline.run_hybrid(lr, valid, rgb, s) if kind == "hybrid" else pipeline.run_dl_only(lr, valid, rgb, s)
            errs.append(np.sqrt(((pred - dep) ** 2).mean()) * 100)
    net.train()
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hybrid", "dl_only"], required=True)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=400)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    net = HybridNet() if a.kind == "hybrid" else DLOnlyNet()
    pipeline._models[a.kind] = net
    print(f"{a.kind}: {n_params(net):,} params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(PatchStream(a.kind), batch_size=a.batch, num_workers=a.workers)
    t0, budget, step, best = time.time(), a.minutes * 60, 0, 1e9
    net.train()
    log = []
    for x, d_bic, d_gate, gt, _ in dl:
        frac = min((time.time() - t0) / budget, 1.0)
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * frac))) * min(1.0, (step + 1) / 100)
        if a.kind == "hybrid":
            out, d_dl, w = net(x, d_bic, d_gate)
            loss = loss_fn(out, gt, d_bic) + 0.5 * loss_fn(d_dl, gt, d_bic)
        else:
            loss = loss_fn(net(x, d_bic), gt, d_bic)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
        if step % a.val_every == 0 or frac >= 1.0:
            v = validate(net, a.kind)
            log.append((step, v))
            print(f"== step {step} val RMSE(cm, clean 2/4/8) {v:.3f}", flush=True)
            if v < best:
                best = v
                torch.save(net.state_dict(), pipeline.WEIGHTS / f"{a.kind}.pt")
        if frac >= 1.0:
            break
    print("best val", best, flush=True)


if __name__ == "__main__":
    main()
