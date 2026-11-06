"""Train HybridNet / DLOnlyNet with on-the-fly degradation. CPU or GPU.

CPU (what produced the committed small models):
    python -m depthup.train --kind hybrid --steps 6200 --minutes 120 --workers 3 --threads 4

GPU (recommended; see docs/GPU_TRAINING.md):
    python -m depthup.train --kind hybrid  --tag L --width 48 --nblocks 2 --mid-rep 2 --patch 192 --batch 16 \
        --steps 60000 --lr 1e-3 --amp --ema --train-range 0 950 --workers 8
    python -m depthup.train --kind dl_only --tag L  (same flags)

Checkpoints are dicts {"cfg": {...}, "state": state_dict, ...}; pipeline.load_model reads cfg so any size loads.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from . import data, degrade, pipeline
from .model import DLOnlyNet, HybridNet, n_params

cv2.setNumThreads(1)


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


def jitter(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild photometric augmentation (brightness, contrast, per-channel gain)."""
    x = rgb.astype(np.float32)
    x = (x - 128) * rng.uniform(0.8, 1.2) + 128 + rng.uniform(-20, 20)
    x = x * rng.uniform(0.95, 1.05, 3)
    return np.clip(x, 0, 255).astype(np.uint8)


class PatchStream(IterableDataset):
    def __init__(self, kind: str, indices, patch: int = 128, seed: int = 0):
        self.kind, self.indices, self.patch, self.seed = kind, list(indices), patch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 1000 + int(time.time()) % 1000)
        P = self.patch
        while True:
            rgb, dep = data.get_pair(int(rng.choice(self.indices)))
            y, x = rng.integers(0, (data.H - P) // 8 + 1) * 8, rng.integers(0, (data.W - P) // 8 + 1) * 8
            rgb, dep = rgb[y:y + P, x:x + P], dep[y:y + P, x:x + P]
            if rng.random() < 0.5:
                rgb, dep = rgb[:, ::-1], dep[:, ::-1]
            rgb, dep = np.ascontiguousarray(rgb), np.ascontiguousarray(dep)
            if rng.random() < 0.7:
                rgb = jitter(rgb, rng)
            d = sample_degradation(rng)
            lr, valid = degrade.degrade(dep, d, rng)
            if self.kind == "hybrid":
                xin, d_bic, d_gate, m, _ = pipeline.hybrid_features(lr, valid, rgb, d.scale)
                yield (torch.from_numpy(xin), torch.from_numpy(d_bic - m)[None], torch.from_numpy(d_gate - m)[None],
                       torch.from_numpy(dep - m)[None])
            else:
                xin, d_bic, m = pipeline.dl_features(lr, valid, rgb, d.scale)
                yield (torch.from_numpy(xin), torch.from_numpy(d_bic - m)[None], torch.zeros(1, P, P),
                       torch.from_numpy(dep - m)[None])


def grad(a):
    return a[..., :, 1:] - a[..., :, :-1], a[..., 1:, :] - a[..., :-1, :]


def loss_fn(pred, gt):
    """L1 + gradient matching + 7x weight on GT depth discontinuities."""
    gx, gy = grad(gt)
    mag = F.pad(gx.abs(), (0, 1, 0, 0)) + F.pad(gy.abs(), (0, 0, 0, 1))
    w = 1.0 + 6.0 * (mag > 0.1).float()
    l1 = (w * (pred - gt).abs()).mean()
    px, py = grad(pred)
    return l1 + 0.5 * ((px - gx).abs().mean() + (py - gy).abs().mean())


def make_net(kind, cfg):
    return (HybridNet if kind == "hybrid" else DLOnlyNet)(**cfg)


def forward(net, kind, x, d_bic, d_gate):
    return net(x, d_bic, d_gate)[0] if kind == "hybrid" else net(x, d_bic)


def build_val(kind, idxs):
    """Pre-compute validation inputs once (float16 to save RAM): clean 2x/4x/8x on full frames."""
    items = []
    rng = np.random.default_rng(123)
    for i in idxs:
        rgb, dep = data.get_pair(i)
        for s in (2, 4, 8):
            lr, valid = degrade.degrade(dep, degrade.Degradation(scale=s), rng)
            if kind == "hybrid":
                x, d_bic, d_gate, m, _ = pipeline.hybrid_features(lr, valid, rgb, s)
                items.append((x.astype(np.float16), (d_bic - m).astype(np.float16), (d_gate - m).astype(np.float16), (dep - m).astype(np.float16)))
            else:
                x, d_bic, m = pipeline.dl_features(lr, valid, rgb, s)
                items.append((x.astype(np.float16), (d_bic - m).astype(np.float16), None, (dep - m).astype(np.float16)))
    return items


@torch.no_grad()
def validate(net, kind, items, device):
    net.eval()
    errs = []
    for x, d_bic, d_gate, gt in items:
        t = lambda a: torch.from_numpy(a.astype(np.float32))[None].to(device)  # noqa: E731
        pred = forward(net, kind, t(x), t(d_bic)[None], t(d_gate)[None] if d_gate is not None else None)
        errs.append(float(((pred[0, 0] - t(gt)[0]) ** 2).mean().sqrt()) * 100)
    net.train()
    return float(np.mean(errs))


def atomic_save(obj, path):
    """Write to a temp file then rename, so a crash mid-write can never leave a corrupt checkpoint."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save(path, net, cfg, **extra):
    atomic_save({"cfg": cfg, "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}, **extra}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hybrid", "dl_only"], required=True)
    ap.add_argument("--tag", default="", help="suffix for the weight file, e.g. L -> weights/hybrid_L.pt (empty = weights/hybrid.pt)")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--nblocks", type=int, default=1)
    ap.add_argument("--mid-rep", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="schedule/stop by step count (equal-budget comparisons)")
    ap.add_argument("--minutes", type=float, default=20, help="time budget when --steps is 0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128, help="HR crop size, multiple of 8")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (CPU training only)")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--train-range", type=int, nargs=2, default=[0, 800], metavar=("START", "STOP"))
    ap.add_argument("--val-range", type=int, nargs=2, default=[800, 1000], metavar=("START", "STOP"))
    ap.add_argument("--val-frames", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (CUDA)")
    ap.add_argument("--ema", action="store_true", help="evaluate/save an exponential moving average of the weights")
    ap.add_argument("--init", default="", help="initialise from an existing checkpoint (fine-tune)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="seed for init and data order (ensemble members use different seeds)")
    ap.add_argument("--probe", action="store_true", help="measure steps/s for 150 steps, print PROBE line, exit (no files written)")
    a = ap.parse_args()

    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(a.seed)
    if device.type == "cpu":
        torch.set_num_threads(a.threads)
    else:
        torch.backends.cudnn.benchmark = True
    cfg = dict(w=a.width, nblocks=a.nblocks, mid_rep=a.mid_rep)
    name = a.kind + (f"_{a.tag}" if a.tag else "")
    out_path, resume_path = pipeline.WEIGHTS / f"{name}.pt", pipeline.WEIGHTS / f"{name}.resume.pt"
    net = make_net(a.kind, cfg).to(device)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["state"] if "state" in ck else ck)
        print("initialised from", a.init, flush=True)
    ema = copy.deepcopy(net) if a.ema else None
    print(f"{name}: {n_params(net):,} params, cfg={cfg}, device={device}, amp={a.amp}, ema={a.ema}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    step, best = 0, 1e9
    if a.resume and resume_path.exists():
        try:
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
            if ema is not None and ck.get("ema") is not None:
                ema.load_state_dict(ck["ema"])
            step, best = ck["step"], ck["best"]
            print(f"resumed at step {step}", flush=True)
        except Exception as e:      # e.g. file truncated by a crash before atomic saves existed
            print(f"WARNING: could not load {resume_path.name} ({e}); starting this model from step 0", flush=True)
            resume_path.rename(str(resume_path) + ".corrupt")

    val_items = build_val(a.kind, list(range(*a.val_range))[::max(1, (a.val_range[1] - a.val_range[0]) // a.val_frames)][:a.val_frames])
    dl = DataLoader(PatchStream(a.kind, range(*a.train_range), a.patch, seed=a.seed), batch_size=a.batch, num_workers=a.workers,
                    pin_memory=device.type == "cuda", persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    t0, budget = time.time(), a.minutes * 60
    net.train()
    use_amp = a.amp and device.type == "cuda"
    for x, d_bic, d_gate, gt in dl:
        frac = min(step / a.steps, 1.0) if a.steps else min((time.time() - t0) / budget, 1.0)
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * frac))) * min(1.0, (step + 1) / 200)
        x, d_bic, d_gate, gt = (t.to(device, non_blocking=True) for t in (x, d_bic, d_gate, gt))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = forward(net, a.kind, x, d_bic, d_gate)
        loss = loss_fn(pred.float(), gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if ema is not None:
            d = min(0.9995, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.mul_(d).add_(p.detach(), alpha=1 - d)
        if a.probe:
            if step == 30:
                t_probe = time.time()
            if step == 150:
                print(f"PROBE its_per_s {120 / (time.time() - t_probe):.3f}", flush=True)
                return
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} t={time.time() - t0:.0f}s", flush=True)
        if step % a.val_every == 0 or frac >= 1.0:
            v = validate(ema if ema is not None else net, a.kind, val_items, device)
            print(f"== step {step} val RMSE(cm, clean 2/4/8) {v:.3f}", flush=True)
            if v < best:
                best = v
                save(out_path, ema if ema is not None else net, cfg, step=step, val_rmse=v)
        if step % a.save_every == 0:
            atomic_save({"net": net.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                         "step": step, "best": best}, resume_path)
        if frac >= 1.0:
            break
    print("best val", best, flush=True)


if __name__ == "__main__":
    main()
