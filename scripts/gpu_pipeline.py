"""One command that runs EVERYTHING needing local resources (GPU training, GPU evaluation, CPU latency benchmark,
ablations, failure mining, report) and bundles the results to send back.

    python scripts/gpu_pipeline.py                       # full run, about 5-8 h on an RTX 5080 laptop
    python scripts/gpu_pipeline.py --hours-per-model 1.5 # shorter
    python scripts/gpu_pipeline.py --smoke               # 2-minute end-to-end check on tiny settings (CPU ok)

Safe to interrupt and re-run: finished stages are skipped (markers in results/gpu_run/) and training resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "gpu_run"
PY = sys.executable
L_ARMS = ["dl_only_L", "hybrid_L", "auto_L", "auto_L_tta", "auto_tta"]


def sh(cmd, log=None, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    e = {**os.environ, **(env or {})}
    out = open(RUN / log, "a") if log else None
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=e, stdout=out, stderr=subprocess.STDOUT if out else None)
    if out:
        out.close()
    if r.returncode:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(map(str, cmd))}  (see results/gpu_run/{log})")


def stage(name):
    marker = RUN / f"{name}.done"
    return marker.exists(), marker


def probe(kind, cfg_args, common):
    r = subprocess.run([PY, "-m", "depthup.train", "--kind", kind, "--probe", "--tag", "probe", *cfg_args, *common],
                       cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"PROBE its_per_s ([\d.]+)", r.stdout)
    if not m:
        sys.exit("speed probe failed:\n" + r.stdout[-800:] + r.stderr[-800:])
    return float(m.group(1))


PROFILES = {   # hours per model, ensemble members, max steps
    "max": dict(hours=3.0, ensemble=2, max_steps=200000),
    "balanced": dict(hours=2.0, ensemble=1, max_steps=120000),
    "quick": dict(hours=0.75, ensemble=1, max_steps=40000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="max",
                    help="max: 2 seeds x 2 models x 3 h (about 13 h incl. evaluation); balanced: ~5 h; quick: ~2 h")
    ap.add_argument("--hours-per-model", type=float, default=None, help="override the profile (wall-clock budget per trained model)")
    ap.add_argument("--ensemble", type=int, default=None, help="override: number of seeds per model type (1 = no ensemble)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--nblocks", type=int, default=3)
    ap.add_argument("--mid-rep", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32, help="starting batch size; halves automatically on GPU out-of-memory")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    ap.add_argument("--eval-procs", type=int, default=min(12, max(2, (os.cpu_count() or 8) // 2)))
    ap.add_argument("--ablation-frames", type=int, default=200)
    ap.add_argument("--tag", default="L")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    prof = PROFILES[a.profile]
    hours = a.hours_per_model if a.hours_per_model is not None else prof["hours"]
    ensemble = a.ensemble if a.ensemble is not None else prof["ensemble"]
    max_steps = a.max_steps if a.max_steps is not None else prof["max_steps"]
    RUN.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.width, a.nblocks, a.mid_rep, a.batch, a.patch, a.tag, a.workers, ensemble = 16, 1, 1, 4, 64, "L", 2, 2

    # ---- 0. environment
    import torch
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"torch {torch.__version__}  cuda available: {cuda}" + (f"  gpu: {torch.cuda.get_device_name(0)}" if cuda else ""))
    if not cuda and not a.smoke:
        sys.exit("CUDA not available. An RTX 5080 (Blackwell) needs a CUDA 12.8 PyTorch build: "
                 "pip install torch --index-url https://download.pytorch.org/whl/cu128   (see docs/GPU_TRAINING.md)")
    # ---- 1. data
    done, mk = stage("1_data")
    if not done:
        if not a.skip_download:
            sh([PY, "scripts/download_nyu.py"], "1_data.log")
        mk.write_text("ok")
    # ---- 2. train (equal STEP budget for both models, derived from a speed probe of the slower one)
    cfg = ["--width", a.width, "--nblocks", a.nblocks, "--mid-rep", a.mid_rep]

    def common_for(batch):
        c = ["--batch", batch, "--patch", a.patch, "--workers", a.workers, "--device", device, "--train-range", 0, 950,
             "--val-range", 950, 1000, "--val-frames", 16]
        return c + (["--amp"] if device == "cuda" else [])

    done, mk = stage("2_probe")
    if not done:
        batch = a.batch
        while True:        # largest batch that fits in GPU memory (probe both kinds at the final size)
            try:
                ips = min(probe(k, [str(c) for c in cfg], [str(c) for c in common_for(batch)]) for k in ("hybrid", "dl_only"))
                break
            except SystemExit as e:
                if batch <= 4 or "out of memory" not in str(e).lower():
                    raise
                batch //= 2
                print(f"GPU out of memory: retrying with batch {batch}", flush=True)
        steps = 30 if a.smoke else int(min(max_steps, hours * 3600 * ips * 0.92))
        mk.write_text(json.dumps({"its_per_s": ips, "steps": steps, "batch": batch}))
    info = json.loads(mk.read_text())
    steps, batch = info["steps"], info["batch"]
    common = common_for(batch)
    print(f"batch {batch}; {steps} steps per model ({steps / info['its_per_s'] / 3600:.1f} h each at {info['its_per_s']:.2f} it/s); "
          f"{ensemble} seed(s) x 2 model types", flush=True)
    tags = [a.tag] + [f"{a.tag}{i + 1}" for i in range(1, ensemble)]          # L, L2, ...
    for seed, tag in enumerate(tags):
        for kind in ("dl_only", "hybrid"):
            done, mk = stage(f"3_train_{kind}_{tag}")
            if not done:
                sh([PY, "-m", "depthup.train", "--kind", kind, "--tag", tag, "--seed", seed, "--steps", steps, "--lr", "1e-3", "--ema",
                    "--val-every", max(10, steps // 40), "--save-every", max(10, steps // 20), "--resume",
                    *[str(c) for c in cfg], *[str(c) for c in common]], f"3_train_{kind}_{tag}.log")
                mk.write_text("ok")
    # ---- 4. evaluation (GPU inference); baselines' rows are reused if results/per_image_*.jsonl came with the project
    n = {"clean": 449, "noisy": 449, "noisy+holes": 449}          # full test split for every protocol
    if a.smoke:
        n = {k: 4 for k in n}
    arms = L_ARMS + (["auto_E_tta"] if ensemble > 1 else [])
    for prot, k in n.items():
        done, mk = stage(f"4_eval_{prot}")
        if not done:
            have = (ROOT / "results" / f"per_image_{prot}.jsonl").exists()
            methods = arms if have else ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical", "dl_only", "hybrid", "auto", *arms]
            sh([PY, "-m", "depthup.evaluate", "--protocol", prot, "--n", k, "--device", device, "--procs", a.eval_procs,
                "--methods", *methods, *(["--append"] if have else [])], f"4_eval_{prot}.log")
            mk.write_text("ok")
    # ---- 5-8. benchmark (CPU latency on purpose: that is what a free host runs), ablation, failures, report
    for name, cmd in [("5_benchmark", [PY, "-m", "depthup.benchmark"]),
                      ("6_ablate", [PY, "-m", "depthup.ablate", "--n", 20 if a.smoke else a.ablation_frames, "--device", device, "--procs", a.eval_procs]),
                      ("7_failures", [PY, "-m", "depthup.failures", "--kind", "hybrid_L", "--device", device]),
                      ("8_report", [PY, "-m", "depthup.report", "--team", os.environ.get("TEAM", "T01"),
                                    "--usns", os.environ.get("USNS", "USN1,USN2,USN3,USN4")]),
                      ("8_docs", [PY, "scripts/make_docs.py"])]:
        done, mk = stage(name)
        if not done:
            sh(cmd, f"{name}.log")
            mk.write_text("ok")
    # ---- 9. bundle everything to send back
    out = ROOT / ("gpu_results_smoke.zip" if a.smoke else "gpu_results.zip")
    # hybrid_L / dl_only_L must exist for the arms above; ensure the ensemble arm name resolves only with >= 2 members
    keep = [*ROOT.glob("weights/*_L.pt"), *ROOT.glob("weights/*_L[0-9].pt"), *ROOT.glob("results/*.json"), *ROOT.glob("results/*.jsonl"),
            *ROOT.glob("results/logs/*.log"), *ROOT.glob("results/gpu_run/*"), *ROOT.glob("docs/*.md"), *ROOT.glob("docs/figures/*.png"),
            *ROOT.glob("eval_P4_*.md"), ROOT / "README.md"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in keep:
            if f.is_file():
                z.write(f, f.relative_to(ROOT))
    print(f"\nALL DONE. Results bundle: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
