"""Download + unzip the NYU Depth V2 labelled set used by this project (about 1.9 GB) and build the cache.

    python scripts/download_nyu.py
Source: https://huggingface.co/datasets/xcll/nyu_depth_train_float32_1449 (research use; NYU Depth V2 licence applies).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URL = "https://huggingface.co/datasets/xcll/nyu_depth_train_float32_1449/resolve/main/nyu_depth_train_float32_1449.zip"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "nyu"


def main():
    from depthup import data
    if (data.CACHE_DIR / "rgb.npy").exists() and (data.CACHE_DIR / "depth.npy").exists():
        print("cache already built:", data.CACHE_DIR)
        return
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "nyu1449.zip"
    if not zpath.exists() or zpath.stat().st_size < 1_800_000_000:
        print("downloading (~1.9 GB) ...", flush=True)
        def hook(b, bs, total):
            if b % 2000 == 0:
                print(f"  {min(100, b * bs * 100 // max(total, 1))}%", flush=True)
        urllib.request.urlretrieve(URL, zpath, hook)
    if not (data.RAW_DIR / "images").exists():
        print("unzipping ...", flush=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(OUT)
    print("building cache ...", flush=True)
    data.build_cache()
    print("done")


if __name__ == "__main__":
    main()
