"""Download-free helper: converts the unzipped NYU folder into a memory-mappable cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthup import data  # noqa: E402

if __name__ == "__main__":
    print("raw dir  :", data.RAW_DIR)
    print("cache dir:", data.CACHE_DIR)
    data.build_cache()
    print("done")
