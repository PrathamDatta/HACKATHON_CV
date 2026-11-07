"""Create (or update) a Hugging Face Docker Space and upload this repository to it.

    python scripts/deploy_space.py --repo <your-hf-username>/edgegate

The token is read from the HF_TOKEN environment variable or typed at a hidden prompt; it is never stored in the repo.
Needs: pip install -U huggingface_hub     and a WRITE-access token from https://huggingface.co/settings/tokens
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = [".git/*", ".venv/*", "venv/*", "data/*", "results/gpu_run/*", "results/logs/*", "weights/*.resume.pt", "weights/*.tmp",
          "*.zip", "handoff/*", "**/__pycache__/*", "*.pyc", ".pytest_cache/*", "pipeline_*.log"]
REQUIRED = ["Dockerfile", "README.md", "app.py", "requirements.txt", "depthup/pipeline.py", "weights/dl_only.pt", "weights/hybrid.pt",
            "weights/dl_only_L.pt", "samples/nyu_1005_rgb.jpg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<hf-username>/<space-name>, e.g. pratham/edgegate")
    ap.add_argument("--private", action="store_true", help="create as private (judges cannot open it: leave off for submission)")
    a = ap.parse_args()
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    if missing:
        sys.exit(f"missing files (run `git pull` in the repo first): {missing}")
    if "/" not in a.repo:
        sys.exit("--repo must look like <username>/<space-name>")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("run: pip install -U huggingface_hub")
    token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face WRITE token (input hidden): ").strip()
    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"logged in as {who}")
    api.create_repo(a.repo, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    print("uploading (the 36 MB model goes through Git LFS automatically) ...", flush=True)
    api.upload_folder(folder_path=str(ROOT), repo_id=a.repo, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy EdgeGate depth upsampler")
    user, name = a.repo.split("/")
    print(f"\nUploaded. Build logs / status : https://huggingface.co/spaces/{a.repo}")
    print(f"Live app (once 'Running')      : https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
