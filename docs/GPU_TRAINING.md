# GPU run guide (RTX 5080 laptop, 64 GB RAM, Core Ultra 9)

Everything that needs local resources is automated by **one command**: `python scripts/gpu_pipeline.py`.

## 1. One-time setup
```bash
git clone https://github.com/PrathamDatta/HACKATHON_CV && cd HACKATHON_CV
git checkout claude/practical-pasteur-0j1ipi          # (or unzip the project zip you were given)
python -m venv .venv                                  # Python 3.11 or 3.12
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128     # RTX 5080 (Blackwell) needs a CUDA 12.8 build
pip install -r requirements.txt                       # torch is already satisfied by the CUDA wheel above, so it is not replaced
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True + RTX 5080
```
If `cuda.is_available()` is False: update the NVIDIA driver (needs a Blackwell-capable driver, 570+), and make sure the wheel is the `cu128` build, not the CPU build. (`requirements.txt` lists `torch` unpinned so the CUDA wheel installed above is kept.)

## 2. Check the whole pipeline (about 5 minutes)
```bash
python scripts/gpu_pipeline.py --smoke --skip-download   # tiny settings; will fail fast if something is broken. Needs data: run step 3's download first, or omit --skip-download.
```
Remove `results/gpu_run/` and `weights/*_L*.pt` afterwards so the real run starts clean.

## 3. The real run
```bash
python scripts/gpu_pipeline.py                       # profile "max" (default)
python scripts/gpu_pipeline.py --profile balanced    # about 5 h
python scripts/gpu_pipeline.py --profile quick       # about 2 h
```
| Profile | Models trained | Budget | Notes |
|---|---|---|---|
| `max` (default) | 2 seeds x (deep-only, hybrid), about 9 M parameters each | 3 h per model, about 13 h total with evaluation | ensemble + flip TTA, full-test evaluation |
| `balanced` | 1 seed x 2 models | 2 h per model | no ensemble |
| `quick` | 1 seed x 2 models | 45 min per model | sanity result |

What it does, in order (each stage writes `results/gpu_run/<stage>.done`; re-running skips finished stages and training resumes from its last checkpoint):
1. download NYU Depth V2 (1.9 GB) and build the memory-mapped cache
2. speed probe: finds the largest batch that fits the GPU and sets an **equal step budget** for every model
3. train `dl_only_L`, `hybrid_L` (and seed-2 `*_L2` for the ensemble): bf16 autocast, EMA weights, crops 256, train frames 0-949, validation frames 950-999, **test frames 1000-1448 never touched**
4. evaluate every arm on the **full 449-frame test split for all three protocols** (GPU inference, parallel CPU workers)
5. CPU latency/memory benchmark (on purpose CPU: that is what a free web host runs)
6. ablations (200 test frames), 7. failure analysis, 8. report + regenerate `eval_P4_T01.md`, `README.md`, `docs/ABLATION.md`, `docs/FAILURES.md`, figures
9. bundle: `gpu_results.zip`

Set the team details for the eval file header before running (optional, can be done later):
```bash
# Windows PowerShell:  $env:TEAM="T07"; $env:USNS="1XX23CS001,1XX23CS002,..."
# Linux/macOS:         export TEAM=T07 USNS="1XX23CS001,1XX23CS002"
```

## 4. Sending the results back
Either push the branch (`git add -A && git commit -m "GPU results" && git push`) or give the Claude session the `gpu_results.zip` file and tell it to extract it into the repository root.

## 5. Troubleshooting
* **CUDA out of memory**: handled automatically in the probe (batch halves). If a *later* stage fails, rerun with `--batch 16 --patch 192`.
* **Windows data-loader errors**: rerun with `--workers 4`.
* **Interrupted**: just run the same command again.
* **Evaluation too slow**: lower `--eval-procs`; or `--ablation-frames 80`.

## 6. Short on time? (custom budget)
`python scripts/gpu_pipeline.py --profile balanced --hours-per-model 1.25 --workers 6 --eval-procs 8` trains one seed of each model for about 1.25 h each (about 2.5 h) plus about 1.5 h of evaluation, roughly 4 h in total. Use `--ablation-frames 80` to trim the ablation stage further.

## 7. Deploying on Hugging Face Spaces (free CPU)
Hugging Face rejects a plain `git push` that contains files over 10 MB without Git LFS (`weights/dl_only_L.pt` is 36 MB). Use the upload API, which handles large files itself:
```bash
pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import HfApi, login
login()                                   # paste a WRITE-access token from huggingface.co/settings/tokens
api = HfApi()
repo = "<your-hf-username>/edgegate"      # pick any name
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path=".", repo_id=repo, repo_type="space",
                  ignore_patterns=[".git/*", "data/*", ".venv/*", "results/gpu_run/*", "weights/*.resume.pt", "*.zip", "handoff/*", "__pycache__/*", "results/logs/*"])
PY
```
The README front matter already declares `sdk: docker` and `app_port: 8501`; the first build takes several minutes (watch the Space's "Logs" tab). Then open the URL, press Reconstruct, upload a bad file once to confirm the error message, and test on a phone. Keep your token out of the repository.
