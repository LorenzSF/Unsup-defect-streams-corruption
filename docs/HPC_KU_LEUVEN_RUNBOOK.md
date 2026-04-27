# HPC KU Leuven (VSC) — JobA_trained runbook

How to run one `(model, category)` pair from `jobA_trained` on **wICE / gpu_a100** through the **OnDemand** web portal. Distilled from the 2026-04-25/26 sessions that brought up CSFlow and DRAEM on `audiojack`.

This runbook documents the current `jobA_trained` path implemented by
[wice_trained.yaml](../src/benchmark_AD/configs/wice_trained.yaml): camera
`C1`, good-only training, and threshold calibration with `val_f1`. The
patched splitter in `data.py` routes a small anomaly holdout into validation,
so `val_f1` sees both classes while the one-class training setup stays intact.

> **Audience:** future-you, no VPN access, runs everything through https://ondemand.hpc.kuleuven.be.

---

## 1. Cluster session request

OnDemand → **Interactive Apps** → **VSCode Server** (or Desktop). Use these values:

| Field | Value | Why |
|---|---|---|
| Cluster | `wice` | Has A100 partition; faster than `genius` V100 for these adapters |
| Partition | `gpu_a100` | A100 — finishes one C1 model in well under 2 h |
| Number of hours | `2` (interactive) / `8-12` (batch sweeps) | C1-only one-model fits in 2 h. All-cameras (5×) does not. |
| Number of nodes | `1` | The pipeline is single-node. |
| Processes per node | `1` | The driver runs models sequentially per category. |
| Memory per core | `16000` MB | GPU is the bottleneck; 16 GB host memory per core is plenty. |
| GPUs | `1` | Adapters are single-device. |

**Do not** request more GPUs — they sit idle. **Do not** request multiple nodes — there's no multi-node code path.

---

## 2. Per-session setup (paste at the start of every fresh terminal)

```bash
export PATH="/scratch/leuven/381/vsc38124/uv-x86_64-unknown-linux-gnu:$PATH"
export UV_CACHE_DIR=/scratch/leuven/381/vsc38124/.uv_cache
export UV_PROJECT_ENVIRONMENT=/scratch/leuven/381/vsc38124/.venv
export UV_PYTHON_INSTALL_DIR=/scratch/leuven/381/vsc38124/.uv_python
export PYTHONPATH=src
cd /data/leuven/381/vsc38124/Real-time-visual-defect-detection
```

Sanity-check the GPU and venv (only when you suspect something is off — skip on a known-good day):

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True NVIDIA A100 ...`. If `False`, you're not on a GPU node — abandon and re-launch the OnDemand session.

If `uv sync --extra gpu` has never been run on this account, run it once. It populates `/scratch/leuven/381/vsc38124/.venv` with the CUDA 12.4 wheels (see [pyproject.toml:51-62](../pyproject.toml#L51-L62)).

---

## 3. Run one `(model, category)` pair

### 3a. Pick the category and extract its zip

Real-IAD ships as `<category>.zip` files under `/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/`. Pick a category — the alphabetically-first is `audiojack` — and extract from inside that directory so the layout is single-nested (`<cat>/OK/...`, **not** `<cat>/<cat>/OK/...`):

```bash
CATEGORY=bottle_cap
cd /scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/

# Wipe any half-extracted folder that may exist from a previous broken run
rm -rf "${CATEGORY}"

# Extract
unzip -q "${CATEGORY}.zip"

# Verify
ls "${CATEGORY}"                                              # must show: OK  NG
find "${CATEGORY}/OK" -type f -name "*.jpg" | wc -l           # > 0
find "${CATEGORY}/NG" -type f -name "*.jpg" | wc -l           # > 0
```

The 30 Real-IAD categories: `audiojack`, `bottle_cap`, `button_battery`, `end_cap`, `eraser`, `fire_hood`, `mint`, `mounts`, `pcb`, `phone_battery`, `plastic_nut`, `plastic_plug`, `porcelain_doll`, `regulator`, `rolled_strip_base`, `sim_card_set`, `switch`, `tape`, `terminalblock`, `toothbrush`, `toy`, `toy_brick`, `transistor1`, `u_block`, `usb`, `usb_adaptor`, `vcpill`, `wooden_beads`, `woodstick`, `zipper`.

### 3b. Set the dataset path and return to the repo

```bash
DATASET_ROOT=/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/${CATEGORY}
cd /data/leuven/381/vsc38124/Real-time-visual-defect-detection
```

### 3c. Run the model — one line, no `\` continuations

The four trained models in `jobA_trained` are:

| `--model` value | Epochs | Batch | Per-epoch on A100 (C1) | Total wall (C1) |
|---|---|---|---|---|
| `anomalib_stfpm` | 100 | 8 | ~3-5 s | ~10 min |
| `anomalib_csflow` | 240 | 8 | ~8 s | ~33 min |
| `anomalib_draem` | 200 | 8 | ~9 s | ~30 min |
| `rd4ad` | 200 | 16 | ~10-15 s | ~35-50 min |

Launch (paste this on **one line** — the OnDemand terminal mangles `\` continuations):

```bash
MODEL=anomalib_draem
uv run python main.py --config src/benchmark_AD/configs/wice_trained.yaml --model "$MODEL" --dataset-path "$DATASET_ROOT" --extract-dir "$DATASET_ROOT" --run-name "jobA_${MODEL}_${CATEGORY}"
```

**Use `wice_trained.yaml`, not `colab_trained.yaml`.** The wICE overlay sets `output_dir: /scratch/.../runs` and `num_workers: 4` while inheriting everything else from the colab base. No sed-edits required.

### 3d. What you should see in the first ~2 minutes

- `[Benchmark] Running model: <model>`
- `[<model>] Stage 1/5 fit: started | train_samples=354` (354 = audiojack C1; varies per category)
- `[<model>] fit 1/N: ...`

Steady-state per-epoch wall time × N epochs ≈ total training time. If the first non-warmup epoch is materially slower than the table above, you're probably on `cameras: "all"` — check [src/benchmark_AD/configs/colab_trained.yaml:29](../src/benchmark_AD/configs/colab_trained.yaml#L29) (or use `--all-cameras` only intentionally with longer wall time).

---

## 4. Retrieve the run outputs

Run dirs land in `/scratch/leuven/381/vsc38124/runs/jobA_<model>_<category>_<UTC-timestamp>/`. They're tiny (~250 KB of JSON) for the trained models.

### 4a. Inspect inline (no download needed for the headline number)

```bash
RUN=$(ls -1dt /scratch/leuven/381/vsc38124/runs/jobA_${MODEL}_${CATEGORY}_* | head -n1)
cat "$RUN/benchmark_summary.json"
```

The metrics that matter for cross-model comparison:

- **`auroc`** — primary. Threshold-independent. Paper-standard for Real-IAD. Use this as the headline.
- **`aupr`** — secondary. Threshold-independent.
- **`ms_per_image`**, **`fps`**, **`peak_vram_mb`** — runtime characterisation for the real-time chapter.
- `f1`, `precision`, `recall` — useful as a *high-precision operating point* result but read with caveats (see §6).

### 4b. Download via OnDemand File Browser

The File Browser only sees `/data/...`, not `/scratch/...`. Bridge:

```bash
mkdir -p /data/leuven/381/vsc38124/downloads
cp -r "$RUN" /data/leuven/381/vsc38124/downloads/
```

OnDemand → **Files** → navigate to `/data/leuven/381/vsc38124/downloads/` → tick the run folder → **Download**.

Clean up after a successful local copy so you don't blow `/data` quota:

```bash
rm -rf /data/leuven/381/vsc38124/downloads/jobA_${MODEL}_${CATEGORY}_*
```

`/scratch` keeps the original until purge.

---

## 5. Lessons learned — pitfalls of the OnDemand environment

### 5.1 The OnDemand web terminal mangles long pasted commands

Confirmed failure modes (multiple times in one session):

- **Heredocs** (`cat > file <<'EOF' ... EOF`) write the **literal `cat`/`<<'EOF'`/`EOF` lines** into the file. Bash never executes them.
- **Long single-line `printf '...' > file`** invocations get mangled the same way.
- **Backslash line continuations (`\` at end of line)** silently drop. Each `--flag` becomes its own command, producing `bash: --flag: command not found`.
- **Ctrl+C inside a stuck heredoc** terminates the SSH connection (Slurm job survives, just reconnect from the dashboard).

**How to write/edit files reliably:**

- **Use `nano`** for multi-line edits — paste into the editor, Ctrl+O Enter Ctrl+X. Interactive editor, no shell interpretation.
- **Use `sed -i 's|old|new|' file`** for short single-substitution edits. One short command, zero quoting drama.
- **Never paste heredocs.**
- **Avoid `\` continuations**; put commands on one line, after `cd`-ing into a short relative path.

### 5.2 Quota landscape

| Mount | Capacity | Visible in OnDemand File Browser? | Purged? |
|---|---|---|---|
| `$HOME` (`/user/...`) | ~3 GB | yes | no |
| `/data/leuven/381/vsc38124/...` | ~50 GB | yes | no |
| `/scratch/leuven/381/vsc38124/...` | ~475 GB | **no** | yes (periodic) |

**Implications:**
- All caches (`UV_*`, torch hub, HF) MUST go to `/scratch`. The env exports in §2 do this.
- Repo lives in `/data` (committed code). Run outputs live in `/scratch` (regenerable).
- To download: copy `/scratch → /data` first, then File Browser.

### 5.3 Dataset extraction nesting

Real-IAD `<category>.zip` archives contain a top-level `<category>/` folder. So:

- `cd /scratch/.../realiad_1024/ && unzip -q audiojack.zip` produces `realiad_1024/audiojack/OK/...` ✓
- `unzip -q audiojack.zip -d audiojack/` produces `audiojack/audiojack/OK/...` ✗ (double-nested, will trip the loader)

**Always extract from the parent dir.** If you find a double-nested folder, `rm -rf` it and re-extract from the parent.

### 5.4 Camera filter

[colab_trained.yaml](../src/benchmark_AD/configs/colab_trained.yaml) sets `dataset.cameras: "C1"` to keep one of the five Real-IAD camera angles. This is consistent with the published Real-IAD benchmarks and gives ~354 training samples per category — comfortable inside 2 h interactive wall time.

`cameras: "all"` (5× more data, ~1770 train samples) gives ~3 h wall time on CSFlow/DRAEM. Use `sbatch` with longer wall, not interactive sessions.

### 5.5 Threshold calibration

The current split (`train_on_good_only: true`) keeps training good-only but
routes a `val_ratio` slice of NG samples into validation. Consequence:

- `val_f1` now calibrates on both classes, which matches the current methodology baseline in [METHOD.md](../METHOD.md).
- `val_quantile` is no longer the `jobA_trained` default.
- If a category somehow reaches validation with only one class, `val_f1` falls back to `val_quantile` (see [pipeline.py:311-345](../src/benchmark_AD/pipeline.py#L311-L345)).

→ **Use AUROC for cross-model comparison first**, and report F1/precision/recall as calibrated operating-point metrics under the explicit `val_f1` policy.

### 5.6 Cache locations and first-run cost

Every model's first invocation downloads its backbone weights to `$TORCH_HUB / $HF_HOME`. With the env exports in §2 these go to `/scratch/.../.uv_cache` (or wherever uv sets them). Subsequent runs reuse the cache — you only pay this once per backbone (e.g. EfficientNet-B5 for CSFlow, wide_resnet50_2 for RD4AD/anomalib base, resnet18 for STFPM/PaDiM).

---

## 6. Cross-model consistency

For the four `jobA_trained` models to give **directly comparable** numbers across categories:

| Variable | Keep constant | Source |
|---|---|---|
| Camera filter | `C1` | inherited from `colab_trained.yaml` |
| Image size / preprocessing | 256 / ImageNet normalize | shared anomalib block |
| Threshold mode | `val_f1` | shared `model.thresholding` |
| Splits / seed | `test_ratio: 0.2`, `val_ratio: 0.1`, `seed: 42` | `default.yaml` |
| Per-model hyperparams | epochs/lr from `colab_trained.yaml` | per-model block (paper-comparable) |

Run the same 30 categories with each of the four models and tabulate `auroc` from `benchmark_summary.json`. That's the comparable cross-model row.

---

## 7. Scaling up: from one pair to all 30 × 4

Sitting through 120 interactive runs is impractical. Use `sbatch` for the full sweep:

- Wall: 8-12 h per (model × 30 categories).
- One job per model, four jobs in parallel — they don't share GPUs.
- The dedicated wICE driver [scripts/run_jobA_trained_wice.sh](../scripts/run_jobA_trained_wice.sh) already implements the per-category loop with `.done` markers for resumability and the correct `/scratch` + `uv` environment exports.
- What is still missing is only the **Slurm wrapper** (`sbatch`) around that script. The batch job should call `bash scripts/run_jobA_trained_wice.sh <model> <cat1> ... <catN>` after the env exports from §2.

---

## 8. Quick reference card

```bash
# 1. Per-session env (paste once per terminal)
export PATH="/scratch/leuven/381/vsc38124/uv-x86_64-unknown-linux-gnu:$PATH"
export UV_CACHE_DIR=/scratch/leuven/381/vsc38124/.uv_cache
export UV_PROJECT_ENVIRONMENT=/scratch/leuven/381/vsc38124/.venv
export UV_PYTHON_INSTALL_DIR=/scratch/leuven/381/vsc38124/.uv_python
export PYTHONPATH=src
cd /data/leuven/381/vsc38124/Real-time-visual-defect-detection

# 2. Pick category + extract
CATEGORY=audiojack
cd /scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/
rm -rf "${CATEGORY}" && unzip -q "${CATEGORY}.zip"
DATASET_ROOT=/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/${CATEGORY}
cd /data/leuven/381/vsc38124/Real-time-visual-defect-detection

# 3. Run (single line, no backslashes)
MODEL=anomalib_draem
uv run python main.py --config src/benchmark_AD/configs/wice_trained.yaml --model "$MODEL" --dataset-path "$DATASET_ROOT" --extract-dir "$DATASET_ROOT" --run-name "jobA_${MODEL}_${CATEGORY}"

# 4. Inspect headline metric
RUN=$(ls -1dt /scratch/leuven/381/vsc38124/runs/jobA_${MODEL}_${CATEGORY}_* | head -n1)
cat "$RUN/benchmark_summary.json"

# 5. Stage for download
mkdir -p /data/leuven/381/vsc38124/downloads && cp -r "$RUN" /data/leuven/381/vsc38124/downloads/
# Then: OnDemand → Files → /data/leuven/381/vsc38124/downloads/ → Download
```
