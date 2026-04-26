---
name: wICE / KU Leuven HPC setup for this thesis
description: Cluster-specific paths, env vars, and quotas for running jobA on wICE / genius (VSC)
type: project
---

The user runs thesis compute on **VSC (Flemish Supercomputer Center)**, accessed via **OnDemand** (https://ondemand.hpc.kuleuven.be). Confirmed working configuration as of 2026-04-25: cluster `wice`, partition `gpu_a100`, 1 node, 1 process, 1 GPU, 16000 MB/core, 2 h interactive wall.

**Why:** No KU Leuven VPN → no external SSH from the user's PC → all work happens through OnDemand sessions. CSFlow on Real-IAD audiojack (C1-only, 354 train samples) runs at ~8 s/epoch on A100, ~33 min for full 240 epochs.

**How to apply:** Reuse these paths verbatim at the start of every wICE session. Do not place caches in `$HOME` (only 3 GB and almost full).

**Repo (data partition, persists, OnDemand file browser sees this):**
- `/data/leuven/381/vsc38124/Real-time-visual-defect-detection`

**Scratch (475 GB, OnDemand file browser does NOT see this, can be purged):**
- Datasets: `/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/<category>.zip`
- uv toolchain: `/scratch/leuven/381/vsc38124/uv-x86_64-unknown-linux-gnu`
- Run outputs: `/scratch/leuven/381/vsc38124/runs/`
- uv cache, venv, python: under `/scratch/leuven/381/vsc38124/.uv_cache`, `.venv`, `.uv_python`

**Per-session env (must export every new shell):**
```
export PATH="/scratch/leuven/381/vsc38124/uv-x86_64-unknown-linux-gnu:$PATH"
export UV_CACHE_DIR=/scratch/leuven/381/vsc38124/.uv_cache
export UV_PROJECT_ENVIRONMENT=/scratch/leuven/381/vsc38124/.venv
export UV_PYTHON_INSTALL_DIR=/scratch/leuven/381/vsc38124/.uv_python
export PYTHONPATH=src
cd /data/leuven/381/vsc38124/Real-time-visual-defect-detection
```

**Toolchain notes:**
- `pip` not directly available; use `uv sync --extra gpu` (CUDA 12.4 wheels per [pyproject.toml:51-62](../pyproject.toml#L51-L62)).
- 2 h interactive wall is enough for ONE category × ONE model on C1-only data. All-cameras (5×) blows past 2 h on CSFlow — use `sbatch` for those.

**Proven dataset layout on scratch:** `unzip -q <cat>.zip` from within `realiad_1024/` produces single-nested `<cat>/OK/...` and `<cat>/NG/...`. Pre-existing extracted folders may have double nesting (`<cat>/<cat>/OK`); if so, re-extract cleanly rather than pointing deeper.
