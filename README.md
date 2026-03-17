# Real-Time Visual Defect Detection

## Project Goals

- Build a reproducible visual anomaly detection benchmark pipeline.
- Compare multiple models under the same data split and corruption settings.
- Keep CPU/GPU execution explicit and reproducible (including NVIDIA devices).

## Current Models

- `dummy_distance`
- `rd4ad`
- `anomalib_patchcore`
- `anomalib_padim`
- `anomalib_stfpm`
- `anomalib_csflow`
- `anomalib_draem`
- `subspacead` (DINOv2 + PCA residual scoring)

## Pipeline Flow

`scripts/main.py` calls `run_pipeline()` in `src/real_time_visual_defect_detection/pipelines/run_pipeline.py`.

1. Load YAML/JSON config.
2. Resolve dataset from folder or ZIP.
3. Build train/val/test split.
4. Resolve runtime/device (`auto`, `cpu`, `cuda`, `cuda:<id>`).
5. Fit on train, optionally calibrate threshold on val, evaluate on test.
6. Save per-model predictions and benchmark summary.

## Repository Structure

```text
.
|-- scripts/main.py
|-- src/real_time_visual_defect_detection/
|   |-- config/default.yaml
|   |-- pipelines/run_pipeline.py
|   |-- models/
|   |   |-- registry.py
|   |   |-- anomalib_adapter.py
|   |   |-- subspacead_model.py
|   |   `-- rd4ad/
|   |-- io/
|   |-- preprocessing/
|   |-- evaluation/
|   `-- visualization/
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
`-- pyproject.toml
```

## Requirements

- Python `>=3.11,<3.12`
- PyTorch + torchvision
- Anomalib `>=2.2,<3.0`
- Transformers (for SubspaceAD)

## Installation

### Option A: pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Option B: uv

```powershell
uv sync
```

## Configuration Overview

Default config: `src/real_time_visual_defect_detection/config/default.yaml`

- `run`: seed/output/run name
- `runtime`: device, precision, workers, CUDA tuning
- `dataset`: source + split policy
- `dataset.split`: includes `test_ratio` and `val_ratio`
- `preprocessing`: resize + normalization
- `corruption`: synthetic corruption controls
- `model`: single-model mode, with `thresholding` policy
- `benchmark.models`: multi-model benchmark mode

## Run the Pipeline

Interactive launcher (single command):

```powershell
$env:PYTHONPATH='src'
python scripts/main.py
```

The launcher will prompt for:

- dataset path (config/history/detected candidates)
- model name (from registry: `dummy_distance`, `rd4ad`, `anomalib_patchcore`, `anomalib_padim`, `anomalib_stfpm`, `anomalib_csflow`, `anomalib_draem`, `subspacead`)
- press `Q` in menus to exit immediately

Direct (non-interactive) run:

```powershell
$env:PYTHONPATH='src'
python scripts/main.py --config src/real_time_visual_defect_detection/config/default.yaml --dataset-path "C:\path\to\dataset" --model anomalib_patchcore --no-interactive
```

Before the run starts, the launcher performs model-specific dependency preflight checks.
During execution, the pipeline prints `Stage X/Y` start/end messages and progress bars for fit/inference loops.

Interactive dataset selection:

```powershell
$env:PYTHONPATH='src'
python scripts/main.py --choose-dataset --choose-model
```

## Outputs

Each run creates:

```text
data/runs/<run_name>_<UTC_YYYYMMDD_HHMMSS>/
```

Artifacts:

- `config_snapshot.json`
- `runtime_info.json`
- `validation_predictions_<model>.json`
- `predictions_<model>.json`
- `predictions.json` (single-model compatibility)
- `benchmark_summary.json`
- `benchmark_summary.csv`
- `plots/embedding_umap_<model>.html` (if embeddings are available)

## NVIDIA / GPU Usage

Set runtime in config:

```yaml
runtime:
  device: "cuda"
```

The run writes `runtime_info.json` with:

- CUDA availability
- selected device
- GPU name
- torch/cuda versions

If `runtime.device` is `cuda` and CUDA is unavailable, the run fails fast.

## Docker Compose

Available services:

- `dev`
- `benchmark` (CPU/default)
- `benchmark-gpu` (requires NVIDIA runtime)
- `gpu-smoke` (quick CUDA check inside container)
- `test`

Examples:

```powershell
docker compose run --rm benchmark
docker compose --profile bench-gpu run --rm gpu-smoke
docker compose --profile bench-gpu run --rm benchmark-gpu
docker compose run --rm test
```

## Testing

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
pytest -q -p no:cacheprovider
```

## License

Apache-2.0 (see `LICENSE`).
