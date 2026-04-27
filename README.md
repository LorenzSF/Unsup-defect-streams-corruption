# Real-Time Visual Defect Detection


## What This Project Does

- Runs a reproducible anomaly-detection benchmark pipeline on image datasets.
- Supports folder and ZIP dataset sources.
- Splits data into train/val/test with one-class-friendly defaults.
- Trains and evaluates one or multiple models in a single run.
- Saves per-model predictions, metrics summaries, runtime metadata, and optional UMAP plots.

## Current Repository Structure

```text
.
|-- main.py
|-- pyproject.toml
|-- Dockerfile
|-- docker-compose.yml
|-- src/
|   |-- benchmark_AD/
|   |   |-- __init__.py
|   |   |-- data.py
|   |   |-- evaluation.py
|   |   |-- models.py
|   |   |-- pipeline.py
|   |   |-- default.yaml
|   |   `-- configs/
|   |       |-- realiad.yaml
|   |       `-- industrial.yaml
|   `-- corruptions/
|       |-- test_loader.py
|       |-- test_pipeline_benchmark.py
|       `-- test_registry.py
`-- notebooks/
    |-- benchmark_graphs and tables.ipynb
    `-- example_notebook.ipynb
```

## Supported Models

- `rd4ad`
- `anomalib_patchcore`
- `anomalib_padim`
- `anomalib_stfpm`
- `anomalib_csflow`
- `anomalib_draem`
- `subspacead` (DINOv2 + PCA residual scoring)

## Python And Dependencies

- Python: `>=3.11,<3.12` (declared in `pyproject.toml`)
- Core dependencies include: `torch`, `torchvision`, `anomalib`, `lightning`, `transformers`, `opencv-python`, `scikit-learn`, `plotly`, `umap-learn`

Install with pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Or with uv:

```powershell
uv sync
```

## Configuration

Base config:

```text
src/benchmark_AD/default.yaml
```

Per-dataset overlays (apply only the fields that differ from the base, via the
top-level `_extends` key resolved by `load_config`):

```text
src/benchmark_AD/configs/realiad.yaml      # standard benchmark (Real-IAD)
src/benchmark_AD/configs/industrial.yaml   # Deceuninck industrial dataset
```

Main config sections:

- `run`: seed, output directory, run name
- `runtime`: device/precision/runtime tuning
- `dataset`: source (`folder` or `zip`), path, extraction, split config
- `preprocessing`: resize/normalization
- `model`: single-model defaults and thresholding policy
- `benchmark.models`: list of models for multi-model benchmark runs (empty by
  default; populated automatically when `--all-models` is passed)

Thresholding modes currently used by pipeline:

- `fixed`
- `val_f1`
- `val_quantile`


## Run The Pipeline

`main.py` is non-interactive. All choices are passed via flags or config files.
Set `PYTHONPATH=src` so the `benchmark_AD` package resolves.

### Single model

```powershell
$env:PYTHONPATH='src'
python main.py --config src/benchmark_AD/default.yaml `
  --model anomalib_patchcore `
  --dataset-path "C:\path\to\dataset_or_zip" `
  --run-name patchcore_smoke
```

### All models (every registered entry that passes dependency preflight)

```powershell
$env:PYTHONPATH='src'
python main.py --config src/benchmark_AD/default.yaml `
  --all-models `
  --dataset-path "C:\path\to\dataset_or_zip" `
  --run-name allmodels_run
```

### Per-dataset overlays

```powershell
# Job A — Real-IAD (standard benchmark), all models
$env:PYTHONPATH='src'
python main.py --config src/benchmark_AD/configs/realiad.yaml --all-models

# Job B — Deceuninck (industrial dataset), all models
$env:PYTHONPATH='src'
python main.py --config src/benchmark_AD/configs/industrial.yaml --all-models
```

CLI behavior highlights:

- `--model` and `--all-models` are mutually exclusive.
- Performs model dependency preflight; unavailable models are skipped (with
  reason logged to stderr) instead of aborting the whole run.
- `dataset.path` is required; the pipeline aborts early if it is missing or
  the path does not exist.
- Fails fast when device config is invalid (for example CUDA requested but
  unavailable).

### Corruption (test-set robustness)

Corruptions live in `src/corruptions/corruption_registry.py`. They are applied
to test images only (training and validation stay clean so threshold
calibration is unaffected). Enable from the CLI:

```powershell
$env:PYTHONPATH='src'
python main.py --config src/benchmark_AD/configs/industrial.yaml `
  --all-models --corruption gaussian_blur --severity 3 `
  --run-name industrial_gblur_s3
```

Or set `corruption.enabled: true` in the YAML. Available types: `gaussian_blur`,
`motion_blur`, `jpeg_compression` (the three essentials per PLAN.md §1.2).
Severity is an integer in `1..5`. Each summary row in
`benchmark_summary.json` carries `corruption_type` and `corruption_severity`
so robustness curves can be built directly from the file. The same
`--corruption` / `--severity` flags exist on `runtime_main.py` for symmetry;
the streaming app picks them up in block 2.2.

## Dataset Conventions

Label discovery priority in `data.py`:

1. JSON labels file at dataset root (well-known names like `labels.json`, `annotations.json`, `metadata.json`)
2. Directory convention with `good/` and one of `bad/defects/defective/anomaly/anomalous`
3. Flat fallback where all images are unlabeled (`label = -1`)

Supported image extensions:

- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

## Outputs

Each run creates:

```text
data/outputs/<run_name>_<UTC_YYYYMMDD_HHMMSS>/
```

Main artifacts:

- `runtime_info.json`
- `benchmark_summary.json`
- `validation_predictions_<model>.json`
- `predictions_<model>.json`
- `plots/embedding_umap_<model>.html` (only when embeddings are available)



## License

Apache-2.0. See `LICENSE`.
