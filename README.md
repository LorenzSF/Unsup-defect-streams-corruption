# Evaluation of Unsupervised Defect Detection Models on Industrial Data Streams Under Corruption


## Goals

1. Streaming inference and visualization for industrial image streams.
2. Benchmarking a main detector against an online baseline on the same stream.
3. Measuring robustness under synthetic image corruptions.

## Install

```bash
pip install -r requirements.txt
```

`anomalib` pulls the heavy model stack used by the pipeline. If you need a
specific CUDA build, install the matching `torch` / `torchvision` pair first
and then install `requirements.txt`.

## Data Layout

The active stream loader expects extracted Real-IAD categories under:

```text
data/
  NAME_dataset/
      Category of the piece/
        OK/
            img1.jpg
        NG/
          Defect name/
              img1.jpg
```

Rules:

- `stream.category` must match the category directory name.
- Normal frames live under `OK/`.
- Anomalous frames live under `NG/<defect_name>/`.
- Images are discovered recursively.
- Only `.jpg` and `.jpeg` files are streamed.

## Config
Main sections:

- `seed`, `output_dir`, `log_every`
- `stream`: dataset, category, data_root, extensions, shuffle, max_frames
- `warmup`: warmup_steps, use_clean_frames
- `model`: name, backbone, device, checkpoint
- `corruption`: enabled, specs
- `metrics`: window_size, threshold_mode, manual_threshold, threshold_quantile
- `visualization`: mode, every_n_frames, overlay_alpha
- `benchmark`: enabled, baseline, learning_rate

Current implementation notes:

- `stream.dataset` is a free-form dataset name used in the output dir;
  any dataset under `stream.data_root/<category>/{OK,NG}/...` works.
- `model.name` supports `pca`, `patchcore`, `padim`, `subspacead`, `stfpm`, `csflow`, `draem`, `rd4ad`, and `efficientad`.
- `efficientad` currently expects `model.checkpoint` to point to trained weights.
- `visualization.mode: file` is the default path.
- `metrics.threshold_mode` currently supports `manual` and `quantile`.
- `quantile` re-scores the warm-up OK frames after fit and sets the
  threshold to `np.quantile(scores, metrics.threshold_quantile)`. The
  default `threshold_quantile: 1.0` uses the max OK score (strict).

## Run

```bash
python main.py
```

Reports are written under `output_dir/<experiment_name>/report.json`. The
experiment name is derived per run from
`{model.name}_{stream.dataset}_{stream.category}_{corruption_kind}_s{severity}_{YYYYMMDD-HHMMSS}`,
omitting the corruption block when `corruption.enabled` is false.

Current per-run report fields include:

- image-level quality: `auroc`, `aupr`, `precision`, `recall`, `f1`, `accuracy`
- operational: `mean_latency_ms`, `p95_latency_ms`, `throughput_fps`
- setup/runtime: `runtime.cold_start_s`, `runtime.peak_vram_mb`
- threshold metadata: `threshold.mode`, `threshold.threshold`

Rendered frames from `visualization.mode: file` are written into the same
per-run directory as the report: `output_dir/<experiment_name>/`.

## Active Modules

- [src/schemas.py](src/schemas.py) — dataclasses and strict config loading
- [src/stream.py](src/stream.py) — Dataset input stream construction
- [src/models.py](src/models.py) — model construction and warm-up
- [src/corruption.py](src/corruption.py) — per-frame corruptions
- [src/metrics.py](src/metrics.py) — online metrics
- [src/visualization.py](src/visualization.py) — streaming outputs
- [src/benchmark.py](src/benchmark.py) — online baseline

## Extending

To add a model, extend the dispatch in [src/models.py](src/models.py).

To add a corruption, register a new kernel in [src/corruption.py](src/corruption.py).

## License

Apache-2.0. See `LICENSE`.
