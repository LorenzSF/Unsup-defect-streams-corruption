# Evaluation of Unsupervised Defect Detection Models on Industrial Data Streams Under Corruption


## Goals

1. Streaming inference and visualization for industrial image streams.
2. Benchmarking SOTA detectors against each other on the same stream:
   each model listed under `model.name` is run in turn with the rest of
   `config.yaml` held fixed, and the resulting `report.json` files
   under `output_dir/<experiment_name>/` form the side-by-side
   comparison.
3. Measuring robustness under synthetic image corruptions.

## Install

```bash
pip install -r requirements.txt
```

`anomalib` pulls the heavy model stack used by the pipeline. If you need a
specific CUDA build, install the matching `torch` / `torchvision` pair first
and then install `requirements.txt`.

## Data Layout

Datasets are not shipped with this repository. The `data/` directory is
git-tracked (via `data/.gitkeep`) but its contents are ignored — download the
dataset locally and lay it out as described below.

The default `config.yaml` targets **Real-IAD** (multi-view industrial anomaly
detection, 30 categories, `OK` / `NG` samples):

- Project page: <https://realiad4ad.github.io/Real-IAD/>
- License: CC BY-NC-SA 4.0 — research use only; access granted via the
  request form on the project page.

Extract the `realiad_1024` split so the tree matches:

```text
data/
  Real-IAD_dataset/
    realiad_1024/
      <category>/             # e.g. audiojack, bottle_cap, ...
        OK/
          <specimen>/
            *.jpg
        NG/
          <defect>/            # e.g. BX, ...
            <specimen>/
              *.jpg
```

Rules:

- `stream.data_root` points at the resolution split (default
  `data/Real-IAD_dataset/realiad_1024`).
- `stream.category` must match a category directory name under `data_root`.
- Normal frames live under `OK/`, anomalous frames under `NG/<defect>/`.
- Images are discovered recursively. Only `.jpg` and `.jpeg` are streamed.
- Any dataset that follows the same `OK/` / `NG/<defect>/` convention works
  by pointing `stream.data_root` and `stream.category` at it.

## Config
Main sections:

- `seed`, `output_dir`, `log_every`
- `stream`: dataset, category, data_root, extensions, shuffle, max_frames
- `warmup`: warmup_steps, use_clean_frames
- `model`: name, backbone, device, checkpoint
- `corruption`: enabled, specs
- `metrics`: window_size, threshold_mode, manual_threshold, pot_risk
- `visualization`: mode, every_n_frames, overlay_alpha, dashboard_enabled, dashboard_host, dashboard_port, dashboard_max_live_points

Current implementation notes:

- `stream.dataset` is a free-form dataset name used in the output dir;
  any dataset under `stream.data_root/<category>/{OK,NG}/...` works.
- `model.name` supports `pca`, `patchcore`, `padim`, `subspacead`, `stfpm`, `csflow`, `draem`, `rd4ad`, and `efficientad`.
- `efficientad` currently expects `model.checkpoint` to point to trained weights.
- `visualization.mode: file` is the default path.
- `visualization.dashboard_enabled: true` runs a FastAPI + WebSocket
  server in a daemon thread alongside `main.py`. Open
  `http://<dashboard_host>:<dashboard_port>/` to see the live dashboard
  (six metric tiles, the current frame with heatmap overlay, and a
  PCA(2) projection of the per-frame embedding with the warm-up frames
  as a reference cloud). The dashboard runs orthogonally to
  `visualization.mode` — any combination is valid. To use it from
  Google Colab, expose the port with `pyngrok` or
  `google.colab.output.serve_kernel_port_as_window`; from a remote
  HPC node, use SSH local port forwarding
  (`ssh -L 8765:localhost:8765 user@host`).
- `metrics.threshold_mode` currently supports `manual` and `pot`.
- `pot` is the unsupervised threshold from Siffer et al. 2017 (KDD,
  "Anomaly Detection in Streams with Extreme Value Theory"). It fits a
  Generalized Pareto Distribution to the upper tail (above the 0.98
  quantile) of the warm-up OK scores and derives the threshold that
  corresponds to a target false-positive rate `metrics.pot_risk`
  (default `1e-3`). It does not require labels. The chosen threshold
  is reported in `report.json` under `threshold.{pot_u, pot_ksi,
  pot_sigma, pot_n_tail, ...}` for traceability. Threshold-free
  evaluation (`auroc`, `aupr` in the report) is unaffected by this
  choice — the threshold only determines the binary metrics
  (`precision`, `recall`, `f1`).

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

The same per-run directory also contains `frames.jsonl` — one JSON object
per frame (`{idx, label, score, latency_ms}`) written line-buffered during
the streaming loop. Use it to recompute metrics at arbitrary thresholds,
inspect OK/NG score distributions, or separate pure model latency from
stream pacing. Load with `pandas.read_json(path, lines=True)`.

Rendered frames from `visualization.mode: file` are written into the same
per-run directory as the report: `output_dir/<experiment_name>/`.

## Active Modules

- [src/schemas.py](src/schemas.py) — dataclasses and strict config loading
- [src/stream.py](src/stream.py) — Dataset input stream construction
- [src/models.py](src/models.py) — model construction and warm-up
- [src/corruption.py](src/corruption.py) — per-frame corruptions
- [src/metrics.py](src/metrics.py) — online metrics
- [src/visualization.py](src/visualization.py) — streaming outputs

## Extending

To add a model, extend the dispatch in [src/models.py](src/models.py).

To add a corruption, register a new kernel in [src/corruption.py](src/corruption.py).

## License

Apache-2.0. See `LICENSE`.
