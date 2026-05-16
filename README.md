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

## Input Data

Datasets are not shipped with this repository. The pipeline input is one flat
folder configured through `stream.input_path`:

```text
data/
  input_images/
    image_0001.jpg
    image_0002.jpg
    image_0003.jpg
    labels.json        # optional, only for benchmark metrics
```

Rules:

- `stream.input_path` must point directly to the folder containing the images.
- Subfolders are not accepted by the pipeline. If your dataset is nested,
  flatten/copy the images you want to stream into the input folder first.
- Images are discovered from that folder only. Only suffixes listed in
  `stream.extensions` are streamed.
- `image_id` is the filename without extension. These ids must be unique.
- If `labels.json` exists, it must be an object mapping `image_id` to exactly
  `"OK"` or `"NG"`:

```json
{
  "image_0001": "OK",
  "image_0002": "NG"
}
```

Missing ids are treated as unknown labels. Precision, recall, F1, accuracy,
AUROC, and AUPR in `report.json` only use frames with labels.

For Real-IAD, first extract the dataset, then prepare a flat input folder from
the images you want to stream. The original Real-IAD tree looks like:

```text
data/
  Real-IAD_dataset/
    realiad_1024/
      <category>/
        OK/
          <specimen>/
            *.jpg
        NG/
          <defect>/
            <specimen>/
              *.jpg
```

## Config

Main sections:

- `seed`, `output_dir`, `log_every`
- `stream`: dataset, input_path, extensions, shuffle, max_frames
- `warmup`: warmup_steps
- `model`: name, backbone, device, checkpoint
- `corruption`: enabled, specs
- `metrics`: window_size, threshold_mode, calibration_steps, initial_threshold, pot_risk
- `visualization`: mode, every_n_frames, overlay_alpha, dashboard_enabled, dashboard_host, dashboard_port, dashboard_max_live_points

Current implementation notes:

- `stream.dataset` is a free-form dataset name used in the output dir.
- Warm-up uses the first `warmup.warmup_steps` images from the configured
  sorted input order.
- Threshold calibration uses the first `metrics.calibration_steps` post-warmup
  frames from that same sorted order. With `stream.shuffle: true`, only the
  remaining post-calibration stream is shuffled.
- `model.name` supports `pca`, `patchcore`, `padim`, `subspacead`, `stfpm`, `csflow`, `draem`, `rd4ad`, and `efficientad`.
- `efficientad` currently expects `model.checkpoint` to point to trained weights.
- `visualization.mode: file` is the default path.
- `visualization.dashboard_enabled: true` runs a FastAPI + WebSocket
  server in a daemon thread alongside `main.py`. Open
  `http://<dashboard_host>:<dashboard_port>/` to see the live dashboard
  (six metric tiles, the current frame with heatmap overlay, and a
  StandardScaler + PCA(2) projection of a per-frame vector combining
  embedding, score, and anomaly-map statistics with the warm-up frames
  as a reference cloud). Live points are colored by predicted label:
  green for `0`, red for `1`. The dashboard runs orthogonally to
  `visualization.mode` - any combination is valid. To use it from
  Google Colab, expose the port with `pyngrok` or
  `google.colab.output.serve_kernel_port_as_window`; from a remote
  HPC node, use SSH local port forwarding
  (`ssh -L 8765:localhost:8765 user@host`).
- `metrics.threshold_mode` currently supports `max_score_ok` and `pot`.
- Supported `corruption.specs[].kind` values are `gaussian_noise`,
  `shot_noise`, `motion_blur`, `defocus_blur`, `brightness`, and `contrast`.
- Both threshold modes start with `metrics.initial_threshold`, score the first
  `metrics.calibration_steps` post-warmup frames, then switch to the calibrated
  threshold for subsequent frames. `max_score_ok` uses the maximum finite score
  from the calibration window and assumes that window is OK-only by construction;
  `pot` fits the unsupervised threshold from Siffer et al. 2017 (KDD, "Anomaly
  Detection in Streams with Extreme Value Theory") using a Generalized Pareto
  Distribution over the upper tail (above the 0.98 quantile). Neither mode
  requires labels for threshold calibration.
  The chosen threshold is reported in `report.json`; POT also reports
  `threshold.{pot_u, pot_ksi, pot_sigma, pot_n_tail, ...}` for traceability.
  Threshold-free evaluation (`auroc`, `aupr` in the report) is unaffected by
  this choice; the threshold only determines the binary metrics
  (`precision`, `recall`, `f1`, `accuracy`).

## Run

```bash
python main.py
```

Reports are written under `output_dir/<experiment_name>/report.json`. The
experiment name is derived per run from
`{model.name}_{stream.dataset}_{input_folder}_{corruption_kind}_s{severity}_{YYYYMMDD-HHMMSS}`,
omitting the corruption block when `corruption.enabled` is false.

Current per-run report fields include:

- image-level quality: `auroc`, `aupr`, `precision`, `recall`, `f1`, `accuracy`
- operational: `mean_latency_ms`, `p95_latency_ms`, `throughput_fps`
- setup/runtime: `runtime.cold_start_s`, `runtime.peak_vram_mb`, `hardware`
- threshold metadata: `threshold.mode`, `threshold.threshold`
- evaluation metadata: calibration frames excluded from benchmark metrics

The same per-run directory also contains `frames.jsonl` - one JSON object
per frame (`{idx, image_id, phase, score, pred_label, threshold_used,
true_label, latency_ms}`) written line-buffered during the run. Warm-up and
threshold-calibration frames are logged with `pred_label: -1` and are excluded
from benchmark metrics. Evaluation frames start only after threshold calibration
has completed. `pred_label` and `true_label` use `0` for OK, `1` for NG, and
`-1` for unknown/unavailable.
Load with `pandas.read_json(path, lines=True)`.

Rendered frames from `visualization.mode: file` are written into the same
per-run directory as the report: `output_dir/<experiment_name>/`.

## Active Modules

- [src/schemas.py](src/schemas.py) - dataclasses and strict config loading
- [src/stream.py](src/stream.py) - input stream construction
- [src/models.py](src/models.py) - model construction and warm-up
- [src/corruption.py](src/corruption.py) - per-frame corruptions
- [src/metrics.py](src/metrics.py) - online metrics and final benchmark report
- [src/visualization.py](src/visualization.py) - streaming outputs and dashboard

## Extending

To add a model, extend the dispatch in [src/models.py](src/models.py).

To add a corruption, register a new kernel in [src/corruption.py](src/corruption.py).

## License

Apache-2.0. See `LICENSE`.
