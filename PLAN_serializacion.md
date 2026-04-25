# Plan: Checkpoint Serialization for Trained Adapters

## 1. Context

The benchmark pipeline currently learns model state in RAM and discards it on
process exit. A run dir under `data/outputs/<run_name>_<UTC>/` only stores
`benchmark_summary.json` and per-model predictions — **no weights**.
Consequence: every time `src/streaming_input/app.py` starts, it rebuilds the
model from the summary's config and re-runs `fit()` on the historical dataset
(`fit_policy: auto` in [src/streaming_input/settings.yaml](src/streaming_input/settings.yaml)).

Cost of the re-fit varies by adapter:

| Adapter | Re-fit cost | Worth checkpointing? |
| --- | --- | --- |
| `anomalib_patchcore` | feature extraction + coreset (~30s–2min) | marginal |
| `anomalib_padim` | feature extraction + Gaussian fit (~1min) | marginal |
| `anomalib_stfpm` | real backprop, N epochs | **yes** |
| `anomalib_csflow` | real backprop, N epochs | **yes** |
| `anomalib_draem` | real backprop, N epochs | **yes** |
| `subspacead` | DINOv2 forward + PCA (~1min) | marginal |
| `rd4ad` | real backprop, N epochs (now an anomalib wrapper) | **yes** |

### Scope decision (from chat 2026-04-25)

Job A_trained on Colab produces 4 models × 30 Real-IAD categories = 120
`(model, category)` pairs. We do **not** want to checkpoint all 120 — only the
champion pairs that will drive the streaming demo. The `.done` markers in
`scripts/run_jobA_trained_colab.sh` are per-pair, so champions can be
re-targeted by deleting their markers and re-running selectively. **Do not
interrupt the in-flight Job A_trained run** — wait for completion, then iterate
on a small champion subset with checkpointing enabled.

> **Update 2026-04-25:** the RD4AD stub was replaced with a real
> `anomalib.reverse_distillation` wrapper (`AnomalibRd4adModel`). RD4AD now
> joins STFPM/CS-Flow/DRAEM as a trained model that needs checkpointing.

## 2. Goal

Persist enough learned state per `(model, category)` pair that the streaming
loop can run with `fit_policy: skip_fit` and skip the re-fit entirely. Backwards
compatible with existing runs (no checkpoint → fall back to current re-fit
path).

## 3. Design

### 3.1 On-disk layout

```
data/outputs/<run_name>_<UTC>/
├── benchmark_summary.json
├── predictions_<model>.json
└── checkpoints/
    └── <model_name>.ckpt        # torch.save payload, see §3.3
```

The `checkpoints/` subdir is created by the pipeline only if at least one
adapter implements `save_checkpoint`. `benchmark_summary.json` records a
`checkpoint_path` field per model row (relative to the run dir) so the stream
can locate it without scanning the filesystem.

### 3.2 Adapter API (extend `BaseModel` in [src/benchmark_AD/models.py](src/benchmark_AD/models.py))

```python
class BaseModel:
    def save_checkpoint(self, path: Path) -> None:
        """No-op default. Override in adapters that have learnable state."""
        return None

    def load_checkpoint(self, path: Path) -> None:
        """No-op default. Adapters that override must set self._is_fitted = True."""
        return None
```

Default no-op (rather than `NotImplementedError`) keeps PaDiM/PatchCore-style
adapters compatible while we only roll this out for the three trained models.

### 3.3 Per-adapter payload

All payloads are a single dict serialized via `torch.save` with
`map_location='cpu'` on load. Each payload includes a `schema_version: 1` key
and an `anomalib_version` string captured from `anomalib.__version__` at save
time, so future loads can fail fast on incompatible upgrades.

| Adapter | Payload keys |
| --- | --- |
| `AnomalibStfpmModel` | `state_dict` (full model — student weights are what was trained, teacher is reconstructable but cheap to ship), `backbone`, `layers`, `image_size` |
| `AnomalibCsflowModel` | `state_dict`, `cross_conv_hidden_channels`, `n_coupling_blocks`, `clamp`, `image_size` |
| `AnomalibDraemModel` | `state_dict` (reconstruction + segmentation networks), `image_size` |
| `AnomalibRd4adModel` | `state_dict` (encoder is pretrained-frozen but ship it for parity; bottleneck + decoder are what was trained), `backbone`, `layers`, `image_size`, `anomaly_map_mode` |
| `AnomalibPatchcoreModel` (optional, phase 2) | `memory_bank` tensor, `backbone`, `layers`, `coreset_sampling_ratio`, `image_size` |
| `AnomalibPadimModel` (optional, phase 2) | `gaussian.mean`, `gaussian.inv_covariance`, `n_features`, `backbone`, `layers`, `image_size` |

The `threshold_used` is **not** stored in the checkpoint — it already lives in
`benchmark_summary.json` and must remain the single source of truth so it can
be edited without rewriting the binary artifact.

### 3.4 Pipeline integration ([src/benchmark_AD/pipeline.py](src/benchmark_AD/pipeline.py))

After the existing `model.fit(...)` call inside `_run_single_model`, add:

```python
ckpt_dir = out_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / f"{model_name}.ckpt"
try:
    model.save_checkpoint(ckpt_path)
    summary_row["checkpoint_path"] = f"checkpoints/{model_name}.ckpt"
except Exception as exc:
    # Non-fatal: record the failure, continue without checkpoint.
    summary_row["checkpoint_error"] = str(exc)
```

Failure to save must not abort the benchmark — metrics are still valid without
a checkpoint.

### 3.5 Stream integration ([src/streaming_input/inference.py](src/streaming_input/inference.py))

Modify `FrameInference._prepare_model` so the decision tree becomes:

1. Read `checkpoint_path` from the summary row.
2. If a checkpoint file exists at `<run_dir>/<checkpoint_path>`:
   - call `self.model.load_checkpoint(...)`,
   - skip `fit()` regardless of `fit_policy`.
3. Else, fall back to the current logic (`auto` / `historical_fit` / `skip_fit`
   error path).

`fit_policy: skip_fit` becomes a hard assertion that a checkpoint must exist
— useful for production-style stream runs where a silent re-fit would be a
regression.

## 4. Implementation phases

| Phase | Deliverable | Files |
| --- | --- | --- |
| **1. Adapter API** | Default no-op `save_checkpoint` / `load_checkpoint` on `BaseModel` | `src/benchmark_AD/models.py` |
| **2. Trained adapters** | Override save/load in STFPM, CS-Flow, DRAEM, RD4AD | `src/benchmark_AD/models.py` |
| **3. Pipeline write** | Persist checkpoint after fit, record path in summary | `src/benchmark_AD/pipeline.py` |
| **4. Stream load** | Prefer checkpoint over re-fit; tighten `skip_fit` semantics | `src/streaming_input/inference.py` |
| **5. Champion re-run** | Pick winners, delete `.done` markers, rerun on Colab | `scripts/run_jobA_trained_colab.sh` (no code change) |
| **6. Feature-based (opt.)** | Override save/load for PatchCore + PaDiM if disk budget allows | `src/benchmark_AD/models.py` |

Phases 1–4 are mergeable as a single PR. Phase 5 is operational. Phase 6 is
deferred until the trained-model flow is proven in the streaming demo.

## 5. Risks and mitigations

- **anomalib version drift.** A future `pip install -U anomalib` may rename
  layer keys and break stored `state_dict`s. *Mitigation:* pin
  `anomalib==<current>` in `requirements.txt`, embed the version string in the
  payload, refuse to load on mismatch with a clear error.
- **Cross-device load.** Checkpoints are written from Colab GPU and loaded on
  the local Windows machine (CPU or different GPU). *Mitigation:* always pass
  `map_location='cpu'` on load, then move the model to the configured device.
- **Disk footprint.** PatchCore memory banks at `coreset_sampling_ratio=0.1`
  on Real-IAD reach hundreds of MB per category. *Mitigation:* Phase 6 only
  triggers for chosen champions, never for the full 4×30 matrix.
- **Pickle trust.** `torch.save` uses pickle. *Mitigation:* checkpoints are
  produced and consumed by the same project; do not load checkpoints from
  untrusted sources.
- **RD4AD legacy results.** The `rd4ad` adapter was a stub returning
  `np.var(x)` until 2026-04-25; it is now an anomalib `reverse_distillation`
  wrapper ([memory/project_rd4ad_stub.md](memory/project_rd4ad_stub.md)). Any
  pre-2026-04-25 RD4AD benchmark numbers must be discarded — do **not** treat
  old `.done` markers as authoritative when picking champions.

## 6. Acceptance criteria

- [ ] `BaseModel` exposes default-noop `save_checkpoint` / `load_checkpoint`.
- [ ] STFPM, CS-Flow, DRAEM, RD4AD adapters round-trip through save → load and
  produce identical predictions on a fixed batch (within fp32 tolerance).
- [ ] Benchmark pipeline writes `<run>/checkpoints/<model>.ckpt` and records
  `checkpoint_path` in `benchmark_summary.json` for each trained model row.
- [ ] Stream with `fit_policy: skip_fit` loads from checkpoint when present
  and raises a clear error when absent (no silent re-fit fallback under
  `skip_fit`).
- [ ] One champion `(model, category)` pair re-run end-to-end on Colab
  produces a usable checkpoint, synced to Drive, then to local
  `data/outputs/`, then served by the streaming app without re-fit.
- [ ] Existing runs without `checkpoint_path` continue to work (regression
  guard for pre-checkpoint runs already on disk).

## 7. Out of scope

- A registry-style checkpoint store decoupled from run dirs.
- Cross-version compatibility shims (mismatched `anomalib_version` is a hard
  failure for now).
- Quantization or compression of stored weights.
