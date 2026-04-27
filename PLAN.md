# PLAN.md — Thesis Project: Real-Time Visual Defect Detection

> **For Claude Code:** Read this file before making any edit.
> Every code change must serve one of the goals listed here.
> Do not refactor outside task scope. Do not add new dependencies without asking first.
> Prefer simple, readable code over clever abstractions.
>
> **Methodology decisions** (split rules, threshold calibration, metric
> choices) live in [METHOD.md](METHOD.md). Update both files when those
> change.

---

## Project Goals

1. **Benchmarking pipeline** — Evaluate SOTA anomaly detection models on a standard dataset (Real-IAD) and a real industrial dataset (Deceuninck). Produce reproducible, comparable metrics.
2. **Streaming inference + XAI dashboard** — Run the best-performing model in a simulated production environment. Show explainable outputs (heatmaps, live metrics, live embedding plot) suitable for a factory floor operator.
3. **Corruption robustness evaluation** — Apply synthetic image corruptions in both batch and streaming modes to measure how model performance degrades under real-world visual degradations.

---

## Repository Structure (current + planned)

```
.
├── main.py                        # Entry point: batch benchmark pipeline
├── runtime_main.py                # Entry point: streaming inference pipeline
├── pyproject.toml
├── PLAN.md                        # ← this file
│
├── src/
│   ├── benchmark_AD/              # [EXISTS] Batch benchmarking pipeline
│   │   ├── pipeline.py            # Main benchmark orchestrator
│   │   ├── models.py              # Model registry and wrappers
│   │   ├── data.py                # Dataset loading and splitting
│   │   ├── evaluation.py          # Metrics: AUROC, F1, inference time
│   │   └── default.yaml           # Config file for all pipeline options
│   │
│   ├── corruptions/               # [PARTIAL] Synthetic corruption module
│   │   ├── corruption_registry.py # [TO BUILD] Functions per corruption type + severity
│   │   ├── test_loader.py
│   │   ├── test_pipeline_benchmark.py
│   │   └── test_registry.py
│   │
│   └── streaming_input/           # [TO BUILD] Streaming inference module
│       ├── app.py                 # StreamingInputApp — main loop
│       ├── settings.py            # load_settings(), DEFAULT_SETTINGS_FILE
│       ├── inference.py           # Per-frame model inference wrapper
│       └── dashboard.py           # Live XAI dashboard (Streamlit or Dash)
│
├── notebooks/
│   ├── benchmark_graphs and tables.ipynb  # Results analysis
│   └── example_notebook.ipynb
│
└── data/
    └── runs/                      # Auto-generated output per run
```

---

## Coding Conventions

These apply to every file touched or created in this project:

- **Language:** Python 3.11. Type hints on all function signatures.
- **Style:** Follow PEP 8. Max line length: 100 characters.
- **Comments:** Write comments that explain *why*, not *what*. A reader who knows Python should understand the purpose of every block without needing to trace the full codebase.
- **Functions:** One responsibility per function. If a function does more than one logical thing, split it.
- **Naming:** `snake_case` for variables and functions. Descriptive names — avoid `x`, `tmp`, `data2`.
- **Error handling:** Use explicit `try/except` with meaningful messages. Never silently swallow exceptions.
- **No hidden state:** Avoid global variables. Pass config/state explicitly through function arguments.
- **Dependencies:** Only use packages already declared in `pyproject.toml`. Ask before adding new ones.
- **No over-engineering:** No abstract base classes or plugin systems unless already in place. Flat is better than nested.

---

## Status snapshot — 2026-04-27

| Block | State | Notes |
|---|---|---|
| Pipeline + metrics | ✅ done | AUROC/AUPR, Recall@FPR=1%/5%, macro/weighted/per-defect recall, runtime cost — all in `benchmark_summary.json`. |
| JobA clean (val_quantile) | ✅ 30 cats × 3 feature-based + partial trained | Headline numbers untrustworthy on F1/Recall; AUROC valid. |
| **JobA val_defect (val_f1)** | ✅ 19 / 30 cats × 3 feature-based | Cleared §9 gates; default switched to `val_f1` on 2026-04-27. See [data/outputs/jobA_val_defect_V1/_analysis/REPORT.md](data/outputs/jobA_val_defect_V1/_analysis/REPORT.md). |
| JobA val_f1 missing 11 cats | 🟡 todo | porcelain_doll, regulator, tape, toy, toy_brick, u_block, usb, usb_adaptor, vcpill, wooden_beads, woodstick. |
| JobA trained (CSFlow/DRAEM/STFPM/RD4AD on HPC) | 🟡 partial | csflow audiojack only, *and on a pre-val_f1 clone* (image_size 256, mode=val_quantile). Re-run after `git pull` on wICE — see [docs/HPC_KU_LEUVEN_RUNBOOK.md §5.5](docs/HPC_KU_LEUVEN_RUNBOOK.md). |
| JobB Deceuninck clean (val_quantile) | ✅ 1 cell × 3 models | AUROC ≥ 0.995 across the 3 models. |
| **JobB Deceuninck val_f1** | 🟡 todo | Re-run with the new default. Driver: [scripts/run_jobB_val_defect_colab.sh](scripts/run_jobB_val_defect_colab.sh). |
| Corruption module | 🟡 partial | See `src/corruptions/` — registry + tests exist; pipeline wiring + JobC pending. |
| Streaming module + dashboard | ⛔ to build | Block 2 below. |
| Notebook tables/plots | 🟡 partial | Tables A–D (TSV) generated under `_analysis/`; not yet imported into the notebook. |

---

## Work Plan

### 1 — Benchmark + Corruptions + GPU Runs

#### 1.1 — Close the headline benchmark (val_f1 default, full grid)
**Goal:** every cell of the headline table reports AUROC, F1, **Recall@FPR=1%**, **macro/per-defect recall** and inference cost under the same `val_f1` calibration policy.

- ✅ Pipeline runs end-to-end on all supported models.
- ✅ `default.yaml` now uses `val_f1`. All Job A/B configs inherit it; the `*_val_defect.yaml` overlays are kept as historical aliases.
- 🟡 **JobA val_f1 — finish the remaining 11 cats** (porcelain_doll, regulator, tape, toy, toy_brick, u_block, usb, usb_adaptor, vcpill, wooden_beads, woodstick). Reuse [scripts/run_jobA_val_defect_colab.sh](scripts/run_jobA_val_defect_colab.sh) (or its successor) and append outputs into `data/outputs/jobA_val_defect_V1/`.
- 🟡 **JobA trained — wICE re-run.** `git pull` on the HPC clone (so `colab_trained.yaml` picks up `val_f1` and `image_size: 512` from `default.yaml`), then re-launch `csflow audiojack` and extend to the 30 categories × 4 trained models matrix described in [HPC_KU_LEUVEN_RUNBOOK.md §6–7](docs/HPC_KU_LEUVEN_RUNBOOK.md).
- 🟡 **JobB Deceuninck val_f1.** Re-run via [scripts/run_jobB_val_defect_colab.sh](scripts/run_jobB_val_defect_colab.sh); confirm the F1/Recall lift on Deceuninck mirrors the Real-IAD result (or document the difference if not).
- 🟡 **Re-run regression cell.** `plastic_plug × PaDiM` showed ΔAUROC = −0.096 in the val_defect rerun; re-launch with a fresh seed to confirm whether the drop is noise or real before publishing the table.
- After results: regenerate Tables A–D via [scripts/compare_clean_vd.py](scripts/compare_clean_vd.py) and identify the **best-performing model** by inspecting the consolidated `benchmark_summary.json` set manually → this name is then passed to `runtime_main.py --model <name>` for streaming. Never auto-select.

#### 1.2 — Build corruption module
**Goal:** `src/corruptions/corruption_registry.py` integrated into the batch pipeline.

- Implement the following corruption types, each as a standalone function accepting `(image: np.ndarray, severity: int) -> np.ndarray` where severity ∈ {1, 2, 3, 4, 5}:
  - `gaussian_noise` — additive pixel noise
  - `gaussian_blur` — spatial blurring
  - `motion_blur` — directional blur simulating camera motion
  - `brightness_shift` — uniform exposure change
  - `contrast_reduction` — reduce dynamic range
  - `jpeg_compression` — lossy compression artifact

- Expose a `get_corruption(name, severity)` factory function as the public API.
- Wire into `default.yaml` via existing `corruption:` section (key: `type`, `severity`).
- Quick test: one model, one corruption, one dataset category.

#### 1.3 — Launch corruption benchmark + start streaming module
**Goal:** Job C running on GPU, JSON corruption outputs saved, `streaming_input/` ready.

- Configure `src/benchmark_AD/default.yaml` and per-dataset configs (`configs/realiad.yaml`, `configs/industrial.yaml`) for:
  - Full model list, same as Jobs A/B. No automatic model selection.
  - `corruption:` enabled with all 6 types from §1.2 and severities `{1, 3, 5}`.
- Launch on GPU cluster:
  - **Job C:** Batch benchmark with corruptions, all models × 6 corruptions × severities `{1, 3, 5}`, on Real-IAD and Deceuninck.
- Save Job C outputs under `data/runs/<run_name>/`, one session folder per `(model × corruption × severity × dataset)`, using JSON only:
  - `predictions.json`: per-frame records `{model, path, label, defect_type, score, pred_is_anomaly, heatmap_path, corruption_type, severity, dataset}`.
  - `live_status.json`: session summary with streaming status keys (`active_model`, `frames_seen`, `decisions_emitted`, `mean_latency_ms`, `p95_latency_ms`, `threshold`, ...) plus `AUROC`, `F1`, `corruption_type`, `severity`, `dataset`.
  - Shape mirrors `data/streaming_input/streaming_input_20260411_160237/`.
- Start building `src/streaming_input/`:
  - `settings.py`: keep public surface `DEFAULT_SETTINGS_FILE`, `load_settings(path) -> dict`.
  - `inference.py`: wrap the selected model to process a single image and return `{anomaly_score, anomaly_map, embedding}`.
  - `app.py`: `StreamingInputApp` class — init loads model, `run()` iterates over image source (folder or simulated camera), calls inference per frame, saves `predictions.json` and `live_status.json`.
  - `dashboard.py`: empty placeholder until §2.1.
- Clean `src/streaming_input/` to contain only `app.py`, `settings.py`, `inference.py`, `dashboard.py`, and `__init__.py`:
  - Fold folder input handling, model loading, inference contracts, and session-output writing into the flat module structure.
  - Defer dashboard, live metrics, reports, and web app concerns to §2.1.
  - Update `__init__.py` to export only `StreamingInputApp`, `DEFAULT_SETTINGS_FILE`, `load_settings`.
  - Update `runtime_main.py` for manual model selection via CLI flags such as `--model <name>` and `--run-dir <path>`.
- Verify `StreamingInputApp().run()` iterates a folder source, calls inference per frame, and writes the JSON outputs above.
---

### 2 — Streaming + Dashboard + Results

#### 2.1 — Live XAI dashboard
**Goal:** `src/streaming_input/dashboard.py` — a dashboard that runs alongside the streaming loop and updates in real time.

Dashboard must show (in a single screen, readable by a factory operator):

| Panel | Content |
|---|---|
| Current frame | Input image with anomaly heatmap overlaid (colormap: green→red) |
| Anomaly score | Numeric gauge or bar with decision threshold marked |
| Inference throughput | FPS counter (rolling average over last 10 frames) |
| Anomaly rate | % of frames classified as anomalous since session start |
| Score history | Line chart of the last N anomaly scores (N configurable, default 100) |
| Live embedding plot | 2D scatter updated each frame — new points colored by score |

**Implementation notes for the embedding plot:**
- Pre-fit a UMAP or PCA projection on the training set embeddings (offline, once at startup).
- Each new frame: extract embedding → project using the pre-fit transformer → add point to scatter.
- Color scale: green (score near 0) → red (score near threshold and above).
- No ground-truth labels are available at inference time — color by score only.

**Recommended stack:** Streamlit (simple, already available). Use `st.empty()` placeholders and `time.sleep()` loop for live updates. If Streamlit is not suitable, use Plotly Dash with a background thread.

#### 2.2 — Streaming with corruptions
**Goal:** The streaming loop can optionally apply corruptions frame-by-frame, reusing the same registry from Block 2.

- Add `--corruption` and `--severity` flags to `runtime_main.py`.
- In `app.py`, if corruption is configured: apply `get_corruption(name, severity)` to each frame before inference.
- This enables direct comparison: dashboard metrics with vs. without corruption active.
- Capture dashboard screenshots for thesis figures.

#### 2.3 — Results analysis and thesis writing
**Goal:** All figures and tables needed for the thesis are generated and exported.

- Import the val_defect TSVs already generated into `notebooks/benchmark_graphs and tables.ipynb`:
  - `data/outputs/jobA_val_defect_V1/_analysis/tableA_per_cat_model.tsv`
  - `tableB_per_model.tsv`, `tableC_threshold_shift.tsv`, `tableD_val_sanity.tsv`
- Update the notebook with the post-rerun run outputs (clean baseline + 30/30 val_f1 cells once §1.1 is closed).
- Generate the following figures:
  1. **Threshold-calibration evidence** — clean vs val_f1 per-model bar chart (F1 / Recall lift, AUROC stability). Direct support for METHOD.md §2.
  2. **Model comparison table:** AUROC and F1 on Real-IAD (val_f1) vs Deceuninck (val_f1).
  3. **F1 lift bar chart and PR scatter** described in [PLAN job A_analize val_defect.md §7](PLAN%20job%20A_analize%20val_defect.md).
  4. **Robustness curves:** AUROC vs corruption severity, per corruption type (line chart, output of §1.3 / Job C).
  5. **Streaming dashboard screenshots:** clean run and corrupted run side by side.
  6. **UMAP embedding plot snapshot** from a streaming session (if available).
- Export all figures at 300 DPI (`.png`) and `.pdf`.
- Write or complete in thesis:
  - **Methodology:** pipeline architecture, threshold-calibration policy (val_f1 + patched splitter, justified in METHOD.md §2-3), corruption module design, streaming setup.
  - **Results:** model selection rationale, robustness analysis, dashboard demonstration.

---

## Integral action map — what's left to ship the thesis

Sequenced by dependency (each block unblocks the next). Status: ✅ done · 🟡 in progress · ⛔ blocked / not started.

### Phase A — Lock the headline numbers (val_f1 default)

1. ✅ Switch `default.yaml` to `val_f1`. Annotate clean configs.
2. 🟡 **JobA val_f1 — finish the 11 missing cats**
   (porcelain_doll, regulator, tape, toy, toy_brick, u_block, usb,
   usb_adaptor, vcpill, wooden_beads, woodstick) on Colab via
   [scripts/run_jobA_val_defect_colab.sh](scripts/run_jobA_val_defect_colab.sh).
   Output dir: `data/outputs/jobA_val_defect_V1/`.
3. 🟡 **Re-run plastic_plug × PaDiM** with a fresh seed. The 19-cat rerun
   showed ΔAUROC = −0.096 there; confirm noise vs. real regression
   before the headline table goes into the thesis.
4. 🟡 **JobB Deceuninck val_f1** via
   [scripts/run_jobB_val_defect_colab.sh](scripts/run_jobB_val_defect_colab.sh).
   Compare against the clean (val_quantile) Deceuninck baseline using
   [scripts/compare_val_defect.py](scripts/compare_val_defect.py).
5. 🟡 **HPC `git pull`** on `/data/leuven/.../Real-time-visual-defect-detection`,
   then re-run `anomalib_csflow audiojack` and extend to the
   30 cats × 4 trained models matrix. Sanity check: the resulting
   `benchmark_summary.json` must show `threshold_mode: val_f1` and
   `image_size: 512`.
6. 🟡 **Regenerate Tables A–D** by re-running
   [scripts/compare_clean_vd.py](scripts/compare_clean_vd.py) once the
   30 cells are complete. Update
   [data/outputs/jobA_val_defect_V1/_analysis/REPORT.md](data/outputs/jobA_val_defect_V1/_analysis/REPORT.md).

### Phase B — Robustness (Job C, corruptions)

7. 🟡 Finalize [src/corruptions/corruption_registry.py](src/corruptions/corruption_registry.py)
   with the 6 functions from §1.2 and `get_corruption(name, severity)` factory.
8. 🟡 Wire the corruption block into the pipeline (test-set only, training/val stay clean).
9. ⛔ **Launch Job C** on GPU: 6 corruption types × {1, 3, 5} severities
   × 3 feature-based models × Real-IAD (subset acceptable per fallback) + Deceuninck.
10. ⛔ Aggregate Job C → robustness curves (AUROC vs severity per corruption).

### Phase C — Streaming + dashboard

11. ⛔ **Pick the production model** by reading the consolidated headline
    table (do not auto-select). PaDiM is the current frontrunner on both
    quality and cost; confirm after the 30/30 val_f1 grid is complete.
12. ⛔ Build [src/streaming_input/](src/streaming_input/) — `app.py`,
    `settings.py`, `inference.py` per §1.3. Wire `runtime_main.py
    --model <name>` for manual selection.
13. ⛔ Build [src/streaming_input/dashboard.py](src/streaming_input/dashboard.py)
    per §2.1 (current frame + heatmap, score gauge, FPS, anomaly rate,
    score history, live UMAP embedding).
14. ⛔ Add `--corruption` / `--severity` to `runtime_main.py` for the
    corrupted-stream comparison (§2.2). Capture dashboard screenshots
    (clean + corrupted) for the thesis figures.

### Phase D — Thesis figures and writing

15. ⛔ Open `notebooks/benchmark_graphs and tables.ipynb`, import the
    Phase A TSVs, and render:
    a. Calibration evidence (clean vs val_f1).
    b. Model comparison (Real-IAD val_f1 vs Deceuninck val_f1).
    c. F1 lift bar chart + PR scatter per
       [PLAN job A_analize val_defect.md §7](PLAN%20job%20A_analize%20val_defect.md).
    d. Robustness curves from Job C.
    e. Streaming dashboard screenshots.
    f. UMAP embedding snapshot.
16. ⛔ Export every figure at 300 DPI `.png` + `.pdf` per
    [PLAN job A_analize val_defect.md §2.3](PLAN%20job%20A_analize%20val_defect.md).
17. ⛔ Methodology chapter — pipeline architecture, val_f1 + splitter
    policy (cite METHOD.md §2-3), corruption module design, streaming
    setup. Limitations table from
    [METHOD.md §8](METHOD.md#8-open-methodological-issues-limitations-to-disclose).
18. ⛔ Results chapter — model selection rationale, headline benchmark
    table, robustness analysis, dashboard demonstration.

### Critical-path gates (block downstream phases)

- **Phase A → B**: Job C corruption sweep should reuse the chosen
  calibration (val_f1) so its numbers stack on top of Phase A's
  headline table. Do not start Job C until §A.6 is done.
- **Phase B → C**: streaming model selection must be informed by the
  *robust* AUROC (degradation under corruption), not just the clean
  AUROC. Wait until §B.10 is aggregated.
- **Phase C → D**: dashboard screenshots are required figures (§2.1);
  no thesis figure pass before the dashboard runs end-to-end.

---

## Fallback Priorities (if time runs short)

| Component | Required for thesis? | Fallback |
|---|---|---|
| Batch benchmark (clean) | ✅ Yes | — |
| Corruption batch benchmark | ✅ Yes | Reduce to 3 types, severities 1/3/5 |
| Streaming inference loop | ✅ Yes | Folder-based, no live camera |
| XAI dashboard | ✅ Yes | Static heatmap figures only |
| Live embedding plot | ⚠️ Desirable | Static UMAP from a recorded session |
| Streaming + corruptions | ✅ Yes | Single corruption type only |
| Interactive demo | ❌ Optional | Skip entirely |

---

## Key Output Files (what the thesis needs)

```
data/runs/<run_name>/
├── benchmark_summary.json              # Model comparison table + run context.
│                                       # Per-model entry now includes:
│                                       #   recall_at_fpr_1pct, recall_at_fpr_5pct,
│                                       #   macro_recall, weighted_recall,
│                                       #   per_defect_recall, per_defect_support
├── predictions_<model>.json            # Per-frame records, streaming-shape JSON (with corruption_type, severity, dataset)
├── live_status_<model>.json            # Per-(model × corruption × severity × dataset) session summary, streaming-shape JSON (with AUROC, F1)
├── plots/
│   ├── robustness_curves.png           # AUROC vs severity per corruption type
│   ├── model_comparison.png            # Clean benchmark comparison
│   ├── heatmap_sample_<model>.png      # XAI heatmap examples
│   └── embedding_umap_<model>.html     # Embedding plot
└── streaming_session/
    ├── dashboard_clean.png             # Dashboard screenshot, no corruption
    └── dashboard_corrupted.png         # Dashboard screenshot, with corruption
```

Robustness results are emitted as JSON only (mirroring `data/streaming_input/<session>/`), not CSV.

---

## Out of Scope (do not implement)

- Training new models from scratch (use pretrained weights only).
- Support for video streams or live camera input (folder simulation is sufficient).
- Any web API or REST endpoint.
- Docker deployment or CI/CD pipeline changes.
- Unit test coverage beyond what already exists in `tests/`.
