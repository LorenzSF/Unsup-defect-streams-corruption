# PLAN — JobA val_defect: comparison + tables

> Companion to [PLAN.md](PLAN.md). Scope: analyse the JobA "val_defect"
> re-run against the clean JobA baseline,
> and produce the tables/plots needed for the thesis discussion of
> threshold calibration.

---

## 1. Why this plan exists

The first JobA run (30 categories × 3 models = 90 cells) reported
`val_precision = val_recall = val_f1 = 0` everywhere because the splitter
sent 100% of anomalies to test. With `thresholding.mode: val_quantile`,
threshold = 99th percentile of clean val scores → very conservative →
test recall collapsed (mean ≈ 0.36–0.48) while precision pinned at ~0.99.
AUROC / AUPR were trustworthy; F1 / recall / accuracy were not.

This plan covers the analysis of the second run, which uses:

1. **Patched splitter** — 10% of anomalies routed into val.
2. **`val_f1` thresholding** — uses both classes to pick the operating point.
3. **Restricted scope** — top-10 highest mean-AUPR categories from JobA clean.

Goal of the analysis: prove (or disprove) that the new threshold
calibration recovers F1/recall **without harming AUROC/AUPR**, and
produce publication-ready tables for the thesis methodology section.

---

## 2. What changed in the codebase

| File | Change |
|---|---|
| [src/benchmark_AD/data.py:594-624](src/benchmark_AD/data.py#L594-L624) | `apply_dataset_split` now carves `val_ratio` of bads into val when `train_on_good_only=True`. Train remains one-class. |
| [src/benchmark_AD/configs/colab_featurebased_val_defect.yaml](src/benchmark_AD/configs/colab_featurebased_val_defect.yaml) | New overlay. Same 3 models as `colab_featurebased.yaml` but `thresholding.mode: val_f1` for each. |
| [scripts/run_jobA_val_defect_colab.sh](scripts/run_jobA_val_defect_colab.sh) | Driver that loops only the 10 top-AUPR categories. |
| [scripts/analyze_jobA.py](scripts/analyze_jobA.py) | Aggregator. Already prints top-10 by mean AUPR; will be reused/extended for the comparison. |

Nothing else in the pipeline was modified. The threshold-calibration
fallback in [pipeline.py:250-263](src/benchmark_AD/pipeline.py#L250-L263)
already handles the degenerate case (`val_f1` → falls back to
`val_quantile` if PR curve has no usable point), so the patch is safe.

---

## 3. Expected outputs of the new run

Each category produces `data/outputs/jobA_val_defect_<cat>_<UTC>/benchmark_summary.json`
with the same schema as JobA clean. Key fields to inspect:

- `val_precision`, `val_recall`, `val_f1`, `val_auroc`, `val_aupr` — **must
  now be > 0**. If any of these is still 0, val ended up with no anomalies
  → splitter patch did not run (check that the new code is on the box).
- `threshold_used` — should differ from JobA clean. `val_f1` typically
  picks a lower threshold than `val_quantile` (more permissive).
- `auroc`, `aupr` — should match JobA clean within ±0.01 (run-to-run noise
  from sample-grouping shuffles). **Big deltas here are an alarm**:
  threshold-independent metrics shouldn't move because of a calibration
  change.
- `precision`, `recall`, `f1`, `accuracy` — the actual deliverables.

---

## 5. Comparison methodology

### Pairing
Match by `(category, model)`. JobA clean run path:
`data/outputs/jobA_<cat>_<UTC>/`. Val_defect run path:
`data/outputs/jobA_val_defect_<cat>_<UTC>/`. The category slug is in the
folder name; ignore the timestamp suffix when joining.

### Direction-of-change expectations

| Metric | Expectation | Alarm if... |
|---|---|---|
| AUROC | unchanged (±0.01) | drift > 0.02 → seed/grouping issue |
| AUPR | unchanged (±0.01) | drift > 0.02 → same |
| Recall | **substantially up** (e.g. +0.2 to +0.4) | unchanged or down → val_f1 not active |
| F1 | up | unchanged → see above |
| Precision | mildly down (calibration trade-off) | crashes < 0.5 → threshold too low |
| `threshold_used` | lower than clean | identical to clean → calibration silently fell back to val_quantile |
| `val_*` metrics | non-zero | zero → splitter patch did not run |

### Cells to drop / annotate
- `usb_adaptor` clean: VRAM=0, ms/img×60. Drop from clean→val_defect
  latency comparison; keep for AUROC/AUPR.
- Per-cell anomaly: if `val_recall == 0` in val_defect, log it but do
  not aggregate that cell into the F1/recall delta means.

---

## 6. Tables to generate

All tables will go into [notebooks/benchmark_graphs and tables.ipynb](notebooks/benchmark_graphs%20and%20tables.ipynb).

### Table A — Per-(category, model) clean vs val_defect
Columns: `category, model, AUROC_clean, AUROC_vd, ΔAUROC,
AUPR_clean, AUPR_vd, ΔAUPR, F1_clean, F1_vd, ΔF1,
Precision_clean, Precision_vd, Recall_clean, Recall_vd,
threshold_clean, threshold_vd`. 30 rows.

### Table B — Per-model summary (mean and median over the 10 cats)
Columns: `model, AUROC_clean, AUROC_vd, ΔAUROC, AUPR_clean, AUPR_vd,
F1_clean, F1_vd, ΔF1, Recall_clean, Recall_vd`. 3 rows.
Use this in the thesis as the headline before/after table.

### Table C — Threshold-shift summary
For each model, report mean and stdev of `threshold_vd / threshold_clean`
ratio. Confirms the operating point moved consistently and not erratically.

### Table D — Val-set sanity
Columns: `category, model, val_n_good, val_n_bad, val_f1, val_auroc,
val_aupr, threshold_used`. 30 rows. Used as a methodology appendix to
show val now contains real anomalies.

---

## 7. Plots (optional, for the thesis figures)

- **F1 lift bar chart**: per category, three bars (one per model),
  showing ΔF1 = F1_vd − F1_clean. Sorted by mean lift.
- **Precision–recall scatter**: clean vs val_defect on the same axes,
  arrows from clean point to val_defect point per (category, model).
  Shows the trade-off the new threshold makes.
- **AUROC stability check**: scatter `AUROC_clean` (x) vs `AUROC_vd` (y).
  Should hug y=x; outliers flag random-seed or grouping issues.

Export at 300 DPI `.png` and `.pdf` per [PLAN.md §2.3](PLAN.md).

---

## 8. Step-by-step analysis recipe

1. Confirm all 10 val_defect runs landed under `data/outputs/jobA_val_defect_*/`.
2. Extend [scripts/analyze_jobA.py](scripts/analyze_jobA.py): add a
   `--prefix jobA_val_defect_` flag (or copy to `analyze_jobA_val_defect.py`)
   so the existing aggregator can be reused on the new outputs.
3. Build a `compare_clean_vd.py` that loads both directories, joins on
   `(category, model)`, emits Tables A–D as TSV (so they paste into the
   notebook cleanly).
4. Run sanity checks:
   - Every val_defect cell has `val_recall > 0`.
   - `|ΔAUROC|` median across 30 cells < 0.01.
   - At least 2/3 models per category show ΔF1 > 0.
5. Open [notebooks/benchmark_graphs and tables.ipynb](notebooks/benchmark_graphs%20and%20tables.ipynb)
   and import the TSVs. Render the four tables, the three plots.
6. Write methodology paragraph: splitter change, threshold mode, what the
   numbers prove. Cite Table B for the headline.

---

## 9. Decision points (defer until results land)

- **Roll out to all 30 categories?** Only if Table B shows ΔF1 > 0.10
  averaged across the three models AND |ΔAUROC| < 0.01. Otherwise
  diagnose first.
- **Switch the project default to `val_f1`?** Update
  [src/benchmark_AD/default.yaml:61](src/benchmark_AD/default.yaml#L61)
  and the other config overlays only after the 10-category result is
  positive. Until then, keep `val_quantile` as the documented baseline.
- **JobB (Deceuninck)?** Do not run val_defect on JobB until JobA
  result is confirmed. Same dataset format assumption (`OK/` + `NG/`)
  applies, so the patch is portable.
- **`usb_adaptor` re-inclusion**: check whether the new run's
  `peak_vram_mb > 0` and `ms_per_image` is in the normal range. If yes,
  the latency table can include all 10 cats; if not, exclude again.

---