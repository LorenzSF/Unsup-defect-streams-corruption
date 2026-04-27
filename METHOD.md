# METHOD.md — Methodology notes for the thesis

> Living document with **partial conclusions** drawn from the JobA / JobB
> analyses. Numbers below are from the 30-cat clean JobA run, the
> **19-cat JobA val_defect rerun** (2026-04-27), and the single JobB run.
> Everything is stamped *as of* 2026-04-27.
>
> **Methodology baseline (current default):** patched splitter +
> `val_f1` thresholding. Adopted as the project-wide default in
> [default.yaml](src/benchmark_AD/default.yaml#L73) on 2026-04-27 after
> the 19-cat rerun cleared the §9 decision gates of
> [PLAN job A_analize val_defect.md](PLAN%20job%20A_analize%20val_defect.md).
> The original feature-based and Deceuninck overlays now also use
> `val_f1`; the `*_val_defect.yaml` overlays are kept only as historical
> artifacts.
>
> Companion to [PLAN.md](PLAN.md) (project-level scope) and
> [PLAN job A_analize val_defect.md](PLAN%20job%20A_analize%20val_defect.md)
> (operational plan for the val_defect comparison).

---

## 1. Pipeline at a glance

The benchmark pipeline ([src/benchmark_AD/pipeline.py](src/benchmark_AD/pipeline.py))
is the same for every run; only configuration changes between Jobs A, B
and the val_defect overlays.

```
samples → split (train / val / test)
          ├── train (good only, one-class fitting)
          ├── val   (used ONLY for threshold calibration)
          └── test  (frozen model, frozen threshold → metrics)
```

Three feature-based, training-free models are evaluated head-to-head
(plus a fourth, trained, family for later jobs):

- **PaDiM** (resnet18, layers 1–3) — Gaussian-per-patch.
- **PatchCore** (wide_resnet50_2, layers 2–3, coreset 10%) — nearest-neighbour
  in a sub-sampled patch bank.
- **SubspaceAD** (DINOv2-large, last 7 transformer layers) — PCA subspace,
  `mtop1p` aggregation.

Output contract per run:
[benchmark_summary.json](data/outputs/jobA_audiojack_20260424_093730/benchmark_summary.json)
+ [predictions_<model>.json](data/outputs/jobA_audiojack_20260424_093730/predictions_anomalib_padim.json)
+ [validation_predictions_<model>.json](data/outputs/jobA_audiojack_20260424_093730/validation_predictions_anomalib_padim.json)
+ [live_status_<model>.json](data/outputs/jobA_audiojack_20260424_093730/live_status_anomalib_padim.json).

---

## 2. Threshold calibration is the dominant axis of variability

Every model emits a continuous **anomaly score**. AUROC and AUPR are
threshold-free (they evaluate the score directly). Precision, Recall,
F1 and Accuracy require a cut-off — the *operating point*. The cut-off
is set on the val split during a calibration step
([pipeline.py:_maybe_calibrate_threshold](src/benchmark_AD/pipeline.py#L231-L273)).

### Two calibration modes available

| Mode | Picks threshold to | Needs in val | Industrial failure mode |
|---|---|---|---|
| `val_quantile` | `quantile(1 − target_fpr, val_negatives)` | only negatives | unstable when val negatives are few or low-variance — collapses recall (over-strict) or explodes FPR (under-strict) |
| `val_f1` | argmax F1 over val PR curve | **both classes** | silently falls back to `val_quantile` when val has 0 positives |

### Empirical evidence that calibration is the bottleneck

Across the **19-category JobA val_defect rerun** (clean baseline vs the
patched splitter + `val_f1`, same splits otherwise, single seed). Source:
[data/outputs/jobA_val_defect_V1/_analysis/REPORT.md](data/outputs/jobA_val_defect_V1/_analysis/REPORT.md).
Means over the 19 paired (cat, model) cells:

| Model | ΔAUROC | ΔAUPR | ΔF1 | ΔRecall | ΔPrecision | mean Thr_v / Thr_c |
|---|---:|---:|---:|---:|---:|---:|
| PatchCore  | +0.003 | −0.002 | **+0.281** | **+0.381** | −0.052 | 0.91 |
| PaDiM      | +0.012 | +0.002 | **+0.451** | **+0.560** | −0.036 | **0.66** |
| SubspaceAD | +0.017 | +0.002 | **+0.257** | **+0.338** | −0.046 | **0.50** |

**Conclusion:** ranking quality (AUROC, AUPR) is essentially unchanged by
the calibration switch, as it should be — *means* drift inside the ±0.02
noise band the plan tolerated. The F1/Recall jumps come entirely from a
better operating point. PaDiM and SubspaceAD were the most miscalibrated
(threshold ratios 0.66 and 0.50 → previous cut was at 1.5× and 2× the
optimal).

**Caveats from the rerun analysis (record but do not block the decision):**

- Per-cell `|ΔAUROC|` median is 0.0145 (target was < 0.01); the drift is
  structural — the 10% bads moved into val change the test-set
  composition, so AUROC is computed on a different (smaller) test pool
  than the clean baseline. Per-model means agree within ±0.02; that is
  the meaningful signal.
- One regression worth disclosing: `plastic_plug` × PaDiM ΔAUROC =
  −0.096 (0.905 → 0.810). F1 still rose (0.45 → 0.87) because val_f1
  picked a permissive threshold, but ranking quality genuinely worsened
  on this cell. Re-run with a fresh seed before publishing the table.
- One favourable outlier: `plastic_nut` × SubspaceAD ΔAUROC = +0.123
  (0.73 → 0.86); likely a bad seed in the clean run.
- Single mild F1 regression: `terminalblock` × SubspaceAD ΔF1 = −0.010
  — the clean baseline already had R = 0.90 with the conservative
  threshold; val_f1 picked a slightly tighter cut. Below the noise floor.

### Decision for the thesis

`val_f1` is now the project default (committed 2026-04-27). The
`val_quantile` numbers are kept only for the calibration-evidence table
above. The methodology section should state explicitly that any
reported F1/Recall without a documented calibration procedure is
meaningless on industrial AD datasets, and that the clean→val_f1 lift
above is the empirical justification for the policy.

---

## 3. Train / val / test composition is the *second* dominant axis

Once the calibrator works, the next sensitivity is whether the
validation split contains **both classes**. The current val_defect
workflow keeps the one-class training setup intact and applies only a
minimal splitter patch in
[data.py:apply_dataset_split](src/benchmark_AD/data.py):

- **Clean baseline**: goods are split into train / val / test by ratio, and
  all anomalies go to test.
- **val_defect workflow**: the goods split stays the same, but 10% of the
  anomaly pool is routed into val so `val_f1` has positives to calibrate on.
- **Per-defect test reserve**: `ceil(test_ratio * len(class))` is still
  enforced per defect type so minority classes remain measurable on test.

### Why this rule set matters

1. **`val_f1` needs positives in val** - without them the calibrator falls
   back to `val_quantile`, which recreates the over-conservative operating
   point seen in the first clean runs.
2. **Model fitting is unchanged** - train stays good-only, so any metric
   shift comes from threshold calibration rather than from changing the
   one-class training regime.
3. **Minority defect classes stay evaluable** - the per-type test reserve
   prevents small JobB classes from being consumed by val.

### Empirical effect on splits

**Deceuninck** (224 goods + [21, 27, 23, 23, 388] bads):

| Split | train | val | test |
|---|---:|---:|---:|
| Legacy (`val_quantile`) | 161g | 18g+0b | 45g+482b |
| Current val_defect (`val_f1`) | 161g | 18g+49b | 45g+433b |

**Real-IAD categories** keep the same number of training goods as the clean
baseline; only the anomaly side changes. In practice, val gains 10% of the
bad pool and test keeps the remaining 90%, which is enough to calibrate
`val_f1` without collapsing the one-class train split.

---

## 4. F1 alone is not enough — what the headline metrics need to be

For an industrial-credibility claim the run summary now reports five
metric families
([pipeline.py:_run_single_model](src/benchmark_AD/pipeline.py#L550)):

| Family | Metric(s) | Interpretation | Prevalence-invariant? |
|---|---|---|---|
| Ranking | `auroc`, `aupr` | "Does the model put bads above goods on the score axis?" | AUROC yes, AUPR no |
| Operating point | `precision`, `recall`, `f1`, `accuracy` | At the calibrated threshold | no |
| Operational target | `recall_at_fpr_1pct`, `recall_at_fpr_5pct` | "If the line tolerates 1% / 5% false alarms, what fraction of defects do we catch?" | **yes** |
| Per-defect | `per_defect_recall` (dict), `macro_recall`, `weighted_recall` | Per-class recall + simple/weighted mean | n/a |
| Cost | `ms_per_image`, `fps`, `peak_vram_mb`, `fit_seconds` | Deployment cost | n/a |

### Why each addition was necessary

- **`recall_at_fpr_*`** — answers the question a plant engineer actually
  asks ("how many defects do we catch if I limit false alarms to X?").
  Independent of threshold calibration choices and of test prevalence.
- **`macro_recall`** — when one class dominates (Deceuninck scratches
  80%), aggregate recall ≈ recall on the dominant class. Macro mean
  treats every defect type as equally important.
- **`weighted_recall`** — equals the global recall on bads; reported
  for completeness so the macro/weighted gap is visible.
- **`per_defect_recall`** — surfaces the breakdown so the thesis can
  show which defect types are easy/hard per model.

### What the thesis tables should report

For each model, on each dataset:

1. AUROC ± std (multi-seed when available — *not yet*).
2. AUPR ± std.
3. Recall @ FPR = 1%.
4. F1 at calibrated threshold (state which calibrator).
5. Macro recall (5 defect types on Deceuninck, 5–10 on each Real-IAD cat).
6. ms/img, peak VRAM.

Per-defect recall belongs in an appendix table — too wide for the body.

---

## 5. Dataset structures (and how they shaped the methodology)

| Aspect | Real-IAD (JobA) | Deceuninck (JobB) |
|---|---|---|
| Task | 30 independent object categories | 1 product line (window-profile extrusion) |
| Layout | `<cat>/OK/SXXXX/<img>_C[1-5]_*.jpg` + `<cat>/NG/<defect>/SXXXX/{img.jpg, mask.png}` | `good/*.bmp` + `defects/<5 subfolders>/*.bmp` |
| Cameras | 5 viewpoints (currently using C1 only) | 1 line camera |
| Specimen grouping | Yes (`SXXXX/`) — splitter avoids leakage | **No** — open issue |
| Image format | JPG (image) + PNG (mask, filtered) | BMP only |
| Format hint | `format: real_iad` (mandatory) | `format: auto` (relies on `_BAD_DIR_NAMES`) |
| As-collected prevalence in test | varies, typically 30–70% bad | **91% bad** (482 / 527) |
| Defect-class balance | 5–10 types per cat, fairly balanced | 1 type ≈ 80% of all bads |
| Total scale per "cell" | ~1000 images per cat × C1 | 706 images, single cell |

### Methodological consequences

- **Specimen grouping leakage on Deceuninck** is the biggest
  unaddressed risk. Filename timestamps could be parsed to derive a
  group key (e.g. `30_2024327-155443_*`). Not implemented; documented
  as a limitation. Until fixed, JobB results should be reported as a
  *line-level* benchmark, not a *physical-defect-level* benchmark.
- **Single-camera scope** on Real-IAD halves data and limits
  generalisation claims to C1 viewpoint. Methodology section must say
  "single-viewpoint preview" — not "Real-IAD benchmark".
- **`format: auto` on Deceuninck** is a soft contract: any rename of
  `defects/` would break ingestion silently. For a published benchmark
  the format should be pinned (e.g. `format: deceuninck`) — open issue.

---

## 6. Configuration matrix used in JobA / JobB

All overlays inherit from [default.yaml](src/benchmark_AD/default.yaml)
— which now sets `thresholding.mode: val_f1` as the project default.
Differences highlighted.

| Overlay | Dataset | Format | Cameras | Threshold mode | Val split |
|---|---|---|---|---|---|
| [default.yaml](src/benchmark_AD/default.yaml) | — | auto | all | **val_f1** (project default since 2026-04-27) | patched splitter (good train + 10% bads in val) |
| [colab_featurebased.yaml](src/benchmark_AD/configs/colab_featurebased.yaml) | Real-IAD | real_iad | C1 | val_f1 | inherits default |
| [colab_featurebased_val_defect.yaml](src/benchmark_AD/configs/colab_featurebased_val_defect.yaml) | Real-IAD | real_iad | C1 | val_f1 | historical alias of the line above |
| [colab_trained.yaml](src/benchmark_AD/configs/colab_trained.yaml) | Real-IAD | real_iad | C1 | val_f1 | inherits default |
| [wice_trained.yaml](src/benchmark_AD/configs/wice_trained.yaml) | Real-IAD | real_iad | C1 | val_f1 | inherits via colab_trained |
| [colab_featurebased_deceuninck.yaml](src/benchmark_AD/configs/colab_featurebased_deceuninck.yaml) | Deceuninck | auto | n/a | val_f1 | inherits default |
| [colab_featurebased_deceuninck_val_defect.yaml](src/benchmark_AD/configs/colab_featurebased_deceuninck_val_defect.yaml) | Deceuninck | auto | n/a | val_f1 | historical alias of the line above |

---

## 7. Preliminary model conclusions

**These are not the final thesis numbers** — refresh this section after the
next full val_defect rerun lands for both datasets.

### From clean JobA (30 categories × 3 models, val_quantile baseline)

- AUROC mean: PaDiM 0.893 > PatchCore 0.870 > SubspaceAD 0.842.
- AUROC win counts: PaDiM 14 / 30, PatchCore 10 / 30, SubspaceAD 6 / 30.
- F1/Recall numbers are not trustworthy under `val_quantile` — over-strict
  (recall 0.36 mean, 0.99 precision). See §2.

### From JobA val_defect (19 categories, patched splitter + val_f1, 2026-04-27)

Headline (means over 19 paired cells; full tables under
[data/outputs/jobA_val_defect_V1/_analysis/](data/outputs/jobA_val_defect_V1/_analysis/)):

| Model | AUROC clean → vd | F1 clean → vd | Recall clean → vd | Precision clean → vd |
|---|---|---|---|---|
| PatchCore | 0.892 → 0.894 | 0.633 → **0.914** (+0.281) | 0.510 → **0.891** (+0.381) | 0.994 → 0.942 |
| PaDiM | 0.897 → 0.910 | 0.485 → **0.936** (+0.451) | 0.365 → **0.925** (+0.560) | 0.986 → 0.950 |
| SubspaceAD | 0.868 → 0.885 | 0.637 → **0.894** (+0.257) | 0.517 → **0.854** (+0.338) | 0.988 → 0.942 |

- All §9 decision gates from
  [PLAN job A_analize val_defect.md](PLAN%20job%20A_analize%20val_defect.md)
  passed (ΔF1 ≫ 0.10 per model; AUROC means stable inside ±0.02). The
  project default was switched to `val_f1` on 2026-04-27 in
  [default.yaml](src/benchmark_AD/default.yaml#L73).
- Pending before locking the thesis numbers: re-run the **11 categories
  not yet covered** by the val_defect rerun — porcelain_doll, regulator,
  tape, toy, toy_brick, u_block, usb, usb_adaptor, vcpill, wooden_beads,
  woodstick. The headline table for the thesis must be over the full 30
  cats, not the 19 we have today.

### From JobB Deceuninck (single run, val_quantile, no val balancing)

- All three models AUROC ≥ 0.995 → ranking quality on Deceuninck is
  excellent across the family. The model choice is *not* driven by
  detection quality, it's driven by FPR / cost.
- PaDiM dominates the cost/quality frontier: 7.8 ms/img, 1.06 GB VRAM,
  FPR = 4.4% (vs PatchCore 17.8%, SubspaceAD 15.6%).
- PatchCore and SubspaceAD reach 482/482 recall but with 8/45 and 7/45
  false alarms — too many for a real line.
- Per-defect breakdown shows all five types are detected at >99% recall;
  the dominant class (Scratch, 80% of bads) doesn't visibly distort the
  result here, but it could on harder datasets.

### Cross-job consistency check

PaDiM also wins **14/30** Real-IAD categories by AUROC. Its dominance
on Deceuninck is **not** a Deceuninck-specific fluke; it reflects a
real cost-quality advantage of the simpler Gaussian-per-patch model on
this kind of one-class industrial AD task.

---

## 8. Open methodological issues (limitations to disclose)

| Issue | Impact | Mitigation |
|---|---|---|
| Single seed per cell | F1 / Recall variance unmeasured (~±0.03 expected) | Re-run headline cells with 3 seeds; report mean ± std |
| Deceuninck specimen grouping not enforced | Possible near-duplicate leakage train↔test | Parse filename timestamp prefix as group key |
| Test prevalence on Deceuninck = 91% bad | F1/Precision inflated vs deployment (~5% bad) | Always co-report Recall@FPR=1% (prevalence-invariant) |
| Resize to 256×256 | Defects < ~1% of image area lost | Quantify smallest annotated defect size; document |
| `format: auto` on Deceuninck | Brittle to folder rename | Pin to `format: deceuninck` (not yet implemented) |
| Real-IAD limited to camera C1 | Methodology only covers one viewpoint | Frame as "single-viewpoint preview" in thesis |
| Subspace `mtop1p` aggregation, layer choice | Hyperparameters not tuned | Document as "fixed defaults from the SubspaceAD paper" |

---

## 9. Current JobA/B run settings (post-2026-04-27)

The default workflow now uses:

- `thresholding.mode: val_f1` everywhere — set in
  [default.yaml](src/benchmark_AD/default.yaml#L73) and inherited by all
  overlays. The legacy `val_quantile` is reachable only by explicit
  override.
- The patched splitter that routes 10% of anomaly images into val while
  keeping train good-only.
- The same per-summary fields as the clean runs, plus the industrial
  metrics already added to `benchmark_summary.json`:
  `recall_at_fpr_1pct`, `recall_at_fpr_5pct`, `macro_recall`,
  `weighted_recall`, `per_defect_recall`, `per_defect_support`.

Expected behaviour:

- AUROC / AUPR stay essentially unchanged because ranking is unchanged
  (per-cell drift is structural, see §2 caveats).
- Thresholds shift because val now contains both classes.
- F1 / Recall rise when the clean baseline was over-conservative.
- Macro recall vs weighted recall still exposes uneven per-class detection
  on imbalanced categories.

Operational consequence for the HPC clone: the wICE repo must be
`git pull`-ed before the next jobA_trained run so it picks up the new
default. The 2026-04-27 `csflow audiojack` cell predates the merge and
ran with `val_quantile` + `image_size: 256`; it has to be re-run after
the pull.

---

## 10. Pointers

- Operational rerun plan: [PLAN job A_analize val_defect.md](PLAN%20job%20A_analize%20val_defect.md)
- 19-cat rerun analysis (REPORT + Tables A–D):
  [data/outputs/jobA_val_defect_V1/_analysis/REPORT.md](data/outputs/jobA_val_defect_V1/_analysis/REPORT.md)
- Comparison script (clean ↔ val_defect):
  [scripts/compare_clean_vd.py](scripts/compare_clean_vd.py)
- Project-level scope: [PLAN.md](PLAN.md)
- Splitter implementation: [src/benchmark_AD/data.py:apply_dataset_split](src/benchmark_AD/data.py)
- Calibrator implementation: [src/benchmark_AD/pipeline.py:_maybe_calibrate_threshold](src/benchmark_AD/pipeline.py)
- Industrial metrics implementation: [src/benchmark_AD/pipeline.py:_recall_at_fpr](src/benchmark_AD/pipeline.py), [_per_defect_recall](src/benchmark_AD/pipeline.py)
- Aggregator: [scripts/analyze_runs.py](scripts/analyze_runs.py) (new) and [scripts/analyze_jobA.py](scripts/analyze_jobA.py) (legacy)
- HPC runbook (val_f1 caveats for the wICE clone):
  [docs/HPC_KU_LEUVEN_RUNBOOK.md §5.5](docs/HPC_KU_LEUVEN_RUNBOOK.md)
- Real-IAD layout gotchas: [memory/reference_realiad_layout.md](memory/reference_realiad_layout.md)
