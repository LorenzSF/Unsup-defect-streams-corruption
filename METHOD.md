# METHOD.md — Methodology notes for the thesis

> Living document with **partial conclusions** drawn from the JobA / JobB
> analyses. Numbers below are from the 30-cat clean JobA run, the 7-cat
> val_defect rerun, and the single JobB run. Everything is stamped *as of*
> 2026-04-26 and will be updated when the new (val_balance=equal +
> per-defect-type) reruns finish.
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

Across the **7-category JobA val_defect rerun** (clean vs val_f1, same
splits otherwise, single seed):

| Model | ΔAUROC | ΔAUPR | ΔF1 | ΔRecall | ΔPrecision | mean Thr_v / Thr_c |
|---|---:|---:|---:|---:|---:|---:|
| PatchCore | +0.001 | −0.001 | **+0.103** | **+0.168** | −0.024 | 0.87 |
| PaDiM | +0.004 | ±0 | **+0.295** | **+0.438** | −0.032 | **0.50** |
| SubspaceAD | −0.001 | −0.001 | **+0.089** | **+0.152** | −0.026 | 0.87 |

**Conclusion (preliminary):** ranking quality (AUROC, AUPR) is unchanged
by the calibration switch, as it should be. F1 and Recall jumps come
entirely from a better operating point. PaDiM was the most miscalibrated
(threshold ratio 0.50 → it was running at twice the optimal cut).

### Decision for the thesis

The `val_quantile` numbers are kept only for completeness; the headline
metrics use `val_f1` plus the val-composition rules in §3. The
methodology section should explicitly state that any reported F1 / Recall
without a documented calibration procedure is meaningless on industrial
AD datasets.

---

## 3. Train / val / test composition is the *second* dominant axis

Once the calibrator works, the next sensitivity is the **composition of
each split**. The pipeline now exposes three knobs in
[default.yaml#dataset.split](src/benchmark_AD/default.yaml):

| Knob | Default | val_defect overlay | Effect |
|---|---|---|---|
| `val_balance` | `"natural"` | `"equal"` | when "equal", val ends up 50/50 good/bad. F1 curve no longer biased by val prevalence (which varies wildly across categories). |
| `val_bad_balance_by_type` | `false` | `true` | per `defect_type`, val bad pool capped at `ceil(min_class * (1 + tolerance))`. Dominant defect class no longer monopolises calibration. |
| `val_balance_tolerance` | `0.15` | `0.15` | wiggle room around the smallest class. |

A test reserve (`ceil(test_ratio * len(class))` per defect type) is
applied unconditionally so per-defect recall stays measurable on test —
a bug found during smoke testing where minority Deceuninck classes
(Black spots = 21 images) would otherwise be fully consumed by val.

### Why each rule matters

1. **`val_balance: equal`** — F1 is prevalence-sensitive. With val ≈ 73%
   bads (JobB pre-patch) the F1 maximiser tolerates large FPR; with val
   ≈ 18% bads (some JobA cats) it picks an over-conservative threshold.
   A 50/50 val makes the picked threshold comparable across cats and
   across jobs.
2. **`val_bad_balance_by_type: true`** — On Deceuninck, *Scratch in
   extrusion direction* is 80% of all bads (388/482). Without per-type
   balancing the calibrator fits the operating point on scratches and
   the four other defect types are accidentally calibrated. With the
   cap each type contributes ~equally to the F1 curve.
3. **Test reserve per type** — Without it, the per-defect recall on
   minority classes becomes undefined (0 samples in test). With
   `test_reserve = ceil(0.2 * class_size)` every type retains ≥1
   sample (≥5 for Deceuninck minority classes).

### Empirical effect on Deceuninck splits (verified on smoke test)

| Split | train | val | test | val per type | test per type |
|---|---:|---:|---:|---|---|
| Legacy (val_quantile) | 161g | 18g+0b | 45g+482b | n/a | natural |
| val_defect v1 (val_ratio of bads → val) | 161g | 18g+49b | 45g+433b | Scratch=35, others 4–5 | natural |
| **val_defect v2** (equal + per-type, **new**) | 81g | 98g+98b | 45g+384b | Scratch=25, Cleaning=21, Degassing=18, Pigment=18, Black=16 | Scratch=363, Cleaning=6, Black/Degassing/Pigment=5 |

The price for v2 is a smaller train (81 goods vs 161). Acceptable for
one-class learning on a homogeneous extrusion product; would be
re-examined on harder categories.

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

All overlays inherit from [default.yaml](src/benchmark_AD/default.yaml).
Differences highlighted.

| Overlay | Dataset | Format | Cameras | Threshold mode | val_balance | val_bad_balance_by_type |
|---|---|---|---|---|---|---|
| [colab_featurebased.yaml](src/benchmark_AD/configs/colab_featurebased.yaml) | Real-IAD | real_iad | C1 | val_quantile (target_fpr 0.01) | natural | false |
| [colab_featurebased_val_defect.yaml](src/benchmark_AD/configs/colab_featurebased_val_defect.yaml) | Real-IAD | real_iad | C1 | val_f1 | **equal** | **true** |
| [colab_featurebased_deceuninck.yaml](src/benchmark_AD/configs/colab_featurebased_deceuninck.yaml) | Deceuninck | auto | n/a | val_quantile | natural | false |
| [colab_featurebased_deceuninck_val_defect.yaml](src/benchmark_AD/configs/colab_featurebased_deceuninck_val_defect.yaml) | Deceuninck | auto | n/a | val_f1 | **equal** | **true** |

---

## 7. Preliminary model conclusions

**These are not the final thesis numbers** — they predate the
val_balance=equal + per-type rerun. Update this section once those
numbers land.

### From clean JobA (30 categories × 3 models)

- AUROC mean: PaDiM 0.893 > PatchCore 0.870 > SubspaceAD 0.842.
- AUROC win counts: PaDiM 14 / 30, PatchCore 10 / 30, SubspaceAD 6 / 30.
- F1/Recall numbers are not trustworthy under `val_quantile` — over-strict
  (recall 0.36 mean, 0.99 precision). See §2.

### From JobA val_defect (7 of 10 categories, val_f1, no val_balance)

- Calibration switch alone gives ΔF1 = +0.10 / +0.30 / +0.09 (PatchCore /
  PaDiM / SubspaceAD), ΔRecall = +0.17 / +0.44 / +0.15. AUROC unchanged.
- Decision gates from
  [PLAN job A_analize val_defect.md](PLAN%20job%20A_analize%20val_defect.md) §9
  passed → switching the project default to `val_f1` is justified.

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

## 9. What changes in JobA/B re-runs (after 2026-04-26)

The next batch of runs (val_defect v2) uses:

- `thresholding.mode: val_f1` (carried over from v1).
- `val_balance: equal`, `val_bad_balance_by_type: true`, `val_balance_tolerance: 0.15` (**new**).
- New per-summary fields: `recall_at_fpr_1pct`, `recall_at_fpr_5pct`,
  `macro_recall`, `weighted_recall`, `per_defect_recall`,
  `per_defect_support`.

Expected behaviour:

- AUROC / AUPR unchanged (model and ranking are unchanged).
- Threshold per cell shifts slightly (calibrator now sees a balanced val).
- Macro recall vs weighted recall gap exposes uneven per-class detection
  on categories with imbalanced defect classes.
- On JobB specifically, the calibrator's permissiveness should drop
  (val no longer 18 clean images), so FPR should fall. PaDiM's lead on
  the FPR axis may shrink as PatchCore/SubspaceAD catch up.

---

## 10. Pointers

- Operational rerun plan: [PLAN job A_analize val_defect.md](PLAN%20job%20A_analize%20val_defect.md)
- Project-level scope: [PLAN.md](PLAN.md)
- Splitter implementation: [src/benchmark_AD/data.py:apply_dataset_split](src/benchmark_AD/data.py)
- Calibrator implementation: [src/benchmark_AD/pipeline.py:_maybe_calibrate_threshold](src/benchmark_AD/pipeline.py)
- Industrial metrics implementation: [src/benchmark_AD/pipeline.py:_recall_at_fpr](src/benchmark_AD/pipeline.py), [_per_defect_recall](src/benchmark_AD/pipeline.py)
- Aggregator: [scripts/analyze_runs.py](scripts/analyze_runs.py) (new) and [scripts/analyze_jobA.py](scripts/analyze_jobA.py) (legacy)
- Real-IAD layout gotchas: [memory/reference_realiad_layout.md](memory/reference_realiad_layout.md)
