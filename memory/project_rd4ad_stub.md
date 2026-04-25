---
name: RD4AD adapter rewritten — was a stub returning image variance
description: rd4ad was a no-op stub until 2026-04-25; it is now a real anomalib reverse_distillation wrapper. Old benchmark numbers under the rd4ad name are invalid.
type: project
---

The `rd4ad` model in [src/benchmark_AD/models.py](../src/benchmark_AD/models.py) **used to be a stub** and was replaced on 2026-04-25 with `AnomalibRd4adModel`, a wrapper around `anomalib.models.image.reverse_distillation` modeled on `AnomalibStfpmModel`.

**What the old stub did (now removed):**

- `fit()` was a no-op — just set `self._is_fitted = True`.
- `_load_checkpoint()` tried to read `data/checkpoints/rd4ad.pth` (a file that did not exist) and silently passed.
- `predict()` returned `float(np.var(x))` — pixel-variance, not anomaly detection.

**Symptoms in old benchmark output:** `fit_seconds: 0.00`, `predict_seconds: ~0.2s`, threshold drifting with image content, `heatmap=None` on every frame. Discovered mid-Job-A on Colab; ~20 RD4AD `.done` markers were deleted because the AUROC/F1 numbers were meaningless.

**How to apply:**

- Any benchmark result for `rd4ad` from runs **before 2026-04-25** is image-variance noise — discard.
- The current adapter trains the bottleneck + decoder with Adam(lr=0.005, betas=(0.5, 0.999)); encoder is frozen by `ReverseDistillationModel.forward()`. Defaults: `backbone=wide_resnet50_2`, `layers=[layer1, layer2, layer3]`, `epochs=200` for paper-comparable runs (`epochs=1` in `default.yaml` is the smoke-test default — see `project_default_epochs.md`).
- `rd4ad` now belongs in the trained-model serialization plan (`PLAN_serializacion.md` Phase 2) alongside STFPM/CS-Flow/DRAEM. Re-running RD4AD before checkpoint serialization lands means re-training on every stream startup.
