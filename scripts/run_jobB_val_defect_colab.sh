#!/usr/bin/env bash
# Job B "val_defect" driver: Deceuninck dataset with the patched splitter
# (10% of anomalies routed into val) and val_f1 thresholding.
#
# Single dataset (no per-category loop). Resumable via jobB_val_defect.done
# under RESULTS_DIR — delete the marker to rerun.
#
# Environment (override with `KEY=value bash run_jobB_val_defect_colab.sh`):
#   DATASET_DIR   Directory on Drive containing good/ and defects/
#   RESULTS_DIR   Directory on Drive where the run is copied
#   WORK_DIR      Colab-local scratch dir (fast SSD)
#   REPO_DIR      Local clone of the thesis repo
#   CONFIG        Path to the val_defect Job B config YAML

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-/content/drive/MyDrive/Deceuninck_dataset}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobB_val_defect}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_featurebased_deceuninck_val_defect.yaml}"

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}/good" || ! -d "${DATASET_DIR}/defects" ]]; then
  echo "Expected good/ and defects/ under ${DATASET_DIR}" >&2
  ls -la "${DATASET_DIR}" >&2
  exit 1
fi

marker="${RESULTS_DIR}/jobB_val_defect.done"
if [[ -f "${marker}" ]]; then
  echo "[jobB_val_defect] already done (${marker}). Delete it to rerun." >&2
  exit 0
fi

run_name="jobB_val_defect_deceuninck"
local_dataset="${WORK_DIR}/${run_name}"

echo "==============================================================="
echo "[jobB_val_defect] starting at $(date -u +%H:%M:%S)"
echo "[jobB_val_defect] config:     ${CONFIG}"
echo "[jobB_val_defect] dataset:    ${DATASET_DIR}"
echo "[jobB_val_defect] results to: ${RESULTS_DIR}"
echo "==============================================================="

rm -rf "${local_dataset}"
mkdir -p "${local_dataset}"
echo "[jobB_val_defect] copying dataset from Drive to ${local_dataset}..."
cp -r "${DATASET_DIR}/good"    "${local_dataset}/good"
cp -r "${DATASET_DIR}/defects" "${local_dataset}/defects"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "[jobB_val_defect] running pipeline (3 models, val_f1)..."
python main.py \
  --config "${CONFIG}" \
  --dataset-path "${local_dataset}" \
  --extract-dir "${local_dataset}" \
  --run-name "${run_name}"

latest_run="$(ls -1dt "${WORK_DIR}/runs/${run_name}"_* 2>/dev/null | head -n1)"
if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
  echo "[jobB_val_defect] ERROR: pipeline produced no output dir" >&2
  rm -rf "${local_dataset}"
  exit 1
fi

echo "[jobB_val_defect] syncing ${latest_run} -> Drive..."
dest="${RESULTS_DIR}/$(basename "${latest_run}")"
rsync -a --remove-source-files "${latest_run}/" "${dest}/"
find "${latest_run}" -type d -empty -delete

touch "${marker}"
echo "[jobB_val_defect] done -> ${dest}"

rm -rf "${local_dataset}"

# Release Colab resources once the run is finished and synced.
python - <<'PY'
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
gc.collect()
try:
    from google.colab import runtime
    try:
        runtime.unassign()
    except Exception as exc:
        print(f"[cleanup] runtime.unassign() skipped: {exc}")
except Exception:
    pass
PY
