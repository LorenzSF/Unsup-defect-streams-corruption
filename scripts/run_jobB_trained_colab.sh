#!/usr/bin/env bash
# Job B driver for Colab: stage the Deceuninck dataset from Drive to local SSD,
# run one trained model at a time, and sync results back to Drive.
#
# The dataset is single-shot (good/ + defects/). Resumable via a
# `jobB_trained__<model>.done` marker under RESULTS_DIR.
#
# Environment (override with `KEY=value bash run_jobB_trained_colab.sh`):
#   MODEL        Model registry name (one of: anomalib_stfpm, anomalib_csflow,
#                anomalib_draem, rd4ad). Default: anomalib_stfpm.
#   DATASET_DIR  Directory on Drive containing good/ and defects/
#   RESULTS_DIR  Directory on Drive where the run is copied
#   WORK_DIR     Colab-local scratch dir (fast SSD)
#   REPO_DIR     Local clone of the thesis repo
#   CONFIG       Path to the trained-models config YAML

set -euo pipefail

MODEL="${MODEL:-anomalib_stfpm}"
DATASET_DIR="${DATASET_DIR:-/content/drive/MyDrive/Deceuninck_dataset}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobB_trained}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_trained_deceuninck.yaml}"

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

marker="${RESULTS_DIR}/jobB_trained__${MODEL}.done"
if [[ -f "${marker}" ]]; then
  echo "[jobB_trained] already done (${marker}). Delete it to rerun." >&2
  exit 0
fi

run_name="jobB_trained_${MODEL}"
local_dataset="${WORK_DIR}/jobB_trained_deceuninck"

echo "==============================================================="
echo "[jobB_trained] starting at $(date -u +%H:%M:%S)"
echo "[jobB_trained] model:      ${MODEL}"
echo "[jobB_trained] config:     ${CONFIG}"
echo "[jobB_trained] dataset:    ${DATASET_DIR}"
echo "[jobB_trained] results to: ${RESULTS_DIR}"
echo "==============================================================="

rm -rf "${local_dataset}"
mkdir -p "${local_dataset}"
echo "[jobB_trained] copying dataset from Drive to ${local_dataset}..."
cp -r "${DATASET_DIR}/good"    "${local_dataset}/good"
cp -r "${DATASET_DIR}/defects" "${local_dataset}/defects"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "[jobB_trained] running pipeline..."
python main.py \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --dataset-path "${local_dataset}" \
  --extract-dir "${local_dataset}" \
  --run-name "${run_name}"

latest_run="$(ls -1dt "${WORK_DIR}/runs/${run_name}"_* 2>/dev/null | head -n1)"
if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
  echo "[jobB_trained] ERROR: pipeline produced no output dir" >&2
  rm -rf "${local_dataset}"
  exit 1
fi

echo "[jobB_trained] syncing ${latest_run} -> Drive..."
dest="${RESULTS_DIR}/$(basename "${latest_run}")"
rsync -a --remove-source-files "${latest_run}/" "${dest}/"
find "${latest_run}" -type d -empty -delete

touch "${marker}"
echo "[jobB_trained] done -> ${dest}"

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
    runtime.unassign()
except Exception:
    pass
PY
