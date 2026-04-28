#!/usr/bin/env bash
# Job B driver for Colab: stage the Deceuninck dataset from Drive to local SSD,
# run the feature-based benchmark (PatchCore + PaDiM + SubspaceAD) once, and
# sync results back to Drive.
#
# Deceuninck is a single dataset (good/ + defects/<5 subfolders>/*.jpg), so
# unlike Job A there is no per-category loop. The script is still resumable:
# if RESULTS_DIR/jobB.done exists it exits early. Delete the marker to rerun.
#
# Environment (override with `KEY=value bash run_jobB_colab.sh`):
#   DATASET_DIR   Directory on Drive containing good/ and defects/
#   RESULTS_DIR   Directory on Drive where the run is copied
#   WORK_DIR      Colab-local scratch dir (fast SSD)
#   REPO_DIR      Local clone of the thesis repo
#   CONFIG        Path to the Colab Job B config YAML

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-/content/drive/MyDrive/Deceuninck_dataset}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobB}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_featurebased_deceuninck.yaml}"

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

marker="${RESULTS_DIR}/jobB.done"
if [[ -f "${marker}" ]]; then
  echo "[jobB] already done (${marker}). Delete it to rerun." >&2
  exit 0
fi

run_name="jobB_deceuninck"
local_dataset="${WORK_DIR}/${run_name}"

echo "==============================================================="
echo "[jobB] starting at $(date -u +%H:%M:%S)"
echo "[jobB] config:     ${CONFIG}"
echo "[jobB] dataset:    ${DATASET_DIR}"
echo "[jobB] results to: ${RESULTS_DIR}"
echo "==============================================================="

# Stage dataset on local SSD — Drive mount is slow for many small JPEGs.
rm -rf "${local_dataset}"
mkdir -p "${local_dataset}"
echo "[jobB] copying dataset from Drive to ${local_dataset}..."
cp -r "${DATASET_DIR}/good"    "${local_dataset}/good"
cp -r "${DATASET_DIR}/defects" "${local_dataset}/defects"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "[jobB] running pipeline (3 models)..."
python main.py \
  --config "${CONFIG}" \
  --dataset-path "${local_dataset}" \
  --extract-dir "${local_dataset}" \
  --run-name "${run_name}"

# Pipeline emits /content/work/runs/<run_name>_<UTC-timestamp>/
latest_run="$(ls -1dt "${WORK_DIR}/runs/${run_name}"_* 2>/dev/null | head -n1)"
if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
  echo "[jobB] ERROR: pipeline produced no output dir" >&2
  rm -rf "${local_dataset}"
  exit 1
fi

echo "[jobB] syncing ${latest_run} -> Drive..."
dest="${RESULTS_DIR}/$(basename "${latest_run}")"
rsync -a --remove-source-files "${latest_run}/" "${dest}/"
find "${latest_run}" -type d -empty -delete

touch "${marker}"
echo "[jobB] done -> ${dest}"

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
