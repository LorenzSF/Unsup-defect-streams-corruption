#!/usr/bin/env bash
# §1.2 smoke test: one model × one corruption × one dataset category.
#
# Runs anomalib_padim on the local Deceuninck dataset with gaussian_blur at
# severity 3. Confirms the corruption block is wired end-to-end:
#   - cfg['corruption'] is read from default.yaml + CLI overrides
#   - corruption is applied to test images only (train/val stay clean)
#   - per-row corruption_type / severity / dataset are stamped
#   - per-(model × corruption × severity × dataset) live_status_*.json is emitted
#   - corrupted TP/FP/FN/TN samples are exported under corrupted_confusion_samples/
#
# Run from the repo root:
#   bash scripts/run_smoke_corruption.sh
#
# Override defaults by env var, e.g.:
#   CORRUPTION=motion_blur SEVERITY=5 bash scripts/run_smoke_corruption.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/industrial.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/data/Deceuninck_dataset}"
MODEL="${MODEL:-anomalib_padim}"
CORRUPTION="${CORRUPTION:-gaussian_blur}"
SEVERITY="${SEVERITY:-3}"
RUN_NAME="${RUN_NAME:-smoke_corruption_${MODEL}_${CORRUPTION}_s${SEVERITY}}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}/good" || ! -d "${DATASET_DIR}/defects" ]]; then
  echo "Expected good/ and defects/ under ${DATASET_DIR}" >&2
  ls -la "${DATASET_DIR}" >&2 || true
  exit 1
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "==============================================================="
echo "[smoke] config:     ${CONFIG}"
echo "[smoke] dataset:    ${DATASET_DIR}"
echo "[smoke] model:      ${MODEL}"
echo "[smoke] corruption: ${CORRUPTION} (severity=${SEVERITY})"
echo "[smoke] run_name:   ${RUN_NAME}"
echo "==============================================================="

python main.py \
  --config "${CONFIG}" \
  --dataset-path "${DATASET_DIR}" \
  --extract-dir "${DATASET_DIR}" \
  --model "${MODEL}" \
  --corruption "${CORRUPTION}" \
  --severity "${SEVERITY}" \
  --run-name "${RUN_NAME}"

# The pipeline writes outputs under cfg['run']['output_dir']; industrial.yaml
# inherits "data/outputs" from default.yaml. Surface the latest run dir so the
# caller can inspect benchmark_summary.json without hunting for the timestamp.
output_root="data/outputs"
latest_run="$(ls -1dt "${output_root}/${RUN_NAME}"_* 2>/dev/null | head -n1 || true)"
if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
  echo "[smoke] WARNING: no run dir found under ${output_root}/${RUN_NAME}_*" >&2
  exit 1
fi

echo
echo "[smoke] run output: ${latest_run}"
echo "[smoke] sanity check — files written:"
ls -1 "${latest_run}"
