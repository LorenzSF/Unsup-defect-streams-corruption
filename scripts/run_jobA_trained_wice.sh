#!/usr/bin/env bash
# JobA driver for wICE / VSC — TRAINED models, model-major iteration.
#
# Runs ONE model across a list of Real-IAD categories on /scratch.
# Resumable: writes <model>__<category>.done markers under
# ${RUNS_DIR}/.done_markers/. Re-running skips finished pairs.
#
# Usage:
#   bash scripts/run_jobA_trained_wice.sh <model> <cat1> [<cat2> ...]
#
# Examples:
#   bash scripts/run_jobA_trained_wice.sh anomalib_draem audiojack bottle_cap eraser
#   bash scripts/run_jobA_trained_wice.sh rd4ad audiojack
#
# Models: anomalib_stfpm, anomalib_csflow, anomalib_draem, rd4ad
#
# Environment overrides (rarely needed):
#   ZIPS_DIR, RUNS_DIR, REPO_DIR, CONFIG, MARKERS_DIR

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 <model> <cat1> [<cat2> ...]" >&2
  echo "  Models: anomalib_stfpm, anomalib_csflow, anomalib_draem, rd4ad" >&2
  exit 1
fi

MODEL="$1"
shift
CATEGORIES=("$@")

# --- Paths (override via env if your VSC layout differs) ---------------------
ZIPS_DIR="${ZIPS_DIR:-/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024}"
RUNS_DIR="${RUNS_DIR:-/scratch/leuven/381/vsc38124/runs}"
REPO_DIR="${REPO_DIR:-/data/leuven/381/vsc38124/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/wice_trained.yaml}"
MARKERS_DIR="${MARKERS_DIR:-${RUNS_DIR}/.done_markers}"

# --- uv toolchain (idempotent re-export so the script works even if you forgot
#     to export them in the parent shell). -----------------------------------
export PATH="/scratch/leuven/381/vsc38124/uv-x86_64-unknown-linux-gnu:${PATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/leuven/381/vsc38124/.uv_cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/leuven/381/vsc38124/.venv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/scratch/leuven/381/vsc38124/.uv_python}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
# Force unbuffered stdout/stderr so progress is visible through `tee`.
export PYTHONUNBUFFERED=1

mkdir -p "${RUNS_DIR}" "${MARKERS_DIR}"

if [[ ! -d "${ZIPS_DIR}" ]]; then
  echo "ZIPS_DIR not found: ${ZIPS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG not found: ${CONFIG}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not on PATH. Expected at /scratch/.../uv-x86_64-unknown-linux-gnu" >&2
  exit 1
fi

echo "[jobA-wice] model:      ${MODEL}"
echo "[jobA-wice] categories: ${#CATEGORIES[@]} -> ${CATEGORIES[*]}"
echo "[jobA-wice] config:     ${CONFIG}"
echo "[jobA-wice] runs to:    ${RUNS_DIR}"
echo "[jobA-wice] markers in: ${MARKERS_DIR}"
echo

cd "${REPO_DIR}"

for CATEGORY in "${CATEGORIES[@]}"; do
  marker="${MARKERS_DIR}/${MODEL}__${CATEGORY}.done"
  zip_path="${ZIPS_DIR}/${CATEGORY}.zip"
  dataset_root="${ZIPS_DIR}/${CATEGORY}"

  if [[ -f "${marker}" ]]; then
    echo "[skip] ${MODEL} on ${CATEGORY} (${marker})"
    continue
  fi

  if [[ ! -f "${zip_path}" ]]; then
    echo "[warn] zip missing for category '${CATEGORY}': ${zip_path}" >&2
    continue
  fi

  echo "==============================================================="
  echo "[${MODEL} / ${CATEGORY}] starting at $(date -u +%H:%M:%S) UTC"
  echo "==============================================================="

  # Extract from inside ZIPS_DIR so the layout is single-nested
  # (<cat>/OK, <cat>/NG) instead of double-nested (<cat>/<cat>/OK).
  if [[ ! -d "${dataset_root}/OK" ]]; then
    echo "[${CATEGORY}] extracting..."
    rm -rf "${dataset_root}"
    (cd "${ZIPS_DIR}" && unzip -q "${CATEGORY}.zip")
  else
    echo "[${CATEGORY}] already extracted -> ${dataset_root}"
  fi

  if [[ ! -d "${dataset_root}/OK" ]]; then
    echo "[${CATEGORY}] ERROR: expected OK/ under ${dataset_root}" >&2
    ls -la "${dataset_root}" >&2 || true
    continue
  fi

  run_name="jobA_${MODEL}_${CATEGORY}"
  echo "[${MODEL} / ${CATEGORY}] running pipeline..."

  uv run python main.py \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --dataset-path "${dataset_root}" \
    --extract-dir "${dataset_root}" \
    --run-name "${run_name}"

  touch "${marker}"
  echo "[${MODEL} / ${CATEGORY}] done -> ${RUNS_DIR}/${run_name}_*"
done

echo
echo "[jobA-wice] model '${MODEL}' processed (or skipped) for all ${#CATEGORIES[@]} categories."
