#!/usr/bin/env bash
# Job C driver — corruption robustness benchmark (PLAN.md §1.3).
#
# One invocation = one cell of the corruption grid:
#   (1 model × 1 dataset × N categories × 1 corruption × 1 severity)
#
# Per-cell .done markers under MARKERS_DIR make the script resumable across
# Colab sessions and HPC job submissions. The notebook drives the
# (corruption × severity) grid by re-invoking this script for each combo.
#
# Usage:
#   bash scripts/run_jobC.sh <model> <dataset:realiad|deceuninck> <corruption> <severity> [<cat1> <cat2> ...]
#
# Categories:
#   - realiad: optional. Defaults to the 5-cat subset
#       audiojack button_battery plastic_plug regulator woodstick
#     (chosen to span object types without paying for the full 30-cat grid).
#   - deceuninck: ignored (single-cell dataset).
#
# Examples (Colab):
#   bash scripts/run_jobC.sh anomalib_padim realiad gaussian_blur 3
#   bash scripts/run_jobC.sh anomalib_padim realiad motion_blur 5 audiojack bottle_cap
#   bash scripts/run_jobC.sh anomalib_padim deceuninck jpeg_compression 1
#
# HPC (wICE) — Real-IAD only; Deceuninck is intentionally unsupported because
# the dataset is not on /scratch. Override env vars in the parent shell:
#   PYTHON_CMD="uv run python" \
#   REPO_DIR=/data/leuven/381/vsc38124/Real-time-visual-defect-detection \
#   RIAD_ZIPS_DIR=/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024 \
#   WORK_DIR=/scratch/leuven/381/vsc38124/jobC_work \
#   OUTPUT_ROOT=/scratch/leuven/381/vsc38124/runs \
#   RESULTS_DIR= \
#   CONFIG_REALIAD=src/benchmark_AD/configs/wice_jobC_realiad.yaml \
#   bash scripts/run_jobC.sh anomalib_padim realiad gaussian_blur 3 audiojack
#
# Env vars (all optional — Colab defaults shown):
#   REPO_DIR          repo root (default: $PWD)
#   WORK_DIR          local SSD scratch dir (default: /content/work)
#   OUTPUT_ROOT       where the pipeline writes runs; MUST match the
#                     output_dir set in CONFIG_* (default: ${WORK_DIR}/runs)
#   RESULTS_DIR       Drive dir to rsync runs to; empty = no sync
#                     (default: /content/drive/MyDrive/thesis_runs/jobC)
#   MARKERS_DIR       where to put per-cell .done markers
#                     (default: ${RESULTS_DIR}/.done_markers if non-empty,
#                      else ${OUTPUT_ROOT}/.done_markers)
#   RIAD_ZIPS_DIR     dir containing <category>.zip files
#                     (default: /content/drive/MyDrive/Real-IAD_dataset/realiad_1024)
#   DECEU_DIR         Deceuninck root with good/ + defects/
#                     (default: /content/drive/MyDrive/Deceuninck_dataset)
#   CONFIG_REALIAD    Real-IAD config YAML
#                     (default: ${REPO_DIR}/src/benchmark_AD/configs/colab_jobC_realiad.yaml)
#   CONFIG_DECEUNINCK Deceuninck config YAML
#                     (default: ${REPO_DIR}/src/benchmark_AD/configs/colab_jobC_deceuninck.yaml)
#   PYTHON_CMD        python command (default: python; HPC: "uv run python")

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash $0 <model> <dataset:realiad|deceuninck> <corruption> <severity> [<cat1> ...]" >&2
  exit 1
fi

MODEL="$1"
DATASET="$2"
CORRUPTION="$3"
SEVERITY="$4"
shift 4
CATEGORIES_IN=("$@")

# Default 5-cat subset for Real-IAD per PLAN.md §1.3 fallback. Easy to override
# by passing categories as positional args.
DEFAULT_CATEGORIES=(audiojack button_battery plastic_plug regulator woodstick)

REPO_DIR="${REPO_DIR:-$(pwd)}"
WORK_DIR="${WORK_DIR:-/content/work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/runs}"
RESULTS_DIR="${RESULTS_DIR-/content/drive/MyDrive/thesis_runs/jobC}"
RIAD_ZIPS_DIR="${RIAD_ZIPS_DIR:-/content/drive/MyDrive/Real-IAD_dataset/realiad_1024}"
DECEU_DIR="${DECEU_DIR:-/content/drive/MyDrive/Deceuninck_dataset}"
CONFIG_REALIAD="${CONFIG_REALIAD:-${REPO_DIR}/src/benchmark_AD/configs/colab_jobC_realiad.yaml}"
CONFIG_DECEUNINCK="${CONFIG_DECEUNINCK:-${REPO_DIR}/src/benchmark_AD/configs/colab_jobC_deceuninck.yaml}"
PYTHON_CMD="${PYTHON_CMD:-python}"

# Argument validation — fail before creating any directories so the error
# message isn't masked by a downstream "mkdir: cannot create" on a misspelled
# DATASET in environments where /content doesn't exist (e.g. local Windows).
case "${DATASET}" in
  realiad|deceuninck) ;;
  *) echo "DATASET must be 'realiad' or 'deceuninck', got '${DATASET}'" >&2; exit 1 ;;
esac
case "${CORRUPTION}" in
  gaussian_blur|motion_blur|jpeg_compression) ;;
  *) echo "CORRUPTION must be one of gaussian_blur|motion_blur|jpeg_compression, got '${CORRUPTION}'" >&2; exit 1 ;;
esac
case "${SEVERITY}" in
  1|2|3|4|5) ;;
  *) echo "SEVERITY must be 1..5, got '${SEVERITY}'" >&2; exit 1 ;;
esac

if [[ -z "${MARKERS_DIR:-}" ]]; then
  if [[ -n "${RESULTS_DIR}" ]]; then
    MARKERS_DIR="${RESULTS_DIR}/.done_markers"
  else
    MARKERS_DIR="${OUTPUT_ROOT}/.done_markers"
  fi
fi

mkdir -p "${OUTPUT_ROOT}" "${WORK_DIR}" "${MARKERS_DIR}"
[[ -n "${RESULTS_DIR}" ]] && mkdir -p "${RESULTS_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

if [[ "${DATASET}" == "realiad" ]]; then
  if [[ ${#CATEGORIES_IN[@]} -gt 0 ]]; then
    CATEGORIES=("${CATEGORIES_IN[@]}")
  else
    CATEGORIES=("${DEFAULT_CATEGORIES[@]}")
  fi
  CONFIG="${CONFIG_REALIAD}"
else
  CATEGORIES=()  # Deceuninck has no per-category loop
  CONFIG="${CONFIG_DECEUNINCK}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG not found: ${CONFIG}" >&2
  exit 1
fi

echo "==============================================================="
echo "[jobC] model:       ${MODEL}"
echo "[jobC] dataset:     ${DATASET}"
echo "[jobC] corruption:  ${CORRUPTION} (severity=${SEVERITY})"
if [[ "${DATASET}" == "realiad" ]]; then
  echo "[jobC] categories:  ${CATEGORIES[*]}"
fi
echo "[jobC] config:      ${CONFIG}"
echo "[jobC] output_root: ${OUTPUT_ROOT}"
echo "[jobC] results_dir: ${RESULTS_DIR:-<none>}"
echo "[jobC] markers_dir: ${MARKERS_DIR}"
echo "==============================================================="

# ---------------------------------------------------------------------------
# Per-cell runner. Idempotent via .done marker; rsyncs to Drive on Colab.
# ---------------------------------------------------------------------------
run_one_cell() {
  local cat_label="$1"     # category name, or 'deceuninck' for the single-cell case
  local dataset_root="$2"  # path to the staged dataset on local SSD

  local run_name="jobC_${DATASET}_${cat_label}_${MODEL}_${CORRUPTION}_s${SEVERITY}"
  local marker="${MARKERS_DIR}/${run_name}.done"

  if [[ -f "${marker}" ]]; then
    echo "[skip] ${run_name} already done"
    return 0
  fi

  echo
  echo "---------------------------------------------------------------"
  echo "[${run_name}] starting at $(date -u +%H:%M:%S) UTC"
  echo "---------------------------------------------------------------"

  ${PYTHON_CMD} main.py \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --dataset-path "${dataset_root}" \
    --extract-dir "${dataset_root}" \
    --corruption "${CORRUPTION}" \
    --severity "${SEVERITY}" \
    --run-name "${run_name}"

  local latest_run
  latest_run="$(ls -1dt "${OUTPUT_ROOT}/${run_name}"_* 2>/dev/null | head -n1 || true)"
  if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
    echo "[${run_name}] ERROR: no output dir under ${OUTPUT_ROOT}/${run_name}_*" >&2
    return 1
  fi

  if [[ -n "${RESULTS_DIR}" && "${RESULTS_DIR}" != "${OUTPUT_ROOT}" ]]; then
    echo "[${run_name}] syncing ${latest_run} -> ${RESULTS_DIR}..."
    local dest="${RESULTS_DIR}/$(basename "${latest_run}")"
    rsync -a --remove-source-files "${latest_run}/" "${dest}/"
    find "${latest_run}" -type d -empty -delete
  fi

  touch "${marker}"
  echo "[${run_name}] done"
}

# ---------------------------------------------------------------------------
# Stage data once per category (cached in WORK_DIR), then run the cell.
# Re-extraction is skipped when the dataset root is already on local SSD —
# important because the same category is hit many times across the grid.
# ---------------------------------------------------------------------------
if [[ "${DATASET}" == "realiad" ]]; then
  if [[ ! -d "${RIAD_ZIPS_DIR}" ]]; then
    echo "RIAD_ZIPS_DIR not found: ${RIAD_ZIPS_DIR}" >&2
    exit 1
  fi
  for category in "${CATEGORIES[@]}"; do
    cat_work="${WORK_DIR}/${category}"
    dataset_root="${cat_work}/${category}"

    if [[ ! -d "${dataset_root}/OK" ]]; then
      zip_path="${RIAD_ZIPS_DIR}/${category}.zip"
      if [[ ! -f "${zip_path}" ]]; then
        echo "[${category}] ERROR: zip missing at ${zip_path}" >&2
        continue
      fi
      echo "[${category}] staging zip to ${cat_work}..."
      rm -rf "${cat_work}"
      mkdir -p "${cat_work}"
      cp "${zip_path}" "${cat_work}/${category}.zip"
      unzip -q -o "${cat_work}/${category}.zip" -d "${cat_work}"
      rm -f "${cat_work}/${category}.zip"
    else
      echo "[${category}] already staged at ${dataset_root}"
    fi

    if [[ ! -d "${dataset_root}/OK" ]]; then
      echo "[${category}] ERROR: expected OK/ under ${dataset_root}" >&2
      continue
    fi
    run_one_cell "${category}" "${dataset_root}"
  done
else
  if [[ ! -d "${DECEU_DIR}/good" || ! -d "${DECEU_DIR}/defects" ]]; then
    echo "DECEU_DIR missing good/ + defects/: ${DECEU_DIR}" >&2
    echo "(Deceuninck on HPC is intentionally unsupported — dataset only on Colab Drive.)" >&2
    exit 1
  fi
  staged="${WORK_DIR}/jobC_deceuninck"
  if [[ ! -d "${staged}/good" || ! -d "${staged}/defects" ]]; then
    echo "[deceuninck] copying from Drive to ${staged}..."
    rm -rf "${staged}"
    mkdir -p "${staged}"
    cp -r "${DECEU_DIR}/good"    "${staged}/good"
    cp -r "${DECEU_DIR}/defects" "${staged}/defects"
  else
    echo "[deceuninck] already staged at ${staged}"
  fi
  run_one_cell "deceuninck" "${staged}"
fi

echo
echo "[jobC] cell complete."
