#!/usr/bin/env bash
# Job A driver for Colab — TRAINED models, model-major iteration.
#
# One model at a time across all (or a subset of) Real-IAD categories.
# Per-session pattern:  run-this-model on cat1 -> save -> cat2 -> save -> ...
#
# Resumable: writes a marker `<model>__<category>.done` per finished pair.
# Re-running skips finished pairs only — switching MODEL re-uses any prior
# work for the new model (its markers are independent).
#
# Environment (override with `KEY=value bash run_jobA_trained_colab.sh`):
#   MODEL         Model registry name (one of: anomalib_stfpm, anomalib_csflow,
#                 anomalib_draem, rd4ad). Default: anomalib_stfpm.
#   CATEGORIES    Space-separated list to restrict (e.g. "audiojack bottle_cap").
#                 Empty = all 30 categories found in ZIPS_DIR.
#   ZIPS_DIR      Directory on Drive containing <category>.zip files.
#   RESULTS_DIR   Directory on Drive where per-(model,category) runs are copied.
#   WORK_DIR      Colab-local scratch dir.
#   REPO_DIR      Local clone of the thesis repo.
#   CONFIG        Path to the trained-models config YAML.

set -euo pipefail

MODEL="${MODEL:-anomalib_stfpm}"
CATEGORIES="${CATEGORIES:-}"
ZIPS_DIR="${ZIPS_DIR:-/content/drive/MyDrive/Real-IAD_dataset/realiad_1024}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobA_trained}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_trained.yaml}"

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"

if [[ ! -d "${ZIPS_DIR}" ]]; then
  echo "ZIPS_DIR not found: ${ZIPS_DIR}" >&2
  exit 1
fi

# Build the category list — either restricted or auto-discovered from Drive.
declare -a cat_list
if [[ -n "${CATEGORIES}" ]]; then
  read -r -a cat_list <<< "${CATEGORIES}"
else
  shopt -s nullglob
  zips=("${ZIPS_DIR}"/*.zip)
  if [[ ${#zips[@]} -eq 0 ]]; then
    echo "No .zip files found under ${ZIPS_DIR}" >&2
    exit 1
  fi
  for z in "${zips[@]}"; do
    cat_list+=("$(basename "${z}" .zip)")
  done
fi

echo "[jobA-trained] model:      ${MODEL}"
echo "[jobA-trained] categories: ${#cat_list[@]} -> ${cat_list[*]}"
echo "[jobA-trained] config:     ${CONFIG}"
echo "[jobA-trained] results to: ${RESULTS_DIR}"
echo

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
# Force unbuffered Python stdout/stderr so progress is visible through `tee`
# (otherwise long-running stages like DRAEM training can stay silent for minutes
# while output sits in the 4 KB pipe buffer).
export PYTHONUNBUFFERED=1

for category in "${cat_list[@]}"; do
  marker="${RESULTS_DIR}/${MODEL}__${category}.done"
  cat_work="${WORK_DIR}/${category}"
  zip_path="${ZIPS_DIR}/${category}.zip"

  if [[ -f "${marker}" ]]; then
    echo "[skip] ${MODEL} on ${category} (${marker})"
    continue
  fi

  if [[ ! -f "${zip_path}" ]]; then
    echo "[warn] zip missing for category '${category}': ${zip_path}" >&2
    continue
  fi

  echo "==============================================================="
  echo "[${MODEL} / ${category}] starting at $(date -u +%H:%M:%S)"
  echo "==============================================================="

  rm -rf "${cat_work}"
  mkdir -p "${cat_work}"

  echo "[${category}] copying zip from Drive..."
  local_zip="${cat_work}/${category}.zip"
  cp "${zip_path}" "${local_zip}"

  echo "[${category}] extracting..."
  unzip -q -o "${local_zip}" -d "${cat_work}"
  rm -f "${local_zip}"

  dataset_root="${cat_work}/${category}"
  if [[ ! -d "${dataset_root}/OK" ]]; then
    echo "[${category}] ERROR: expected OK/ under ${dataset_root}" >&2
    ls -la "${cat_work}" >&2
    rm -rf "${cat_work}"
    continue
  fi

  run_name="jobA_${MODEL}_${category}"
  echo "[${MODEL} / ${category}] running pipeline..."
  python main.py \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --dataset-path "${dataset_root}" \
    --extract-dir "${dataset_root}" \
    --run-name "${run_name}"

  latest_run="$(ls -1dt "${WORK_DIR}/runs/${run_name}"_* 2>/dev/null | head -n1)"
  if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
    echo "[${category}] ERROR: pipeline produced no output dir" >&2
    rm -rf "${cat_work}"
    continue
  fi

  echo "[${category}] syncing ${latest_run} -> Drive..."
  dest="${RESULTS_DIR}/$(basename "${latest_run}")"
  rsync -a --remove-source-files "${latest_run}/" "${dest}/"
  find "${latest_run}" -type d -empty -delete

  touch "${marker}"
  echo "[${MODEL} / ${category}] done -> ${dest}"

  rm -rf "${cat_work}"
done

echo
echo "[jobA-trained] model '${MODEL}' processed (or skipped) for all selected categories."
