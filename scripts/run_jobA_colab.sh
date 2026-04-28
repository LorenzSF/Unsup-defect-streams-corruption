#!/usr/bin/env bash
# Job A driver for Colab: iterate Real-IAD category zips on Drive, run the
# feature-based benchmark (PatchCore + PaDiM + SubspaceAD) on each one,
# and sync results back to Drive.
#
# Resumable: if a marker file `<category>.done` exists under RESULTS_DIR,
# the category is skipped. Delete the marker to force a rerun.
#
# Environment (override with `KEY=value bash run_jobA_colab.sh`):
#   ZIPS_DIR      Directory on Drive containing <category>.zip files
#   RESULTS_DIR   Directory on Drive where per-category runs are copied
#   WORK_DIR      Colab-local scratch dir (fast SSD, cleared between cats)
#   REPO_DIR      Local clone of the thesis repo
#   CONFIG        Path to the Colab config YAML

set -euo pipefail

ZIPS_DIR="${ZIPS_DIR:-/content/drive/MyDrive/Real-IAD_dataset/realiad_1024}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobA}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_featurebased.yaml}"

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"

if [[ ! -d "${ZIPS_DIR}" ]]; then
  echo "ZIPS_DIR not found: ${ZIPS_DIR}" >&2
  exit 1
fi

shopt -s nullglob
zips=("${ZIPS_DIR}"/*.zip)
if [[ ${#zips[@]} -eq 0 ]]; then
  echo "No .zip files found under ${ZIPS_DIR}" >&2
  exit 1
fi

echo "[jobA] found ${#zips[@]} category zips"
echo "[jobA] config:     ${CONFIG}"
echo "[jobA] results to: ${RESULTS_DIR}"
echo

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for zip in "${zips[@]}"; do
  category="$(basename "${zip}" .zip)"
  marker="${RESULTS_DIR}/${category}.done"
  cat_work="${WORK_DIR}/${category}"

  if [[ -f "${marker}" ]]; then
    echo "[skip] ${category} already done (${marker})"
    continue
  fi

  echo "==============================================================="
  echo "[${category}] starting at $(date -u +%H:%M:%S)"
  echo "==============================================================="

  rm -rf "${cat_work}"
  mkdir -p "${cat_work}"

  # Stage zip locally so unzip reads from fast SSD (Drive mount is slow).
  local_zip="${cat_work}/${category}.zip"
  echo "[${category}] copying zip from Drive..."
  cp "${zip}" "${local_zip}"

  echo "[${category}] extracting..."
  unzip -q -o "${local_zip}" -d "${cat_work}"
  rm -f "${local_zip}"

  # Zips embed a top-level <category>/ folder, so dataset root is nested.
  dataset_root="${cat_work}/${category}"
  if [[ ! -d "${dataset_root}/OK" ]]; then
    echo "[${category}] ERROR: expected OK/ under ${dataset_root}" >&2
    ls -la "${cat_work}" >&2
    rm -rf "${cat_work}"
    continue
  fi

  run_name="jobA_${category}"
  echo "[${category}] running pipeline (3 models)..."
  python main.py \
    --config "${CONFIG}" \
    --dataset-path "${dataset_root}" \
    --extract-dir "${dataset_root}" \
    --run-name "${run_name}"

  # Pipeline emits /content/work/runs/<run_name>_<UTC-timestamp>/
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
  echo "[${category}] done -> ${dest}"

  rm -rf "${cat_work}"
done

echo
echo "[jobA] all categories processed."

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
