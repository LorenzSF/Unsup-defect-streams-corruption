#!/usr/bin/env bash
# Job A "val_defect" driver: runs the patched splitter + val_f1 thresholding
# on the pending Real-IAD categories relative to the 30-category dataset.
# The categories already present under data/outputs/jobA_val_defect_V1 were
# removed from this loop so the next Colab batch covers only the remaining
# v1 cells.
#
# Resumable: a `<category>.done` marker under RESULTS_DIR skips the run.
#
# Environment (override with `KEY=value bash run_jobA_val_defect_colab.sh`):
#   ZIPS_DIR     Directory containing <category>.zip files
#   RESULTS_DIR  Directory where per-category runs are copied
#   WORK_DIR     Local scratch dir
#   REPO_DIR     Local clone of the thesis repo
#   CONFIG       Path to the val_defect config YAML

set -euo pipefail

ZIPS_DIR="${ZIPS_DIR:-/content/drive/MyDrive/Real-IAD_dataset/realiad_1024}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/jobA_val_defect}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/benchmark_AD/configs/colab_featurebased_val_defect.yaml}"

CATEGORIES=(
  audiojack
  bottle_cap
  button_battery
  end_cap
  pcb
  eraser
  fire_hood
  mint
  mounts
  phone_battery
  plastic_nut
  plastic_plug
  porcelain_doll
  regulator
  tape
  toy
  toy_brick
  u_block
  usb
  usb_adaptor
  vcpill
  wooden_beads
  woodstick
)

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"

if [[ ! -d "${ZIPS_DIR}" ]]; then
  echo "ZIPS_DIR not found: ${ZIPS_DIR}" >&2
  exit 1
fi

echo "[jobA_val_defect] ${#CATEGORIES[@]} categories"
echo "[jobA_val_defect] config:     ${CONFIG}"
echo "[jobA_val_defect] results to: ${RESULTS_DIR}"
echo

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

for category in "${CATEGORIES[@]}"; do
  zip="${ZIPS_DIR}/${category}.zip"
  marker="${RESULTS_DIR}/${category}.done"
  cat_work="${WORK_DIR}/${category}"

  if [[ -f "${marker}" ]]; then
    echo "[skip] ${category} already done (${marker})"
    continue
  fi

  if [[ ! -f "${zip}" ]]; then
    echo "[${category}] ERROR: zip not found at ${zip}" >&2
    continue
  fi

  echo "==============================================================="
  echo "[${category}] starting at $(date -u +%H:%M:%S)"
  echo "==============================================================="

  rm -rf "${cat_work}"
  mkdir -p "${cat_work}"

  local_zip="${cat_work}/${category}.zip"
  echo "[${category}] copying zip..."
  cp "${zip}" "${local_zip}"

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

  run_name="jobA_val_defect_${category}"
  echo "[${category}] running pipeline (3 models, val_f1)..."
  python main.py \
    --config "${CONFIG}" \
    --dataset-path "${dataset_root}" \
    --extract-dir "${dataset_root}" \
    --run-name "${run_name}"

  latest_run="$(ls -1dt "${WORK_DIR}/runs/${run_name}"_* 2>/dev/null | head -n1)"
  if [[ -z "${latest_run}" || ! -d "${latest_run}" ]]; then
    echo "[${category}] ERROR: pipeline produced no output dir" >&2
    rm -rf "${cat_work}"
    continue
  fi

  echo "[${category}] syncing ${latest_run} -> results dir..."
  dest="${RESULTS_DIR}/$(basename "${latest_run}")"
  rsync -a --remove-source-files "${latest_run}/" "${dest}/"
  find "${latest_run}" -type d -empty -delete

  touch "${marker}"
  echo "[${category}] done -> ${dest}"

  rm -rf "${cat_work}"
done

echo
echo "[jobA_val_defect] all categories processed."

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
