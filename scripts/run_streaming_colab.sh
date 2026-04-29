#!/usr/bin/env bash
# Launch the streaming runtime on Google Colab GPU against an existing benchmark run.
#
# This driver stages the historical dataset and the live input folder onto the
# Colab local SSD, then runs `runtime_main.py` with path overrides so the
# benchmark_summary.json does not need to embed machine-specific paths.
#
# Environment (override with `KEY=value bash run_streaming_colab.sh`):
#   MODEL            Model row to use from benchmark_summary.json
#   BENCH_RUN_DIR    Existing benchmark run directory on Drive or local disk
#   DATASET_DIR      Historical dataset root used to rebuild the training split
#   INPUT_DIR        Live input folder to stream; defaults to DATASET_DIR
#   RESULTS_DIR      Destination directory for streaming sessions/logs
#   WORK_DIR         Colab-local scratch dir (fast SSD)
#   REPO_DIR         Local clone of the thesis repo
#   CONFIG           Runtime settings YAML
#   FIT_POLICY       auto | historical_fit | skip_fit
#   MAX_FRAMES       Max frames for this session (default: 300)
#   TARGET_FPS       Runtime target FPS (default: 10)
#   PORT             Dashboard port (default: 8765)
#   LOOP             1 => keep looping the input folder, 0 => single pass
#
# Example:
#   MODEL=anomalib_patchcore \
#   BENCH_RUN_DIR=/content/drive/MyDrive/thesis_runs/jobB_deceuninck_20260425_101433 \
#   DATASET_DIR=/content/drive/MyDrive/Deceuninck_dataset \
#   RESULTS_DIR=/content/drive/MyDrive/thesis_runs/streaming \
#   bash scripts/run_streaming_colab.sh

set -euo pipefail

MODEL="${MODEL:-anomalib_patchcore}"
BENCH_RUN_DIR="${BENCH_RUN_DIR:-}"
DATASET_DIR="${DATASET_DIR:-/content/drive/MyDrive/Deceuninck_dataset}"
INPUT_DIR="${INPUT_DIR:-${DATASET_DIR}}"
RESULTS_DIR="${RESULTS_DIR:-/content/drive/MyDrive/thesis_runs/streaming}"
WORK_DIR="${WORK_DIR:-/content/work}"
REPO_DIR="${REPO_DIR:-/content/Real-time-visual-defect-detection}"
CONFIG="${CONFIG:-${REPO_DIR}/src/streaming_input/settings.yaml}"
FIT_POLICY="${FIT_POLICY:-auto}"
MAX_FRAMES="${MAX_FRAMES:-300}"
TARGET_FPS="${TARGET_FPS:-10}"
PORT="${PORT:-8765}"
LOOP="${LOOP:-0}"

if [[ -z "${BENCH_RUN_DIR}" ]]; then
  echo "BENCH_RUN_DIR is required." >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}"

if [[ ! -d "${BENCH_RUN_DIR}" ]]; then
  echo "BENCH_RUN_DIR not found: ${BENCH_RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BENCH_RUN_DIR}/benchmark_summary.json" ]]; then
  echo "benchmark_summary.json not found under ${BENCH_RUN_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "INPUT_DIR not found: ${INPUT_DIR}" >&2
  exit 1
fi

local_dataset="${WORK_DIR}/streaming_history_dataset"
local_input="${WORK_DIR}/streaming_live_input"
local_run_dir="${WORK_DIR}/streaming_benchmark_run"
log_path="${RESULTS_DIR}/streaming_${MODEL}.log"

echo "==============================================================="
echo "[streaming_colab] starting at $(date -u +%H:%M:%S)"
echo "[streaming_colab] model:         ${MODEL}"
echo "[streaming_colab] benchmark run: ${BENCH_RUN_DIR}"
echo "[streaming_colab] dataset:       ${DATASET_DIR}"
echo "[streaming_colab] input:         ${INPUT_DIR}"
echo "[streaming_colab] fit_policy:    ${FIT_POLICY}"
echo "[streaming_colab] max_frames:    ${MAX_FRAMES}"
echo "[streaming_colab] target_fps:    ${TARGET_FPS}"
echo "[streaming_colab] port:          ${PORT}"
echo "[streaming_colab] loop:          ${LOOP}"
echo "[streaming_colab] results to:    ${RESULTS_DIR}"
echo "==============================================================="

rm -rf "${local_dataset}" "${local_input}" "${local_run_dir}"
mkdir -p "${local_dataset}" "${local_input}" "${local_run_dir}"

echo "[streaming_colab] copying benchmark run to local SSD..."
rsync -a "${BENCH_RUN_DIR}/" "${local_run_dir}/"

echo "[streaming_colab] copying historical dataset to local SSD..."
rsync -a "${DATASET_DIR}/" "${local_dataset}/"

if [[ "${INPUT_DIR}" == "${DATASET_DIR}" ]]; then
  echo "[streaming_colab] reusing historical dataset as live input."
  rsync -a --delete "${local_dataset}/" "${local_input}/"
else
  echo "[streaming_colab] copying live input folder to local SSD..."
  rsync -a "${INPUT_DIR}/" "${local_input}/"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

loop_flag=()
if [[ "${LOOP}" == "1" ]]; then
  loop_flag+=(--loop)
fi

echo "[streaming_colab] launching runtime..."
python runtime_main.py \
  --config "${CONFIG}" \
  --run-dir "${local_run_dir}" \
  --model "${MODEL}" \
  --fit-policy "${FIT_POLICY}" \
  --dataset-path "${local_dataset}" \
  --extract-dir "${local_dataset}" \
  --input-dir "${local_input}" \
  --max-frames "${MAX_FRAMES}" \
  --target-fps "${TARGET_FPS}" \
  --port "${PORT}" \
  "${loop_flag[@]}" 2>&1 | tee "${log_path}"

latest_session="$(ls -1dt "${REPO_DIR}"/data/streaming_output/streaming_output_* 2>/dev/null | head -n1)"
if [[ -z "${latest_session}" || ! -d "${latest_session}" ]]; then
  echo "[streaming_colab] ERROR: runtime produced no streaming output dir" >&2
  exit 1
fi

dest="${RESULTS_DIR}/$(basename "${latest_session}")"
echo "[streaming_colab] syncing ${latest_session} -> ${dest}"
rsync -a --delete "${latest_session}/" "${dest}/"

echo "[streaming_colab] done -> ${dest}"
echo "[streaming_colab] If running inside a notebook, expose the dashboard with:"
echo "from google.colab import output; output.serve_kernel_port_as_iframe(${PORT}, height=1000)"
