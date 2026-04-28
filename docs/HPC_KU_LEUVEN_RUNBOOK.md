# HPC KU Leuven (VSC) - JobA_trained runbook

`jobA_trained` is the trained Real-IAD path on wICE / `gpu_a100`, run through
OnDemand. The current baseline uses `wice_trained.yaml`, camera `C1`,
good-only training, and `val_f1` calibration with the patched splitter from
`data.py`, which routes a validation slice of anomalies without changing the
one-class training setup.

Use one category at a time for a quick check, or pass multiple categories to
the driver for consecutive runs. Keep repo files in `/data`, caches and outputs
in `/scratch`, and remember that the OnDemand terminal mangles long pasted
commands.

## Context

- Cluster: `wice`
- Partition: `gpu_a100`
- Nodes: `1`
- GPUs: `1`
- Typical wall time: `2 h` interactive for one `(model, category)` pair
- Dataset layout: one Real-IAD category at a time, extracted as
  `<category>/OK/...` and `<category>/NG/...`

Do not request multiple GPUs or multiple nodes. The pipeline is single-node and
the trained adapters are single-GPU.

## Command Summary

```bash
# 1. Per-session environment
export PATH="/scratch/leuven/381/vsc38124/uv-x86_64-unknown-linux-gnu:$PATH"
export UV_CACHE_DIR=/scratch/leuven/381/vsc38124/.uv_cache
export UV_PROJECT_ENVIRONMENT=/scratch/leuven/381/vsc38124/.venv
export UV_PYTHON_INSTALL_DIR=/scratch/leuven/381/vsc38124/.uv_python
export PYTHONPATH=src
cd /data/leuven/381/vsc38124/Real-time-visual-defect-detection

# 2. Optional sanity check / first-time sync
uv sync --extra gpu
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Pick one category and extract it on scratch
CATEGORY=audiojack
cd /scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/
rm -rf "${CATEGORY}"
unzip -q "${CATEGORY}.zip"
ls "${CATEGORY}"
find "${CATEGORY}/OK" -type f -name "*.jpg" | wc -l
find "${CATEGORY}/NG" -type f -name "*.jpg" | wc -l

# 4. Set the dataset root and return to the repo
DATASET_ROOT=/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/${CATEGORY}
cd /data/leuven/381/vsc38124/Real-time-visual-defect-detection

# 5. Run one trained model on one category
MODEL=anomalib_csflow
uv run python main.py --config src/benchmark_AD/configs/wice_trained.yaml --model "$MODEL" --dataset-path "$DATASET_ROOT" --extract-dir "$DATASET_ROOT" --run-name "jobA_${MODEL}_${CATEGORY}"

# 6. Run the same model on several categories consecutively
bash scripts/run_jobA_trained_wice.sh anomalib_draem plastic_nut plastic_plug porcelain_doll regulator rolled_strip_base sim_card_set

# 7. Inspect or stage the outputs
RUN=$(ls -1dt /scratch/leuven/381/vsc38124/runs/jobA_anomalib_csflow * | head -n1)
cat "$RUN/benchmark_summary.json"
mkdir -p /data/leuven/381/vsc38124/downloads
cp -r "$RUN" /data/leuven/381/vsc38124/downloads/

# 8. Clean old outputs in scratch before rerunning an old category
for CATEGORY in audiojack bottle_cap button_battery; do
  rm -rf /scratch/leuven/381/vsc38124/runs/jobA_${MODEL}_${CATEGORY}_*
  rm -f /scratch/leuven/381/vsc38124/runs/.done_markers/${MODEL}__${CATEGORY}.done
done


# 9. Massive copy from scratch to downlads
mkdir -p /data/leuven/381/vsc38124/downloads
rsync -a /scratch/leuven/381/vsc38124/runs/jobA_* /data/leuven/381/vsc38124/downloads/

# 10. See the content of some folder or directory
find /scratch/leuven/381/vsc38124/runs -maxdepth 1 -type d -name 'jobA_*' | sort

# 11. Clean massive in some directory
rm -rf /scratch/leuven/381/vsc38124/runs/jobA_*


```

If you also want a full reset of the extracted datasets for those categories,
delete `/scratch/leuven/381/vsc38124/datasets/Real-IAD_dataset/realiad_1024/${CATEGORY}`
inside the same loop.

## Stage 1

Set the wICE `uv` toolchain, scratch caches, and `PYTHONPATH` before doing
anything else. This keeps Python wheels, Hugging Face downloads, and torch
cache off `$HOME` and avoids quota issues.

## Stage 2

Only run the GPU check when you need to verify the node. `uv sync --extra gpu`
is the one-time setup that installs the CUDA wheel set into the scratch venv.

## Stage 3

Extract exactly one Real-IAD category per run and always do it from the parent
`realiad_1024/` directory. That produces the single-nested layout the loader
expects and avoids `audiojack/audiojack/...` mistakes.

## Stage 4

Keep `DATASET_ROOT` pointed at the extracted category and return to the repo
before launching Python. This makes the command lines shorter and keeps the run
path stable.

## Stage 5

This is the direct single-pair launch for `jobA_trained`. `wice_trained.yaml`
already carries the current method baseline, so the run uses `val_f1` and the
patched split behavior without requiring a second config file.

## Stage 6

The batch driver loops over every category you pass after the model name. It
skips any pair that already has a `.done` marker, so it is resumable by design.

## Stage 7

Run outputs live under `/scratch/leuven/381/vsc38124/runs/`. Copy the folder to
`/data/leuven/381/vsc38124/downloads/` if you want to inspect or download it
through the OnDemand file browser.

## Stage 8

Delete the matching run folders and `.done` markers when you want to rerun the
same category with a newer commit. If you need a completely clean dataset
state, remove the extracted category folder too.

## Categories And Models

Possible Real-IAD categories:

`audiojack`, `bottle_cap`, `button_battery`, `end_cap`, `eraser`, `fire_hood`,
`mint`, `mounts`, `pcb`, `phone_battery`, `plastic_nut`, `plastic_plug`,
`porcelain_doll`, `regulator`, `rolled_strip_base`, `sim_card_set`, `switch`,
`tape`, `terminalblock`, `toothbrush`, `toy`, `toy_brick`, `transistor1`,
`u_block`, `usb`, `usb_adaptor`, `vcpill`, `wooden_beads`, `woodstick`,
`zipper`.

Possible `jobA_trained` models:

`anomalib_stfpm`, `anomalib_csflow`, `anomalib_draem`, `rd4ad`.
