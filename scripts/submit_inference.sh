#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=sam_inference
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

# SAM Inference SLURM Batch Script
# Supports TIFF and OME-Zarr inputs (auto-detected)
# Usage:
#   TIFF: sbatch scripts/submit_inference.sh <model_path> <input_dir> <output_dir> [extra_args]
#   Zarr: sbatch scripts/submit_inference.sh <model_path> <input.zarr> "" [extra_args]
#         (output_dir not used for zarr - labels written back to zarr)
#   Pretrained/base model (no custom checkpoint): pass "" as <model_path>, and specify
#   the desired base model via --model-type in [extra_args] (default vit_b_lm if omitted):
#     sbatch scripts/submit_inference.sh "" images/ masks/ --model-type vit_b_lm
#
# Selecting a specific channel (OME-Zarr: by label name, wavelength ID, or index; TIFF: index only):
#   sbatch scripts/submit_inference.sh model.pt images/ masks/ --channel BF
#   sbatch scripts/submit_inference.sh model.pt images/ masks/ --channel A02_C04
#   sbatch scripts/submit_inference.sh model.pt images/ masks/ --channel 2

set -eu

if [ $# -lt 3 ]; then
    echo "Usage: sbatch scripts/submit_inference.sh <model_path> <input_path> <output_dir> [extra_args]"
    echo "  For OME-Zarr: output_dir can be empty (\"\"), labels are written back to zarr"
    echo "  For the pretrained/base model: pass \"\" as <model_path> and set --model-type in [extra_args]"
    exit 1
fi

# Activate environment
WD="$(pwd)"
export PATH="$PATH:$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"

PIXIBIN="$WD/infrastructure/apps/pixi/bin/pixi"
if [[ ! -x "$PIXIBIN" ]]; then
    echo "[INFO] Pixi binary not found; running install.sh"
    bash "$WD/install.sh"
fi

# Ensure environment is properly installed
echo "[INFO] Ensuring GPU environment is ready..."
pixi install -e gpu

echo "[INFO] Using Pixi GPU environment"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"

MODEL_PATH=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
shift 3

echo "[INFO] Starting inference job $SLURM_JOB_ID"
echo "[INFO] Model: ${MODEL_PATH:-<none, using pretrained micro-sam model via --model-type>}"
echo "[INFO] Input: $INPUT_DIR"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Extra Args: $@"

# Ensure output directory exists (if not empty/zarr mode)
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Only pass --model when a checkpoint path was given; an empty MODEL_PATH means
# "use the pretrained micro-sam model" (run_inference.py downloads/caches it based
# on --model-type, which the caller should pass in [extra_args]).
MODEL_ARGS=()
if [ -n "$MODEL_PATH" ]; then
    MODEL_ARGS=(--model "$MODEL_PATH")
fi

# Run inference - pass all extra arguments through
pixi run -e gpu python sam_trainer/run_inference.py \
    "${MODEL_ARGS[@]}" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    "$@"

echo "[INFO] Inference complete"
