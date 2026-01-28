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

set -eu

if [ $# -lt 3 ]; then
    echo "Usage: sbatch scripts/submit_inference.sh <model_path> <input_path> <output_dir> [extra_args]"
    echo "  For OME-Zarr: output_dir can be empty (\"\"), labels are written back to zarr"
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
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Input: $INPUT_DIR"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Extra Args: $@"

# Ensure output directory exists (if not empty/zarr mode)
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# TODO make model path not required 

# Run inference - pass all extra arguments through
pixi run -e gpu python sam_trainer/run_inference.py \
    --model "$MODEL_PATH" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    "$@"

echo "[INFO] Inference complete"
