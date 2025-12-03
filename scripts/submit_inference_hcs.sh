#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=sam_hcs_inf
#SBATCH --output=logs/hcs_inf_%j.out
#SBATCH --error=logs/hcs_inf_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main,several
#SBATCH --time=16:00:00

# USAGE:
# 1. To submit a SINGLE plate:
#    sbatch scripts/submit_inference_hcs.sh /path/to/plate.zarr
#
# 2. To submit ALL plates in a parent folder (sequential processing):
#    sbatch scripts/submit_inference_hcs.sh /path/to/parent_folder

INPUT_PATH="$1"

# Check if input path is provided
if [[ -z "$INPUT_PATH" ]]; then
    echo "Error: No input path provided"
    echo "Usage: sbatch scripts/submit_inference_hcs.sh <path> [--all]"
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

echo "[INFO] Using existing Pixi environment"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"

# Validate input path
if [[ ! -d "$INPUT_PATH" ]]; then
    echo "Error: Input path does not exist or is not a directory: $INPUT_PATH"
    exit 1
fi

# Run inference (Python script handles both single plate and batch mode)
echo "[INFO] Running inference on: $INPUT_PATH"
pixi run -e gpu python sam_trainer/run_inference_hcs.py \
    --input "$INPUT_PATH" \
    -v

echo ""
echo "[INFO] Job finished."
