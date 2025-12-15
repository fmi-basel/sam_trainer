#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=inf_hcs
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
#    sbatch scripts/submit_inference_hcs.sh /path/to/plate.zarr label_name
#
# 2. To submit ALL plates in a parent folder (sequential processing):
#    sbatch scripts/submit_inference_hcs.sh /path/to/parent_folder/ label_name
#
# 3. With additional flags (e.g., AMG mode, custom thresholds):
#    sbatch scripts/submit_inference_hcs.sh plate.zarr sam_labels --use-amg
#    sbatch scripts/submit_inference_hcs.sh folder/ ais_relaxed --center-dist-thresh 0.6

INPUT_PATH="$1"
LABEL_NAME="$2"
shift 2  # Remove first 2 arguments, keep the rest as extra args

# Check if required arguments are provided
if [[ -z "$INPUT_PATH" ]] || [[ -z "$LABEL_NAME" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch scripts/submit_inference_hcs.sh <input_path> <label_name> [extra_args]"
    echo ""
    echo "Examples:"
    echo "  Single plate:  sbatch scripts/submit_inference_hcs.sh plate.zarr sam_labels"
    echo "  Batch mode:    sbatch scripts/submit_inference_hcs.sh plates_folder/ ais_default"
    echo "  With flags:    sbatch scripts/submit_inference_hcs.sh plate.zarr sam_amg --use-amg"
    echo ""
    echo "Model path uses default from run_inference_hcs.py (override with --model flag)"
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

# Validate input path
if [[ ! -d "$INPUT_PATH" ]]; then
    echo "Error: Input path does not exist or is not a directory: $INPUT_PATH"
    exit 1
fi

# Run inference (Python script handles both single plate and batch mode)
# Model path uses default from run_inference_hcs.py unless --model is provided in extra args
echo "[INFO] Input: $INPUT_PATH"
echo "[INFO] Label name: $LABEL_NAME"
echo "[INFO] Extra arguments: $@"
pixi run -e gpu python sam_trainer/run_inference_hcs.py \
    --input "$INPUT_PATH" \
    --label-name "$LABEL_NAME" \
    -v \
    "$@"

echo ""
echo "[INFO] Job finished."
