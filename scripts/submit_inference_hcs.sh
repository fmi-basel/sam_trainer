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
#SBATCH --time=08:00:00

# USAGE:
# 1. To submit a SINGLE plate:
#    sbatch scripts/submit_inference_hcs.sh /path/to/plate.zarr
#
# 2. To submit ALL plates in a parent folder (sequential processing):
#    sbatch scripts/submit_inference_hcs.sh /path/to/parent_folder --all

INPUT_PATH="$1"
MODE="${2:-}"

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

# MODE 1: Process all .zarr plates in parent folder sequentially
if [[ "$MODE" == "--all" ]]; then
    echo "[INFO] Processing all .zarr plates in $INPUT_PATH sequentially..."
    
    # Find all directories ending in .zarr
    zarr_plates=($(find "$INPUT_PATH" -maxdepth 1 -type d -name "*.zarr"))
    plate_count=${#zarr_plates[@]}
    
    if [[ $plate_count -eq 0 ]]; then
        echo "Error: No .zarr plates found in $INPUT_PATH"
        exit 1
    fi
    
    echo "[INFO] Found $plate_count plates to process"
    
    for plate in "${zarr_plates[@]}"; do
        echo ""
        echo "[INFO] Processing plate: $plate"
        pixi run -e gpu python sam_trainer/run_inference_ngio.py \
            --input "$plate"
            # -v
        
        if [[ $? -ne 0 ]]; then
            echo "[WARNING] Error processing $plate, continuing with next..."
        fi
    done
    
    echo ""
    echo "[INFO] All plates processed."

# MODE 2: Process single .zarr plate
else
    echo "[INFO] Processing single plate: $INPUT_PATH"
    
    if [[ ! -d "$INPUT_PATH" ]]; then
        echo "Error: Input path does not exist or is not a directory: $INPUT_PATH"
        exit 1
    fi
    
    pixi run -e gpu python sam_trainer/run_inference_ngio.py \
        --input "$INPUT_PATH" \
        -v
fi

echo ""
echo "[INFO] Job finished."
