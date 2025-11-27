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
#SBATCH --partition=main,several,short
#SBATCH --time=04:00:00

# USAGE:
# 1. To submit a SINGLE plate directly:
#    sbatch scripts/submit_inference_hcs.sh /path/to/plate.zarr
#
# 2. To submit ALL plates in a folder (Parallel):
#    ./scripts/submit_inference_hcs.sh /path/to/parent_folder --submit-all

INPUT_PATH="$1"
MODE="${2:-}"

# Activate environment
# Assuming pixi is available or environment is already set up
# If using pixi run, the python command below changes to `pixi run python ...`
# Ensure Pixi (local install) is available in the job environment
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

# Force Pixi to use existing environment without checking for updates
# This prevents network access attempts when lock file has changed
export PIXI_FROZEN=true
export PIXI_LOCKED=true

echo "[INFO] Using existing Pixi environment (frozen mode - no network updates)"

# Run the appropriate mode

if [[ "$MODE" == "--submit-all" ]]; then
    # MASTER MODE: Loop through folder and submit jobs
    echo "Searching for .zarr plates in $INPUT_PATH..."
    
    # Find all directories ending in .zarr
    find "$INPUT_PATH" -maxdepth 1 -type d -name "*.zarr" | while read plate; do
        echo "Submitting job for plate: $plate"
        sbatch scripts/submit_inference_.sh "$plate"
    done
    exit 0
fi

# WORKER MODE: Process a single plate
echo "Running inference on plate: $INPUT_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Run the python script
# Using 'pixi run' to ensure environment is correct
pixi run -e gpu python sam_trainer/run_inference_ngio.py \
    --input "$INPUT_PATH"

echo "Job finished."
