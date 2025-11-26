#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=sam_hcs_inf
#SBATCH --output=logs/hcs_inf_%j.out
#SBATCH --error=logs/hcs_inf_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=main
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

if [[ "$MODE" == "--submit-all" ]]; then
    # MASTER MODE: Loop through folder and submit jobs
    echo "Searching for .zarr plates in $INPUT_PATH..."
    
    # Find all directories ending in .zarr
    find "$INPUT_PATH" -maxdepth 1 -type d -name "*.zarr" | while read plate; do
        echo "Submitting job for plate: $plate"
        sbatch scripts/submit_inference_hcs.sh "$plate"
    done
    exit 0
fi

# WORKER MODE: Process a single plate
echo "Running inference on plate: $INPUT_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Run the python script
# Using 'pixi run' to ensure environment is correct
pixi run -e gpu python scripts/run_inference_hcs.py \
    --input "$INPUT_PATH"

echo "Job finished."
