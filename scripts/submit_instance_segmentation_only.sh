#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=sam_training_mip164
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=main
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constraint="infiniband&gpuram48gb"
#SBATCH --time=48:00:00
# Write logs into logs/<job-name>-<jobid>.out|.err under the working directory
# NOTE: the directory is created below before running
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# SAM Training SLURM Batch Script
# Usage: sbatch submit_training.sh <path_to_config.yaml>

set -eu

function display_memory_usage() {
    set +eu
    echo -n "[INFO] [$(date -Iseconds)] [$$] Max memory usage in bytes: "
    cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes
    echo
}

trap display_memory_usage EXIT

START=$(date +%s)
STARTDATE=$(date -Iseconds)
echo "[INFO] [$STARTDATE] [$$] Starting SLURM job $SLURM_JOB_ID"
echo "[INFO] [$STARTDATE] [$$] Running in $(hostname -s)"
echo "[INFO] [$STARTDATE] [$$] Working directory: $(pwd)"

### SAM TRAINING SCRIPT

# Check if config file was provided
if [ $# -eq 0 ]; then
    echo "[ERROR] No config file provided"
    echo "Usage: sbatch submit_training.sh <path_to_config.yaml>"
    exit 1
fi

CONFIG_PATH=$1

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "[INFO] [$STARTDATE] [$$] Using config: $CONFIG_PATH"

# Load modules if needed (uncomment and adjust as needed)
# module purge
# module load cuda/11.8

# Optional: limit thread oversubscription for better dataloader performance
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

# Activate pixi environment and run training
echo "[INFO] [$STARTDATE] [$$] Starting SAM training..."

# Option 1: Run with pixi (recommended)
# Use the GPU environment we defined in pixi.toml (requires cuda drivers on the node)
pixi run -e gpu python -m sam_trainer.cli train --config "$CONFIG_PATH" -vv

# Option 2: If pixi is already activated, use directly
# python -m sam_trainer.cli train --config "$CONFIG_PATH" -vv

### END OF SAM TRAINING SCRIPT

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time (seconds): $(( $END-$START ))"
