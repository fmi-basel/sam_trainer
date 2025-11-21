#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=full_img_sam_train_a40
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=main
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint="infiniband&gpuram48gb"
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# SAM Training SLURM Batch Script (A40 / 48GB GPU)
# Usage: sbatch scripts/submit_training_a40.sh <path_to_config.yaml>

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

# Check if config file was provided
if [ $# -eq 0 ]; then
    echo "[ERROR] No config file provided"
    echo "Usage: sbatch scripts/submit_training_a40.sh <path_to_config.yaml>"
    exit 1
fi

CONFIG_PATH=$1

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "[INFO] [$STARTDATE] [$$] Using config: $CONFIG_PATH"

# Set threading for optimal CPU usage with 20 cores
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# Run training
echo "[INFO] [$STARTDATE] [$$] Starting SAM training on A40..."
pixi run -e gpu python -m sam_trainer.cli train --config "$CONFIG_PATH" -vv

# Move logs to output directory
echo "[INFO] Moving logs to experiment directory..."
# Extract experiment info using grep (simple parsing)
EXP_NAME=$(grep "^experiment_name:" "$CONFIG_PATH" | sed 's/experiment_name: //;s/"//g;s/ //g')
BASE_DIR=$(grep "^output_base_dir:" "$CONFIG_PATH" | sed 's/output_base_dir: //;s/"//g;s/ //g')
BASE_DIR=${BASE_DIR:-runs}
OUTPUT_DIR="$BASE_DIR/$EXP_NAME"

if [ -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR/logs"
    # Copy first to be safe, then remove? Or just move.
    # We use cp then rm because moving an open file (stdout/stderr) can be tricky while script is running
    # But usually safe in bash scripts at the very end.
    mv "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$OUTPUT_DIR/logs/" || echo "Warning: Could not move .out log"
    mv "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "$OUTPUT_DIR/logs/" || echo "Warning: Could not move .err log"
    echo "[INFO] Logs moved to $OUTPUT_DIR/logs/"
else
    echo "[WARNING] Output directory $OUTPUT_DIR not found, logs remain in logs/"
fi

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time (seconds): $(( $END-$START ))"
