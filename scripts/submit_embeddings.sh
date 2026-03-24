#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=sam_embeddings
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=main
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="infiniband&gpuram80gb"
#SBATCH --time=08:00:00

# Usage:
#   Local CPU: bash scripts/submit_embeddings.sh --local --config configs/embeddings_vit_b_cpu_debug.yaml
#   Cluster GPU: sbatch scripts/submit_embeddings.sh --config configs/embeddings_vit_b_a100.yaml

set -eu

function display_memory_usage() {
    set +eu
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo -n "[INFO] [$(date -Iseconds)] [$$] Max memory usage in bytes: "
        cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes
        echo
    fi
}

trap display_memory_usage EXIT

LOCAL_MODE=false
CONFIG_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_PATH" ]]; then
    echo "[ERROR] Missing --config"
    echo "Usage: bash/sbatch scripts/submit_embeddings.sh --config <config.yaml> [--local] [extra args]"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

START=$(date +%s)
STARTDATE=$(date -Iseconds)

if [[ "$LOCAL_MODE" == true ]]; then
    echo "[INFO] [$STARTDATE] [$$] Starting local embeddings run"
else
    echo "[INFO] [$STARTDATE] [$$] Starting SLURM job $SLURM_JOB_ID"
    echo "[INFO] [$STARTDATE] [$$] Running in $(hostname -s)"
fi

echo "[INFO] [$STARTDATE] [$$] Working directory: $(pwd)"
echo "[INFO] [$STARTDATE] [$$] Config: $CONFIG_PATH"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

if [[ "$LOCAL_MODE" == true ]]; then
    ENV_NAME="cpu"
else
    ENV_NAME="gpu"
fi

echo "[INFO] [$STARTDATE] [$$] Using pixi environment: $ENV_NAME"
pixi run -e "$ENV_NAME" python -m sam_trainer.cli embeddings --config "$CONFIG_PATH" -v "${EXTRA_ARGS[@]}"

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time (seconds): $(( END-START ))"
