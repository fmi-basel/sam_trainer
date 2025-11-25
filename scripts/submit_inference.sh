#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=sam_inference
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

# SAM Inference SLURM Batch Script
# Usage: sbatch scripts/submit_inference.sh <model_path> <input_dir> <output_dir> [extra_args]

set -eu

if [ $# -lt 3 ]; then
    echo "Usage: sbatch scripts/submit_inference.sh <model_path> <input_dir> <output_dir> [extra_args]"
    exit 1
fi

MODEL_PATH=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
shift 3
EXTRA_ARGS="$@"

echo "[INFO] Starting inference job $SLURM_JOB_ID"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Input: $INPUT_DIR"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Extra Args: $EXTRA_ARGS"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run inference
pixi run -e gpu python scripts/run_inference.py \
    --model "$MODEL_PATH" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo "[INFO] Inference complete"
