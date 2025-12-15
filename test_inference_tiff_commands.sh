#!/bin/bash

# Test commands for run_inference.py (TIFF and single zarr)
# Compare different decoder threshold configurations and AMG mode
# Uses unseen test set for evaluation

MODEL_PATH="final_models/full_img_vit_b_lm.pt"
INPUT_PATH="inference_test/unseen_test_set_168_8"
OUTPUT_BASE="inference_test/tiff_comparison"

echo "=== SAM Inference Testing Suite (TIFF/Single Zarr) ==="
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_PATH"
echo ""

# Test 1: AIS with default thresholds (0.5, 0.5, 0.5)
echo "[Test 1/4] AIS mode with default thresholds..."
sbatch scripts/submit_inference.sh \
    "$MODEL_PATH" \
    "$INPUT_PATH" \
    "${OUTPUT_BASE}/ais_default" \
    --center-dist-thresh 0.5 \
    --boundary-dist-thresh 0.5 \
    --foreground-thresh 0.5

echo ""

# Test 2: AIS with relaxed thresholds (0.6, 0.6, 0.4)
echo "[Test 2/4] AIS mode with relaxed thresholds (more permissive)..."
sbatch scripts/submit_inference.sh \
    "$MODEL_PATH" \
    "$INPUT_PATH" \
    "${OUTPUT_BASE}/ais_relaxed" \
    --center-dist-thresh 0.6 \
    --boundary-dist-thresh 0.6 \
    --foreground-thresh 0.4

echo ""

# Test 3: AIS with strict thresholds (0.4, 0.4, 0.6)
echo "[Test 3/4] AIS mode with strict thresholds (more conservative)..."
sbatch scripts/submit_inference.sh \
    "$MODEL_PATH" \
    "$INPUT_PATH" \
    "${OUTPUT_BASE}/ais_strict" \
    --center-dist-thresh 0.4 \
    --boundary-dist-thresh 0.4 \
    --foreground-thresh 0.6

echo ""

# Test 4: AMG mode (no thresholds needed)
echo "[Test 4/4] AMG mode (Automatic Mask Generation)..."
sbatch scripts/submit_inference.sh \
    "$MODEL_PATH" \
    "$INPUT_PATH" \
    "${OUTPUT_BASE}/amg_default" \
    --use-amg

echo ""
echo "All jobs submitted! Check logs/ for progress."
echo "Compare results in: $OUTPUT_BASE/"
