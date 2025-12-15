#!/bin/bash
# Test inference commands for HCS plate segmentation
# Run these to test different segmentation modes and parameters

# Model path
MODEL="final_models/full_img_vit_b_lm.pt"
INPUT="inference_test/exp168-diff8.zarr"

# # Test 1: Decoder mode with default thresholds
# echo "Test 1: AIS (decoder) with default thresholds"
# sbatch scripts/submit_inference_hcs.sh "$INPUT" \
#     --model "$MODEL" \
#     --label-name ais_default \
#     -v

# Test 2: Decoder mode with relaxed thresholds (may find more instances)
echo "Test 2: AIS (decoder) with relaxed thresholds"
sbatch scripts/submit_inference_hcs.sh "$INPUT" \
    --model "$MODEL" \
    --label-name ais_relaxed \
    --center-dist-thresh 0.7 \
    --boundary-dist-thresh 0.7 \
    --foreground-thresh 0.4 \
    -v

# Test 3: Decoder mode with strict thresholds (fewer false positives)
echo "Test 3: AIS (decoder) with strict thresholds"
sbatch scripts/submit_inference_hcs.sh "$INPUT" \
    --model "$MODEL" \
    --label-name ais_strict \
    --center-dist-thresh 0.2 \
    --boundary-dist-thresh 0.2 \
    --foreground-thresh 0.6 \
    -v

# Test 4: AMG mode
echo "Test 4: AMG (automatic mask generation)"
sbatch scripts/submit_inference_hcs.sh "$INPUT" \
    --model "$MODEL" \
    --label-name amg_default \
    --use-amg \
    -v

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/hcs_inf_<jobid>.out"
echo ""
echo "Results will be in: $INPUT/*.zarr/labels/<label_name>/"
