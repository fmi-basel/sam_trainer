# Inference Refactor Summary

## What Was Fixed

### Root Cause

The inference code was broken because:

1. **AMG mode**: Was using `get_amg()` which doesn't load your trained checkpoint - it used pretrained SAM weights only
2. **Decoder mode**: Wasn't calling `segmenter.initialize(image)` before `segmenter.generate()`, which is required for `InstanceSegmentationWithDecoder`
3. **Missing parameters**: No decoder thresholds (`center_distance_threshold`, etc.) were being passed

### Solution

Reverted to the working pattern from commit 41f1708:

- Use `get_predictor_and_segmenter()` with `amg=True/False` parameter
- For decoder mode: call `initialize()` then `generate()` with threshold parameters
- For AMG mode: use `automatic_instance_segmentation()`

## Key Changes

### `sam_trainer/utils/inference_utils.py`

- **`load_model_with_decoder()`**: Now uses `get_predictor_and_segmenter()` instead of manual model loading
- **`segment_image()`**: Added decoder mode handling with `initialize()` + `generate()` pattern
- Added `generate_kwargs` parameter for decoder thresholds

### `sam_trainer/run_inference_hcs.py` & `sam_trainer/run_inference.py`

Both inference scripts updated with:

- Added CLI parameters for decoder thresholds:
  - `--center-dist-thresh` (default: 0.5)
  - `--boundary-dist-thresh` (default: 0.5)
  - `--foreground-thresh` (default: 0.5)
- Pass these through the entire processing chain
- Both use the same refactored `inference_utils.py`

### SLURM Scripts

Both updated to pass extra arguments through:

- `scripts/submit_inference_hcs.sh` - for HCS plates
- `scripts/submit_inference.sh` - for TIFF/single zarr

## Test Commands

**HCS Plates:** See `test_inference_commands.sh` for HCS zarr plates
**TIFF/Single Zarr:** See `test_inference_tiff_commands.sh` for TIFF directories

Both test:

1. **AIS with default thresholds** - baseline decoder segmentation
2. **AIS with relaxed thresholds** (0.6, 0.6, 0.4) - may find more instances
3. **AIS with strict thresholds** (0.4, 0.4, 0.6) - fewer false positives
4. **AMG mode** - automatic mask generation with your trained model

### Run Tests

```bash
# HCS plates
bash test_inference_commands.sh

# TIFF/single zarr
bash test_inference_tiff_commands.sh
```

**HCS results** will be saved as separate label layers in your zarr files:

- `ais_default/`
- `ais_relaxed/`
- `ais_strict/`
- `amg_default/`

**TIFF results** will be saved as separate directories:

- `inference_test/tiff_comparison/ais_default/`
- `inference_test/tiff_comparison/ais_relaxed/`
- `inference_test/tiff_comparison/ais_strict/`
- `inference_test/tiff_comparison/amg_default/`

## Next Steps

1. Run the test commands on the cluster
2. View results in napari side-by-side to compare modes
3. Adjust thresholds based on which works best
4. Use best mode for production runs

## Important Notes

- The decoder thresholds are CRITICAL - they control how instances are identified from the predicted distance maps
- Lower thresholds = more permissive (more instances, possible false positives)
- Higher thresholds = more strict (fewer instances, possible false negatives)
- AMG mode doesn't use these thresholds - it uses different internal parameters
