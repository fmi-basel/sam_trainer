# SAM Trainer - AI Agent Instructions

A training and inference framework for microscopy instance segmentation using [micro-SAM](https://github.com/computational-cell-analytics/micro-sam). Built on PyTorch, manages full pipeline from data augmentation to model export.

## Core Architecture

**Three-Module Structure:**
- `sam_trainer/config.py`: Pydantic schemas (`PipelineConfig`, `TrainingConfig`, `AugmentationConfig`) with strict validation
- `sam_trainer/training.py`: Wraps `micro_sam.training` API (train_sam, train_instance_segmentation, export_instance_segmentation_model)
- `sam_trainer/cli.py`: Typer-based CLI orchestrating augmentation → training → export

**Data Flow:**
1. Raw images/labels → `augmentation.py` (optional Albumentations pipeline) → augmented dataset
2. Augmented data → `training.py` (torch-em dataloaders + micro-SAM trainer) → checkpoints
3. Best checkpoint → `export_instance_segmentation_model` → final `.pt` (includes decoder state)

**Two Training Modes:**
- **Traditional**: Separate `images_dir/` and `labels_dir/` with matched filenames, auto train/val split
- **Zarr**: Pre-split `train_zarr_path` and `val_zarr_path` containers with `raw_key` and `label_key` for hierarchical data access

## Critical Conventions

### Line Endings (LF Only)
- This repo is primarily executed on Linux/HPC. Text files must use LF line endings.
- Enforce LF via `.gitattributes` (`* text=auto eol=lf`) and `.editorconfig` (`end_of_line = lf`).
- Avoid introducing CRLF from Windows editors; if Git shows a “full-file changed” diff with no visible edits, check for CRLF↔LF conversion.

### Config Files (`configs/*.yaml`)
- **Naming**: `{purpose}_{model}_{gpu}[_variant].yaml` (e.g., `full_sam_vit_b_a100_high_lr.yaml`)
- **Distinguish hardware**: A100 configs (80GB VRAM) use `batch_size: 4`, A40 (48GB) use `batch_size: 1`
- **Always set**: `checkpoint_name`, `export_path`, `output_base_dir` for deterministic artifact paths
- **Training modes are mutually exclusive**: Use `train_zarr_path`/`val_zarr_path` OR `images_dir`/`labels_dir`, never both

### Checkpoint Structure
```
runs/<experiment>/
├── config.yaml              # Frozen copy of input config
├── checkpoints/
│   └── <checkpoint_name>/
│       ├── best.pt          # Tracked by micro-SAM trainer
│       └── latest.pt
└── <experiment>_model.pt    # Exported with decoder state
```

### Preprocessing Pipeline
1. **Percentile normalization**: `PercentileNormalizer` clips to [1.0, 99.5] percentiles by default, maps to uint8
2. **Assumption**: Foreground objects are **darker** than background (brightfield microscopy)
3. Keep normalization parameters in `TrainingConfig` (`normalize_lower_percentile`, `normalize_upper_percentile`)

### Inference Scripts

**IMPORTANT: Always use NGIO for OME-Zarr operations.** NGIO is the preferred library for all OME-Zarr I/O throughout this codebase.

`run_inference.py`: Unified inference supporting:
    - TIFF files (single/batch) with optional tiling (`--tile-shape`, `--halo`)
    - Single OME-Zarr images using NGIO (labels written back to zarr)
    - Auto-detects input type (TIFF vs OME-Zarr)
`run_inference_hcs.py`: HCS OME-Zarr plates using NGIO (wells/fields structure)
`inference_utils.py`: Shared utilities (model loading, segmentation, postprocessing) for reuse across scripts
### Adding Experiments
1. Copy nearest config from `configs/` (match backbone and GPU)
2. Adjust `experiment_name`, `checkpoint_name`, `export_path`
3. Tune hyperparameters (start with `learning_rate`, `batch_size`, `n_samples`)
4. Submit via `scripts/submit_training_a100.sh <config.yaml>`

### SLURM Specifics
- **Default partition**: `main` with A100 GPUs (up to 4/node, 2 nodes/job)
- **Max walltime**: 56h (set in `submit_training_a100.sh`)
- **Memory**: 150GB default for A100 jobs, scale up to 600GB if needed
- **Threading**: `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` (preset in scripts)
- **Environment**: Use `pixi run -e gpu` for CUDA 12.6 + PyTorch GPU build

### Package Management (Pixi)
- **Tasks**: `pixi run train`, `pixi run augment`, `pixi run config` (defined in `pixi.toml`)
- **Environments**: `gpu` (CUDA 12.6) for cluster, `cpu` for local dev (auto-selected)
- **Editable install**: sam_trainer installed via `pypi-dependencies` with `editable = true`

### Data Organization
**Traditional Mode:**
```
dat/augmented_training_data/
├── images/
│   ├── img001.tif
│   └── img002.tif
└── labels/
    ├── img001.tif
    └── img002.tif
```

**Zarr Mode:** Pre-split containers like `accumulated_train.zarr` with keys `"0"` (raw) and `"labels/mask/0"` (labels)

## Key Micro-SAM Integration Points

### Training Entry Points
- **Full model fine-tuning**: `train_sam()` with `train_instance_segmentation_only=False`
- **Decoder-only**: `train_instance_segmentation()` for faster iteration
- **Dataloader**: `default_sam_loader()` with custom `raw_transform=PercentileNormalizer(...)`

### Sampler Configuration
- **MinInstanceSampler**: Rejects patches with <N instances (set `min_instances_per_patch`, `min_instance_size`)
- **Disable for full images**: Set `use_min_instance_sampler: false` when `patch_shape == image_size`

### Export Format
```python
export_instance_segmentation_model(
    checkpoint_path=...,
    model_type="vit_b_lm",
    save_path=export_path
)
# Saves: {image_encoder, prompt_encoder, mask_decoder, decoder_state} dict
```

## Common Pitfalls

1. **Batch size OOM**: A100 (80GB) handles batch_size=4 for 512×512 patches. Reduce if training larger patches or full images.
2. **Zarr key errors**: Ensure `raw_key` and `label_key` match actual hierarchy (use `zarr.open()` to inspect)
3. **Val split in Zarr mode**: Ignored! Data must be pre-split into separate train/val containers.
4. **Checkpoint resume**: Provide absolute path in `resume_from_checkpoint`, not relative to experiment dir
5. **Model cache**: First run downloads SAM backbones to `~/.cache/micro_sam` (requires internet once)
6. **Inference tiling**: Use `--tile-shape 512,512 --halo 64,64` for >2048×2048 images to avoid OOM

## Testing & Validation

- **Quick test**: Use `configs/quick_test_vit_b_a100.yaml` (5 epochs, 10 samples/epoch)
- **Validation predictions**: Set `save_validation_predictions_frequency: 5` to visualize every N epochs (saved to `runs/<exp>/validation_preds/`)
- **Inference sanity check**: Run `scripts/submit_inference.sh` on held-out data in `dat/test_images/`

## When Modifying Core Logic

- **Config schema changes**: Update `sam_trainer/config.py` Pydantic models + add `@field_validator` for constraints
- **Dataloader changes**: Modify `training.py:prepare_data_splits()` or `build_loaders_zarr()`, test both modes
- **New augmentations**: Extend `augmentation.py:create_augmentation_pipeline()` (Albumentations compatible)
- **Inference modes**: Add new scripts to `sam_trainer/` (not `scripts/`), keep SLURM wrappers in `scripts/`

## References

- micro-SAM docs: https://computational-cell-analytics.github.io/micro-sam/
- torch-em (underlying dataloader): https://github.com/constantinpape/torch-em
- NGIO (OME-Zarr handling): https://github.com/fractal-analytics-platform/ngio
- Existing project instructions: `prompts/project.instructions.md` (HPC-specific conventions)
