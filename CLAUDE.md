# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM Trainer is a Python package for fine-tuning [micro-SAM](https://github.com/computational-cell-analytics/micro-sam) models on microscopy data. It wraps `micro_sam.training` and `torch-em` with a Pydantic config system, Typer CLI, and SLURM scripts for HPC workflows.

## Commands

All tasks run through Pixi (never `pip install` or `conda`):

```bash
pixi run train --config configs/<name>.yaml    # Full pipeline (augment → train → export)
pixi run augment                                # Augmentation only
pixi run embeddings --config configs/<name>.yaml
pixi run config                                 # Interactive config builder
```

For local dev (CPU environment):
```bash
pixi run -e cpu train --config configs/<name>.yaml
```

SLURM submission (HPC cluster):
```bash
sbatch scripts/submit_training_a100.sh configs/<name>.yaml
sbatch scripts/submit_training_a40.sh configs/<name>.yaml
sbatch scripts/submit_inference.sh
sbatch scripts/submit_inference_hcs.sh
```

No formal test suite — inference is validated via shell scripts:
```bash
bash test_inference_commands.sh
bash test_inference_tiff_commands.sh
```

Linting: no pre-commit config present. Run `ruff check . && ruff format .` manually if needed.

## Architecture

**Core data flow:**
```
configs/*.yaml (Pydantic-validated)
  → cli.py (Typer orchestrator)
  → augmentation.py (Albumentations pipeline, optional)
  → training.py (torch-em loaders + micro_sam trainer)
  → runs/<experiment>/ (checkpoints, exported .pt, optional val preds)
```

**Key modules:**

- `config.py` — Pydantic schemas: `PipelineConfig`, `AugmentationConfig`, `TrainingConfig`, `EmbeddingsExtractionConfig`. All YAML configs map to these. Two mutually exclusive training data modes: traditional (`images_dir`/`labels_dir`) and Zarr (`train_zarr_path`/`val_zarr_path` with `raw_key`/`label_key`).
- `training.py` — Wraps `micro_sam.training.train_sam` and `train_instance_segmentation`. Percentile normalization (clips to configurable percentiles → uint8). Uses `MinInstanceSampler` for balanced patch sampling.
- `augmentation.py` — Albumentations-based; supports 2D and slice-wise 3D. Outputs OME-Zarr, TIF, or HDF5.
- `cli.py` — Typer CLI; all commands accept `-v`/`-vv`/`-vvv` verbosity.
- `io.py` — Format-agnostic reader (OME-Zarr via `ngio`, TIFF, HDF5). OME-Zarr preferred.
- `run_inference.py` — TIFF/Zarr inference; AIS (decoder) and AMG modes; optional tiling.
- `run_inference_hcs.py` — HCS plate inference (well/field traversal).
- `embeddings.py` — Per-organoid encoder embeddings from HCS plates; exports parquet artifacts.
- `utils/inference_utils.py` — Model loading, postprocessing, channel resolution shared across inference scripts.

**Output layout:**
```
runs/<experiment_name>/
  config.yaml             # frozen config copy
  checkpoints/<name>/best.pt, latest.pt
  <experiment_name>_model.pt   # exported model
  validation_preds/            # optional
```

## Config Conventions

Config files live in `configs/` named `{purpose}_{model}_{gpu}[_variant].yaml`.
Hardware notes: A100 → `batch_size: 4`; A40 → `batch_size: 1`.

Model variants: `vit_t/b/l/h` (general) or `vit_t/b/l_lm` (light microscopy, recommended for fluorescence).

AIS inference thresholds (all default 0.5): lower (0.3–0.4) → more instances; higher (0.6–0.7) → fewer.

## Key Dependencies

| Library | Role |
|---|---|
| `micro_sam` ≥1.6.2 | SAM training/inference API |
| `torch-em` | DataLoader + MinInstanceSampler |
| `pydantic` ≥2.0 | Config validation |
| `typer` ≥0.12 | CLI |
| `albumentations` ≥1.4 | Augmentation |
| `ngio` ≥0.4.4 | OME-Zarr I/O (prefer over zarr directly) |
| `tifffile` ≥2024 | TIFF I/O |

Environments: `gpu` (CUDA 12.6, cluster) and `cpu` (local dev) defined in `pixi.toml`.
