# SAM Trainer

A lightweight Python package for training [micro-SAM](https://github.com/computational-cell-analytics/micro-sam) models with optional data augmentation. Designed for fast iteration on bio-imaging instance segmentation tasks.

## Features

- 🔄 **Flexible data augmentation** with multiple output formats (OME-Zarr, TIF, HDF5)
- 🎯 **Instance segmentation training** for SAM models
- 🖥️ **GPU/CPU auto-detection** with automatic fallback
- ⚙️ **Interactive config builder** with validation
- 🚀 **HPC-ready** with SLURM batch scripts
- 📊 **Multiple input formats**: OME-Zarr, TIF, HDF5 (2D and 3D)
- 🎨 **Rich CLI** with progress indicators and colored output

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd sam_trainer

# Install dependencies with pixi
pixi install

# Or update an existing environment
pixi update
```

## Quick Start

### 1. Create Configuration

Use the interactive config builder:

```bash
pixi run config --output my_experiment.yaml
```

Or use pixi tasks:

```bash
pixi run config --output my_experiment.yaml
```

This will guide you through:

- Experiment name and output directories
- Augmentation settings (optional)
- Training hyperparameters
- Model selection

### 2. Run Training

Train with the generated config:

```bash
pixi run train --config my_experiment.yaml -vv
```

Verbosity levels:

- `-v`: INFO level logging
- `-vv`: DEBUG level logging  
- `-vvv`: Maximum verbosity

### 3. Run on HPC Cluster

Submit to SLURM:

```bash
sbatch scripts/submit_training.sh my_experiment.yaml
```

The script automatically:

- Allocates GPU resources (V100 compatible)
- Activates pixi environment
- Runs training with proper logging
- Tracks memory usage

## Usage Examples

### Augmentation Only

- 90-degree rotations
- Horizontal/vertical flips
- Gaussian blur (range 3-11 pixels)
- Multiplicative noise (±5% intensity variation)
- Very subtle brightness (±5%) and contrast (±2%)

Run data augmentation without training:

```bash
pixi run augment \
    --images data/raw/images \
    --labels data/raw/labels \
    --output data/augmented \
    --n-aug 5 \
    --format ome-zarr \
    -vv
```

### Training Without Augmentation

Create a config without augmentation:

```yaml
experiment_name: "my_training"
output_base_dir: "runs"
augmentation: null  # Skip augmentation
training:
  images_dir: "data/train/images"
  labels_dir: "data/train/labels"
  model_type: "vit_b_lm"
  patch_shape: [512, 512]
  batch_size: 1
  n_epochs: 100
  learning_rate: 1.0e-05
  val_split: 0.1
  checkpoint_name: "my_training"
```

Then train:

```bash
pixi run train --config config.yaml -v
```

### Resume from Checkpoint

In your config:

```yaml
training:
  resume_from_checkpoint: "runs/previous_exp/checkpoints/best.pt"
  # ... other settings
```

### Inference

Run the bash script on the cluster or run `run_inference.py` directly:

```bash
sbatch scripts/submit_inference.sh \
    runs/my_experiment/checkpoints/best.pt \
    dat/test_images/ \
    runs/my_experiment/predictions/ \
    --use-decoder \
    --center-dist-thresh 0.4 \
    --boundary-dist-thresh 0.4
```

## Configuration Reference

### Augmentation Config

```yaml
augmentation:
  input_images_dir: "data/raw/images"
  input_labels_dir: "data/raw/labels"
  output_dir: "data/augmented"
  output_format: "ome-zarr"  # or "tif", "hdf5"
  n_augmentations: 3
  rotation_range: 45
  flip_horizontal: true
  flip_vertical: true
  gaussian_blur_prob: 0.3
  gaussian_noise_prob: 0.3
  brightness_contrast: true
  elastic_transform: false
```

### Training Config

```yaml
training:
  images_dir: "data/train/images"
  labels_dir: "data/train/labels"
  model_type: "vit_b_lm"  # vit_t, vit_b, vit_l, vit_h, vit_t_lm, vit_b_lm, vit_l_lm
  patch_shape: [512, 512]
  batch_size: 1
  n_epochs: 100
  learning_rate: 1.0e-05
  val_split: 0.1
  checkpoint_name: "my_model"
  resume_from_checkpoint: null
  export_path: null  # Auto-generated if null
```

## Supported Formats

### Input Formats

- **OME-Zarr**: Directory-based format (detects via `.zattrs`/`.zgroup`)
- **TIF/TIFF**: Single or multi-page TIFF files
- **HDF5**: `.h5` or `.hdf5` files (auto-detects dataset)
- **Zarr**: `.zarr` files

### Output Formats (Augmentation)

- **OME-Zarr** (default): Best for large datasets, cloud-ready
- **TIF**: Standard format, good compression
- **HDF5**: Good for complex metadata

## Data Organization

### For Non-Zarr Formats

```text
data/
├── images/
│   ├── img001.tif
│   ├── img002.tif
│   └── img003.tif
└── labels/
    ├── img001.tif
    ├── img002.tif
    └── img003.tif
```

### For OME-Zarr

```text
data/
├── images/
│   ├── img001.zarr/
│   ├── img002.zarr/
│   └── img003.zarr/
└── labels/
    ├── img001.zarr/
    ├── img002.zarr/
    └── img003.zarr/
```

## Output Structure

After training, your experiment directory will look like:

```text
runs/
└── my_experiment/
    ├── config.yaml                    # Copy of configuration
    ├── checkpoints/
    │   └── my_model/
    │       ├── best.pt                # Best checkpoint
    │       └── latest.pt              # Latest checkpoint
    └── my_experiment_model.pt         # Exported model
```

## Model Types

- `vit_t`: ViT-Tiny (fastest, least accurate)
- `vit_b`: ViT-Base (good balance)
- `vit_l`: ViT-Large (slow, most accurate)
- `vit_h`: ViT-Huge (very slow)
- `vit_t_lm`, `vit_b_lm`, `vit_l_lm`: Light microscopy variants (recommended for microscopy)

**Recommendation**: Use `vit_b_lm` for light microscopy images.

## Hardware Requirements

- **GPU**: V100 or better recommended (32GB VRAM)
- **CPU**: Multi-core recommended for augmentation
- **RAM**: 32GB+ recommended for larger images
- **Storage**: SSD recommended for fast I/O

The package automatically detects GPU availability and falls back to CPU if needed.

## Troubleshooting

### Import Errors

If you see import errors after installation, run:

```bash
pixi update
```

### GPU Not Detected

Check PyTorch CUDA availability:

```bash
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

Reduce:

- `batch_size` (try 1)
- `patch_shape` (e.g., 256x256 instead of 512x512)
- `n_augmentations`

### Data Format Issues

Enable debug logging to see what's being loaded:

```bash
pixi run train --config config.yaml -vvv
```

## Development

```bash
# Run with verbose logging
pixi run train --config config.yaml -vv

# Test augmentation
pixi run augment --images test_data/images --labels test_data/labels --output test_out -n 2 -vv

# Build config interactively
pixi run config -o test_config.yaml
```

## Project Structure

```text
sam_trainer/
├── sam_trainer/              # Main package
│   ├── __init__.py
│   ├── config.py             # Pydantic schemas
│   ├── io.py                 # Multi-format I/O
│   ├── augmentation.py       # Data augmentation
│   ├── training.py           # Training logic
│   └── cli.py                # CLI commands
├── scripts/
│   └── submit_training.sh    # SLURM batch script
├── dat/                      # Local data directory (gitignored)
├── runs/                     # Training outputs (gitignored)
├── pixi.toml                 # Environment definition
├── pixi.lock                 # Locked dependencies
├── example_config.yaml       # Example configuration
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Contributing

This is a lightweight, project-specific package. For issues or feature requests, please open an issue.

## License

[Your license here]

## Acknowledgments

Built on top of:

- [micro-SAM](https://github.com/computational-cell-analytics/micro-sam)
- [torch-em](https://github.com/constantinpape/torch-em)
- [albumentations](https://github.com/albumentations-team/albumentations)

# Basic usage

python scripts/run_inference.py \
    --model runs/training_original_images/checkpoints/training_original_40_images/best.pt \
    --input dat/test_images/ \
    --output results/predictions/

# With verbosity

python scripts/run_inference.py -m model.pt -i images/ -o masks/ -vv

# With tiling for large images (recommended for 2048x2048+ images)

python scripts/run_inference.py \
    -m model.pt -i images/ -o masks/ \
    --tile-shape 512,512 --halo 64,64

# Different file pattern

python scripts/run_inference.py -m model.pt -i images/ -o masks/ --pattern "*.tiff"
