# SAM Trainer

A lightweight Python package for training [micro-SAM](https://github.com/computational-cell-analytics/micro-sam) models with optional data augmentation. Designed for fast iteration on bio-imaging instance segmentation tasks.

## Features

- 🔄 **Flexible data augmentation** with multiple output formats (OME-Zarr, TIF, HDF5)
- 🎯 **Instance segmentation training** for SAM models
- 🔬 **Production inference** with two modes:
  - **AIS (Decoder)**: Fast instance segmentation with tunable thresholds
  - **AMG**: Automatic Mask Generation for zero-shot segmentation
- 🗃️ **NGIO-powered OME-Zarr support**:
  - Single OME-Zarr images with in-place label writing
  - HCS plate structures (wells/fields)
  - TIFF files with optional tiling for large images
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

#### Two Segmentation Modes

**1. AIS (Decoder-Based) - Default**

- Uses your trained decoder for fast instance segmentation
- Requires decoder threshold tuning for optimal results
- Best for production inference on similar data to training

**2. AMG (Automatic Mask Generation)**

- Zero-shot segmentation using trained encoder
- No threshold tuning needed
- Better for diverse or unseen data

#### TIFF/Single OME-Zarr Inference

```bash
# AIS mode with default thresholds (0.5, 0.5, 0.5)
pixi run python sam_trainer/run_inference.py \
    --model final_models/best.pt \
    --input dat/test_images/ \
    --output results/predictions/ \
    -vv

# AIS mode with tuned thresholds
pixi run python sam_trainer/run_inference.py \
    --model final_models/best.pt \
    --input dat/test_images/ \
    --output results/predictions/ \
    --center-dist-thresh 0.4 \
    --boundary-dist-thresh 0.4 \
    --foreground-thresh 0.6 \
    -vv

# AMG mode
pixi run python sam_trainer/run_inference.py \
    --model final_models/best.pt \
    --input dat/test_images/ \
    --output results/predictions/ \
    --use-amg \
    -vv

# OME-Zarr input (labels written back to zarr)
pixi run python sam_trainer/run_inference.py \
    --model final_models/best.pt \
    --input dat/test_data.zarr \
    --label-name sam_segmentation \
    -vv

# Large images with tiling
pixi run python sam_trainer/run_inference.py \
    --model final_models/best.pt \
    --input dat/large_images/ \
    --output results/predictions/ \
    --tile-shape 512,512 \
    --halo 64,64 \
    -vv
```

#### HCS Plate Inference

```bash
# Process entire HCS plate with AIS mode
pixi run python sam_trainer/run_inference_hcs.py \
    --input dat/exp168-diff8.zarr \
    --model final_models/best.pt \
    --label-name ais_default \
    -vv

# Process specific wells only
pixi run python sam_trainer/run_inference_hcs.py \
    --input dat/exp168-diff8.zarr \
    --model final_models/best.pt \
    --wells B02 B03 C02 \
    -vv

# AMG mode on HCS plate
pixi run python sam_trainer/run_inference_hcs.py \
    --input dat/exp168-diff8.zarr \
    --model final_models/best.pt \
    --use-amg \
    -vv
```

#### SLURM Submission

```bash
# TIFF/single zarr inference
sbatch scripts/submit_inference.sh \
    final_models/best.pt \
    dat/test_images/ \
    results/predictions/ \
    --center-dist-thresh 0.4 \
    --boundary-dist-thresh 0.4

# HCS plate inference
sbatch scripts/submit_inference_hcs.sh \
    final_models/best.pt \
    dat/exp168-diff8.zarr \
    ais_relaxed \
    --center-dist-thresh 0.6 \
    --boundary-dist-thresh 0.6 \
    --foreground-thresh 0.4
```

#### Decoder Threshold Tuning

The three decoder thresholds control how distance maps convert to instance masks:

- **`center_distance_threshold`** (default: 0.5): Controls center point detection
- **`boundary_distance_threshold`** (default: 0.5): Controls boundary detection
- **`foreground_threshold`** (default: 0.5): Controls foreground/background separation

**Guidelines:**

- **Lower thresholds** (0.3-0.4): More permissive, finds more instances, possible false positives
- **Higher thresholds** (0.6-0.7): More strict, fewer instances, possible false negatives
- **Default (0.5)**: Balanced, good starting point

**Recommended approach:**

1. Start with defaults (0.5, 0.5, 0.5)
2. If under-segmenting: try relaxed (0.6, 0.6, 0.4)
3. If over-segmenting: try strict (0.4, 0.4, 0.6)
4. Use test scripts to compare multiple configurations

#### Testing Multiple Configurations

```bash
# Test HCS plate with 4 different configurations
bash test_inference_commands.sh

# Test TIFF/zarr with 4 different configurations
bash test_inference_tiff_commands.sh
```

Results will be saved as separate label layers for side-by-side comparison in napari.

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

## Model Loading (Critical for Inference)

**IMPORTANT**: Always use the correct micro-SAM API to load your trained model.

### ✅ Correct Pattern

```python
from micro_sam.automatic_segmentation import get_predictor_and_segmenter

# Load model with decoder (AIS mode)
predictor, segmenter = get_predictor_and_segmenter(
    model_type="vit_b_lm",
    checkpoint="final_models/best.pt",
    device="cuda",
    amg=False  # Use trained decoder
)

# For decoder mode: MUST initialize before generate
segmenter.initialize(image)
predictions = segmenter.generate(
    center_distance_threshold=0.5,
    boundary_distance_threshold=0.5,
    foreground_threshold=0.5
)

# Load model with AMG
predictor, segmenter = get_predictor_and_segmenter(
    model_type="vit_b_lm",
    checkpoint="final_models/best.pt",
    device="cuda",
    amg=True  # Use automatic mask generation
)

# For AMG mode: use automatic_instance_segmentation
from micro_sam.instance_segmentation import automatic_instance_segmentation
masks = automatic_instance_segmentation(
    predictor, segmenter, input_path, output_path
)
```

### ❌ Common Mistakes

```python
# DON'T: Manually construct segmenter without loading checkpoint
from micro_sam.automatic_segmentation import get_amg
segmenter = get_amg(predictor, ...)  # Uses pretrained SAM, ignores your checkpoint!

# DON'T: Call generate without initialize (decoder mode)
predictions = segmenter.generate(...)  # Will produce garbage without initialize()

# DON'T: Forget decoder thresholds
predictions = segmenter.generate()  # Missing threshold parameters!
```

### Implementation Details

See `sam_trainer/utils/inference_utils.py` for the reference implementation:

- **`load_model_with_decoder()`**: Wraps `get_predictor_and_segmenter()` with proper checkpoint loading
- **`segment_image()`**: Handles both AIS and AMG modes with correct initialization patterns
- **`postprocess_masks()`**: Optional filtering by area, border margin, and instance count

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
├── sam_trainer/                      # Main package
│   ├── __init__.py
│   ├── config.py                     # Pydantic schemas
│   ├── io.py                         # Multi-format I/O
│   ├── augmentation.py               # Data augmentation
│   ├── training.py                   # Training logic
│   ├── cli.py                        # CLI commands
│   ├── run_inference.py              # TIFF/single zarr inference
│   ├── run_inference_hcs.py          # HCS plate inference
│   └── utils/
│       ├── inference_utils.py        # Shared inference utilities
│       └── logging.py                # Centralized logging
├── scripts/
│   ├── submit_training_a100.sh       # SLURM training (A100)
│   ├── submit_training_a40.sh        # SLURM training (A40)
│   ├── submit_inference.sh           # SLURM inference (TIFF/zarr)
│   └── submit_inference_hcs.sh       # SLURM inference (HCS)
├── configs/                          # Training configurations
│   ├── full_sam_vit_b_a100.yaml
│   ├── zarr_a100.yaml
│   └── ...
├── dat/                              # Local data directory (gitignored)
├── runs/                             # Training outputs (gitignored)
├── final_models/                     # Exported models
├── test_inference_commands.sh        # HCS test suite
├── test_inference_tiff_commands.sh   # TIFF/zarr test suite
├── INFERENCE_REFACTOR.md             # Inference refactor documentation
├── pixi.toml                         # Environment definition
├── pixi.lock                         # Locked dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
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
- [NGIO](https://github.com/fractal-analytics-platform/ngio) - OME-Zarr I/O

## Additional Resources

- **Inference Refactor Documentation**: See `INFERENCE_REFACTOR.md` for details on the correct micro-SAM API usage
- **AI Agent Instructions**: See `.github/copilot-instructions.md` for codebase conventions
- **Test Suites**: Run `test_inference_commands.sh` or `test_inference_tiff_commands.sh` to compare segmentation modes
