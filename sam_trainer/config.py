"""Configuration schemas for SAM training pipeline using Pydantic."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AugmentationConfig(BaseModel):
    """Configuration for data augmentation."""

    input_images_dir: Path = Field(..., description="Directory containing input images")
    input_labels_dir: Path = Field(..., description="Directory containing label masks")
    output_dir: Path = Field(..., description="Directory to save augmented data")
    output_format: Literal["ome-zarr", "tif", "hdf5", "original"] = Field(
        default="original",
        description="Output format for augmented data (original=preserve input format)",
    )
    include_original: bool = Field(
        default=True,
        description="If True, save original images alongside augmented versions",
    )

    # Augmentation parameters
    n_augmentations: int = Field(
        default=3,
        ge=0,
        description="Number of augmentations per image (0=no augmentation)",
    )
    treat_3d_as_2d: bool = Field(
        default=False,
        description="If True, treat 3D stacks as separate 2D slices with independent augmentations. If False, apply same augmentation to all slices in a stack.",
    )
    rotation_range: int = Field(
        default=90, ge=0, le=180, description="Random rotation range in degrees"
    )
    flip_horizontal: bool = Field(
        default=True, description="Apply random horizontal flips"
    )
    flip_vertical: bool = Field(default=True, description="Apply random vertical flips")
    gaussian_blur_prob: float = Field(
        default=0.3, ge=0, le=1, description="Probability of applying Gaussian blur"
    )
    gaussian_noise_prob: float = Field(
        default=0.3, ge=0, le=1, description="Probability of adding Gaussian noise"
    )
    brightness_contrast: bool = Field(
        default=True, description="Apply random brightness/contrast adjustments"
    )

    @field_validator("input_images_dir", "input_labels_dir")
    @classmethod
    def validate_input_dirs_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Directory does not exist: {v}")
        return v

    class Config:
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Configuration for SAM model training."""

    # Data paths - Traditional mode
    images_dir: Optional[Path] = Field(
        default=None, 
        description="Directory containing training images (traditional mode only)"
    )
    labels_dir: Optional[Path] = Field(
        default=None, 
        description="Directory containing label masks (traditional mode only)"
    )
    
    # Data paths - Zarr mode
    train_zarr_path: Optional[Path] = Field(
        default=None,
        description="Path to training zarr container (zarr mode only)"
    )
    val_zarr_path: Optional[Path] = Field(
        default=None,
        description="Path to validation zarr container (zarr mode only)"
    )
    raw_key: Optional[str] = Field(
        default=None,
        description="Key for raw data in zarr containers (e.g., '0'). Required for zarr mode."
    )
    label_key: Optional[str] = Field(
        default=None,
        description="Key for labels in zarr containers (e.g., 'labels/mask/0'). Required for zarr mode."
    )

    # Model configuration
    model_type: Literal[
        "vit_t", "vit_b", "vit_l", "vit_h", "vit_t_lm", "vit_b_lm", "vit_l_lm"
    ] = Field(default="vit_b_lm", description="SAM model type to train")

    # Training parameters
    patch_shape: tuple[int, int] = Field(
        default=(512, 512), description="Patch size for training"
    )
    batch_size: int = Field(default=1, ge=1, description="Training batch size")
    n_epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    n_samples: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of random patches per image per epoch (None=auto, typically 16)",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of dataloader workers for parallel data loading (0=single-threaded)",
    )
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate")
    early_stopping: int = Field(
        default=10,
        ge=1,
        description="Early stopping patience: number of epochs without improvement before stopping",
    )
    val_split: float = Field(
        default=0.1, gt=0, lt=1, description="Validation split ratio"
    )
    shuffle_data: bool = Field(
        default=True,
        description="Whether to shuffle data before splitting into train/validation",
    )
    shuffle_seed: Optional[int] = Field(
        default=None,
        description="Optional random seed for shuffling (None = random seed)",
    )
    train_instance_segmentation_only: bool = Field(
        default=True,
        description="If True, train only the instance decoder; if False, fine-tune the full SAM",
    )
    normalize_inputs: bool = Field(
        default=True,
        description="Normalize raw intensities to 8-bit using percentiles before training",
    )
    normalize_lower_percentile: float = Field(
        default=1.0,
        ge=0,
        lt=100,
        description="Lower percentile for intensity clipping",
    )
    normalize_upper_percentile: float = Field(
        default=99.5,
        gt=0,
        le=100,
        description="Upper percentile for intensity clipping",
    )
    use_min_instance_sampler: bool = Field(
        default=True,
        description="Use a MinInstanceSampler to ensure patches contain foreground instances",
    )
    min_instances_per_patch: int = Field(
        default=2,
        ge=1,
        description="Minimum number of distinct instances required per sampled patch",
    )
    min_instance_size: int = Field(
        default=25,
        ge=1,
        description="Minimum instance size (in pixels) to consider for the sampler",
    )

    # Checkpoint configuration
    checkpoint_name: str = Field(..., description="Name for this training run")
    resume_from_checkpoint: Optional[Path] = Field(
        default=None, description="Path to checkpoint to resume from"
    )
    save_validation_predictions_frequency: Optional[int] = Field(
        default=None,
        ge=1,
        description="Save validation predictions every N epochs for visualization (None=disabled)",
    )

    # Output configuration
    export_path: Optional[Path] = Field(
        default=None, description="Path to export final model (auto-generated if None)"
    )

    @field_validator("images_dir", "labels_dir")
    @classmethod
    def validate_data_dirs_exist(cls, v: Path) -> Path:
        # Don't validate existence during config creation
        # Directories might be created by augmentation step before training
        # Validation will happen when training actually starts
        return v

    @field_validator("resume_from_checkpoint")
    @classmethod
    def validate_checkpoint_exists(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"Checkpoint file does not exist: {v}")
        return v

    @field_validator("normalize_upper_percentile")
    @classmethod
    def validate_percentiles(cls, v: float, info) -> float:
        lower = info.data.get("normalize_lower_percentile", 0.0)
        if v <= lower:
            raise ValueError(
                "normalize_upper_percentile must be greater than normalize_lower_percentile"
            )
        return v
    
    @field_validator("label_key")
    @classmethod
    def validate_data_paths(cls, v: Optional[str], info) -> Optional[str]:
        """Validate data path configuration - must use either traditional or zarr mode, not both."""
        train_zarr = info.data.get("train_zarr_path")
        val_zarr = info.data.get("val_zarr_path")
        images_dir = info.data.get("images_dir")
        labels_dir = info.data.get("labels_dir")
        raw_key = info.data.get("raw_key")
        
        # Check if zarr mode
        zarr_mode = train_zarr is not None or val_zarr is not None
        # Check if traditional mode
        trad_mode = images_dir is not None or labels_dir is not None
        
        if zarr_mode and trad_mode:
            raise ValueError(
                "Cannot use both zarr mode (train_zarr_path/val_zarr_path) and "
                "traditional mode (images_dir/labels_dir) simultaneously. Choose one."
            )
        
        if not zarr_mode and not trad_mode:
            raise ValueError(
                "Must specify either zarr paths (train_zarr_path, val_zarr_path) "
                "or traditional directories (images_dir, labels_dir)"
            )
        
        # Validate zarr mode requirements
        if zarr_mode:
            if not train_zarr or not val_zarr:
                raise ValueError(
                    "Zarr mode requires both train_zarr_path and val_zarr_path"
                )
            if not raw_key or not v:
                raise ValueError(
                    "Zarr mode requires both raw_key and label_key. "
                    "Example: raw_key='0', label_key='labels/mask/0'"
                )
        
        # Validate traditional mode requirements
        if trad_mode:
            if not images_dir or not labels_dir:
                raise ValueError(
                    "Traditional mode requires both images_dir and labels_dir"
                )
        
        return v
    
    def is_zarr_mode(self) -> bool:
        """Check if configuration is in zarr container mode."""
        return (
            self.train_zarr_path is not None and 
            self.val_zarr_path is not None
        )

    class Config:
        extra = "forbid"


class PipelineConfig(BaseModel):
    """Complete pipeline configuration including augmentation and training."""

    experiment_name: str = Field(..., description="Name of the experiment")
    output_base_dir: Path = Field(
        default=Path("runs"), description="Base directory for all outputs"
    )

    augmentation: Optional[AugmentationConfig] = Field(
        default=None, description="Augmentation config (None to skip)"
    )
    training: TrainingConfig = Field(..., description="Training configuration")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    @property
    def experiment_dir(self) -> Path:
        """Get the experiment output directory."""
        return self.output_base_dir / self.experiment_name

    class Config:
        extra = "forbid"
