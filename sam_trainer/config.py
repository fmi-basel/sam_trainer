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

    # Data paths
    images_dir: Path = Field(..., description="Directory containing training images")
    labels_dir: Path = Field(..., description="Directory containing label masks")

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
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate")
    val_split: float = Field(
        default=0.1, gt=0, lt=1, description="Validation split ratio"
    )

    # Checkpoint configuration
    checkpoint_name: str = Field(..., description="Name for this training run")
    resume_from_checkpoint: Optional[Path] = Field(
        default=None, description="Path to checkpoint to resume from"
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
