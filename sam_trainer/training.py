"""Training module for SAM instance segmentation models."""

import logging
from pathlib import Path

import torch
from micro_sam.training import (
    default_sam_loader,
    export_instance_segmentation_model,
    train_instance_segmentation,
)

from sam_trainer.config import TrainingConfig
from sam_trainer.io import get_image_paths

logger = logging.getLogger(__name__)


def prepare_data_splits(
    images_dir: Path, labels_dir: Path, val_split: float
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """Prepare train/validation splits from data directories.

    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing label masks
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    image_paths = get_image_paths(images_dir)
    label_paths = get_image_paths(labels_dir)

    if len(image_paths) != len(label_paths):
        raise ValueError(
            f"Number of images ({len(image_paths)}) and labels ({len(label_paths)}) must match"
        )

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_dir}")

    # Calculate split index
    n_total = len(image_paths)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    logger.info(f"Total samples: {n_total}, Train: {n_train}, Val: {n_val}")

    # Split data (last n_val samples for validation)
    train_images = image_paths[:n_train]
    train_labels = label_paths[:n_train]
    val_images = image_paths[n_train:]
    val_labels = label_paths[n_train:]

    return train_images, train_labels, val_images, val_labels


def run_training(config: TrainingConfig, output_dir: Path) -> dict[str, Path]:
    """Run SAM model training.

    Args:
        config: Training configuration
        output_dir: Directory to save checkpoints and exported model

    Returns:
        Dictionary with paths to checkpoint and exported model
    """
    logger.info("Starting SAM training...")
    logger.info(f"Model type: {config.model_type}")
    logger.info(f"Patch shape: {config.patch_shape}")
    logger.info(f"Batch size: {config.batch_size}")

    # Validate data directories exist before starting
    if not config.images_dir.exists():
        raise FileNotFoundError(
            f"Images directory does not exist: {config.images_dir}\n"
            "If using augmentation, make sure the augmentation step completed successfully."
        )
    if not config.labels_dir.exists():
        raise FileNotFoundError(
            f"Labels directory does not exist: {config.labels_dir}\n"
            "If using augmentation, make sure the augmentation step completed successfully."
        )

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cpu":
        logger.warning("No GPU detected - training will be slow!")

    # Prepare data splits
    train_images, train_labels, val_images, val_labels = prepare_data_splits(
        config.images_dir, config.labels_dir, config.val_split
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = default_sam_loader(
        raw_paths=[str(p) for p in train_images],
        label_paths=[str(p) for p in train_labels],
        raw_key=None,
        label_key=None,
        batch_size=config.batch_size,
        patch_shape=config.patch_shape,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        is_train=True,
        n_samples=config.n_samples,  # Number of patches per image per epoch
    )

    val_loader = default_sam_loader(
        raw_paths=[str(p) for p in val_images],
        label_paths=[str(p) for p in val_labels],
        raw_key=None,
        label_key=None,
        batch_size=config.batch_size,
        patch_shape=config.patch_shape,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        is_train=False,
        n_samples=config.n_samples,  # Number of patches per image per epoch
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Set up checkpoint directory
    # Note: micro-SAM's train_instance_segmentation will create checkpoints/<name>/
    # under save_root, so we pass the experiment dir directly
    checkpoint_dir = output_dir / "checkpoints" / config.checkpoint_name

    # Training arguments
    train_kwargs = {
        "name": config.checkpoint_name,
        "model_type": config.model_type,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "n_epochs": config.n_epochs,
        "lr": config.learning_rate,  # micro-SAM uses 'lr' not 'learning_rate'
        "save_root": str(output_dir),  # micro-SAM will add checkpoints/<name> itself
    }

    # Add checkpoint resume if specified
    if config.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        train_kwargs["checkpoint_path"] = str(config.resume_from_checkpoint)

    # Run training
    logger.info("Starting training loop...")
    train_instance_segmentation(**train_kwargs)

    # Export model
    best_checkpoint = checkpoint_dir / "best.pt"
    if not best_checkpoint.exists():
        logger.warning(f"Best checkpoint not found at {best_checkpoint}")
        # Try to find any checkpoint
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            best_checkpoint = checkpoints[0]
            logger.info(f"Using checkpoint: {best_checkpoint}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Determine export path
    if config.export_path is not None:
        export_path = config.export_path
    else:
        export_path = output_dir / f"{config.checkpoint_name}_model.pt"

    export_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting model to {export_path}...")
    export_instance_segmentation_model(
        str(best_checkpoint), str(export_path), config.model_type
    )

    logger.info("Training complete!")

    return {
        "checkpoint": best_checkpoint,
        "exported_model": export_path,
        "checkpoint_dir": checkpoint_dir,
    }
