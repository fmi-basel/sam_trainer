"""Training module for SAM instance segmentation models."""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from micro_sam.training import (
    default_sam_loader,
    export_instance_segmentation_model,
    train_instance_segmentation,
    train_sam,
)
from torch_em.data import MinInstanceSampler

from sam_trainer.config import TrainingConfig
from sam_trainer.io import get_image_paths
from sam_trainer.visualization import (
    create_predictor_and_segmenter,
    save_validation_predictions,
)

logger = logging.getLogger(__name__)

# TODO Add a global var for where to save new models downloaded during the run. Potentially outside of the script here


class PercentileNormalizer:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def __call__(self, raw):
        arr = np.asarray(raw, dtype=np.float32)
        lo, hi = np.percentile(arr, [self.lower, self.upper])
        if hi <= lo:
            lo = float(arr.min())
            hi = float(arr.max())
            if hi == lo:
                hi = lo + 1.0
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        arr = (arr * 255.0).astype(np.uint8)
        return arr


def _build_raw_transform(config: TrainingConfig):
    if not config.normalize_inputs:
        return None

    lower = config.normalize_lower_percentile
    upper = config.normalize_upper_percentile
    logger.info(
        "Applying percentile normalization (lower=%.2f, upper=%.2f)",
        lower,
        upper,
    )

    return PercentileNormalizer(lower, upper)


# TODO adapt data split for OME-ZARR containers, which have labels inside the container (-> only 1 shuffling needed)
def prepare_data_splits(
    images_dir: Path,
    labels_dir: Path,
    val_split: float,
    *,
    shuffle: bool = True,
    seed: Optional[int] = None,
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

    indices = list(range(n_total))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    shuffled_images = [image_paths[i] for i in indices]
    shuffled_labels = [label_paths[i] for i in indices]

    train_images = shuffled_images[:n_train]
    train_labels = shuffled_labels[:n_train]
    val_images = shuffled_images[n_train:]
    val_labels = shuffled_labels[n_train:]

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
    logger.info(f"Number of samples: {config.n_samples}")
    logger.info(f"Number of epochs: {config.n_epochs}")
    logger.info(f"Learning rate: {config.learning_rate:.2e}")
    logger.info(f"Early stopping after: {config.early_stopping} epochs")
    logger.info(f"Validation split: {config.val_split:.2f}")
    logger.info(f"Number of workers: {config.num_workers}")

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
        config.images_dir,
        config.labels_dir,
        config.val_split,
        shuffle=config.shuffle_data,
        seed=config.shuffle_seed,
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    logger.info(f"Using {config.num_workers} dataloader workers")

    raw_transform = _build_raw_transform(config)
    train_sampler = None
    val_sampler = None
    if config.use_min_instance_sampler:
        train_sampler = MinInstanceSampler(
            config.min_instances_per_patch,
            min_size=config.min_instance_size,
        )
        val_sampler = MinInstanceSampler(
            config.min_instances_per_patch,
            min_size=config.min_instance_size,
        )
        logger.info(
            "Using MinInstanceSampler (min_instances=%s, min_size=%s) for both train and validation",
            config.min_instances_per_patch,
            config.min_instance_size,
        )
    loader_kwargs = {
        "raw_key": None,
        "label_key": None,
        "batch_size": config.batch_size,
        "patch_shape": config.patch_shape,
        "with_segmentation_decoder": True,
        "train_instance_segmentation_only": config.train_instance_segmentation_only,
        "n_samples": config.n_samples,
        "num_workers": config.num_workers,
        "raw_transform": raw_transform,
    }

    # TODO add raw_key and label_key to config if needed ("0" and "sam_labels" for OME-ZARR)
    # Potentially
    # raw_key = "0"
    # label_key = "labels/mask/0"
    # loader = default_sam_loader(
    #     raw_paths=str(zarr_path),
    #     raw_key=raw_key,
    #     label_paths=str(zarr_path),
    #     label_key=label_key,
    #     is_train=True,
    #     patch_shape=[2048, 2048],
    #     with_segmentation_decoder=True,
    #     batch_size=1,
    # )
    train_loader = default_sam_loader(
        raw_paths=[str(p) for p in train_images],
        label_paths=[str(p) for p in train_labels],
        is_train=True,
        sampler=train_sampler,
        **loader_kwargs,
    )

    val_loader = default_sam_loader(
        raw_paths=[str(p) for p in val_images],
        label_paths=[str(p) for p in val_labels],
        is_train=False,
        sampler=val_sampler,
        **loader_kwargs,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Set up checkpoint directory
    # Note: micro-SAM's trainers will create checkpoints/<name>/ under save_root
    checkpoint_dir = output_dir / "checkpoints" / config.checkpoint_name

    base_kwargs = {
        "name": config.checkpoint_name,
        "model_type": config.model_type,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "n_epochs": config.n_epochs,
        "lr": config.learning_rate,
        "early_stopping": config.early_stopping,
        "save_root": str(output_dir),
    }

    if config.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        base_kwargs["checkpoint_path"] = str(config.resume_from_checkpoint)

    # Run training
    logger.debug(
        f"Using train_instance_segmentation_only: {config.train_instance_segmentation_only}"
    )
    logger.info("Starting training loop...")

    # Setup validation prediction saving callback if enabled
    if config.save_validation_predictions_frequency is not None:
        logger.info(
            f"Will save validation predictions every {config.save_validation_predictions_frequency} epochs"
        )
        # Note: micro_sam trainers don't support callbacks directly,
        # so we'll save predictions after training completes
        # For real-time predictions during training, would need to modify micro_sam or use custom trainer

    if config.train_instance_segmentation_only:
        train_instance_segmentation(**base_kwargs)
    else:
        # Only override verify_n_labels_in_loader to skip pre-training validation
        # Don't override n_objects_per_batch - let it use default behavior (25 with smart subsampling)
        train_sam(
            with_segmentation_decoder=True,
            verify_n_labels_in_loader=None,
            **base_kwargs,
        )

    # Save final validation predictions if enabled
    if config.save_validation_predictions_frequency is not None:
        try:
            logger.info("Creating final validation predictions...")

            # Load best checkpoint
            best_checkpoint = checkpoint_dir / "best.pt"
            if best_checkpoint.exists():
                # Load model and create predictor/segmenter
                predictor, segmenter = create_predictor_and_segmenter(
                    checkpoint_path=best_checkpoint,
                    model_type=config.model_type,
                    device=device,
                )

                # Save predictions for final epoch
                save_validation_predictions(
                    predictor=predictor,
                    segmenter=segmenter,
                    val_images=val_images,
                    output_dir=output_dir,
                    epoch=config.n_epochs,
                    max_samples=20,
                )
        except Exception as e:
            logger.warning(f"Failed to save final validation predictions: {e}")

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
