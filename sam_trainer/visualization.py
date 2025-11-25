"""Visualization utilities for validation predictions during training."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from matplotlib.colors import ListedColormap

# Use non-interactive backend for saving plots without display
matplotlib.use("Agg")
# Suppress verbose matplotlib debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def create_colormap(n_labels: int, seed: int = 42) -> ListedColormap:
    """Create a colormap for instance visualization with bright colors.

    Args:
        n_labels: Number of unique labels (excluding background)
        seed: Random seed for reproducibility

    Returns:
        Matplotlib colormap
    """
    rng = np.random.RandomState(seed)
    colors = np.zeros((n_labels + 1, 3))
    colors[0] = [0, 0, 0]  # Background is black
    # Generate bright, saturated colors (avoid dark colors for visibility)
    for i in range(1, n_labels + 1):
        colors[i] = rng.rand(3) * 0.5 + 0.5  # Range [0.5, 1.0]
    return ListedColormap(colors)


def save_prediction_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    alpha: float = 0.5,
    dpi: int = 150,
) -> None:
    """Save a prediction overlay visualization without displaying.

    Args:
        image: Input image (2D grayscale)
        prediction: Instance segmentation masks (2D with integer labels)
        output_path: Path to save the overlay image
        alpha: Transparency of mask overlay (0=transparent, 1=opaque)
        dpi: DPI for output image
    """
    n_instances = len(np.unique(prediction)) - 1

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray", interpolation="nearest")

    if n_instances > 0:
        cmap = create_colormap(n_instances)
        masked_labels = np.ma.masked_where(prediction == 0, prediction)
        ax.imshow(masked_labels, cmap=cmap, alpha=alpha, interpolation="nearest")

    ax.set_title(f"Validation Prediction ({n_instances} instances)", fontsize=12)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def run_inference_on_image(image: np.ndarray, predictor, segmenter) -> np.ndarray:
    """Run inference on a single image.

    Args:
        image: Input image (2D grayscale)
        predictor: SAM predictor
        segmenter: Instance segmenter (InstanceSegmentationWithDecoder or AMG)

    Returns:
        Instance segmentation masks
    """
    from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

    if isinstance(segmenter, InstanceSegmentationWithDecoder):
        # Decoder-based segmentation
        segmenter.initialize(image)
        predictions = segmenter.generate(image)
        masks = np.zeros(image.shape, dtype=np.uint16)
        for idx, pred in enumerate(predictions, start=1):
            mask = pred["segmentation"]
            masks[mask > 0] = idx
    else:
        # AMG-based segmentation
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        masks = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            ndim=2,
        )

    return masks


def save_validation_predictions(
    predictor,
    segmenter,
    val_images: list[Path],
    output_dir: Path,
    epoch: int,
    max_samples: int = 5,
) -> None:
    """Run inference on validation images and save visualizations.

    Args:
        predictor: SAM predictor
        segmenter: Instance segmenter
        val_images: List of validation image paths
        output_dir: Base output directory
        epoch: Current epoch number
        max_samples: Maximum number of validation images to process
    """
    logger.info(f"Saving validation predictions for epoch {epoch}...")

    pred_dir = output_dir / "validation_predictions" / f"epoch_{epoch:03d}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Sample a few validation images
    n_samples = min(max_samples, len(val_images))
    sample_images = val_images[:n_samples]

    for img_path in sample_images:
        try:
            # Load image
            image = tifffile.imread(img_path)
            if image.ndim == 3:
                image = image[0]  # Take first slice if 3D

            # Run prediction
            masks = run_inference_on_image(image, predictor, segmenter)

            # Save visualization
            output_path = pred_dir / f"{img_path.stem}_overlay.png"
            save_prediction_overlay(image, masks, output_path)

            # Also save raw masks
            mask_path = pred_dir / f"{img_path.stem}_masks.tif"
            tifffile.imwrite(mask_path, masks.astype(np.uint16), compression="zlib")

        except Exception as e:
            logger.warning(f"Failed to save prediction for {img_path.name}: {e}")
            continue

    logger.info(f"Saved validation predictions to {pred_dir}")


def create_predictor_and_segmenter(
    checkpoint_path: Path,
    model_type: str,
    device: str,
):
    """Load model and create predictor/segmenter from checkpoint.

    This uses the same loading logic as run_inference.py for consistency.
    Assumes checkpoints have decoder_state (from full SAM training).

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: SAM model type
        device: Device to load model on

    Returns:
        Tuple of (predictor, segmenter)
    """
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
        get_decoder,
    )
    from micro_sam.util import get_sam_model

    # Load checkpoint (weights_only=False needed for PyTorch 2.6+ with training checkpoints)
    checkpoint_state = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    # Extract SAM model state (remove decoder_state)
    model_state = {k: v for k, v in checkpoint_state.items() if k != "decoder_state"}

    # Load SAM model
    predictor = get_sam_model(
        model_type=model_type,
        checkpoint_path=None,  # We'll load state manually
        device=device,
        return_sam=False,
    )
    sam = predictor.model
    sam.load_state_dict(model_state, strict=False)

    # Load decoder with the image encoder
    decoder = get_decoder(
        image_encoder=sam.image_encoder,
        decoder_state=checkpoint_state["decoder_state"],
        device=device,
    )

    # Create segmenter
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)

    return predictor, segmenter
