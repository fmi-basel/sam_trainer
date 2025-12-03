"""Shared utilities for SAM inference across different input formats.

This module provides common functions for model loading, image processing,
and segmentation that are used by various inference scripts.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder, get_decoder
from micro_sam.util import get_sam_model
from sam_trainer.utils.logging import get_logger

logger = get_logger(__name__)


def load_model_with_decoder(
    model_path: str, model_type: str, device: str
) -> Tuple:
    """Load an exported model that includes decoder state.

    This function loads a SAM model checkpoint that was exported with
    instance segmentation decoder state. It handles both weights_only
    and full pickle loading modes.

    Args:
        model_path: Path to exported model checkpoint (.pt file)
        model_type: SAM model type (e.g., 'vit_b_lm', 'vit_l_lm')
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (predictor, segmenter) where:
            - predictor: SAM predictor for image encoding
            - segmenter: InstanceSegmentationWithDecoder for generating masks

    Raises:
        Exception: If model loading fails
    """
    # Load the checkpoint (weights_only=False needed for PyTorch 2.6+ with training checkpoints)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.warning(
            f"Failed to load with weights_only=False initially, trying fallback: {e}"
        )
        # Add parent directory to sys.path so sam_trainer module can be found during unpickling
        script_dir = Path(__file__).parent.parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract SAM model state (remove decoder_state)
    model_state = {k: v for k, v in checkpoint.items() if k != "decoder_state"}

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
        decoder_state=checkpoint["decoder_state"],
        device=device,
    )

    # Create segmenter
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)

    return predictor, segmenter


def segment_image(
    image: np.ndarray,
    predictor,
    segmenter,
    use_decoder: bool = True,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Run instance segmentation on a single image.

    Args:
        image: Input image as 2D numpy array
        predictor: SAM predictor
        segmenter: SAM segmenter (AMG or InstanceSegmentationWithDecoder)
        use_decoder: Whether to use decoder-based segmentation (default: True)
        tile_shape: Optional tile shape for large images (e.g., (512, 512))
        halo: Optional overlap for stitching tiles (e.g., (64, 64))

    Returns:
        Instance segmentation masks as 2D numpy array with integer labels
    """
    masks = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
        verbose=False,
    )

    return masks


def postprocess_masks(
    masks: np.ndarray,
    min_area: int = 0,
    border_margin: int = 0,
    max_instances: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Filter and relabel instance masks based on size and position.

    Args:
        masks: Instance masks as 2D array with integer labels
        min_area: Minimum area in pixels (instances smaller are removed)
        border_margin: Margin in pixels from border (touching instances removed)
        max_instances: Maximum number of instances to keep (largest by area)

    Returns:
        Tuple of (filtered_masks, num_removed) where:
            - filtered_masks: Relabeled masks array
            - num_removed: Number of instances that were filtered out
    """
    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids != 0]
    
    if label_ids.size == 0:
        return masks.astype(np.uint16, copy=False), 0

    kept: list[Tuple[int, int]] = []
    height, width = masks.shape
    
    for label_id in label_ids:
        region_mask = masks == label_id
        area = int(region_mask.sum())
        
        # Filter by minimum area
        if min_area > 0 and area < min_area:
            continue

        # Filter by border touching
        if border_margin > 0:
            ys, xs = np.nonzero(region_mask)
            if ys.size == 0:
                continue
            touches_border = (
                (ys < border_margin).any()
                or (ys >= height - border_margin).any()
                or (xs < border_margin).any()
                or (xs >= width - border_margin).any()
            )
            if touches_border:
                continue

        kept.append((area, label_id))

    if not kept:
        return np.zeros_like(masks, dtype=np.uint16), label_ids.size

    # Sort by area (largest first)
    kept.sort(key=lambda item: item[0], reverse=True)
    
    # Limit number of instances
    if max_instances is not None and max_instances > 0:
        kept = kept[:max_instances]

    # Relabel sequentially
    filtered = np.zeros_like(masks, dtype=np.uint16)
    for new_idx, (_, label_id) in enumerate(kept, start=1):
        filtered[masks == label_id] = new_idx

    removed = label_ids.size - len(kept)
    return filtered, removed
