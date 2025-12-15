"""Shared utilities for SAM inference across different input formats.

This module provides common functions for model loading, image processing,
and segmentation that are used by various inference scripts.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)
from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

from sam_trainer.utils.logging import get_logger

logger = get_logger(__name__)


def load_model_with_decoder(
    model_path: str,
    model_type: str,
    device: str,
    use_amg: bool = False,
    **amg_kwargs,
) -> Tuple:
    """Load an exported model with either decoder or AMG segmentation.

    This uses micro-SAM's get_predictor_and_segmenter which properly handles
    both AMG and decoder-based (AIS) modes.

    Args:
        model_path: Path to exported model checkpoint (.pt file)
        model_type: SAM model type (e.g., 'vit_b_lm', 'vit_l_lm')
        device: Device to load model on ('cuda' or 'cpu')
        use_amg: If True, use AMG instead of decoder-based segmentation
        **amg_kwargs: Additional kwargs for AMG (pred_iou_thresh, stability_score_thresh, etc.)

    Returns:
        Tuple of (predictor, segmenter) where:
            - predictor: SAM predictor for image encoding
            - segmenter: InstanceSegmentationWithDecoder or AMG for generating masks

    Raises:
        Exception: If model loading fails
    """
    mode = "AMG" if use_amg else "AIS (decoder-based)"
    logger.info(f"Loading model with {mode} segmentation")

    # Use get_predictor_and_segmenter which properly handles both modes
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=model_path,
        device=device,
        amg=use_amg,  # This tells micro-SAM which mode to use
        **amg_kwargs,
    )

    return predictor, segmenter


def segment_image(
    image: np.ndarray,
    predictor,
    segmenter,
    use_amg: bool = False,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    generate_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Run instance segmentation on a single image.

    Args:
        image: Input image as 2D numpy array
        predictor: SAM predictor
        segmenter: SAM segmenter (AMG or InstanceSegmentationWithDecoder)
        use_amg: Whether using AMG mode (affects processing)
        tile_shape: Optional tile shape for large images (e.g., (512, 512))
        halo: Optional overlap for stitching tiles (e.g., (64, 64))
        generate_kwargs: Optional parameters for generate() method (decoder thresholds)

    Returns:
        Instance segmentation masks as 2D numpy array with integer labels
    """
    generate_kwargs = generate_kwargs or {}

    if isinstance(segmenter, InstanceSegmentationWithDecoder) and not use_amg:
        # Decoder-based segmentation: use initialize + generate pattern
        segmenter.initialize(image)
        predictions = segmenter.generate(**generate_kwargs)

        # Convert predictions (list of dicts) to label image
        masks = np.zeros(image.shape, dtype=np.uint32)
        for idx, pred in enumerate(predictions, start=1):
            mask = pred["segmentation"]
            masks[mask > 0] = idx
    else:
        # AMG-based segmentation: use automatic_instance_segmentation
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
