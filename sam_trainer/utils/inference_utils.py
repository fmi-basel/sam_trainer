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
    model_type: str,
    device: str,
    model_path: Optional[str] = None,
    use_amg: bool = False,
    **amg_kwargs,
) -> Tuple:
    """Load an exported model with either decoder or AMG segmentation.

    This uses micro-SAM's get_predictor_and_segmenter which properly handles
    both AMG and decoder-based (AIS) modes.

    Args:
        model_type: SAM model type (e.g., 'vit_b_lm', 'vit_l_lm')
        device: Device to load model on ('cuda' or 'cpu')
        model_path: Path to a custom model checkpoint (.pt file). If None, the
            pre-trained micro-SAM model for `model_type` is downloaded/used from cache.
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

    # Use get_predictor_and_segmenter which properly handles both modes.
    # When checkpoint=None, micro-SAM downloads/uses the cached pre-trained model for model_type.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=model_path,
        device=device,
        amg=use_amg,  # This tells micro-SAM which mode to use
        **amg_kwargs,
    )

    return predictor, segmenter


def resolve_channel_index(
    channel_labels: list,
    channel: Optional[str],
    wavelength_ids: Optional[list] = None,
) -> int:
    """Resolve a channel specification to a 0-based integer index.

    Lookup order for string names:
    1. ``channel_labels`` (omero channel names, e.g. 'BF', 'DAPI')
    2. ``wavelength_ids`` (e.g. 'A02_C01')

    Args:
        channel_labels: List of channel label strings from image metadata (e.g., from NGIO).
            Can be empty or None if metadata is unavailable.
        channel: Channel specification. One of:
            - None: use the first channel (index 0, default behaviour)
            - An integer string like "0" or "2": use that channel index
            - A channel name like "BF" or "DAPI": look up in channel_labels
            - A wavelength ID like "A02_C01": look up in wavelength_ids
        wavelength_ids: Optional list of wavelength ID strings from image metadata
            (e.g., from NGIO ``image_data.wavelength_ids``). Can be empty or None.

    Returns:
        0-based channel index.

    Raises:
        ValueError: If the channel name is not found or the index is out of range.
    """
    if channel is None:
        return 0

    # Try integer index first
    try:
        idx = int(channel)
        n = (
            len(channel_labels)
            if channel_labels
            else (len(wavelength_ids) if wavelength_ids else 0)
        )
        if n and idx >= n:
            raise ValueError(f"Channel index {idx} out of range for {n} channels")
        return idx
    except ValueError as int_err:
        if "out of range" in str(int_err):
            raise

    # Look up by name: try channel_labels first, then wavelength_ids
    if channel_labels and channel in channel_labels:
        return channel_labels.index(channel)

    if wavelength_ids and channel in wavelength_ids:
        return wavelength_ids.index(channel)

    # Not found — build a helpful error message
    available: list[str] = []
    if channel_labels:
        available += [f"channel_labels: {channel_labels}"]
    if wavelength_ids:
        available += [f"wavelength_ids: {wavelength_ids}"]
    if not available:
        raise ValueError(
            f"Cannot resolve channel name '{channel}': no channel metadata available. "
            "Use a numeric index instead."
        )
    raise ValueError(
        f"Channel '{channel}' not found. Available — " + "; ".join(available)
    )


def _to_2d(image: np.ndarray, channel_index: int = 0) -> np.ndarray:
    """Reduce an ND image to 2D by squeezing singletons and selecting a channel.

    OME-Zarr images arriving from NGIO iterators can have extra leading dimensions
    (e.g. C, Z) even after MIP. SAM only accepts 2D (H, W) or 3D (Z, H, W) input.

    Args:
        image: Input image array of any dimensionality.
        channel_index: Index of the channel to select when the image has multiple
            channels after squeezing. Default: 0 (first channel).
    """
    original_shape = image.shape
    # Squeeze all size-1 dimensions first
    image = np.squeeze(image)
    if image.ndim > 2:
        if channel_index >= image.shape[0]:
            raise ValueError(
                f"Channel index {channel_index} out of range for shape {image.shape} "
                f"(original: {original_shape})"
            )
        logger.debug(
            f"Image shape {original_shape} → squeezed to {image.shape}, "
            f"selecting channel {channel_index} along axis 0"
        )
        image = image[channel_index]
        # Handle remaining extra dims (e.g., Z axis after channel selection)
        while image.ndim > 2:
            image = image[0]
    elif image.ndim != len(original_shape):
        logger.debug(f"Image shape {original_shape} → squeezed to {image.shape}")
    return image


def segment_image(
    image: np.ndarray,
    predictor,
    segmenter,
    use_amg: bool = False,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    generate_kwargs: Optional[Dict[str, Any]] = None,
    channel_index: int = 0,
) -> np.ndarray:
    """Run instance segmentation on a single image.

    Args:
        image: Input image as numpy array (any dimensionality; reduced to 2D internally)
        predictor: SAM predictor
        segmenter: SAM segmenter (AMG or InstanceSegmentationWithDecoder)
        use_amg: Whether using AMG mode (affects processing)
        tile_shape: Optional tile shape for large images (e.g., (512, 512))
        halo: Optional overlap for stitching tiles (e.g., (64, 64))
        generate_kwargs: Optional parameters for generate() method (decoder thresholds)
        channel_index: Index of the channel to select from multi-channel patches.
            Default: 0 (first channel).

    Returns:
        Instance segmentation masks as 2D numpy array with integer labels
    """
    generate_kwargs = generate_kwargs or {}
    image = _to_2d(image, channel_index=channel_index)

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
