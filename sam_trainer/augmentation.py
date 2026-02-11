"""Data augmentation module using albumentations."""

import logging

import albumentations as A
import numpy as np
from albumentations.core.composition import Compose

from sam_trainer.config import AugmentationConfig
from sam_trainer.io import get_image_paths, read_image, write_image

logger = logging.getLogger(__name__)


def relabel_mask(mask: np.ndarray) -> np.ndarray:
    """Relabel instance mask to consecutive integers starting from 1.

    Background (0) is preserved, all other unique values are remapped to 1, 2, 3...

    Args:
        mask: Instance mask with arbitrary integer labels

    Returns:
        Relabeled mask with consecutive integers (0=background, 1, 2, 3...)
    """
    # Get unique values excluding background (0)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        # No instances, return as-is
        return mask.copy()

    # Create output mask
    relabeled = np.zeros_like(mask, dtype=np.uint16)

    # Relabel each instance to consecutive integers
    for new_label, old_label in enumerate(unique_labels, start=1):
        relabeled[mask == old_label] = new_label

    logger.debug(
        f"Relabeled mask: {len(unique_labels)} instances (was: {unique_labels[:5]}..., now: 1-{len(unique_labels)})"
    )

    return relabeled


def create_augmentation_pipeline(config: AugmentationConfig) -> Compose:
    """Create albumentations augmentation pipeline from config.

    Args:
        config: Augmentation configuration

    Returns:
        Composed augmentation pipeline
    """
    transforms = []

    # Geometric transforms (use INTER_NEAREST for masks to preserve labels)
    # Only use 90-degree rotations and flips to avoid padding/reflection artifacts
    geo_transforms = []
    if config.allow_rotate_90:
        geo_transforms.append(A.RandomRotate90(p=0.5))
    if config.flip_horizontal:
        geo_transforms.append(A.HorizontalFlip(p=0.5))
    if config.flip_vertical:
        geo_transforms.append(A.VerticalFlip(p=0.5))
    if geo_transforms:
        transforms.append(A.Compose(geo_transforms))

    # Blur and noise
    if config.gaussian_blur_prob > 0:
        transforms.append(
            A.GaussianBlur(blur_limit=(3, 11), p=config.gaussian_blur_prob)
        )

        # Add noise using multiplicative approach (works better with uint16)
        if config.gaussian_noise_prob > 0:
            transforms.append(
                A.MultiplicativeNoise(
                    multiplier=(0.95, 1.05), p=config.gaussian_noise_prob
                )
            )

        # Brightness and contrast (extremely light adjustments to avoid darkening uint16 images)
    if config.brightness_contrast:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.02, contrast_limit=0.01, p=0.3
            )
        )

    return A.Compose(transforms)


def augment_image_pair(
    image: np.ndarray,
    mask: np.ndarray,
    pipeline: Compose,
    n_augmentations: int,
    treat_3d_as_2d: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply augmentation pipeline to image-mask pair.

    Args:
        image: Input image (2D or 3D)
        mask: Input mask (same shape as image) - should be uint16 with consecutive labels
        pipeline: Augmentation pipeline
        n_augmentations: Number of augmented versions to create
        treat_3d_as_2d: If True, treat each slice independently with different augmentations

    Returns:
        Tuple of (augmented_images, augmented_masks) lists
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask shapes must match: {image.shape} vs {mask.shape}"
        )

    augmented_images = []
    augmented_masks = []

    # Store dtypes to preserve after augmentation
    image_dtype = image.dtype
    mask_dtype = mask.dtype

    # Handle 3D by processing slices
    if image.ndim == 3:
        if treat_3d_as_2d:
            # Each slice gets independent random augmentations
            for aug_idx in range(n_augmentations):
                aug_img_slices = []
                aug_mask_slices = []

                for z_idx in range(image.shape[0]):
                    img_slice = image[z_idx]
                    mask_slice = mask[z_idx]

                    # Apply different random augmentation to each slice
                    augmented = pipeline(image=img_slice, mask=mask_slice)
                    aug_img_slices.append(augmented["image"].astype(image_dtype))
                    aug_mask_slices.append(augmented["mask"].astype(mask_dtype))

                augmented_images.append(np.stack(aug_img_slices))
                augmented_masks.append(np.stack(aug_mask_slices))
        else:
            # Same augmentation applied consistently to all slices
            for aug_idx in range(n_augmentations):
                aug_img_slices = []
                aug_mask_slices = []

                # Apply augmentation to first slice to get the random parameters
                # then apply same transform to all slices
                for z_idx in range(image.shape[0]):
                    img_slice = image[z_idx]
                    mask_slice = mask[z_idx]

                    augmented = pipeline(image=img_slice, mask=mask_slice)
                    aug_img_slices.append(augmented["image"].astype(image_dtype))
                    aug_mask_slices.append(augmented["mask"].astype(mask_dtype))

                augmented_images.append(np.stack(aug_img_slices))
                augmented_masks.append(np.stack(aug_mask_slices))

    # Handle 2D
    else:
        for aug_idx in range(n_augmentations):
            augmented = pipeline(image=image, mask=mask)
            augmented_images.append(augmented["image"].astype(image_dtype))
            augmented_masks.append(augmented["mask"].astype(mask_dtype))

    return augmented_images, augmented_masks


def run_augmentation(config: AugmentationConfig) -> dict[str, int]:
    """Run augmentation pipeline on all images in input directories.

    Args:
        config: Augmentation configuration

    Returns:
        Dictionary with augmentation statistics
    """
    logger.info("Starting data augmentation...")

    # Get image and label paths
    image_paths = get_image_paths(config.input_images_dir)
    label_paths = get_image_paths(config.input_labels_dir)

    if len(image_paths) != len(label_paths):
        raise ValueError(
            f"Number of images ({len(image_paths)}) and labels ({len(label_paths)}) must match"
        )

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {config.input_images_dir}")

    logger.info(f"Found {len(image_paths)} image-label pairs")

    # Create output directories
    output_images_dir = config.output_dir / "images"
    output_labels_dir = config.output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Create augmentation pipeline
    if config.n_augmentations > 0:
        pipeline = create_augmentation_pipeline(config)
        logger.info(
            f"Created augmentation pipeline with {len(pipeline.transforms)} transforms"
        )
        if config.treat_3d_as_2d:
            logger.info("3D mode: Each slice will get independent random augmentations")
        else:
            logger.info(
                "3D mode: Same augmentation applied consistently across all slices"
            )
    else:
        logger.info("No augmentation requested (n_augmentations=0), copying data only")
        pipeline = None

    # Process each image-label pair
    total_generated = 0

    for idx, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
        logger.info(f"Processing {idx + 1}/{len(image_paths)}: {img_path.name}")

        # Read image and label
        image = read_image(img_path)
        label = read_image(label_path)

        # Relabel mask to consecutive integers (0=background, 1, 2, 3...)
        label = relabel_mask(label)

        # Ensure mask dtype matches image for augmentation compatibility
        logger.debug(
            f"Before dtype fix - Image: {image.dtype}, shape: {image.shape}, Mask: {label.dtype}, shape: {label.shape}"
        )
        if label.dtype != image.dtype:
            if image.dtype in [np.uint8, np.uint16]:
                if label.max() < 256:
                    label = label.astype(np.uint8)
                else:
                    label = label.astype(np.uint16)
            else:
                label = label.astype(image.dtype)
            logger.debug(
                f"Converted mask dtype to {label.dtype} to match image {image.dtype}"
            )
        else:
            logger.debug("Dtypes already match, no conversion needed")

        # Determine output format
        if config.output_format == "original":
            # Detect format from input file
            if img_path.is_dir():
                output_fmt = "ome-zarr"
            elif img_path.suffix.lower() in [".tif", ".tiff"]:
                output_fmt = "tif"
            elif img_path.suffix.lower() in [".h5", ".hdf5"]:
                output_fmt = "hdf5"
            elif img_path.suffix.lower() == ".zarr":
                output_fmt = "ome-zarr"
            else:
                output_fmt = "tif"  # fallback
            logger.debug(f"Auto-detected format: {output_fmt}")
        else:
            output_fmt = config.output_format

        # Save original (if requested)
        base_name = img_path.stem
        if config.include_original:
            write_image(image, output_images_dir / f"{base_name}_orig", output_fmt)
            write_image(label, output_labels_dir / f"{base_name}_orig", output_fmt)
            total_generated += 1

        # Generate augmentations
        if pipeline is not None and config.n_augmentations > 0:
            aug_images, aug_labels = augment_image_pair(
                image, label, pipeline, config.n_augmentations, config.treat_3d_as_2d
            )

            for aug_idx, (aug_img, aug_label) in enumerate(zip(aug_images, aug_labels)):
                write_image(
                    aug_img,
                    output_images_dir / f"{base_name}_aug{aug_idx:03d}",
                    output_fmt,
                )
                write_image(
                    aug_label,
                    output_labels_dir / f"{base_name}_aug{aug_idx:03d}",
                    output_fmt,
                )
                total_generated += 1

    stats = {
        "n_input_pairs": len(image_paths),
        "n_augmentations_per_image": config.n_augmentations,
        "total_output_pairs": total_generated,
        "output_format": config.output_format,
    }

    logger.info(f"Augmentation complete: {total_generated} image-label pairs generated")
    return stats
