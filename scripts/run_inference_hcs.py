#!/usr/bin/env python
"""
Run SAM inference (AMG mode) on HCS OME-Zarr plates.
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from ome_zarr.writer import write_labels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default model path on the cluster
DEFAULT_MODEL_PATH = "/tachyon/groups/scratch/gmicro/khosnikl/projects/sam_trainer/sam_trainer/runs/full_img_vit_b_a100_lr-1e-5_full_run/checkpoints/full_image_vit_b_a100_lr-1e-5_full_run/best.pt"


def process_image_node(node, predictor, segmenter, device):
    """Process a single image node (Field) within the HCS plate."""

    # 1. Load Image Data (Level 0 - highest resolution)
    # node.data is a list of dask arrays (multiscales)
    dask_data = node.data[0]

    # Convert to numpy
    image_data = np.array(dask_data)
    original_shape = image_data.shape

    # Handle dimensions: squeeze singleton dimensions for inference
    # Expected input for SAM is (C, Y, X) or (Y, X)
    # HCS data is often (T, C, Z, Y, X) e.g. (1, 1, 1, 2048, 2048)
    inference_image = image_data.squeeze()

    logger.info(
        f"  Processing image shape: {inference_image.shape} (Original: {original_shape})"
    )

    # 2. Run Inference (AMG)
    # automatic_instance_segmentation handles the prediction loop
    prediction = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=None,  # We pass data directly
        ndim=2,  # Force 2D segmentation
        image=inference_image,
        device=device,
        verbose=False,
    )

    # 3. Prepare Labels for Writing
    # We need to reshape the 2D prediction back to the original 5D shape (or whatever it was)
    # to satisfy OME-Zarr expectations if we want to match the image structure.
    # If original was (1, 1, 1, Y, X), we make labels (1, 1, 1, Y, X)

    # Create a view with the original shape
    try:
        labels_data = prediction.reshape(original_shape)
    except ValueError:
        logger.warning(
            f"  Could not reshape prediction {prediction.shape} to {original_shape}. Saving as 2D."
        )
        labels_data = prediction

    # 4. Save as Labels
    # node.zarr is the zarr group for this image.
    image_group = node.zarr

    # Define label name
    label_name = "segmentation"

    # write_labels helper from ome_zarr handles creating the 'labels' group and metadata
    # It will write the data to the group.
    write_labels(
        labels=labels_data, group=image_group, name=label_name, storage_options=None
    )
    logger.info(f"  Saved labels to group: 'labels/{label_name}'")


def process_plate(plate_path, model_path, device):
    """Traverse the HCS plate and process all images."""
    logger.info(f"Opening plate: {plate_path}")

    # Open OME-Zarr
    # mode='r+' is needed to write labels back
    # Note: Using standard ome_zarr. If NGIO is required, replace this section.
    try:
        loc = parse_url(plate_path, mode="r+")
        if loc is None:
            raise ValueError(f"Could not parse URL: {plate_path}")

        reader = Reader(loc)
        nodes = list(reader())
    except Exception as e:
        logger.error(f"Failed to open plate {plate_path}: {e}")
        return

    # Load Model (AMG Mode)
    logger.info(f"Loading model from: {model_path}")
    try:
        predictor, segmenter = get_predictor_and_segmenter(
            model_type="vit_b_lm",  # Assuming model type based on path
            checkpoint=model_path,
            device=device,
            amg=True,  # Use AMG mode
            is_custom_model=True,  # It's a finetuned model
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Traverse nodes
    # In HCS OME-Zarr, nodes are typically the images (Fields)
    count = 0
    for node in nodes:
        # Check if this node is an image (has data)
        if node.data:
            # Construct a readable path/ID for logging
            node_path = node.zarr.path
            logger.info(f"Found image node: {node_path}")

            try:
                process_image_node(node, predictor, segmenter, device)
                count += 1
            except Exception as e:
                logger.error(f"Failed to process {node_path}: {e}")

    logger.info(f"Finished processing plate. Total images: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM inference on HCS OME-Zarr plate"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the OME-Zarr plate (e.g., path/to/plate.zarr)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model checkpoint",
    )

    args = parser.parse_args()

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    process_plate(args.input, args.model, device)


if __name__ == "__main__":
    main()
