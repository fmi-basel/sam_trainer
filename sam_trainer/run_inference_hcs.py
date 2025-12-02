#!/usr/bin/env python
"""
Run SAM inference (AMG mode) on HCS OME-Zarr plates.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder, get_decoder
from micro_sam.util import get_sam_model
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from ome_zarr.writer import write_labels
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="sam-inference-hcs",
    help="Run SAM instance segmentation inference on HCS OME-Zarr plates",
    add_completion=False,
)
console = Console()

# Default model path on the cluster
DEFAULT_MODEL_PATH = "/tachyon/groups/scratch/gmicro/khosnikl/projects/sam_trainer/sam_trainer/runs/full_img_vit_b_a100_lr-1e-5_full_run/checkpoints/full_image_vit_b_a100_lr-1e-5_full_run/best.pt"


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = level_map.get(verbosity, logging.DEBUG)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


logger = logging.getLogger(__name__)


def load_model_with_decoder(model_path: str, model_type: str, device: str) -> tuple:
    """Load an exported model that includes decoder state.

    Args:
        model_path: Path to exported model checkpoint
        model_type: SAM model type
        device: Device to load on

    Returns:
        Tuple of (predictor, segmenter)
    """
    # Load the checkpoint with weights_only=True to avoid unpickling issues
    # This loads only tensors, avoiding module references
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        logger.warning(
            f"Failed to load with weights_only=True, trying weights_only=False: {e}"
        )
        # If that fails, we need to add the parent directory to sys.path temporarily
        # so that sam_trainer module can be found during unpickling
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
        input_path=inference_image,  # We pass data directly
        ndim=2,  # Force 2D segmentation
        # image=inference_image,
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
    label_name = "labels"

    # write_labels helper from ome_zarr handles creating the 'labels' group and metadata
    # It will write the data to the group.
    write_labels(
        labels=labels_data, group=image_group, name=label_name, storage_options=None
    )
    logger.info(f"  Saved labels to group: 'labels/{label_name}'")


def process_plate(plate_path: Path, model_path: Path, model_type: str, device: str):
    """Traverse the HCS plate and process all images."""
    console.print(f"[cyan]Opening plate:[/cyan] {plate_path}")

    # Open OME-Zarr
    # mode='r+' is needed to write labels back
    try:
        loc = parse_url(plate_path, mode="r+")
        if loc is None:
            raise ValueError(f"Could not parse URL: {plate_path}")

        reader = Reader(loc)
        nodes = list(reader())
    except Exception as e:
        console.print(f"[bold red]Error opening plate:[/bold red] {e}")
        logger.exception(f"Failed to open plate {plate_path}")
        raise typer.Exit(1)

    # Load Model
    console.print(f"[cyan]Loading model:[/cyan] {model_path}")
    try:
        predictor, segmenter = load_model_with_decoder(
            model_path=str(model_path),
            model_type=model_type,
            device=device,
        )
        console.print("[green]✓[/green] Model loaded successfully")
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        logger.exception("Model loading failed")
        raise typer.Exit(1)

    # Traverse nodes
    count = 0
    for node in nodes:
        if node.data:
            node_path = node.zarr.path
            logger.info(f"Found image node: {node_path}")

            try:
                process_image_node(node, predictor, segmenter, device)
                count += 1
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Failed to process {node_path}: {e}")
                logger.exception(f"Failed to process {node_path}")

    console.print(f"[green]✓[/green] Finished processing plate. Total images: {count}")


@app.command()
def main(
    input_plate: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to HCS OME-Zarr plate (.zarr file)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    model_path: Path = typer.Option(
        DEFAULT_MODEL_PATH,
        "--model",
        "-m",
        help="Path to trained SAM model (.pt file)",
    ),
    model_type: str = typer.Option(
        "vit_b_lm",
        "--model-type",
        "-t",
        help="SAM model type (vit_b_lm, etc.)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (cuda/cpu). Auto-detect if not specified",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase logging verbosity"
    ),
) -> None:
    """Run SAM instance segmentation inference on HCS OME-Zarr plate."""
    setup_logging(verbose)

    # Validate model path
    if not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found: {model_path}")
        raise typer.Exit(1)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device:[/cyan] {device}")

    process_plate(input_plate, model_path, model_type, device)

    console.print("\n[bold green]✓ Inference complete![/bold green]")
    console.print(f"Labels saved in: {input_plate}")


if __name__ == "__main__":
    app()
