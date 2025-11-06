#!/usr/bin/env python
"""Simple inference script for trained SAM models.

This script runs instance segmentation inference on a folder of images
using a trained SAM model.

Usage:
    python run_inference.py --model path/to/model.pt --input path/to/images/ --output path/to/outputs/
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import torch
import typer
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="sam-inference",
    help="Run SAM instance segmentation inference on images",
    add_completion=False,
)
console = Console()


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


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array
    """
    img = tifffile.imread(image_path)

    # Handle different image shapes
    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        if img.shape[0] <= 4 and img.shape[0] < min(img.shape[1:]):
            # Likely channels-first (C, H, W)
            logger.debug(
                f"  Shape: {img.shape} - treating as multi-channel, using first channel"
            )
            return img[0]
        else:
            # Likely Z-stack or channels-last
            logger.debug(f"  Shape: {img.shape} - taking first slice/channel")
            return img[0]
    else:
        logger.warning(f"  Unexpected shape {img.shape}, taking first 2D slice")
        return img[0]


def save_masks(masks: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, masks.astype(np.uint16), compression="zlib")


def run_inference_on_image(
    image: np.ndarray,
    predictor,
    segmenter,
    tile_shape: Optional[tuple[int, int]] = None,
    halo: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Run inference on a single image using automatic instance segmentation.

    Args:
        image: Input image (2D grayscale)
        predictor: SAM predictor from get_predictor_and_segmenter
        segmenter: Segmenter from get_predictor_and_segmenter
        tile_shape: Optional tile shape for tiling-based segmentation
        halo: Optional overlap for stitching tiles

    Returns:
        Instance segmentation masks
    """
    # Run automatic instance segmentation (handles normalization internally)
    masks = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
    )

    return masks


@app.command()
def main(
    model: Path = typer.Option(
        ..., "--model", "-m", help="Path to trained SAM model (.pt file)"
    ),
    input_dir: Path = typer.Option(
        ..., "--input", "-i", help="Directory containing input images"
    ),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Directory to save segmentation masks"
    ),
    pattern: str = typer.Option(
        "*.tif", "--pattern", "-p", help="Glob pattern for input files"
    ),
    model_type: str = typer.Option(
        "vit_b_lm",
        "--model-type",
        "-t",
        help="SAM model type (vit_t, vit_b, vit_l, vit_h, vit_t_lm, vit_b_lm, vit_l_lm)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (cuda/cpu). Auto-detect if not specified",
    ),
    tile_shape: Optional[str] = typer.Option(
        None,
        "--tile-shape",
        help="Tile shape for large images (e.g., '512,512'). Enables tiling-based segmentation",
    ),
    halo: Optional[str] = typer.Option(
        None,
        "--halo",
        help="Overlap for stitching tiles (e.g., '64,64'). Only used with --tile-shape",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase logging verbosity"
    ),
) -> None:
    """Run SAM instance segmentation inference on a folder of images."""
    setup_logging(verbose)

    # Validate inputs
    if not model.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found: {model}")
        raise typer.Exit(1)
    if not input_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] Input directory not found: {input_dir}"
        )
        raise typer.Exit(1)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device:[/cyan] {device}")

    # Parse tile parameters
    tile_shape_tuple = None
    halo_tuple = None
    if tile_shape:
        try:
            tile_shape_tuple = tuple(map(int, tile_shape.split(",")))
            if len(tile_shape_tuple) != 2:
                raise ValueError
        except ValueError:
            console.print(
                "[bold red]Error:[/bold red] tile-shape must be two integers separated by comma (e.g., '512,512')"
            )
            raise typer.Exit(1)

        if halo:
            try:
                halo_tuple = tuple(map(int, halo.split(",")))
                if len(halo_tuple) != 2:
                    raise ValueError
            except ValueError:
                console.print(
                    "[bold red]Error:[/bold red] halo must be two integers separated by comma (e.g., '64,64')"
                )
                raise typer.Exit(1)

    # Load model using micro-SAM's approach from the notebook
    console.print(f"[cyan]Loading model:[/cyan] {model}")
    try:
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=model_type,
            checkpoint=str(model),
            device=device,
            is_tiled=(tile_shape_tuple is not None),
        )
        console.print("[green]✓[/green] Model loaded successfully")
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        logger.exception("Model loading failed")
        raise typer.Exit(1)

    # Get list of images
    image_files = sorted(input_dir.glob(pattern))
    if not image_files:
        console.print(
            f"[bold red]Error:[/bold red] No images found matching '{pattern}' in {input_dir}"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(image_files)} images to process[/cyan]")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing images...", total=len(image_files))

        for image_path in image_files:
            progress.update(task, description=f"[cyan]Processing {image_path.name}...")

            try:
                # Load image
                image = load_image(image_path)
                logger.debug(f"Loaded {image_path.name} with shape {image.shape}")

                # Run inference
                masks = run_inference_on_image(
                    image, predictor, segmenter, tile_shape_tuple, halo_tuple
                )

                n_instances = len(np.unique(masks)) - 1  # Subtract background
                logger.info(f"  {image_path.name}: Found {n_instances} instances")

                # Save results
                output_path = output_dir / f"{image_path.stem}_masks.tif"
                save_masks(masks, output_path)

                progress.advance(task)

            except Exception as e:
                console.print(
                    f"[yellow]⚠[/yellow] Error processing {image_path.name}: {e}"
                )
                logger.exception(f"Failed to process {image_path.name}")
                progress.advance(task)
                continue

    console.print("\n[bold green]✓ Inference complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    app()
