#!/usr/bin/env python
"""Unified SAM inference script for TIFF and OME-Zarr images.

This script runs instance segmentation inference on:
- Individual TIFF files or directories of TIFFs
- Single OME-Zarr images (non-HCS format) using NGIO
- Batch processing of multiple images

For HCS plate structures, use run_inference_hcs.py instead.

Usage:
    # TIFF inference
    python run_inference.py --input images/ --output masks/ --model model.pt
    
    # OME-Zarr inference (auto-detected, labels written back to zarr)
    python run_inference.py --input image.zarr --model model.pt
    
    # With tiling for large images
    python run_inference.py --input images/ --output masks/ --model model.pt \
        --tile-shape 512,512 --halo 64,64
"""

from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import torch
import typer
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import SegmentationIterator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from sam_trainer.utils.inference_utils import (
    load_model_with_decoder,
    postprocess_masks,
    segment_image,
)
from sam_trainer.utils.logging import get_logger, setup_logging

app = typer.Typer(
    name="sam-inference",
    help="Run SAM instance segmentation inference on TIFF and OME-Zarr images",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


# Helper functions for different input types


def is_zarr_path(path: Path) -> bool:
    """Check if path is an OME-Zarr image."""
    return path.suffix == ".zarr" or (path.is_dir() and (path / ".zattrs").exists())


def load_tiff_image(image_path: Path) -> np.ndarray:
    """Load a TIFF image and return 2D array.

    Args:
        image_path: Path to TIFF file

    Returns:
        2D numpy array (grayscale image)
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


def save_tiff_masks(masks: np.ndarray, output_path: Path) -> None:
    """Save masks as compressed TIFF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, masks.astype(np.uint16), compression="zlib")


def process_zarr_image(
    zarr_path: Path,
    predictor,
    segmenter,
    label_name: str = "sam_labels",
    use_amg: bool = False,
    generate_kwargs: dict = None,
) -> None:
    """Process a single OME-Zarr image and write labels back to zarr.

    Args:
        zarr_path: Path to OME-Zarr image
        predictor: SAM predictor
        segmenter: SAM segmenter
        label_name: Name for the label layer (default: 'sam_labels')
        use_amg: If True, using AMG segmentation mode
        generate_kwargs: Optional decoder parameters
    """
    console.print(f"[cyan]Processing OME-Zarr:[/cyan] {zarr_path.name}")

    try:
        # Open OME-Zarr image using NGIO
        image_container = open_ome_zarr_container(zarr_path)
        image_data = image_container.get_image()

        logger.info(f"  Image shape: {image_data.shape}")

        # Create label container
        label = image_container.derive_label(label_name, overwrite=True)

        # Use SegmentationIterator for efficient processing
        seg = SegmentationIterator(input_image=image_data, output_label=label)

        total_instances = 0
        for img_patch, lbl_writer in seg.iter_as_numpy():
            # Run segmentation on patch
            masks = segment_image(
                img_patch,
                predictor,
                segmenter,
                use_amg=use_amg,
                generate_kwargs=generate_kwargs,
            )

            # Write labels back to zarr
            lbl_writer(patch=masks.astype(np.uint32))

            n_instances = len(np.unique(masks)) - 1
            total_instances += n_instances
            logger.debug(f"  Patch: {n_instances} instances")

        logger.info(f"  Total instances found: {total_instances}")
        console.print(
            f"[green]✓[/green] Labels saved to {zarr_path}/labels/{label_name}"
        )

    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Error processing {zarr_path.name}: {e}")
        logger.exception(f"Failed to process {zarr_path}")


def process_tiff_images(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    predictor,
    segmenter,
    tile_shape: Optional[tuple[int, int]],
    halo: Optional[tuple[int, int]],
    min_instance_area: int,
    border_margin: int,
    max_instances: Optional[int],
    min_file_size_kb: float = 0,
    use_amg: bool = False,
    generate_kwargs: dict = None,
) -> None:
    """Process TIFF images and save masks as separate TIFF files.

    Args:
        input_dir: Directory containing input TIFF images
        output_dir: Directory to save output masks
        pattern: Glob pattern for finding input files
        predictor: SAM predictor
        segmenter: SAM segmenter
        tile_shape: Optional tile shape for large images
        halo: Optional overlap for stitching tiles
        min_instance_area: Minimum area filter for postprocessing
        border_margin: Border margin filter for postprocessing
        max_instances: Maximum number of instances to keep
        min_file_size_kb: Minimum file size in KB to process (0 = no filter)
        use_amg: Whether using AMG mode
        generate_kwargs: Optional decoder parameters
    """
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

            # Filter by file size
            if min_file_size_kb > 0:
                size_kb = image_path.stat().st_size / 1024
                if size_kb < min_file_size_kb:
                    logger.debug(
                        f"Skipping {image_path.name}: too small ({size_kb:.1f} KB < {min_file_size_kb} KB)"
                    )
                    progress.advance(task)
                    continue

            try:
                # Load image
                image = load_tiff_image(image_path)
                logger.debug(f"Loaded {image_path.name} with shape {image.shape}")

                # Run inference
                masks = segment_image(
                    image,
                    predictor,
                    segmenter,
                    use_amg=use_amg,
                    # tile_shape=tile_shape_tuple,
                    # halo=halo_tuple,
                    generate_kwargs=generate_kwargs,
                )

                # Postprocess masks
                masks, removed = postprocess_masks(
                    masks,
                    min_area=min_instance_area,
                    border_margin=border_margin,
                    max_instances=max_instances,
                )
                if removed:
                    logger.debug(f"  Post-processing removed {removed} instances")

                n_instances = len(np.unique(masks)) - 1  # Subtract background
                logger.info(f"  {image_path.name}: Found {n_instances} instances")

                # Save results
                output_path = output_dir / f"{image_path.stem}_masks.tif"
                save_tiff_masks(masks, output_path)

                progress.advance(task)

            except Exception as e:
                console.print(
                    f"[yellow]⚠[/yellow] Error processing {image_path.name}: {e}"
                )
                logger.exception(f"Failed to process {image_path.name}")
                progress.advance(task)
                continue


@app.command()
def main(
    model: Path = typer.Option(
        ..., "--model", "-m", help="Path to trained SAM model (.pt file)"
    ),
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input path: TIFF directory, single OME-Zarr (.zarr), or directory of OME-Zarr images",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for TIFF masks (not used for OME-Zarr, labels written back to zarr)",
    ),
    pattern: str = typer.Option(
        "*.tif", "--pattern", "-p", help="Glob pattern for TIFF input files"
    ),
    label_name: str = typer.Option(
        "sam_labels",
        "--label-name",
        help="Name for label layer in OME-Zarr outputs",
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
        help="Tile shape for large images (e.g., '512,512'). For TIFF only.",
    ),
    halo: Optional[str] = typer.Option(
        None,
        "--halo",
        help="Overlap for stitching tiles (e.g., '64,64'). Only used with --tile-shape for TIFF.",
    ),
    max_instances: Optional[int] = typer.Option(
        None,
        "--max-instances",
        help="Keep at most this many instances per image (largest areas first). TIFF only.",
    ),
    min_instance_area: int = typer.Option(
        0,
        "--min-instance-area",
        help="Drop instances smaller than this area (pixels). TIFF only.",
    ),
    min_file_size_kb: float = typer.Option(
        0,
        "--min-file-size",
        help="Skip files smaller than this size in KB (0 = no filter). TIFF only.",
    ),
    border_margin: int = typer.Option(
        0,
        "--border-margin",
        help="Discard instances touching border within this many pixels. TIFF only.",
    ),
    use_amg: bool = typer.Option(
        False,
        "--use-amg",
        help="Use AMG (Automatic Mask Generation) instead of decoder-based segmentation",
    ),
    center_distance_threshold: float = typer.Option(
        0.5,
        "--center-dist-thresh",
        help="Center distance threshold for decoder mode (default: 0.5)",
    ),
    boundary_distance_threshold: float = typer.Option(
        0.5,
        "--boundary-dist-thresh",
        help="Boundary distance threshold for decoder mode (default: 0.5)",
    ),
    foreground_threshold: float = typer.Option(
        0.5,
        "--foreground-thresh",
        help="Foreground threshold for decoder mode (default: 0.5)",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase logging verbosity"
    ),
) -> None:
    """Run SAM instance segmentation inference on TIFF or OME-Zarr images.

    Automatically detects input type:
    - TIFF: Saves masks as separate .tif files in output directory
    - OME-Zarr: Writes labels back to the zarr container (no output dir needed)

    For HCS plate structures, use run_inference_hcs.py instead.

    Segmentation modes:
    - Default (AIS): Uses trained decoder for instance segmentation
    - AMG (--use-amg): Uses Automatic Mask Generation from micro-SAM

    Decoder thresholds (AIS mode only):
    - Adjust center-dist-thresh, boundary-dist-thresh, foreground-thresh to tune segmentation
    """
    setup_logging(verbose, console)

    # Validate inputs
    if not model.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found: {model}")
        raise typer.Exit(1)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input path not found: {input_path}")
        raise typer.Exit(1)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device:[/cyan] {device}")

    # Parse tile parameters (only for TIFF)
    tile_shape_tuple = None
    halo_tuple = None
    if tile_shape:
        try:
            tile_shape_tuple = tuple(map(int, tile_shape.split(",")))
            if len(tile_shape_tuple) != 2:
                raise ValueError
            console.print(
                f"[cyan]Tiling mode:[/cyan] Using tiles of shape {tile_shape_tuple}"
            )
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
                console.print(f"[cyan]Tile overlap:[/cyan] {halo_tuple}")
            except ValueError:
                console.print(
                    "[bold red]Error:[/bold red] halo must be two integers separated by comma (e.g., '64,64')"
                )
                raise typer.Exit(1)

    # Load model
    console.print(f"[cyan]Loading model:[/cyan] {model}")
    segmentation_mode = "AMG" if use_amg else "AIS (decoder-based)"
    console.print(f"[cyan]Segmentation mode:[/cyan] {segmentation_mode}")

    # Prepare decoder parameters
    generate_kwargs = {
        "center_distance_threshold": center_distance_threshold,
        "boundary_distance_threshold": boundary_distance_threshold,
        "foreground_threshold": foreground_threshold,
    }
    if not use_amg:
        console.print(f"[cyan]Decoder thresholds:[/cyan] {generate_kwargs}")

    try:
        predictor, segmenter = load_model_with_decoder(
            model_path=str(model),
            model_type=model_type,
            device=device,
            use_amg=use_amg,
        )
        console.print("[green]✓[/green] Model loaded successfully")
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        logger.exception("Model loading failed")
        raise typer.Exit(1)

    # Detect input type and process accordingly
    if is_zarr_path(input_path):
        # Single OME-Zarr image
        console.print("[cyan]Input type:[/cyan] OME-Zarr image")
        process_zarr_image(
            input_path, predictor, segmenter, label_name, use_amg, generate_kwargs
        )
        console.print("\n[bold green]✓ Inference complete![/bold green]")

    elif input_path.is_dir():
        # Check if directory contains zarr images or TIFFs
        zarr_images = [p for p in input_path.iterdir() if is_zarr_path(p)]

        if zarr_images:
            # Batch process OME-Zarr images
            console.print(
                f"[cyan]Input type:[/cyan] Directory with {len(zarr_images)} OME-Zarr images"
            )
            for zarr_path in zarr_images:
                process_zarr_image(
                    zarr_path,
                    predictor,
                    segmenter,
                    label_name,
                    use_amg,
                    generate_kwargs,
                )
            console.print("\n[bold green]✓ All inference complete![/bold green]")

        else:
            # Process TIFFs
            console.print(f"[cyan]Input type:[/cyan] TIFF images (pattern: {pattern})")

            if output_dir is None:
                console.print(
                    "[bold red]Error:[/bold red] --output directory required for TIFF inference"
                )
                raise typer.Exit(1)

            if tile_shape_tuple:
                console.print(
                    f"[cyan]Tiling enabled:[/cyan] {tile_shape_tuple} with halo {halo_tuple}"
                )
            else:
                console.print("[cyan]Processing full images (no tiling)[/cyan]")

            process_tiff_images(
                input_dir=input_path,
                output_dir=output_dir,
                pattern=pattern,
                predictor=predictor,
                segmenter=segmenter,
                tile_shape=tile_shape_tuple,
                halo=halo_tuple,
                min_instance_area=min_instance_area,
                border_margin=border_margin,
                max_instances=max_instances,
                min_file_size_kb=min_file_size_kb,
                use_amg=use_amg,
                generate_kwargs=generate_kwargs,
            )
            console.print("\n[bold green]✓ Inference complete![/bold green]")
            console.print(f"Results saved to: {output_dir}")
    else:
        console.print(
            "[bold red]Error:[/bold red] Input must be a directory or OME-Zarr image"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
