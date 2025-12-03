#!/usr/bin/env python
"""SAM inference script for HCS OME-Zarr plates using NGIO.

This script is specialized for High-Content Screening (HCS) plate structures
with wells and fields. For single OME-Zarr images or TIFFs, use run_inference.py instead.

Usage:
    # Single HCS plate
    python run_inference_hcs.py --input plate.zarr --model model.pt

    # Batch process all plates in a directory
    python run_inference_hcs.py --input /path/to/plates/ --model model.pt
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from ngio import open_ome_zarr_plate
from ngio.experimental.iterators import SegmentationIterator
from rich.console import Console

from sam_trainer.utils.inference_utils import (
    load_model_with_decoder,
    segment_image,
)
from sam_trainer.utils.logging import setup_logging, get_logger

# TODO test all inference versions

app = typer.Typer(
    name="sam-inference-hcs",
    help="Run SAM instance segmentation inference on HCS OME-Zarr plates",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)

# Default model path on cluster
DEFAULT_MODEL_PATH = (
    "/tachyon/groups/scratch/gmicro/khosnikl/projects/sam_trainer/sam_trainer/"
    "runs/full_img_vit_b_a100_lr-1e-5_full_run/checkpoints/"
    "full_image_vit_b_a100_lr-1e-5_full_run/best.pt"
)


def process_well(
    well_image_container,
    predictor,
    segmenter,
    label_name: str,
) -> None:
    """Process a single well in an HCS plate.

    Args:
        well_image_container: NGIO image container for the well
        predictor: SAM predictor
        segmenter: SAM segmenter
        label_name: Name for the label layer
    """
    # Get image data
    image_data = well_image_container.get_image()
    logger.info(f"  Image shape: {image_data.shape}")

    # Create label container
    label = well_image_container.derive_label(label_name, overwrite=True)

    # Use SegmentationIterator for efficient processing
    seg = SegmentationIterator(input_image=image_data, output_label=label)

    total_instances = 0
    for img, lbl_writer in seg.iter_as_numpy():
        masks = segment_image(img, predictor, segmenter)

        lbl_writer(patch=masks.astype(np.uint8))

        n_instances = len(np.unique(masks)) - 1
        total_instances += n_instances
        logger.debug(f"    Image: {n_instances} instances")

    logger.info(f"  Total instances: {total_instances}")


def process_wells(
    plate,
    predictor,
    segmenter,
    label_name: str,
) -> None:
    """Process all wells in an HCS plate.

    Args:
        plate: NGIO plate object
        predictor: SAM predictor
        segmenter: SAM segmenter
        label_name: Name for the label layer
    """
    wells = list(plate.get_wells().keys())
    console.print(f"[cyan]Found {len(wells)} wells to process[/cyan]")

    for well_id in wells:
        try:
            # Get the image for this well (assuming one image per well)
            well_image_container = plate.get_image(
                row=well_id[0],
                column=int(well_id[2:]),
                image_path="0",
            )
            logger.info(f"Processing well {well_id}")

            process_well(
                well_image_container=well_image_container,
                predictor=predictor,
                segmenter=segmenter,
                label_name=label_name,
            )

        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Error processing well {well_id}: {e}")
            logger.exception(f"Failed to process well {well_id}")
            continue


def process_single_plate(
    plate_path: Path,
    predictor,
    segmenter,
    label_name: str,
) -> None:
    """Process a single HCS plate.

    Args:
        plate_path: Path to HCS plate zarr
        predictor: SAM predictor
        segmenter: SAM segmenter
        label_name: Name for the label layer
    """
    console.print(f"\n[cyan]Opening plate:[/cyan] {plate_path}")

    try:
        plate = open_ome_zarr_plate(plate_path)
        console.print(f"[green]✓[/green] Plate opened: {plate}")
    except Exception as e:
        console.print(f"[bold red]Error opening plate:[/bold red] {e}")
        logger.exception(f"Failed to open plate {plate_path}")
        return

    try:
        process_wells(plate, predictor, segmenter, label_name)
        console.print(f"[green]✓[/green] Plate complete: {plate_path}")
    except Exception as e:
        console.print(f"[bold red]Error processing plate:[/bold red] {e}")
        logger.exception(f"Failed to process plate {plate_path}")


def process_hcs_plates(
    input_path: Path,
    model_path: Path,
    model_type: str,
    device: str,
    label_name: str,
) -> None:
    """Process either a single plate or all plates in a directory.

    Args:
        input_path: Path to a single .zarr plate or parent directory containing plates
        model_path: Path to model checkpoint
        model_type: SAM model type
        device: Device to use
        label_name: Name for the label layer
    """
    # Load model once
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

    # Determine if input is a single plate or parent directory
    if input_path.name.endswith(".zarr"):
        # Single plate
        console.print("[cyan]Mode:[/cyan] Single plate processing")
        process_single_plate(input_path, predictor, segmenter, label_name)
    else:
        # Parent directory - find all .zarr plates
        console.print("[cyan]Mode:[/cyan] Batch processing")
        zarr_plates = sorted(input_path.glob("*.zarr"))

        if not zarr_plates:
            console.print(
                f"[bold red]Error:[/bold red] No .zarr plates found in {input_path}"
            )
            raise typer.Exit(1)

        console.print(f"[cyan]Found {len(zarr_plates)} plates to process[/cyan]")

        for i, plate_path in enumerate(zarr_plates, 1):
            console.print(f"\n[cyan]Processing plate {i}/{len(zarr_plates)}[/cyan]")
            process_single_plate(plate_path, predictor, segmenter, label_name)


@app.command()
def main(
    input_plate: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to HCS OME-Zarr plate (.zarr file) or parent directory containing plates",
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
    label_name: str = typer.Option(
        "sam_labels",
        "--label-name",
        help="Name for the label layer in OME-Zarr outputs",
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
    """Run SAM instance segmentation inference on HCS OME-Zarr plate.

    This script is specialized for HCS plate structures (wells/fields).
    For single OME-Zarr images or TIFFs, use run_inference.py instead.

    Labels are written back to the zarr structure under labels/<label_name>.
    """
    setup_logging(verbose, console)

    # Validate inputs
    if not input_plate.exists():
        console.print(
            f"[bold red]Error:[/bold red] Input plate not found: {input_plate}"
        )
        raise typer.Exit(1)
    if model_path and not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found: {model_path}")
        raise typer.Exit(1)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device:[/cyan] {device}")

    process_hcs_plates(
        input_path=input_plate,
        model_path=model_path,
        model_type=model_type,
        device=device,
        label_name=label_name,
    )

    console.print("\n[bold green]✓ All inference complete![/bold green]")


if __name__ == "__main__":
    app()
