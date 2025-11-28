#!/usr/bin/env python
"""SAM inference script for HCS OME-Zarr plates using NGIO.

This script runs instance segmentation inference on HCS plates stored as OME-Zarr
using a trained SAM model. Labels are saved in the standard OME-Zarr format.

Usage:
    python run_inference_ngio.py --input path/to/plate.zarr [--model path/to/model.pt]
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder, get_decoder
from micro_sam.util import get_sam_model
from ngio import open_ome_zarr_plate
from ngio.experimental.iterators import SegmentationIterator
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="sam-inference-ngio",
    help="Run SAM instance segmentation inference on HCS OME-Zarr plates",
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


def load_model_with_decoder(model_path: str, model_type: str, device: str):
    """Load an exported model that includes decoder state.

    Args:
        model_path: Path to exported model checkpoint
        model_type: SAM model type
        device: Device to load on

    Returns:
        Tuple of (predictor, segmenter)
    """
    # Load the checkpoint with weights_only=True to avoid unpickling issues
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.warning(
            f"Failed to load with weights_only=True, trying weights_only=False: {e}"
        )
        # If that fails, we need to add the parent directory to sys.path temporarily
        # so that sam_trainer module can be found during unpickling
        import sys

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


def segment_image(image: np.ndarray, predictor, segmenter):
    masks = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        verbose=False,
    )

    return masks


def process_well(well_image_container, predictor, segmenter, device):
    # Process each well and image
    # Get image data as numpy array
    image_data = well_image_container.get_image()
    logger.info(f"Image shape: {image_data.shape}")
    label = well_image_container.derive_label("sam_labels", overwrite=True)
    seg = SegmentationIterator(input_image=image_data, output_label=label)

    for img, lbl in seg.iter_as_numpy():
        mask = segment_image(img, predictor, segmenter)
        lbl(patch=mask.astype(np.uint8))

        logger.info(f"Found {len(np.unique(mask)) - 1} instances in patch")


def process_wells(plate, predictor, segmenter, device):
    # Process each well
    wells = list(plate.get_wells().keys())
    console.print(f"[cyan]Found {len(wells)} wells to process[/cyan]")

    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[progress.description]{task.description}"),
    #     console=console,
    # ) as progress:
    #     task = progress.add_task("[cyan]Processing wells...", total=len(wells))

    for well_id in wells:
        # progress.update(task, description=f"[cyan]Processing well {well_id}...")

        try:
            # Get the image for this well (assuming one image per well)
            well_image_container = plate.get_image(
                row=well_id[0], column=int(well_id[2:]), image_path="0"
            )
            logger.info(f"Processing well {well_id}")
            process_well(
                well_image_container=well_image_container,
                predictor=predictor,
                segmenter=segmenter,
                device=device,
            )
            # progress.advance(task)

        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Error processing well {well_id}: {e}")
            logger.exception(f"Failed to process well {well_id}")
            # progress.advance(task)
        continue


def process_single_plate(plate_path: Path, predictor, segmenter, device: str) -> None:
    """Process a single HCS plate."""
    console.print(f"\n[cyan]Opening plate:[/cyan] {plate_path}")
    try:
        plate = open_ome_zarr_plate(plate_path)
        console.print(f"[green]✓[/green] Plate opened: {plate}")
    except Exception as e:
        console.print(f"[bold red]Error opening plate:[/bold red] {e}")
        logger.exception(f"Failed to open plate {plate_path}")
        return

    try:
        process_wells(plate, predictor, segmenter, device)
        console.print(f"[green]✓[/green] Plate complete: {plate_path}")
    except Exception as e:
        console.print(f"[bold red]Error processing plate:[/bold red] {e}")
        logger.exception(f"Failed to process plate {plate_path}")


def process_hcs_plates(
    input_path: Path, model_path: Path, model_type: str, device: str
) -> None:
    """Process either a single plate or all plates in a directory.

    Args:
        input_path: Path to a single .zarr plate or parent directory containing plates
        model_path: Path to model checkpoint
        model_type: SAM model type
        device: Device to use
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
        process_single_plate(input_path, predictor, segmenter, device)
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
            process_single_plate(plate_path, predictor, segmenter, device)


@app.command()
def main(
    input_plate: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to HCS OME-Zarr plate (.zarr file) or parent directory containing plates",
    ),
    model_path: Path = typer.Option(
        "/tachyon/groups/scratch/gmicro/khosnikl/projects/sam_trainer/sam_trainer/runs/full_img_vit_b_a100_lr-1e-5_full_run/checkpoints/full_image_vit_b_a100_lr-1e-5_full_run/best.pt",
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
    )

    console.print("\n[bold green]✓ All inference complete![/bold green]")


if __name__ == "__main__":
    app()
