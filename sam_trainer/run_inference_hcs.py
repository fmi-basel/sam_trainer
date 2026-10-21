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

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import typer
from joblib import Parallel, delayed
from natsort import natsorted
from ngio import open_ome_zarr_plate
from ngio.experimental.iterators import SegmentationIterator
from rich.console import Console

from sam_trainer.utils.inference_utils import (
    load_model_with_decoder,
    resolve_channel_index,
    segment_image,
)
from sam_trainer.utils.logging import get_logger, setup_logging

# TODO test all inference versions

app = typer.Typer(
    name="sam-inference-hcs",
    help="Run SAM instance segmentation inference on HCS OME-Zarr plates",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


def process_well(
    well_image_container,
    predictor,
    segmenter,
    label_name: str,
    use_amg: bool = False,
    generate_kwargs: dict = None,
    channel: Optional[str] = None,
    normalize: bool = True,
    normalize_lower_percentile: float = 1.0,
    normalize_upper_percentile: float = 99.5,
    invert: bool = False,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
) -> None:
    """Process a single well in an HCS plate.

    Args:
        well_image_container: NGIO image container for the well
        predictor: SAM predictor
        segmenter: SAM segmenter
        label_name: Name for the label layer
        use_amg: Whether using AMG mode
        generate_kwargs: Optional decoder parameters
        channel: Channel to use. Integer index string (e.g. '0') or omero channel
            name (e.g. 'BF'). Default: first channel.
        tile_shape: Tile shape for large well/field images. `SegmentationIterator`'s default
            ROI is the *entire* image as a single patch, so without this, wells much larger
            than the training patch_shape get downscaled by SAM before segmentation.
            `segmenter` must have been built with `is_tiled=True` for this to work.
        halo: Overlap for stitching tiles when `tile_shape` is set.
    """
    # Get image data
    image_data = well_image_container.get_image()
    logger.debug(f"  Image shape: {image_data.shape}")

    # Resolve channel index once per well
    channel_labels = getattr(image_data, "channel_labels", []) or []
    wavelength_ids = getattr(image_data, "wavelength_ids", []) or []
    channel_index = resolve_channel_index(
        channel_labels, channel, wavelength_ids=wavelength_ids
    )
    if channel is not None:
        logger.debug(f"  Using channel index {channel_index} (spec: '{channel}')")

    # Create label container - overwrites existing labels with same name as "label_name"
    label = well_image_container.derive_label(label_name, overwrite=True)

    # Use SegmentationIterator for efficient processing
    seg = SegmentationIterator(input_image=image_data, output_label=label)

    total_instances = 0
    for img, lbl_writer in seg.iter_as_numpy():
        masks = segment_image(
            img,
            predictor,
            segmenter,
            use_amg=use_amg,
            tile_shape=tile_shape,
            halo=halo,
            generate_kwargs=generate_kwargs,
            channel_index=channel_index,
            normalize=normalize,
            normalize_lower_percentile=normalize_lower_percentile,
            normalize_upper_percentile=normalize_upper_percentile,
            invert=invert,
        )

        # NGIO label writer expects a 2D label patch for this ROI.
        lbl_writer(patch=masks.astype(np.uint32, copy=False))

        n_instances = len(np.unique(masks)) - 1
        total_instances += n_instances
        logger.debug(f"    Image: {n_instances} instances")

    logger.debug(f"  Total instances in well: {total_instances}")


def process_wells(
    plate,
    predictor,
    segmenter,
    label_name: str,
    use_amg: bool = False,
    generate_kwargs: dict = None,
    channel: Optional[str] = None,
    normalize: bool = True,
    normalize_lower_percentile: float = 1.0,
    normalize_upper_percentile: float = 99.5,
    invert: bool = False,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
) -> list[str]:
    """Process all wells in an HCS plate.

    Args:
        plate: NGIO plate object
        predictor: SAM predictor
        segmenter: SAM segmenter
        label_name: Name for the label layer
        use_amg: Whether using AMG mode
        generate_kwargs: Optional decoder parameters
        channel: Channel to use. Integer index string or omero channel name. Default: first.
        tile_shape: Tile shape for large well/field images (see process_well).
        halo: Overlap for stitching tiles when `tile_shape` is set.

    Returns:
        IDs of wells that failed (empty if all wells succeeded).
    """
    wells = list(plate.get_wells().keys())
    console.print(f"[cyan]Found {len(wells)} wells to process[/cyan]")

    failed_wells: list[str] = []
    for well_id in wells:
        try:
            # Get the image for this well (assuming one image per well)
            well_image_container = plate.get_image(
                row=well_id[0],
                column=int(well_id[2:]),
                image_path="0",
            )
            logger.debug(f"Processing well {well_id}")

            process_well(
                well_image_container=well_image_container,
                predictor=predictor,
                segmenter=segmenter,
                label_name=label_name,
                use_amg=use_amg,
                generate_kwargs=generate_kwargs,
                channel=channel,
                normalize=normalize,
                normalize_lower_percentile=normalize_lower_percentile,
                normalize_upper_percentile=normalize_upper_percentile,
                invert=invert,
                tile_shape=tile_shape,
                halo=halo,
            )

        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Error processing well {well_id}: {e}")
            logger.exception(f"Failed to process well {well_id}")
            failed_wells.append(well_id)

    return failed_wells


def _done_marker_path(plate_path: Path, label_name: str) -> Path:
    """Path of the sentinel file marking `label_name` as fully written for a plate."""
    return plate_path / f".sam_trainer_{label_name}.done"


def _is_plate_done(plate_path: Path, label_name: str) -> bool:
    """Whether a previous run completed every well of this plate for `label_name`."""
    return _done_marker_path(plate_path, label_name).exists()


def _mark_plate_done(plate_path: Path, label_name: str) -> None:
    """Write the completion sentinel (only after all wells succeeded)."""
    _done_marker_path(plate_path, label_name).touch()


def process_single_plate(
    plate_path: Path,
    predictor,
    segmenter,
    label_name: str,
    use_amg: bool = False,
    generate_kwargs: dict = None,
    channel: Optional[str] = None,
    normalize: bool = True,
    normalize_lower_percentile: float = 1.0,
    normalize_upper_percentile: float = 99.5,
    invert: bool = False,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
) -> bool:
    """Process a single HCS plate.

    Args:
        plate_path: Path to HCS plate zarr
        predictor: SAM predictor
        segmenter: SAM segmenter
        label_name: Name for the label layer
        use_amg: Whether using AMG mode
        generate_kwargs: Optional decoder parameters
        channel: Channel to use. Integer index string or omero channel name. Default: first.
        tile_shape: Tile shape for large well/field images (see process_well).
        halo: Overlap for stitching tiles when `tile_shape` is set.

    Returns:
        True if the plate opened and every well succeeded (completion sentinel written).
    """
    console.print(f"\n[cyan]Opening plate:[/cyan] {plate_path}")

    try:
        plate = open_ome_zarr_plate(plate_path)
        console.print(f"[green]✓[/green] Plate opened: {plate}")
    except Exception as e:
        console.print(f"[bold red]Error opening plate:[/bold red] {e}")
        logger.exception(f"Failed to open plate {plate_path}")
        return False

    try:
        failed_wells = process_wells(
            plate,
            predictor,
            segmenter,
            label_name,
            use_amg,
            generate_kwargs,
            channel=channel,
            normalize=normalize,
            normalize_lower_percentile=normalize_lower_percentile,
            normalize_upper_percentile=normalize_upper_percentile,
            invert=invert,
            tile_shape=tile_shape,
            halo=halo,
        )
    except Exception as e:
        console.print(f"[bold red]Error processing plate:[/bold red] {e}")
        logger.exception(f"Failed to process plate {plate_path}")
        return False

    if failed_wells:
        console.print(
            f"[yellow]⚠[/yellow] Plate finished with {len(failed_wells)} failed "
            f"well(s): {plate_path} ({', '.join(failed_wells)})"
        )
        return False

    _mark_plate_done(plate_path, label_name)
    console.print(f"[green]✓[/green] Plate complete: {plate_path}")
    return True


@lru_cache(maxsize=1)
def _load_model_cached(
    model_type: str,
    device: str,
    model_path: Optional[str],
    use_amg: bool,
    is_tiled: bool,
):
    """Load the model once per process; loky reuses workers, so each loads it once."""
    return load_model_with_decoder(
        model_type=model_type,
        device=device,
        model_path=model_path,
        use_amg=use_amg,
        is_tiled=is_tiled,
    )


def _process_plate_in_worker(
    plate_path: Path,
    model_kwargs: dict,
    plate_kwargs: dict,
    verbose: int,
) -> tuple[Path, bool]:
    """Joblib entry point: segment one plate in a worker process.

    Takes only picklable arguments; the model is loaded lazily in the worker so it is
    never sent across processes. Logging is reconfigured here because a fresh loky
    worker does not inherit the parent's handlers.
    """
    setup_logging(verbose, console, message_prefix=f"[{plate_path.name}] ")
    predictor, segmenter = _load_model_cached(**model_kwargs)
    success = process_single_plate(plate_path, predictor, segmenter, **plate_kwargs)
    return plate_path, success


def _discover_plates(input_path: Path) -> list[Path]:
    """Return the plate(s) to process: the input itself, or every .zarr inside it."""
    if input_path.name.endswith(".zarr"):
        console.print("[cyan]Mode:[/cyan] Single plate processing")
        return [input_path]

    console.print("[cyan]Mode:[/cyan] Batch processing")
    zarr_plates = natsorted(input_path.glob("*.zarr"))
    if not zarr_plates:
        console.print(
            f"[bold red]Error:[/bold red] No .zarr plates found in {input_path}"
        )
        raise typer.Exit(1)
    console.print(f"[cyan]Found {len(zarr_plates)} plates to process[/cyan]")
    return zarr_plates


def _drop_completed_plates(plates: list[Path], label_name: str) -> list[Path]:
    """Remove plates whose completion sentinel for `label_name` already exists."""
    pending = [plate for plate in plates if not _is_plate_done(plate, label_name)]
    n_skipped = len(plates) - len(pending)
    if n_skipped:
        console.print(
            f"[cyan]Skipping {n_skipped} already completed plate(s) "
            f"(--skip-existing)[/cyan]"
        )
    return pending


def _run_plates_sequential(
    plates: list[Path], model_kwargs: dict, plate_kwargs: dict
) -> list[Path]:
    """Segment plates one after another in this process. Returns the failed plates."""
    segmentation_mode = "AMG" if model_kwargs["use_amg"] else "AIS (decoder-based)"
    console.print(f"[cyan]Segmentation mode:[/cyan] {segmentation_mode}")
    try:
        predictor, segmenter = _load_model_cached(**model_kwargs)
        console.print("[green]✓[/green] Model loaded successfully")
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        logger.exception("Model loading failed")
        raise typer.Exit(1)

    failed_plates = []
    for i, plate_path in enumerate(plates, 1):
        console.print(f"\n[cyan]Processing plate {i}/{len(plates)}[/cyan]")
        if not process_single_plate(plate_path, predictor, segmenter, **plate_kwargs):
            failed_plates.append(plate_path)
    return failed_plates


def _run_plates_parallel(
    plates: list[Path],
    model_kwargs: dict,
    plate_kwargs: dict,
    n_workers: int,
    verbose: int,
) -> list[Path]:
    """Segment plates in `n_workers` processes sharing one GPU. Returns the failed plates.

    The parent never loads the model (that would waste VRAM); each worker loads its own
    copy once. Plates are independent zarr stores, so there are no write conflicts.
    """
    console.print(
        f"[cyan]Parallel mode:[/cyan] {n_workers} worker processes, "
        f"{len(plates)} plates (model loaded once per worker)"
    )
    # Resolve the worker via the importable module path: when this file is run as a
    # script its functions live in __main__, which cloudpickle would serialize by value
    # (dragging unpicklable globals such as the rich Console along).
    worker = importlib.import_module("sam_trainer.run_inference_hcs")
    parallel = Parallel(
        n_jobs=n_workers, backend="loky", return_as="generator_unordered"
    )
    results = parallel(
        delayed(worker._process_plate_in_worker)(
            plate_path, model_kwargs, plate_kwargs, verbose
        )
        for plate_path in plates
    )

    failed_plates = []
    for n_done, (plate_path, success) in enumerate(results, 1):
        status = "[green]✓[/green]" if success else "[bold red]✗[/bold red]"
        console.print(f"{status} [{n_done}/{len(plates)}] {plate_path.name}")
        if not success:
            failed_plates.append(plate_path)
    return failed_plates


def process_hcs_plates(
    input_path: Path,
    model_type: str,
    device: str,
    label_name: str,
    model_path: Optional[Path] = None,
    use_amg: bool = False,
    generate_kwargs: dict = None,
    channel: Optional[str] = None,
    normalize: bool = True,
    normalize_lower_percentile: float = 1.0,
    normalize_upper_percentile: float = 99.5,
    invert: bool = False,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    n_workers: int = 1,
    skip_existing: bool = False,
    verbose: int = 0,
) -> None:
    """Process either a single plate or all plates in a directory.

    Args:
        input_path: Path to a single .zarr plate or parent directory containing plates
        model_type: SAM model type
        device: Device to use
        label_name: Name for the label layer
        model_path: Path to a custom model checkpoint (.pt file). If None, the
            pre-trained micro-SAM model for `model_type` is downloaded/used from cache.
        use_amg: If True, use AMG instead of decoder-based segmentation
        generate_kwargs: Optional decoder parameters
        channel: Channel to use. Integer index string or omero channel name. Default: first.
        tile_shape: Tile shape for large well/field images (see process_well).
        halo: Overlap for stitching tiles when `tile_shape` is set.
        n_workers: Number of plate-level worker processes sharing the device. 1 keeps
            the sequential in-process behavior.
        skip_existing: Skip plates already fully completed for `label_name`.
        verbose: Verbosity forwarded to worker processes for logging setup.

    Raises:
        typer.Exit: If no plates are found, the model fails to load, or any plate failed.
    """
    if model_path is not None:
        console.print(f"[cyan]Loading model:[/cyan] {model_path}")
    else:
        console.print(
            f"[cyan]Loading model:[/cyan] pre-trained {model_type} (from cache)"
        )

    model_kwargs = {
        "model_type": model_type,
        "device": device,
        "model_path": str(model_path) if model_path is not None else None,
        "use_amg": use_amg,
        "is_tiled": tile_shape is not None,
    }
    plate_kwargs = {
        "label_name": label_name,
        "use_amg": use_amg,
        "generate_kwargs": generate_kwargs,
        "channel": channel,
        "normalize": normalize,
        "normalize_lower_percentile": normalize_lower_percentile,
        "normalize_upper_percentile": normalize_upper_percentile,
        "invert": invert,
        "tile_shape": tile_shape,
        "halo": halo,
    }

    plates = _discover_plates(input_path)
    if skip_existing:
        plates = _drop_completed_plates(plates, label_name)
    if not plates:
        console.print("[green]Nothing to do: all plates already completed.[/green]")
        return

    # A pool is pointless for a single plate (and would load the model for nothing).
    if n_workers > 1 and len(plates) > 1:
        failed_plates = _run_plates_parallel(
            plates, model_kwargs, plate_kwargs, min(n_workers, len(plates)), verbose
        )
    else:
        failed_plates = _run_plates_sequential(plates, model_kwargs, plate_kwargs)

    if failed_plates:
        console.print(
            f"[bold red]{len(failed_plates)} plate(s) failed:[/bold red] "
            + ", ".join(plate.name for plate in failed_plates)
        )
        raise typer.Exit(1)


@app.command()
def main(
    input_plate: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to HCS OME-Zarr plate (.zarr file) or parent directory containing plates",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to a custom SAM model (.pt file). If omitted, the pre-trained micro-SAM model for --model-type is downloaded/used from cache.",
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
    use_amg: bool = typer.Option(
        False,
        "--use-amg",
        help="Use AMG (Automatic Mask Generation) instead of decoder-based segmentation",
    ),
    tile_shape: Optional[str] = typer.Option(
        None,
        "--tile-shape",
        help="Tile shape for large well/field images (e.g., '1024,1024'). Strongly "
        "recommended whenever a well/field is much larger than the training patch_shape — "
        "without it, AIS/decoder inference silently downscales the whole well image to "
        "SAM's 1024px encoder input before segmenting.",
    ),
    halo: Optional[str] = typer.Option(
        None,
        "--halo",
        help="Overlap for stitching tiles (e.g., '128,128'). Only used with --tile-shape.",
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
    normalize: bool = typer.Option(
        True,
        "--normalize/--no-normalize",
        help="Apply percentile normalization before segmentation. Must match training "
        "settings (default: on, matching training default).",
    ),
    normalize_lower_percentile: float = typer.Option(
        1.0,
        "--normalize-lower-percentile",
        help="Lower percentile for intensity clipping (must match training config)",
    ),
    normalize_upper_percentile: float = typer.Option(
        99.5,
        "--normalize-upper-percentile",
        help="Upper percentile for intensity clipping (must match training config)",
    ),
    invert: bool = typer.Option(
        False,
        "--invert/--no-invert",
        help="Invert intensities after normalization (for dark-foreground images, e.g. "
        "brightfield min-intensity projections). Must match training config.",
    ),
    channel: Optional[str] = typer.Option(
        None,
        "--channel",
        "-c",
        help=(
            "Channel to segment. Integer index (e.g. '0', '1') or omero channel name "
            "(e.g. 'BF', 'DAPI'). Default: first channel."
        ),
    ),
    n_workers: int = typer.Option(
        1,
        "--n-workers",
        "-j",
        min=1,
        help="Number of plates segmented in parallel (separate processes sharing the "
        "one GPU; each holds its own model copy in VRAM and RAM). Only applies to "
        "batch mode (parent directory). Default 1 = sequential.",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing/--no-skip-existing",
        help="Skip plates that already completed for this --label-name (completion "
        "sentinel written after all wells succeed). Makes interrupted batch runs "
        "resumable; do not use when re-running with changed thresholds under the same "
        "label name.",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase logging verbosity"
    ),
) -> None:
    """Run SAM instance segmentation inference on HCS OME-Zarr plate.

    This script is specialized for HCS plate structures (wells/fields).
    For single OME-Zarr images or TIFFs, use run_inference.py instead.

    Labels are written back to the zarr structure under labels/<label_name>.

    Segmentation modes:
    - Default (AIS): Uses trained decoder for instance segmentation
    - AMG (--use-amg): Uses Automatic Mask Generation from micro-SAM

    Decoder thresholds (AIS mode only):
    - Adjust center-dist-thresh, boundary-dist-thresh, foreground-thresh to tune segmentation
    """
    setup_logging(verbose, console)

    # Validate inputs
    if not input_plate.exists():
        console.print(
            f"[bold red]Error:[/bold red] Input plate not found: {input_plate}"
        )
        raise typer.Exit(1)
    if model_path is not None and not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found: {model_path}")
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
            console.print(
                f"[cyan]Tiling mode:[/cyan] Using tiles of shape {tile_shape_tuple}"
            )
        except ValueError:
            console.print(
                "[bold red]Error:[/bold red] tile-shape must be two integers separated by comma (e.g., '1024,1024')"
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
                    "[bold red]Error:[/bold red] halo must be two integers separated by comma (e.g., '128,128')"
                )
                raise typer.Exit(1)
        else:
            halo_tuple = (0, 0)

    # Prepare decoder parameters
    generate_kwargs = {
        "center_distance_threshold": center_distance_threshold,
        "boundary_distance_threshold": boundary_distance_threshold,
        "foreground_threshold": foreground_threshold,
    }
    console.print(f"[cyan]Decoder thresholds:[/cyan] {generate_kwargs}")

    if channel is not None:
        console.print(f"[cyan]Channel:[/cyan] {channel}")

    process_hcs_plates(
        input_path=input_plate,
        model_type=model_type,
        device=device,
        label_name=label_name,
        model_path=model_path,
        use_amg=use_amg,
        generate_kwargs=generate_kwargs,
        channel=channel,
        normalize=normalize,
        normalize_lower_percentile=normalize_lower_percentile,
        normalize_upper_percentile=normalize_upper_percentile,
        invert=invert,
        tile_shape=tile_shape_tuple,
        halo=halo_tuple,
        n_workers=n_workers,
        skip_existing=skip_existing,
        verbose=verbose,
    )

    console.print("\n[bold green]✓ All inference complete![/bold green]")


if __name__ == "__main__":
    app()
