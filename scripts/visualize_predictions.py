#!/usr/bin/env python
"""Visualize instance segmentation predictions overlaid on original images.

Simple script to create overlays of predicted masks on input images for quality checking.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import typer
from matplotlib.colors import ListedColormap
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="visualize-predictions",
    help="Create overlay visualizations of segmentation predictions",
    add_completion=False,
)
console = Console()


def create_random_colormap(n_labels: int, seed: int = 42) -> ListedColormap:
    """Create a random colormap for instance visualization.

    Args:
        n_labels: Number of unique labels (excluding background)
        seed: Random seed for reproducibility

    Returns:
        Matplotlib colormap
    """
    rng = np.random.RandomState(seed)
    # Generate bright, saturated colors (avoid dark colors for visibility)
    colors = np.zeros((n_labels + 1, 3))
    colors[0] = [0, 0, 0]  # Background is black
    # Generate random hues with high saturation and value
    for i in range(1, n_labels + 1):
        colors[i] = rng.rand(3) * 0.5 + 0.5  # Range [0.5, 1.0] for bright colors
    return ListedColormap(colors)


def create_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.5,
) -> tuple[plt.Figure, plt.Axes]:
    """Create an overlay visualization of masks on image.

    Args:
        image: Input image (2D grayscale)
        masks: Instance segmentation masks (2D with integer labels)
        alpha: Transparency of mask overlay (0=transparent, 1=opaque)

    Returns:
        Matplotlib figure and axes
    """
    n_instances = len(np.unique(masks)) - 1  # Exclude background

    fig, ax = plt.subplots(figsize=(12, 12))

    # Show image in grayscale
    ax.imshow(image, cmap="gray", interpolation="nearest")

    # Overlay masks with random colors
    if n_instances > 0:
        cmap = create_random_colormap(n_instances)
        # Create a masked array where 0 (background) is transparent
        masked_labels = np.ma.masked_where(masks == 0, masks)
        ax.imshow(masked_labels, cmap=cmap, alpha=alpha, interpolation="nearest")

    ax.set_title(f"Segmentation Overlay ({n_instances} instances)", fontsize=14)
    ax.axis("off")

    return fig, ax


@app.command()
def main(
    images_dir: Path = typer.Option(
        ..., "--images", "-i", help="Directory containing input images"
    ),
    masks_dir: Path = typer.Option(
        ..., "--masks", "-m", help="Directory containing predicted masks"
    ),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Directory to save visualizations"
    ),
    pattern: str = typer.Option(
        "*.tif", "--pattern", "-p", help="Glob pattern for image files"
    ),
    mask_suffix: str = typer.Option(
        "_masks",
        "--mask-suffix",
        help="Suffix added to mask filenames (e.g., image_masks.tif)",
    ),
    alpha: float = typer.Option(
        0.5, "--alpha", "-a", help="Transparency of mask overlay (0-1)"
    ),
    dpi: int = typer.Option(150, "--dpi", help="DPI for output images"),
) -> None:
    """Create overlay visualizations of segmentation predictions on original images."""

    # Validate inputs
    if not images_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] Images directory not found: {images_dir}"
        )
        raise typer.Exit(1)
    if not masks_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] Masks directory not found: {masks_dir}"
        )
        raise typer.Exit(1)

    # Get list of images
    image_files = sorted(images_dir.glob(pattern))
    if not image_files:
        console.print(
            f"[bold red]Error:[/bold red] No images found matching '{pattern}' in {images_dir}"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(image_files)} images to visualize[/cyan]")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    processed = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Creating visualizations...", total=len(image_files)
        )

        for image_path in image_files:
            # Find corresponding mask file
            mask_name = f"{image_path.stem}{mask_suffix}{image_path.suffix}"
            mask_path = masks_dir / mask_name

            if not mask_path.exists():
                console.print(
                    f"[yellow]⚠[/yellow] Skipping {image_path.name}: mask not found ({mask_name})"
                )
                skipped += 1
                progress.advance(task)
                continue

            progress.update(task, description=f"[cyan]Processing {image_path.name}...")

            try:
                # Load image and masks
                image = tifffile.imread(image_path)
                masks = tifffile.imread(mask_path)

                # Handle 3D images - take first slice
                if image.ndim == 3:
                    image = image[0]
                if masks.ndim == 3:
                    masks = masks[0]

                # Create overlay
                fig, ax = create_overlay(image, masks, alpha=alpha)

                # Save visualization
                output_path = output_dir / f"{image_path.stem}_overlay.png"
                fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

                processed += 1
                progress.advance(task)

            except Exception as e:
                console.print(
                    f"[yellow]⚠[/yellow] Error processing {image_path.name}: {e}"
                )
                skipped += 1
                progress.advance(task)
                continue

    console.print("\n[bold green]✓ Visualization complete![/bold green]")
    console.print(f"Processed: {processed}, Skipped: {skipped}")
    console.print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    app()
