"""Command-line interface for SAM trainer."""

import logging
from pathlib import Path
from typing import Literal

import click
import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

from sam_trainer.augmentation import run_augmentation
from sam_trainer.config import AugmentationConfig, PipelineConfig, TrainingConfig
from sam_trainer.training import run_training

app = typer.Typer(
    name="sam-trainer",
    help="Train micro-SAM models with optional data augmentation",
    add_completion=False,
)
console = Console()


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG, 3+=DEBUG with more detail
    """
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,
    }
    level = level_map.get(verbosity, logging.DEBUG)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def config(
    output: Path = typer.Option(
        Path("config.yaml"), "--output", "-o", help="Output path for configuration file"
    ),
) -> None:
    """Interactive configuration builder for SAM training pipeline."""
    console.print("\n[bold cyan]SAM Trainer Configuration Builder[/bold cyan]\n")

    # Experiment info
    experiment_name = typer.prompt("Experiment name")
    output_base = Path(typer.prompt("Output base directory", default="runs"))

    # Augmentation
    do_augmentation = typer.confirm("Enable data augmentation?", default=True)
    aug_config = None

    if do_augmentation:
        console.print("\n[bold]Augmentation Configuration[/bold]")
        aug_input_images = Path(typer.prompt("Input images directory"))
        aug_input_labels = Path(typer.prompt("Input labels directory"))
        aug_output = Path(typer.prompt("Output directory for augmented data"))
        aug_format = typer.prompt(
            "Output format",
            type=click.Choice(["ome-zarr", "tif", "hdf5"]),
            default="ome-zarr",
        )
        n_aug = typer.prompt("Number of augmentations per image", type=int, default=3)

        aug_config = AugmentationConfig(
            input_images_dir=aug_input_images,
            input_labels_dir=aug_input_labels,
            output_dir=aug_output,
            output_format=aug_format,
            n_augmentations=n_aug,
        )

    # Training
    console.print("\n[bold]Training Configuration[/bold]")

    if do_augmentation and aug_config is not None:
        train_images = aug_config.output_dir / "images"
        train_labels = aug_config.output_dir / "labels"
    else:
        train_images = Path(typer.prompt("Training images directory"))
        train_labels = Path(typer.prompt("Training labels directory"))

    model_type = typer.prompt(
        "Model type",
        type=click.Choice(
            ["vit_t", "vit_b", "vit_l", "vit_h", "vit_t_lm", "vit_b_lm", "vit_l_lm"]
        ),
        default="vit_b_lm",
    )

    patch_h = typer.prompt("Patch height", type=int, default=512)
    patch_w = typer.prompt("Patch width", type=int, default=512)
    batch_size = typer.prompt("Batch size", type=int, default=1)
    n_epochs = typer.prompt("Number of epochs", type=int, default=100)
    learning_rate = typer.prompt("Learning rate", type=float, default=1e-5)
    val_split = typer.prompt("Validation split ratio", type=float, default=0.1)

    checkpoint_name = typer.prompt("Checkpoint name", default=experiment_name)

    resume = typer.confirm("Resume from existing checkpoint?", default=False)
    resume_path = None
    if resume:
        resume_path = Path(typer.prompt("Checkpoint path"))

    train_config = TrainingConfig(
        images_dir=train_images,
        labels_dir=train_labels,
        model_type=model_type,
        patch_shape=(patch_h, patch_w),
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        checkpoint_name=checkpoint_name,
        resume_from_checkpoint=resume_path,
    )

    # Create full config
    pipeline_config = PipelineConfig(
        experiment_name=experiment_name,
        output_base_dir=output_base,
        augmentation=aug_config,
        training=train_config,
    )

    # Save to YAML
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(
            pipeline_config.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    console.print(f"\n[bold green]✓[/bold green] Configuration saved to {output}")
    console.print(
        f"\n[dim]Run training with:[/dim] sam-trainer train --config {output}"
    )


@app.command()
def train(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to configuration YAML file"
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase logging verbosity (-v, -vv, -vvv)",
    ),
) -> None:
    """Train SAM model from configuration file."""
    setup_logging(verbose)

    # Load config
    if not config_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(1)

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    try:
        config = PipelineConfig(**config_dict)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Invalid configuration: {e}")
        raise typer.Exit(1)

    # Create experiment directory
    experiment_dir = config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy to experiment dir
    config_copy = experiment_dir / "config.yaml"
    with open(config_copy, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[bold cyan]Experiment:[/bold cyan] {config.experiment_name}")
    console.print(f"[bold cyan]Output dir:[/bold cyan] {experiment_dir}\n")

    # Run augmentation if configured
    if config.augmentation is not None:
        console.print("[bold]Running data augmentation...[/bold]")
        try:
            aug_stats = run_augmentation(config.augmentation)
            console.print(
                f"[green]✓[/green] Generated {aug_stats['total_output_pairs']} image pairs"
            )
        except Exception as e:
            console.print(f"[bold red]Error during augmentation:[/bold red] {e}")
            raise typer.Exit(1)
    else:
        console.print("[dim]Skipping augmentation[/dim]")

    # Run training
    console.print("\n[bold]Running training...[/bold]")
    try:
        results = run_training(config.training, experiment_dir)
        console.print("\n[bold green]✓ Training complete![/bold green]")
        console.print(f"Exported model: {results['exported_model']}")
    except Exception as e:
        console.print(f"[bold red]Error during training:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def augment(
    images_dir: Path = typer.Option(
        ..., "--images", "-i", help="Input images directory"
    ),
    labels_dir: Path = typer.Option(
        ..., "--labels", "-l", help="Input labels directory"
    ),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    n_augmentations: int = typer.Option(
        3, "--n-aug", "-n", help="Number of augmentations per image"
    ),
    output_format: Literal["ome-zarr", "tif", "hdf5", "original"] = typer.Option(
        "original",
        "--format",
        "-f",
        help="Output format (original, ome-zarr, tif, hdf5)",
    ),
    treat_3d_as_2d: bool = typer.Option(
        False,
        "--treat-3d-as-2d",
        help="Treat 3D stacks as separate 2D slices with independent augmentations",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase logging verbosity"
    ),
) -> None:
    """Run data augmentation only (without training)."""
    setup_logging(verbose)

    # Create config
    aug_config = AugmentationConfig(
        input_images_dir=images_dir,
        input_labels_dir=labels_dir,
        output_dir=output_dir,
        output_format=output_format,
        n_augmentations=n_augmentations,
        treat_3d_as_2d=treat_3d_as_2d,
    )

    # Run augmentation
    try:
        stats = run_augmentation(aug_config)
        console.print("\n[bold green]✓[/bold green] Augmentation complete!")
        console.print(
            f"Generated {stats['total_output_pairs']} image pairs in {output_format} format"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
