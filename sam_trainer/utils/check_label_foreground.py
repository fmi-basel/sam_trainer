"""Utility to verify that each label file contains foreground instances."""

from __future__ import annotations

from pathlib import Path

import tifffile
import typer
from tqdm import tqdm

app = typer.Typer(help="Validate that label masks contain at least one non-zero pixel.")


def _iter_label_files(labels_dir: Path) -> list[Path]:
    pattern_extensions = (".tif", ".tiff", ".h5", ".hdf5", ".zarr")
    files = []
    for path in sorted(labels_dir.rglob("*")):
        if path.suffix.lower() in pattern_extensions and path.is_file():
            files.append(path)
    return files


@app.command()
def main(
    labels_dir: Path = typer.Argument(..., exists=True, dir_okay=True, file_okay=False),
) -> None:
    """Scan all label files and report empty masks."""

    label_files = _iter_label_files(labels_dir)
    if not label_files:
        typer.echo(f"No label files found under {labels_dir}")
        raise typer.Exit(code=1)

    typer.echo(f"Checking {len(label_files)} label files in {labels_dir}")

    empty_files: list[Path] = []
    for path in tqdm(label_files, desc="Scanning labels"):
        data = tifffile.imread(path)
        if data.max() == 0:
            empty_files.append(path)

    if empty_files:
        typer.echo(f"Found {len(empty_files)} label files without foreground pixels:")
        preview = empty_files[:20]
        for idx, empty_path in enumerate(preview, start=1):
            typer.echo(f"[{idx:02d}] {empty_path}")
        if len(empty_files) > len(preview):
            typer.echo("...")
        raise typer.Exit(code=2)

    typer.echo("All label files contain at least one object.")


if __name__ == "__main__":
    app()
