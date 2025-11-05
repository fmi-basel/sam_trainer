"""I/O utilities for reading and writing image data in various formats."""

import logging
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from skimage import io as skio
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite

logger = logging.getLogger(__name__)


def read_image(path: Path, key: Optional[str] = None) -> np.ndarray:
    """Read an image from various formats.

    Args:
        path: Path to the image file or directory (for OME-Zarr)
        key: Dataset key for HDF5 files (ignored for other formats)

    Returns:
        Image array with shape (Z, Y, X) for 3D or (Y, X) for 2D

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    # OME-Zarr (directory-based format)
    if path.is_dir():
        logger.debug(f"Reading OME-Zarr from {path}")
        return _read_ome_zarr(path)

    # File-based formats
    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        logger.debug(f"Reading TIF from {path}")
        return tif_imread(path)

    elif suffix in [".h5", ".hdf5"]:
        logger.debug(f"Reading HDF5 from {path}")
        return _read_hdf5(path, key)

    elif suffix == ".zarr":
        logger.debug(f"Reading Zarr from {path}")
        return _read_zarr_file(path)

    else:
        # Try generic imread as fallback
        logger.debug(f"Attempting generic read for {path}")
        return skio.imread(path)


def _read_ome_zarr(path: Path) -> np.ndarray:
    """Read OME-Zarr format (directory-based)."""
    reader = Reader(parse_url(str(path)))

    # Get the first (highest resolution) image
    nodes = list(reader())
    if not nodes:
        raise ValueError(f"No image data found in OME-Zarr: {path}")

    image_node = nodes[0]
    data = image_node.data[0]  # Get highest resolution level

    # Load into memory (consider lazy loading for very large images)
    return np.asarray(data)


def _read_zarr_file(path: Path) -> np.ndarray:
    """Read a single Zarr file."""
    z = zarr.open(str(path), mode="r")
    return np.asarray(z)


def _read_hdf5(path: Path, key: Optional[str] = None) -> np.ndarray:
    """Read HDF5 file with automatic key detection."""
    with h5py.File(path, "r") as f:
        if key is not None:
            if key not in f:
                raise ValueError(
                    f"Key '{key}' not found in HDF5 file. Available: {list(f.keys())}"
                )
            return np.asarray(f[key])

        # Auto-detect: use first dataset
        datasets = _find_datasets(f)
        if not datasets:
            raise ValueError(f"No datasets found in HDF5 file: {path}")

        logger.debug(f"Auto-detected dataset: {datasets[0]}")
        return np.asarray(f[datasets[0]])


def _find_datasets(h5obj) -> list[str]:
    """Recursively find all datasets in HDF5 file."""
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5obj.visititems(visitor)
    return datasets


def write_image(
    data: np.ndarray,
    path: Path,
    format: Literal["ome-zarr", "tif", "hdf5"],
    key: str = "data",
) -> None:
    """Write image data to various formats.

    Args:
        data: Image array with shape (Z, Y, X) or (Y, X)
        path: Output path
        format: Output format
        key: Dataset name for HDF5 (ignored for other formats)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "ome-zarr":
        _write_ome_zarr(data, path)
    elif format == "tif":
        tif_imwrite(path.with_suffix(".tif"), data, compression="zlib")
    elif format == "hdf5":
        _write_hdf5(data, path.with_suffix(".h5"), key)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.debug(f"Wrote {format} to {path}")


def _write_ome_zarr(data: np.ndarray, path: Path) -> None:
    """Write OME-Zarr format."""
    from ome_zarr.writer import write_image as ome_write

    path.mkdir(parents=True, exist_ok=True)
    store = parse_url(str(path), mode="w").store

    # Write with default chunks and no pyramid for simplicity
    ome_write(
        image=data,
        group=zarr.group(store=store),
        axes="zyx" if data.ndim == 3 else "yx",
        storage_options=dict(
            chunks=(1, min(512, data.shape[-2]), min(512, data.shape[-1]))
        ),
    )


def _write_hdf5(data: np.ndarray, path: Path, key: str) -> None:
    """Write HDF5 format."""
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=data, compression="gzip", chunks=True)


def get_image_paths(directory: Path, pattern: str = "*") -> list[Path]:
    """Get sorted list of image paths from directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match files

    Returns:
        Sorted list of paths (includes both files and directories for OME-Zarr)
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # For OME-Zarr, look for directories containing .zattrs or .zgroup
    zarr_dirs = [
        p.parent for p in directory.rglob(".zattrs") if (p.parent / ".zgroup").exists()
    ]

    # Regular files
    file_patterns = ["*.tif", "*.tiff", "*.h5", "*.hdf5", "*.zarr"]
    files = []
    for pat in file_patterns:
        files.extend(directory.glob(pat))

    all_paths = sorted(set(zarr_dirs + files))
    logger.debug(f"Found {len(all_paths)} images in {directory}")

    return all_paths


def extract_slices_2d(volume: np.ndarray) -> list[np.ndarray]:
    """Extract 2D slices from a 3D volume.

    Args:
        volume: 3D array with shape (Z, Y, X)

    Returns:
        List of 2D arrays, one per Z slice
    """
    if volume.ndim == 2:
        return [volume]
    elif volume.ndim == 3:
        return [volume[i] for i in range(volume.shape[0])]
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {volume.shape}")
