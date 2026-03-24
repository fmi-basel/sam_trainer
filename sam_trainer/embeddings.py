"""Embeddings extraction pipeline for HCS OME-Zarr plates.

This module extracts microSAM encoder embeddings per organoid instance and exports
reproducible parquet artifacts for downstream classification in another repository.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from ngio import open_ome_zarr_plate

from sam_trainer.config import EmbeddingsExtractionConfig
from sam_trainer.utils.inference_utils import (
    _to_2d,
    load_model_with_decoder,
    resolve_channel_index,
)
from sam_trainer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WellRef:
    """Container for identifying one HCS field in a well."""

    plate_path: Path
    plate: str
    well: str
    row: str
    column: int
    field_path: str


def run_embeddings_extraction(
    config: EmbeddingsExtractionConfig,
) -> dict[str, Path | None]:
    """Run end-to-end embedding extraction and export artifacts."""
    _set_seeds(config.seed)

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.parquet"
    mean_path = run_dir / "embeddings_mean.parquet"
    mean_std_max_path = run_dir / "embeddings_mean_std_max.parquet"
    schema_path = run_dir / "schema.json"
    summary_path = run_dir / "extraction_summary.json"
    skipped_path = run_dir / "skipped_or_failed.csv"

    plate_paths = _discover_plate_paths(
        input_path=config.input_path,
        max_plates=config.max_plates,
        include_plate_regex=config.include_plate_regex,
        exclude_plate_regex=config.exclude_plate_regex,
    )
    logger.info("Discovered %d plate(s)", len(plate_paths))

    predictor, segmenter = load_model_with_decoder(
        model_type=config.model_type,
        device=_resolve_device(config.device),
        model_path=str(config.model_path) if config.model_path is not None else None,
        use_amg=False,
    )

    manifest_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    mean_std_max_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    export_mean = "mean" in config.pooling_modes
    export_mean_std_max = "mean_std_max" in config.pooling_modes

    encoder_calls = 0

    for plate_path in plate_paths:
        plate_name = plate_path.name
        logger.info("Processing plate: %s", plate_name)
        plate = open_ome_zarr_plate(plate_path)
        well_ids = sorted(plate.get_wells().keys())
        if config.max_wells is not None:
            well_ids = well_ids[: config.max_wells]

        for well_id in well_ids:
            try:
                row, column = _parse_well_id(well_id)
                well_ref = WellRef(
                    plate_path=plate_path,
                    plate=plate_name,
                    well=well_id,
                    row=row,
                    column=column,
                    field_path=config.field_path,
                )

                label_array = _read_instance_labels(
                    well_ref=well_ref,
                    label_name=config.label_name,
                    label_level=config.label_level,
                )
                if label_array is None:
                    skipped_rows.append(
                        {
                            "plate": plate_name,
                            "well": well_id,
                            "reason": "missing_label_array",
                        }
                    )
                    continue

                label_ids = np.unique(label_array)
                label_ids = label_ids[label_ids > 0]
                if label_ids.size == 0:
                    logger.info("No organoids in %s/%s", plate_name, well_id)
                    skipped_rows.append(
                        {
                            "plate": plate_name,
                            "well": well_id,
                            "reason": "empty_labels",
                        }
                    )
                    continue

                if config.max_organoids is not None:
                    label_ids = label_ids[: config.max_organoids]

                image_data = plate.get_image(
                    row=row,
                    column=column,
                    image_path=config.field_path,
                ).get_image()
                channel_labels = getattr(image_data, "channel_labels", []) or []
                wavelength_ids = getattr(image_data, "wavelength_ids", []) or []
                channel_index = resolve_channel_index(
                    channel_labels=channel_labels,
                    channel=config.channel,
                    wavelength_ids=wavelength_ids,
                )

                raw = image_data.get_as_numpy()
                image_2d = _to_2d(raw, channel_index=channel_index)
                feature_map = _compute_feature_map(image_2d, predictor, segmenter)
                encoder_calls += 1

                for label_id in label_ids.tolist():
                    composite_id = f"{plate_name}|{well_id}|{int(label_id)}"
                    manifest_row = {
                        "composite_id": composite_id,
                        "plate": plate_name,
                        "well": well_id,
                        "image_path": config.field_path,
                        "label_name": config.label_name,
                        "label_level": config.label_level,
                        "label_id": int(label_id),
                        "raw_locator": f"{row}/{column:02d}/{config.field_path}/0",
                        "label_locator": (
                            f"{row}/{column:02d}/{config.field_path}/labels/"
                            f"{config.label_name}/{config.label_level}"
                        ),
                    }

                    pooled = _pool_for_label(feature_map, label_array, int(label_id))
                    if pooled is None:
                        skipped_rows.append(
                            {
                                "plate": plate_name,
                                "well": well_id,
                                "reason": f"label_{int(label_id)}_vanished_after_resize",
                            }
                        )
                        continue

                    manifest_rows.append(manifest_row)
                    if export_mean:
                        mean_rows.append(
                            {
                                **manifest_row,
                                **_vector_to_columns(pooled["mean"], prefix="emb_m_"),
                            }
                        )
                    if export_mean_std_max:
                        mean_std_max_rows.append(
                            {
                                **manifest_row,
                                **_vector_to_columns(pooled["mean"], prefix="emb_m_"),
                                **_vector_to_columns(pooled["std"], prefix="emb_s_"),
                                **_vector_to_columns(pooled["max"], prefix="emb_x_"),
                            }
                        )

            except Exception as exc:
                logger.exception("Failed processing %s/%s", plate_name, well_id)
                skipped_rows.append(
                    {
                        "plate": plate_name,
                        "well": well_id,
                        "reason": f"exception:{exc}",
                    }
                )

    manifest_df = pd.DataFrame(manifest_rows)
    mean_df = _sort_feature_columns(pd.DataFrame(mean_rows)) if export_mean else None
    mean_std_max_df = (
        _sort_feature_columns(pd.DataFrame(mean_std_max_rows))
        if export_mean_std_max
        else None
    )
    skipped_df = pd.DataFrame(skipped_rows)

    _validate_outputs(
        manifest_df=manifest_df,
        mean_df=mean_df,
        mean_std_max_df=mean_std_max_df,
    )

    manifest_df.to_parquet(manifest_path, index=False)
    if mean_df is not None:
        mean_df.to_parquet(mean_path, index=False)
    if mean_std_max_df is not None:
        mean_std_max_df.to_parquet(mean_std_max_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    schema_df = mean_std_max_df if mean_std_max_df is not None else mean_df
    if schema_df is None:
        raise ValueError("No embeddings dataframe available for schema export")

    schema_payload = {
        "identity_columns": ["composite_id", "plate", "well", "label_id"],
        "mean_columns": (
            [c for c in mean_df.columns if c.startswith("emb_m_")] if mean_df is not None else []
        ),
        "mean_std_max_columns": [
            c
            for c in (mean_std_max_df.columns if mean_std_max_df is not None else [])
            if c.startswith("emb_m_") or c.startswith("emb_s_") or c.startswith("emb_x_")
        ],
        "dtypes": {
            c: str(t)
            for c, t in schema_df.dtypes.items()
        },
    }
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "run_name": config.run_name,
        "plate_count": len(plate_paths),
        "manifest_rows": int(len(manifest_df)),
        "mean_rows": int(len(mean_df)) if mean_df is not None else 0,
        "mean_std_max_rows": int(len(mean_std_max_df)) if mean_std_max_df is not None else 0,
        "skipped_rows": int(len(skipped_df)),
        "encoder_forward_calls": encoder_calls,
        "seed": config.seed,
        "model_type": config.model_type,
        "model_path": str(config.model_path) if config.model_path is not None else None,
        "device": _resolve_device(config.device),
        "pooling_modes": list(config.pooling_modes),
        "input_path": str(config.input_path),
        "git_commit": _get_git_commit(),
        "package_versions": _get_package_versions(),
        "manifest_fingerprint": _fingerprint_manifest(manifest_df),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "run_dir": run_dir,
        "manifest": manifest_path,
        "mean_embeddings": mean_path if mean_df is not None else None,
        "mean_std_max_embeddings": mean_std_max_path if mean_std_max_df is not None else None,
        "schema": schema_path,
        "summary": summary_path,
        "skipped": skipped_path,
    }


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _discover_plate_paths(
    input_path: Path,
    max_plates: int | None,
    include_plate_regex: str | None,
    exclude_plate_regex: str | None,
) -> list[Path]:
    if input_path.name.endswith(".zarr"):
        plate_paths = [input_path]
    else:
        plate_paths = sorted(p for p in input_path.glob("*.zarr") if p.is_dir())

    if include_plate_regex is not None:
        include_pattern = re.compile(include_plate_regex)
        plate_paths = [p for p in plate_paths if include_pattern.search(p.name)]

    if exclude_plate_regex is not None:
        exclude_pattern = re.compile(exclude_plate_regex)
        plate_paths = [p for p in plate_paths if not exclude_pattern.search(p.name)]

    if not plate_paths:
        raise FileNotFoundError(f"No .zarr plates found in {input_path}")
    if max_plates is not None:
        plate_paths = plate_paths[:max_plates]
    return plate_paths


def _parse_well_id(well_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)[/_-]?(\d+)", well_id)
    if match is None:
        raise ValueError(f"Unsupported well ID format: {well_id}")
    row = match.group(1)
    column = int(match.group(2))
    return row, column


def _read_instance_labels(
    well_ref: WellRef,
    label_name: str,
    label_level: int,
) -> np.ndarray | None:
    root = zarr.open(str(well_ref.plate_path), mode="r")
    key = (
        f"{well_ref.row}/{well_ref.column:02d}/{well_ref.field_path}/labels/"
        f"{label_name}/{label_level}"
    )
    if key not in root:
        logger.warning("Missing label key %s in %s", key, well_ref.plate_path)
        return None
    labels = np.asarray(root[key])
    labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(
            f"Expected 2D labels at {key}, got shape {labels.shape} in {well_ref.plate_path}"
        )
    return labels


def _compute_feature_map(image: np.ndarray, predictor, segmenter) -> np.ndarray:
    segmenter.initialize(image)
    features = predictor.features
    if features is None:
        raise RuntimeError("Predictor features were not populated after initialize(image)")
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = np.asarray(features)

    if features_np.ndim == 4:
        if features_np.shape[0] != 1:
            raise ValueError(f"Unexpected feature map shape {features_np.shape}")
        features_np = features_np[0]

    if features_np.ndim != 3:
        raise ValueError(f"Expected feature map with 3 dims, got {features_np.shape}")

    return features_np.astype(np.float32, copy=False)


def _pool_for_label(
    feature_map: np.ndarray,
    labels: np.ndarray,
    label_id: int,
) -> dict[str, np.ndarray] | None:
    feature_tensor = torch.from_numpy(feature_map)
    _, feat_h, feat_w = feature_tensor.shape

    label_mask = (labels == label_id).astype(np.float32)
    mask_tensor = torch.from_numpy(label_mask)[None, None]
    mask_small = F.interpolate(mask_tensor, size=(feat_h, feat_w), mode="nearest")[0, 0]
    mask_small_np = mask_small.numpy().astype(bool)

    # Tiny labels can disappear when projected to the lower-resolution encoder grid.
    # Fall back to coordinate projection so each original label pixel votes for a feature cell.
    if not mask_small_np.any():
        ys, xs = np.nonzero(label_mask)
        if ys.size > 0:
            in_h, in_w = labels.shape
            y_idx = np.clip((ys * feat_h) // max(in_h, 1), 0, feat_h - 1)
            x_idx = np.clip((xs * feat_w) // max(in_w, 1), 0, feat_w - 1)
            mask_small_np[y_idx, x_idx] = True

    if not mask_small_np.any():
        return None

    selected = feature_map[:, mask_small_np]
    mean_vec = selected.mean(axis=1)
    std_vec = selected.std(axis=1)
    max_vec = selected.max(axis=1)

    return {"mean": mean_vec, "std": std_vec, "max": max_vec}


def _vector_to_columns(vector: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}{idx:04d}": float(value) for idx, value in enumerate(vector)}


def _sort_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    id_columns = [
        "composite_id",
        "plate",
        "well",
        "image_path",
        "label_name",
        "label_level",
        "label_id",
        "raw_locator",
        "label_locator",
    ]
    present_id_columns = [c for c in id_columns if c in df.columns]
    feature_columns = sorted(c for c in df.columns if c not in present_id_columns)
    return df[present_id_columns + feature_columns]


def _validate_outputs(
    manifest_df: pd.DataFrame,
    mean_df: pd.DataFrame | None,
    mean_std_max_df: pd.DataFrame | None,
) -> None:
    if manifest_df.empty:
        raise ValueError("No valid organoid instances were extracted")

    if manifest_df["composite_id"].duplicated().any():
        raise ValueError("Duplicate composite_id detected in manifest")

    if mean_df is not None and len(manifest_df) != len(mean_df):
        raise ValueError(
            "Row mismatch between manifest and mean embeddings outputs: "
            f"manifest={len(manifest_df)}, mean={len(mean_df)}"
        )
    if mean_std_max_df is not None and len(manifest_df) != len(mean_std_max_df):
        raise ValueError(
            "Row mismatch between manifest and mean_std_max embeddings outputs: "
            f"manifest={len(manifest_df)}, mean_std_max={len(mean_std_max_df)}"
        )

    feature_source_df = mean_std_max_df if mean_std_max_df is not None else mean_df
    if feature_source_df is None:
        raise ValueError("No embeddings table was selected for export")

    feature_columns = [c for c in feature_source_df.columns if c.startswith("emb_")]
    if not feature_columns:
        raise ValueError("No feature columns produced")

    feature_values = feature_source_df[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_values).all():
        raise ValueError("NaN or inf values found in embeddings features")


def _get_package_versions() -> dict[str, str]:
    import importlib.metadata

    packages = ["micro_sam", "ngio", "numpy", "pandas", "torch", "zarr"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _fingerprint_manifest(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    keys = sorted(df["composite_id"].astype(str).tolist())
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
