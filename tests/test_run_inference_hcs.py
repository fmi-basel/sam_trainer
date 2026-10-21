"""Tests for plate-level orchestration in run_inference_hcs (no model or GPU needed)."""

from pathlib import Path

import pytest
import typer

from sam_trainer import run_inference_hcs as hcs


def _make_plates(parent: Path, names: list[str]) -> list[Path]:
    plates = [parent / name for name in names]
    for plate in plates:
        plate.mkdir()
    return plates


def _run(parent: Path, **overrides) -> None:
    hcs.process_hcs_plates(
        input_path=parent,
        model_type="vit_b_lm",
        device="cpu",
        label_name="labels",
        **overrides,
    )


@pytest.fixture
def fake_sequential_backend(monkeypatch):
    """Replace model loading and per-plate work; record which plates were processed."""
    processed: list[str] = []
    monkeypatch.setattr(hcs, "_load_model_cached", lambda **kwargs: (None, None))

    def fake_process_single_plate(plate_path, predictor, segmenter, **kwargs):
        processed.append(plate_path.name)
        hcs._mark_plate_done(plate_path, kwargs["label_name"])
        return True

    monkeypatch.setattr(hcs, "process_single_plate", fake_process_single_plate)
    return processed


def test_plates_are_discovered_in_natural_order(tmp_path, fake_sequential_backend):
    _make_plates(tmp_path, ["p10.zarr", "p2.zarr", "p1.zarr"])

    _run(tmp_path)

    assert fake_sequential_backend == ["p1.zarr", "p2.zarr", "p10.zarr"]


def test_skip_existing_skips_only_completed_plates(tmp_path, fake_sequential_backend):
    done_plate, pending_plate = _make_plates(tmp_path, ["a.zarr", "b.zarr"])
    hcs._mark_plate_done(done_plate, "labels")

    _run(tmp_path, skip_existing=True)

    assert fake_sequential_backend == [pending_plate.name]


def test_sentinel_is_specific_to_label_name(tmp_path, fake_sequential_backend):
    (plate,) = _make_plates(tmp_path, ["a.zarr"])
    hcs._mark_plate_done(plate, "other_labels")

    _run(tmp_path, skip_existing=True)

    assert fake_sequential_backend == ["a.zarr"]


def test_without_skip_existing_completed_plates_are_redone(
    tmp_path, fake_sequential_backend
):
    (plate,) = _make_plates(tmp_path, ["a.zarr"])
    hcs._mark_plate_done(plate, "labels")

    _run(tmp_path)

    assert fake_sequential_backend == ["a.zarr"]


def test_failed_plate_exits_nonzero_and_has_no_sentinel(tmp_path, monkeypatch):
    (plate,) = _make_plates(tmp_path, ["bad.zarr"])
    monkeypatch.setattr(hcs, "_load_model_cached", lambda **kwargs: (None, None))
    monkeypatch.setattr(hcs, "process_single_plate", lambda *args, **kwargs: False)

    with pytest.raises(typer.Exit):
        _run(tmp_path)

    assert not hcs._is_plate_done(plate, "labels")


def test_empty_directory_exits(tmp_path):
    with pytest.raises(typer.Exit):
        _run(tmp_path)
