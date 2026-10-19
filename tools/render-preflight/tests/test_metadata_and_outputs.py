"""Metadata semantics and complete report publication, exercised through the CLI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from render_preflight import LocalSlab, scan
from render_preflight.cli import main


def image_group(path: Path, axes: list[str], *, ome_version: str) -> Path:
    scale = {"axes": [{"name": name, "type": "space"} for name in axes], "datasets": [{"path": "0"}]}
    attrs = {"multiscales": [scale]}
    if ome_version == "0.5":
        attrs = {"ome": {"version": "0.5", **attrs}}
    group = zarr.open_group(str(path), mode="w", zarr_format=3 if ome_version == "0.5" else 2)
    group.attrs.update(attrs)
    group.create_array("0", data=np.arange(24, dtype=np.uint8).reshape(3, 2, 4), chunks=(3, 2, 2))
    return path


@pytest.mark.parametrize("ome_version", ["0.4", "0.5"])
@pytest.mark.parametrize("direct_array", [False, True])
def test_parent_axes_checked_for_group_and_direct_array_paths(tmp_path, ome_version, direct_array):
    group = image_group(tmp_path / "wrong", ["x", "y", "z"], ome_version=ome_version)
    source = group / "0" if direct_array else group
    with pytest.raises(ValueError, match="Expected ZYX multiscale axes"):
        LocalSlab(source)


@pytest.mark.parametrize("ome_version", ["0.4", "0.5"])
@pytest.mark.parametrize("direct_array", [False, True])
def test_correct_ome_axes_and_parent_metadata_are_pinned(tmp_path, ome_version, direct_array):
    group = image_group(tmp_path / "correct", ["z", "y", "x"], ome_version=ome_version)
    source = group / "0" if direct_array else group
    slab = LocalSlab(source)
    report = scan(slab)
    assert str(group) in report["axis_metadata_sources"]
    parent_metadata = group / ("zarr.json" if ome_version == "0.5" else ".zattrs")
    assert str(parent_metadata) in report["inspected_metadata_sha256"]
    parent_metadata.write_text(parent_metadata.read_text() + "\n")
    with pytest.raises(ValueError, match="metadata changed"):
        scan(slab)


def test_axes_for_another_array_do_not_override_selected_array(tmp_path):
    path = image_group(tmp_path / "mixed", ["z", "y", "x"], ome_version="0.5")
    group = zarr.open_group(str(path), mode="r+")
    group.attrs["ome"] = {"version": "0.5", "multiscales": [
        {"axes": ["x", "y", "z"], "datasets": [{"path": "other"}]},
        {"axes": ["z", "y", "x"], "datasets": [{"path": "0"}]},
    ]}
    assert scan(LocalSlab(path))["ray_counts"]["total"] == 8


def test_publish_failure_rolls_back_only_reports_created_by_this_call(tmp_path, monkeypatch, capsys):
    source = image_group(tmp_path / "input", ["z", "y", "x"], ome_version="0.5")
    summary, tiles = tmp_path / "summary.json", tmp_path / "tiles.csv"
    original = os.link

    def racing_link(pending, destination):
        if destination == summary:
            summary.write_text("another writer's result")
            raise FileExistsError("Another writer published the summary")
        return original(pending, destination)

    monkeypatch.setattr(os, "link", racing_link)
    assert main([str(source), "--json", str(summary), "--tiles", str(tiles)]) == 2
    assert summary.read_text() == "another writer's result"
    assert not tiles.exists()
    assert not capsys.readouterr().out
    assert not list(tmp_path.glob(".render-preflight-*"))


@pytest.mark.parametrize("bad", [None, [], "axes", 4])
def test_malformed_ome_metadata_returns_a_clear_input_error(tmp_path, capsys, bad):
    path = image_group(tmp_path / "bad", ["z", "y", "x"], ome_version="0.5")
    group = zarr.open_group(str(path), mode="r+")
    group.attrs["ome"] = bad
    assert main([str(path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "OME metadata must be an object" in captured.err
