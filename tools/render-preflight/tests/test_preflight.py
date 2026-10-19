"""Independent scalar accounting and operational regression cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from render_preflight import LocalSlab, scan
from render_preflight.cli import main


def make_store(path: Path, values: np.ndarray, *, version=2, separator="/", chunks=None,
               fill_value=0, compressed=False, encoding_name=None, **kwargs) -> LocalSlab:
    options = dict(data=values, chunks=chunks or values.shape, zarr_format=version,
                   fill_value=fill_value, config={"write_empty_chunks": True}, **kwargs)
    options["chunk_key_encoding"] = {"name": encoding_name or ("v2" if version == 2 else "default"),
                                     "separator": separator}
    if not compressed:
        options["compressors"] = None
    zarr.create_array(str(path), **options)
    return LocalSlab(path)


def scalar_oracle(values: np.ndarray, present: np.ndarray, bounds: tuple[int, int, int, int]) -> dict:
    """Loop over individual rays/samples, independent of vectorized production code."""
    depth = values.shape[0]
    peaks, sums, counts, common_sums = [0.0] * depth, [0.0] * depth, [0] * depth, [0.0] * depth
    materialized = finite = zero = complete = constant = allzero = varying = tied = edge = edge_only = 0
    y0, y1, x0, x1 = bounds
    for y in range(y0, y1):
        for x in range(x0, x1):
            ray = [float(values[z, y, x]) for z in range(depth)]
            good = True
            for z, value in enumerate(ray):
                if present[z, y, x]:
                    materialized += 1
                    if np.isfinite(value):
                        finite += 1
                        zero += value == 0
                        sums[z] += value
                        counts[z] += 1
                    else:
                        good = False
                else:
                    good = False
            if not good:
                continue
            complete += 1
            common_sums = [s + v for s, v in zip(common_sums, ray)]
            if max(ray) == min(ray):
                constant += 1
                allzero += max(ray) == 0
                continue
            varying += 1
            winners = [z for z, value in enumerate(ray) if value == max(ray)]
            tied += len(winners) > 1
            edge += 0 in winners or depth - 1 in winners
            edge_only += all(z in (0, depth - 1) for z in winners)
            for z in winners:
                peaks[z] += 1.0 / len(winners)
    pixels = (y1 - y0) * (x1 - x0)
    return {
        "sample_counts": dict(total=pixels * depth, materialized=materialized,
                              implicit_fill=pixels * depth - materialized, finite_materialized=finite,
                              nonfinite_materialized=materialized - finite, zero_materialized=zero),
        "ray_counts": dict(total=pixels, complete_finite=complete, incomplete_or_nonfinite=pixels - complete,
                           constant_complete=constant, allzero_complete=allzero, varying_complete=varying,
                           tied_maximum=tied, maximum_includes_endpoint=edge, maximum_at_endpoints_only=edge_only),
        "finite_counts_by_layer": counts,
        "finite_mean_by_layer": [s / n if n else None for s, n in zip(sums, counts)],
        "common_support_mean_by_layer": [s / complete if complete else None for s in common_sums],
        "tie_weighted_maximum_by_layer": peaks,
    }


@pytest.mark.parametrize("version,separator", [(2, "."), (2, "/"), (3, "."), (3, "/")])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_formats_partial_chunks_depth_chunks_and_window_match_scalar_oracle(tmp_path, version, separator, dtype):
    values = (np.arange(7 * 5 * 9).reshape(7, 5, 9) % 17).astype(dtype)
    values[:, 0, 0] = 0
    values[:, 1, 1] = 3
    values[:, 2, 2] = [9, 0, 9, 0, 0, 0, 0]
    slab = make_store(tmp_path / "input", values, version=version, separator=separator,
                      chunks=(3, 2, 4), compressed=True)
    slab.chunk_path((1, 0, 1)).unlink()
    present = np.ones(values.shape, dtype=bool)
    present[3:6, 0:2, 4:8] = False
    bounds = (0, 5, 0, 8)
    actual = scan(slab, window=bounds)
    expected = scalar_oracle(values, present, bounds)
    for key, value in expected.items():
        if "mean" in key or key == "tie_weighted_maximum_by_layer":
            assert actual[key] == pytest.approx(value)
        else:
            assert actual[key] == value
    assert actual["chunk_keys"] == {"materialized": 17, "unmaterialized": 1}
    assert actual["spatial_tiles"] == 6


def test_implicit_fill_and_stored_zero_have_distinct_denominators(tmp_path):
    values = np.zeros((5, 2, 4), dtype=np.uint8)
    slab = make_store(tmp_path / "zeros", values, chunks=(5, 2, 2))
    slab.chunk_path((0, 0, 1)).unlink()
    result = scan(slab)
    assert result["sample_counts"]["zero_materialized"] == 20
    assert result["sample_counts"]["implicit_fill"] == 20
    assert result["ray_counts"]["allzero_complete"] == 4
    assert result["ray_counts"]["incomplete_or_nonfinite"] == 4
    assert result["endpoint_maximum_mass_fraction"] is None
    assert result["tie_weighted_maximum_by_layer"] == [0.0] * 5


@pytest.mark.parametrize("fill", [7.0, float("nan")])
def test_nonzero_or_nan_implicit_fill_never_becomes_an_observation(tmp_path, fill):
    values = np.broadcast_to(np.arange(5)[:, None, None], (5, 2, 4)).astype(np.float32).copy()
    slab = make_store(tmp_path / "fill", values, chunks=(5, 2, 2), fill_value=fill)
    slab.chunk_path((0, 0, 1)).unlink()
    result = scan(slab)
    assert result["sample_counts"]["implicit_fill"] == 20
    assert result["sample_counts"]["nonfinite_materialized"] == 0
    assert result["common_support_mean_by_layer"] == [0, 1, 2, 3, 4]


def test_ties_are_split_flat_profiles_excluded_and_signed_data_retained(tmp_path):
    values = np.array([[0, 2, -8, 1], [9, 2, -4, 1], [0, 2, -3, 1],
                       [9, 2, -9, 1], [0, 2, -7, 1]], dtype=np.int16)[:, None, :]
    result = scan(make_store(tmp_path / "ties", values))
    assert result["tie_weighted_maximum_by_layer"] == [0, 0.5, 1, 0.5, 0]
    assert result["ray_counts"]["constant_complete"] == 2
    assert result["ray_counts"]["tied_maximum"] == 1
    assert result["ray_counts"]["varying_complete"] == 2


def test_nonfinite_ray_excluded_from_common_support_not_silently_zero_filled(tmp_path):
    values = np.array([[1, 10], [float("nan"), 12], [2, 14]], dtype=np.float32)[:, None, :]
    result = scan(make_store(tmp_path / "nan", values))
    assert result["sample_counts"]["nonfinite_materialized"] == 1
    assert result["finite_counts_by_layer"] == [2, 1, 2]
    assert result["common_support_mean_by_layer"] == [10, 12, 14]
    assert result["ray_counts"]["complete_finite"] == 1


def test_memory_limit_fails_before_read(tmp_path, monkeypatch):
    slab = make_store(tmp_path / "input", np.ones((5, 4, 4), dtype=np.uint8))
    class MustNotRead:
        def __getitem__(self, key):
            raise AssertionError("Read happened before block-size validation")
    monkeypatch.setattr(slab, "array", MustNotRead())
    with pytest.raises(ValueError, match="exceeding limit"):
        scan(slab, max_read_bytes=79)


def test_metadata_only_copy_has_no_observations(tmp_path):
    values = np.ones((5, 2, 4), dtype=np.uint8)
    slab = make_store(tmp_path / "empty", values)
    slab.chunk_path((0, 0, 0)).unlink()
    result = scan(slab)
    assert result["sample_counts"]["finite_materialized"] == 0
    assert result["finite_mean_by_layer"] == [None] * 5
    assert result["common_support_mean_by_layer"] == [None] * 5


def test_group_selection_cli_csv_exit_codes_and_read_only_input(tmp_path, capsys):
    root = tmp_path / "group"
    zarr.open_group(str(root), mode="w", zarr_format=2)
    slab = make_store(root / "0", np.ones((5, 2, 4), dtype=np.uint8), chunks=(5, 2, 2))
    slab.chunk_path((0, 0, 1)).unlink()
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file()}
    output, tiles = tmp_path / "summary.json", tmp_path / "tiles.csv"
    assert main([str(root), "--json", str(output), "--tiles", str(tiles), "--require-materialized"]) == 1
    result = json.loads(output.read_text())
    assert result["chunk_keys"]["unmaterialized"] == 1
    assert len(tiles.read_text().splitlines()) == 3
    assert main([str(root), "--json", str(output)]) == 2
    after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_wrong_axes_shards_bad_window_and_corrupt_chunk_are_refused(tmp_path, capsys):
    slab = make_store(tmp_path / "bad", np.ones((5, 2, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="Window"):
        scan(slab, window=(0, 3, 0, 4))
    slab.chunk_path((0, 0, 0)).write_bytes(b"broken")
    assert main([str(slab.root)]) == 2
    assert not capsys.readouterr().out
    with pytest.raises(ValueError, match="dimension names"):
        make_store(tmp_path / "wrong_axes", np.ones((3, 4, 5)), version=3, dimension_names=("x", "y", "z"))
    with pytest.raises(ValueError, match="Sharded"):
        make_store(tmp_path / "sharded", np.ones((8, 4, 4)), version=3, chunks=(4, 2, 2), shards=(8, 4, 4))


def test_metadata_mutation_is_detected(tmp_path):
    slab = make_store(tmp_path / "changing", np.ones((5, 2, 2), dtype=np.uint8))
    slab.meta_path.write_text(slab.meta_path.read_text() + "\n")
    with pytest.raises(ValueError, match="metadata changed"):
        scan(slab)


def test_failed_scan_never_publishes_partial_reports(tmp_path):
    slab = make_store(tmp_path / "broken", np.ones((5, 2, 4), dtype=np.uint8), chunks=(5, 2, 2))
    slab.chunk_path((0, 0, 1)).write_bytes(b"broken")
    summary, tiles = tmp_path / "summary.json", tmp_path / "tiles.csv"
    assert main([str(slab.path), "--json", str(summary), "--tiles", str(tiles)]) == 2
    assert not summary.exists()
    assert not tiles.exists()
    assert not list(tmp_path.glob(".render-preflight-*"))


@pytest.mark.parametrize("separator", [".", "/"])
def test_v3_v2_key_encoding_tracks_actual_materialization(tmp_path, separator):
    slab = make_store(tmp_path / "v3", np.ones((5, 2, 4), dtype=np.uint16),
                      version=3, separator=separator, encoding_name="v2", chunks=(5, 2, 2))
    slab.chunk_path((0, 0, 1)).unlink()
    result = scan(slab)
    assert result["chunk_keys"] == {"materialized": 1, "unmaterialized": 1}
    assert result["sample_counts"]["zero_materialized"] == 0


def test_require_varying_is_explicit_and_not_a_surface_pass(tmp_path, capsys):
    slab = make_store(tmp_path / "flat", np.zeros((5, 2, 4), dtype=np.uint8))
    assert main([str(slab.path)]) == 0
    assert json.loads(capsys.readouterr().out)["ray_counts"]["allzero_complete"] == 8
    assert main([str(slab.path), "--require-varying"]) == 1
    assert json.loads(capsys.readouterr().out)["interpretation"]["scope"].startswith("Descriptive CT")


def test_compressed_corruption_returns_error_without_summary(tmp_path, capsys):
    slab = make_store(tmp_path / "compressed", np.arange(40, dtype=np.uint8).reshape(5, 2, 4), compressed=True)
    slab.chunk_path((0, 0, 0)).write_bytes(b"broken")
    assert main([str(slab.path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "render-preflight:" in captured.err
