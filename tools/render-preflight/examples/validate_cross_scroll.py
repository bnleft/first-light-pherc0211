"""Real byte/scalar-oracle validation, chunk I/O, and lossless layout changes.

No network. All planned windows and their storage populations are retained.
The independent scalar oracle is limited to the plan's uint8 source renders.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import zarr
from fetch_cross_scroll import verify_fixture
from numcodecs import Blosc, get_codec
from validate_public_render import measurement

import render_preflight
from render_preflight import LocalSlab, scan

METRICS = ("sample_counts", "ray_counts", "finite_counts_by_layer", "finite_mean_by_layer",
           "common_support_mean_by_layer", "tie_weighted_maximum_by_layer", "endpoint_maximum_mass_fraction")


def reconstruct(store: Path, dataset: dict, window: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode named physical keys directly, without Zarr's indexing/selection code."""
    meta = dataset["array_metadata"]
    depth = dataset["shape"][0]
    cz, cy, cx = meta["chunks"]
    y0, y1, x0, x1 = window
    values = np.full((depth, y1 - y0, x1 - x0), meta["fill_value"], dtype=np.uint8)
    present = np.zeros(values.shape, dtype=bool)
    separator = meta.get("dimension_separator", ".")
    for iz, iy, ix in itertools.product(range((depth + cz - 1) // cz),
                                      range(y0 // cy, (y1 + cy - 1) // cy),
                                      range(x0 // cx, (x1 + cx - 1) // cx)):
        path = store / "0" / separator.join(map(str, (iz, iy, ix)))
        if not path.exists():
            continue
        encoded = path.read_bytes()
        decoded = get_codec(meta["compressor"]).decode(encoded) if meta["compressor"] else encoded
        chunk = np.frombuffer(decoded, dtype=np.uint8).reshape(cz, cy, cx)
        za, zb = iz * cz, min((iz + 1) * cz, depth)
        ya, yb, xa, xb = max(y0, iy * cy), min(y1, (iy + 1) * cy), max(x0, ix * cx), min(x1, (ix + 1) * cx)
        selection = (slice(za, zb), slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        values[selection] = chunk[:zb - za, ya - iy * cy:yb - iy * cy, xa - ix * cx:xb - ix * cx]
        present[selection] = True
    return values, present


def reference(values: np.ndarray, present: np.ndarray) -> dict:
    """Full independent scalar peak/ray oracle on directly decoded integer values."""
    depth, height, width = values.shape
    complete = present.all(axis=0)
    common = values[:, complete]
    hist = [0.0] * depth
    constant = allzero = varying = tied = endpoint = endpoint_only = 0
    for ray in common.T:
        samples = ray.tolist()
        lo, hi = min(samples), max(samples)
        if hi == lo:
            constant += 1
            allzero += hi == 0
            continue
        varying += 1
        winners = [z for z, value in enumerate(samples) if value == hi]
        tied += len(winners) > 1
        endpoint += 0 in winners or depth - 1 in winners
        endpoint_only += all(z in (0, depth - 1) for z in winners)
        for z in winners:
            hist[z] += 1.0 / len(winners)
    materialized, count = int(present.sum()), int(complete.sum())
    finite_counts = present.sum(axis=(1, 2)).tolist()
    finite_means = [float(values[z][present[z]].mean()) if n else None for z, n in enumerate(finite_counts)]
    return {
        "sample_counts": {"total": values.size, "materialized": materialized,
                          "implicit_fill": values.size - materialized, "finite_materialized": materialized,
                          "nonfinite_materialized": 0, "zero_materialized": int(((values == 0) & present).sum())},
        "ray_counts": {"total": height * width, "complete_finite": count, "incomplete_or_nonfinite": height * width - count,
                       "constant_complete": constant, "allzero_complete": allzero, "varying_complete": varying,
                       "tied_maximum": tied, "maximum_includes_endpoint": endpoint, "maximum_at_endpoints_only": endpoint_only},
        "finite_counts_by_layer": finite_counts, "finite_mean_by_layer": finite_means,
        "common_support_mean_by_layer": common.mean(axis=1).tolist() if count else [None] * depth,
        "tie_weighted_maximum_by_layer": hist,
        "endpoint_maximum_mass_fraction": (hist[0] + hist[-1]) / varying if varying else None,
    }


def compare(actual: dict, expected: dict) -> None:
    for key in METRICS:
        a, b = actual[key], expected[key]
        if key in ("sample_counts", "ray_counts", "finite_counts_by_layer") or a is None:
            assert a == b, key
        elif isinstance(a, list) and any(value is None for value in a):
            assert a == b, key
        else:
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-7, err_msg=key)


def layouts(values: np.ndarray, expected: dict, destination: Path, source_id: str, source_window: list[int]) -> list[dict]:
    results = []
    for version, chunks, dtype in ((2, (7, 31, 47), np.uint8), (3, (17, 64, 64), np.uint16)):
        path = destination / f"{source_id}_v{version}.zarr"
        group = zarr.open_group(str(path), mode="w", zarr_format=version)
        scale = {"axes": [{"name": axis, "type": "space"} for axis in ("z", "y", "x")],
                 "datasets": [{"path": "0", "coordinateTransformations": [
                     {"type": "scale", "scale": [1, 1, 1]},
                     {"type": "translation", "translation": [0, source_window[0], source_window[2]]}]}]}
        attrs = {"multiscales": [scale]} if version == 2 else {"ome": {"version": "0.5", "multiscales": [scale]}}
        attrs["source_index_space_crop"] = {"source_id": source_id, "window_yx": source_window, "no_resampling": True}
        group.attrs.update(attrs)
        kwargs = {"compressors": Blosc(cname="lz4", clevel=3)} if version == 2 else {"dimension_names": ("z", "y", "x")}
        group.create_array("0", data=values.astype(dtype), chunks=chunks, config={"write_empty_chunks": True}, **kwargs)
        report = scan(LocalSlab(path))
        compare(report, expected)
        results.append({"source_id": source_id, "source_window_yx": source_window, "zarr_format": version,
                        "chunks": list(chunks), "dtype": str(np.dtype(dtype)), "chunk_keys": report["chunk_keys"],
                        "all_metrics_match_source": True})
    return results


def validate(root: Path, work: Path) -> dict:
    fixture = verify_fixture(root)
    work.mkdir(parents=True, exist_ok=False)
    windows, derived = [], []
    for dataset in fixture["datasets"]:
        store = root / f"{dataset['id']}.zarr"
        for index, window in enumerate(dataset["windows_yx"]):
            values, present = reconstruct(store, dataset, window)
            slab = LocalSlab(store)
            y0, y1, x0, x1 = window
            np.testing.assert_array_equal(slab.array[:, y0:y1, x0:x1], values)
            expected = reference(values, present)
            strategies = ["layerwise", "preflight"] if index % 2 == 0 else ["preflight", "layerwise"]
            measured = {strategy: measurement(store, tuple(window), strategy) for strategy in strategies}
            baseline, current = measured["layerwise"], measured["preflight"]
            report = current["report"]
            compare(report, expected)
            assert baseline["profile"] == current["profile"] == expected["common_support_mean_by_layer"]
            depth, cz = dataset["shape"][0], dataset["chunks"][0]
            assert cz == depth, "This source I/O comparison is for full-depth source chunks"
            assert baseline["chunk_gets"] == depth * current["chunk_gets"]
            assert baseline["chunk_bytes_read"] == depth * current["chunk_bytes_read"]
            windows.append({"source_id": dataset["id"], "window_yx": window, **{key: report[key] for key in METRICS},
                            "chunk_keys": report["chunk_keys"], "decoded_bytes_match": True, "scalar_oracle_matches": True,
                            "measurements": {key: {k: v for k, v in measured[key].items() if k not in ("report", "profile")}
                                             for key in strategies}})
            if index == 0 and present.all():
                derived.extend(layouts(values, expected, work, dataset["id"], window))
    verify_fixture(root)
    return {"schema": "render_preflight_cross_scroll_validation_v1", "plan_sha256": fixture["plan_sha256"],
            "tool_source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(Path(render_preflight.__file__).parent.glob("*.py"))},
            "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                            for name in ("numpy", "zarr", "numcodecs", "vesuvius-render-preflight")}},
            "source_manifest": "cross_scroll_manifest.json", "windows": windows, "derived_layouts": derived,
            "io_measurement": "Zarr LocalStore.get calls and returned encoded payload lengths on local files; not measured HTTP requests or physical disk traffic.",
            "totals": {"scrolls": len(fixture["datasets"]), "windows": len(windows),
                       "rays": sum(w["ray_counts"]["total"] for w in windows),
                       "samples": sum(w["sample_counts"]["total"] for w in windows)},
            "limits": "Fixed convenience windows, not an unbiased survey; no anatomical/ink accuracy or whole-scroll performance claim. Derived layouts are lossless crops/casts of existing samples, not additional scroll evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.source, args.work), indent=2, allow_nan=False))
