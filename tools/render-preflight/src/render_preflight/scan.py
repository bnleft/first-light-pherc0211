"""Streaming accounting; maxima describe CT intensity, not physical surfaces."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from .storage import LocalSlab

SAMPLE_COUNTS = ("total", "materialized", "implicit_fill", "finite_materialized",
                 "nonfinite_materialized", "zero_materialized")
RAY_COUNTS = ("total", "complete_finite", "incomplete_or_nonfinite", "constant_complete",
              "allzero_complete", "varying_complete", "tied_maximum", "maximum_includes_endpoint",
              "maximum_at_endpoints_only")


def _empty(depth: int) -> dict:
    return {
        "sample_counts": dict.fromkeys(SAMPLE_COUNTS, 0),
        "ray_counts": dict.fromkeys(RAY_COUNTS, 0),
        "finite_counts_by_layer": np.zeros(depth, dtype=np.int64),
        "finite_sums_by_layer": np.zeros(depth, dtype=np.float64),
        "common_sums_by_layer": np.zeros(depth, dtype=np.float64),
        "tie_weighted_maximum_by_layer": np.zeros(depth, dtype=np.float64),
    }


def _peak_stats(values: np.ndarray, totals: dict) -> None:
    rays = totals["ray_counts"]
    if values.shape[1] == 0:
        return
    minima, maxima = values.min(axis=0), values.max(axis=0)
    varying = maxima > minima
    rays["constant_complete"] = int(np.count_nonzero(~varying))
    rays["allzero_complete"] = int(np.count_nonzero((minima == 0) & (maxima == 0)))
    rays["varying_complete"] = int(np.count_nonzero(varying))
    winners = values[:, varying] == maxima[varying]
    ties = winners.sum(axis=0)
    rays["tied_maximum"] = int(np.count_nonzero(ties > 1))
    rays["maximum_includes_endpoint"] = int(np.count_nonzero(winners[0] | winners[-1]))
    rays["maximum_at_endpoints_only"] = int(np.count_nonzero(~winners[1:-1].any(axis=0)))
    totals["tie_weighted_maximum_by_layer"] = (winners / ties).sum(axis=1, dtype=np.float64)


def _tile_stats(values: np.ndarray, materialized: np.ndarray) -> dict:
    depth, height, width = values.shape
    totals = _empty(depth)
    samples, rays = totals["sample_counts"], totals["ray_counts"]
    present = np.broadcast_to(materialized[:, None, None], values.shape)
    finite = np.isfinite(values) & present
    samples.update(total=values.size, materialized=int(materialized.sum()) * height * width,
                   finite_materialized=int(finite.sum()), zero_materialized=int(((values == 0) & finite).sum()))
    samples["implicit_fill"] = samples["total"] - samples["materialized"]
    samples["nonfinite_materialized"] = samples["materialized"] - samples["finite_materialized"]
    complete = finite.all(axis=0)
    rays.update(total=height * width, complete_finite=int(complete.sum()))
    rays["incomplete_or_nonfinite"] = rays["total"] - rays["complete_finite"]
    totals["finite_counts_by_layer"] = finite.sum(axis=(1, 2), dtype=np.int64)
    totals["finite_sums_by_layer"] = values.sum(axis=(1, 2), where=finite, dtype=np.float64)
    common = values[:, complete]
    totals["common_sums_by_layer"] = common.sum(axis=1, dtype=np.float64)
    _peak_stats(common, totals)
    return totals


def _add(total: dict, tile: dict) -> None:
    for key in ("sample_counts", "ray_counts"):
        for name, count in tile[key].items():
            total[key][name] += count
    for key in ("finite_counts_by_layer", "finite_sums_by_layer", "common_sums_by_layer",
                "tie_weighted_maximum_by_layer"):
        total[key] += tile[key]
        if not np.isfinite(total[key]).all():
            raise ValueError("Numeric overflow while accumulating intensity statistics")


def _means(sums: np.ndarray, counts: np.ndarray) -> list[float | None]:
    return [float(s / n) if n else None for s, n in zip(sums, counts)]


def _fill_value(source: LocalSlab) -> dict:
    value = source.array.fill_value
    if value is None:
        return {"value": None, "is_finite": None}
    finite = bool(np.isfinite(value))
    if finite:
        encoded = int(value) if source.dtype.kind in "iu" else float(value)
    else:
        encoded = "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return {"value": encoded, "is_finite": finite}


def _serialize(totals: dict) -> dict:
    rays = totals["ray_counts"]
    depth = len(totals["finite_counts_by_layer"])
    common_counts = np.full(depth, rays["complete_finite"], dtype=np.int64)
    peak = totals["tie_weighted_maximum_by_layer"]
    denominator = rays["varying_complete"]
    return {
        "sample_counts": totals["sample_counts"], "ray_counts": rays,
        "finite_counts_by_layer": totals["finite_counts_by_layer"].tolist(),
        "finite_mean_by_layer": _means(totals["finite_sums_by_layer"], totals["finite_counts_by_layer"]),
        "common_support_mean_by_layer": _means(totals["common_sums_by_layer"], common_counts),
        "tie_weighted_maximum_by_layer": peak.tolist(),
        "endpoint_maximum_mass_fraction": float(peak[[0, -1]].sum() / denominator) if denominator else None,
    }


def _check_accounting(totals: dict, depth: int, pixels: int) -> None:
    samples, rays = totals["sample_counts"], totals["ray_counts"]
    valid = (
        samples["total"] == depth * pixels == samples["materialized"] + samples["implicit_fill"]
        and samples["materialized"] == samples["finite_materialized"] + samples["nonfinite_materialized"]
        and rays["total"] == pixels == rays["complete_finite"] + rays["incomplete_or_nonfinite"]
        and rays["complete_finite"] == rays["constant_complete"] + rays["varying_complete"]
        and int(totals["finite_counts_by_layer"].sum()) == samples["finite_materialized"]
        and math.isclose(float(totals["tie_weighted_maximum_by_layer"].sum()),
                         rays["varying_complete"], rel_tol=1e-10, abs_tol=1e-8)
    )
    if not valid:
        raise ValueError("Internal sample/ray/maximum accounting failed")


def scan(source: LocalSlab, *, window: tuple[int, int, int, int] | None = None,
         max_read_bytes: int = 64 * 1024 * 1024,
         on_tile: Callable[[dict], None] | None = None) -> dict:
    """Inspect every intersecting chunk once; optionally stream each YX tile record.

    max_read_bytes limits a decoded chunk or input tile, not process RSS. NumPy
    intermediates and the decoder need additional memory. The input must remain
    unchanged during this read-only pass; chunk contents are not snapshotted.
    """
    started = time.perf_counter()
    bounds = source.window(window)
    largest_input = source.check_read_size(max_read_bytes)
    totals = _empty(source.shape[0])
    chunks = {"materialized": 0, "unmaterialized": 0}
    tile_count = 0
    for y0, y1, x0, x1 in source.tiles(bounds):
        present, stored, missing = source.materialized_layers(y0, x0)
        chunks["materialized"] += stored
        chunks["unmaterialized"] += missing
        values = np.asarray(source.array[:, y0:y1, x0:x1])
        if values.shape != (source.shape[0], y1 - y0, x1 - x0):
            raise ValueError("Array read returned an unexpected shape")
        tile = _tile_stats(values, present)
        _add(totals, tile)
        tile_count += 1
        if on_tile is not None:
            on_tile({"window_yx": [y0, y1, x0, x1], "chunk_keys": {
                "materialized": stored, "unmaterialized": missing}, **_serialize(tile)})
    source.verify_metadata()
    _check_accounting(totals, source.shape[0], (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]))
    return {
        "schema": "vesuvius_render_preflight_v1", "version": "0.1.0",
        "source": str(source.path), "shape_zyx": list(source.shape), "chunks_zyx": list(source.chunks),
        "zarr_format": source.format, "dtype": str(source.dtype), "axis_order": "ZYX",
        "resolved_fill_value": _fill_value(source),
        "array_metadata_sha256": source.metadata_sha256,
        "inspected_metadata_sha256": {str(path): digest for path, digest in source.metadata_pins.items()},
        "axis_metadata_sources": source.axis_metadata_sources,
        "declared_attributes_unverified": source.attributes,
        "window_yx": list(bounds), "spatial_tiles": tile_count, "chunk_keys": chunks,
        "max_decoded_input_bytes": largest_input, "elapsed_seconds": time.perf_counter() - started,
        **_serialize(totals),
        "interpretation": {
            "unmaterialized": "Absent keys are legal implicit fill; absence alone does not establish lost data.",
            "zeros": "Stored zeros remain observed values, without an anatomical classification.",
            "common_support": "Same rays at every depth: all chunks materialized and all samples finite.",
            "maximum": "Each nonconstant complete ray contributes one unit split equally among exact maxima.",
            "scope": "Descriptive CT intensity and storage QA; no surface, ink or letter validation.",
        },
    }
