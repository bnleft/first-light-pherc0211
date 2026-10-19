"""Validate real-render accounting, injected storage faults, and chunk I/O.

No network access. Run fetch_public_control.py first. The baseline measures
layer-at-a-time Zarr reads; it is not a claim about every existing QA tool.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from fetch_public_control import WINDOWS
from zarr.storage import LocalStore

from render_preflight import LocalSlab, scan


@contextlib.contextmanager
def count_reads():
    original = LocalStore.get
    totals = {"chunk_gets": 0, "chunk_bytes_read": 0}

    async def get(store, key, prototype=None, byte_range=None):
        buffer = await original(store, key, prototype=prototype, byte_range=byte_range)
        parts = key.replace(".", "/").split("/")
        if parts[0] == "c":
            parts = parts[1:]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            totals["chunk_gets"] += 1
            if buffer is not None:
                totals["chunk_bytes_read"] += len(buffer)
        return buffer
    LocalStore.get = get
    try:
        yield totals
    finally:
        LocalStore.get = original


def worker(source: Path, window: tuple[int, int, int, int], strategy: str) -> dict:
    slab = LocalSlab(source)
    started = time.perf_counter()
    with count_reads() as reads:
        if strategy == "preflight":
            result = scan(slab, window=window)
            profile = result["common_support_mean_by_layer"]
        else:
            y0, y1, x0, x1 = window
            common = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            for ya, yb, xa, xb in slab.tiles(window):
                present, _, _ = slab.materialized_layers(ya, xa)
                common[ya - y0:yb - y0, xa - x0:xb - x0] = present.all()
            profile = []
            for z in range(slab.shape[0]):
                layer = np.asarray(slab.array[z, y0:y1, x0:x1])
                profile.append(float(layer[common].mean(dtype=np.float64)) if common.any() else None)
            result = None
    try:
        import resource
    except ImportError:
        peak_rss = None
    else:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak_rss *= 1024
    return {"strategy": strategy, **reads, "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss,
            "profile": profile, "report": result}


def measurement(source: Path, window: tuple[int, int, int, int], strategy: str) -> dict:
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(source), "--worker", strategy,
                                "--window", *(str(value) for value in window)], capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def raw_values(source: Path, window: tuple[int, int, int, int]) -> np.ndarray:
    """Reconstruct the four uncompressed chunks independently of Zarr's decoder."""
    y0, y1, x0, x1 = window
    values = np.empty((28, y1 - y0, x1 - x0), dtype=np.uint8)
    for y in range(y0, y1, 128):
        for x in range(x0, x1, 128):
            raw = (source / f"0/0/{y // 128}/{x // 128}").read_bytes()
            chunk = np.frombuffer(raw, dtype=np.uint8).reshape(28, 128, 128)
            values[:, y - y0:y - y0 + 128, x - x0:x - x0 + 128] = chunk
    return values


def verify_fixture(source: Path) -> dict:
    manifest = json.loads((source / "fixture_manifest.json").read_text())
    for entry in manifest["files"]:
        raw = (source / entry["key"]).read_bytes()
        assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
    return manifest


def validate(source: Path, work: Path) -> dict:
    manifest = verify_fixture(source)
    work.mkdir(parents=True, exist_ok=False)
    results = []
    for index, window in enumerate(WINDOWS):
        # Alternate ordering; these single-run local times are illustrative, not a speed benchmark.
        strategies = ["layerwise", "preflight"] if index % 2 == 0 else ["preflight", "layerwise"]
        measured = {strategy: measurement(source, window, strategy) for strategy in strategies}
        baseline, current = measured["layerwise"], measured["preflight"]
        values = raw_values(source, window)
        assert np.array_equal(values, LocalSlab(source).array[:, window[0]:window[1], window[2]:window[3]])
        assert baseline["profile"] == current["profile"] == values.mean(axis=(1, 2)).tolist()
        assert baseline["chunk_gets"] == 112 and current["chunk_gets"] == 4
        assert baseline["chunk_bytes_read"] == 28 * current["chunk_bytes_read"]
        report = current["report"]
        assert report["chunk_keys"]["unmaterialized"] == 0
        assert report["sample_counts"]["zero_materialized"] == int((values == 0).sum())
        assert report["ray_counts"]["allzero_complete"] == int((values == 0).all(axis=0).sum())
        results.append({"window_yx": window, "baseline": {k: v for k, v in baseline.items() if k not in ("report", "profile")},
                        "preflight": {k: v for k, v in current.items() if k not in ("report", "profile")},
                        "ray_counts": report["ray_counts"], "endpoint_maximum_mass_fraction": report["endpoint_maximum_mass_fraction"]})
    window = WINDOWS[0]
    missing = work / "one_absent_chunk.zarr"
    shutil.copytree(source, missing)
    (missing / "0/0/16/16").unlink()
    missing_report = scan(LocalSlab(missing), window=window)
    assert missing_report["chunk_keys"]["unmaterialized"] == 1
    assert missing_report["sample_counts"]["implicit_fill"] == 28 * 128 * 128
    assert missing_report["ray_counts"]["complete_finite"] == 3 * 128 * 128
    zeros = work / "explicit_stored_zeros.zarr"
    shutil.copytree(source, zeros)
    # Write explicit uncompressed payloads: ordinary Zarr writes may elide
    # fill-valued chunks, which would test implicit fill a second time.
    for y in (16, 17):
        for x in (16, 17):
            (zeros / f"0/0/{y}/{x}").write_bytes(bytes(28 * 128 * 128))
    zero_report = scan(LocalSlab(zeros), window=window)
    assert zero_report["chunk_keys"]["unmaterialized"] == 0
    assert zero_report["ray_counts"]["allzero_complete"] == 256 * 256
    assert zero_report["ray_counts"]["varying_complete"] == 0
    reversed_path = work / "reversed_depth.zarr"
    shutil.copytree(source, reversed_path)
    array = zarr.open_array(str(reversed_path / "0"), mode="r+")
    values = raw_values(source, window)
    array[:, window[0]:window[1], window[2]:window[3]] = values[::-1]
    original_report = scan(LocalSlab(source), window=window)
    reverse_report = scan(LocalSlab(reversed_path), window=window)
    assert reverse_report["common_support_mean_by_layer"] == original_report["common_support_mean_by_layer"][::-1]
    np.testing.assert_allclose(reverse_report["tie_weighted_maximum_by_layer"], original_report["tie_weighted_maximum_by_layer"][::-1], atol=1e-8)
    verify_fixture(source)
    return {"schema": "render_preflight_real_validation_v1", "public_fixture": manifest,
            "windows": results, "controlled_checks_passed": ["one_absent_chunk_distinguished_from_observed_zero",
            "explicit_stored_zero_window_retains_all_denominators", "real_profile_depth_reversal_is_equivariant",
            "raw_byte_reconstruction_matches_zarr", "input_fixture_hashes_unchanged"],
            "io_result": "28x fewer chunk GETs and returned chunk bytes versus layer-at-a-time reads on this full-depth-chunk layout",
            "limits": "Three fixed windows of one published render; not independent anatomical labels, ink validation, or general speedup evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--worker", choices=("layerwise", "preflight"))
    parser.add_argument("--window", type=int, nargs=4)
    args = parser.parse_args()
    if args.worker:
        output = worker(args.source, tuple(args.window), args.worker)
    else:
        if args.work is None:
            parser.error("--work is required")
        output = validate(args.source, args.work)
    print(json.dumps(output, indent=2, allow_nan=False))
