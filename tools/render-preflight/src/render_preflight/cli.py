"""CPU-only inspection with JSON and streaming per-tile CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

from .scan import scan
from .storage import LocalSlab


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local Zarr group or rank-3 ZYX array")
    parser.add_argument("--array", default="0", help="Array key in a group (default: 0)")
    parser.add_argument("--window", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                        help="Inspect this half-open YX window, retaining every depth layer")
    parser.add_argument("--json", type=Path, help="New summary JSON file (default: stdout)")
    parser.add_argument("--tiles", type=Path, help="New per-tile CSV file, streamed as tiles are read")
    parser.add_argument("--max-read-mib", type=int, default=64,
                        help="Largest decoded chunk/input tile in MiB; not a process RSS limit")
    parser.add_argument("--require-materialized", action="store_true",
                        help="Exit 1 if any intersecting chunk key is absent; use only for stores expected to be dense")
    parser.add_argument("--require-varying", action="store_true",
                        help="Exit 1 unless at least one complete finite ray varies with depth; not a surface/ink test")
    parser.add_argument("--require-finite", action="store_true",
                        help="Exit 1 for nonfinite stored values or nonfinite/unknown implicit fill in the window")
    return parser


def _tile_writer(stream: TextIO) -> Callable[[dict], None]:
    writer = csv.writer(stream)
    writer.writerow(["y0", "y1", "x0", "x1", "materialized_chunks", "unmaterialized_chunks",
                     "complete_finite_rays", "constant_rays", "varying_rays", "tied_maxima_rays",
                     "endpoint_maximum_mass_fraction", "common_support_mean_by_layer", "maximum_mass_by_layer"])

    def write(tile: dict) -> None:
        rays = tile["ray_counts"]
        writer.writerow([*tile["window_yx"], *tile["chunk_keys"].values(), rays["complete_finite"],
                         rays["constant_complete"], rays["varying_complete"], rays["tied_maximum"],
                         tile["endpoint_maximum_mass_fraction"], json.dumps(tile["common_support_mean_by_layer"]),
                         json.dumps(tile["tie_weighted_maximum_by_layer"])])
    return write


def _pending_output(path: Path, stack: ExitStack) -> Path:
    temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix=".render-preflight-", dir=path.parent))
    return Path(temporary) / path.name


def _publish(outputs: list[tuple[Path, Path]]) -> None:
    published = []
    try:
        for pending, destination in outputs:
            os.link(pending, destination)
            published.append((pending, destination))
    except OSError:
        for pending, destination in published:
            if destination.exists() and destination.samefile(pending):
                destination.unlink()
        raise


def _operational_checks(report: dict, args: argparse.Namespace) -> bool:
    samples = report["sample_counts"]
    finite = samples["nonfinite_materialized"] == 0 and (
        samples["implicit_fill"] == 0 or report["resolved_fill_value"]["is_finite"] is True)
    checks = {
        "all_chunks_materialized": {"requested": args.require_materialized,
                                     "passed": report["chunk_keys"]["unmaterialized"] == 0},
        "some_complete_rays_vary": {"requested": args.require_varying,
                                    "passed": report["ray_counts"]["varying_complete"] > 0},
        "all_rendered_samples_finite": {"requested": args.require_finite, "passed": finite},
    }
    report["operational_checks"] = checks
    return all(check["passed"] for check in checks.values() if check["requested"])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = LocalSlab(args.source, args.array)
        for output in (args.json, args.tiles):
            if output and output.exists():
                raise ValueError(f"Output already exists: {output}")
            if output and (output.resolve().is_relative_to(source.root)
                           or output.resolve().is_relative_to(source.path)):
                raise ValueError("Write reports outside the input store")
        if args.json and args.tiles and args.json.resolve() == args.tiles.resolve():
            raise ValueError("JSON and CSV outputs must be different files")
        with ExitStack() as stack:
            write_tile = None
            pending_tiles = None
            if args.tiles:
                pending_tiles = _pending_output(args.tiles, stack)
                stream = stack.enter_context(pending_tiles.open("w", newline=""))
                write_tile = _tile_writer(stream)
            report = scan(source, window=tuple(args.window) if args.window else None,
                          max_read_bytes=args.max_read_mib * 1024 * 1024, on_tile=write_tile)
            checks_passed = _operational_checks(report, args)
            encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
            outputs = []
            if pending_tiles:
                stream.close()
                outputs.append((pending_tiles, args.tiles))
            if args.json:
                pending_json = _pending_output(args.json, stack)
                with pending_json.open("w") as stream:
                    stream.write(encoded)
                outputs.append((pending_json, args.json))
            _publish(outputs)
            if not args.json:
                sys.stdout.write(encoded)
        chunks, rays = report["chunk_keys"], report["ray_counts"]
        print(f"{report['spatial_tiles']} tiles; {chunks['unmaterialized']} unmaterialized chunk keys; "
              f"{rays['complete_finite']}/{rays['total']} complete finite rays; "
              f"{rays['varying_complete']} varying rays; "
              f"{report['sample_counts']['nonfinite_materialized']} nonfinite stored samples", file=sys.stderr)
        return 0 if checks_passed else 1
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f"render-preflight: {error}", file=sys.stderr)
        return 2
