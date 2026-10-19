"""Run an explicit inference command only after finite/variation checks pass.

Usage: python preflight_then_infer.py segment.zarr --report qc.json -- python -m your_inference ...
Sparse, finite fill is allowed unless --require-materialized is requested.
The checks do not establish anatomical or ink quality.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-materialized", action="store_true")
    if "--" not in argv or argv[-1] == "--":
        parser.error("Provide the inference command after --")
    separator = argv.index("--")
    args = parser.parse_args(argv[:separator])
    command = argv[separator + 1:]
    preflight = [sys.executable, "-m", "render_preflight", str(args.source), "--json", str(args.report),
                 "--require-finite", "--require-varying"]
    if args.require_materialized:
        preflight.append("--require-materialized")
    result = subprocess.run(preflight, check=False)
    if result.returncode:
        return result.returncode
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as error:
        print(f"Could not start inference command: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
