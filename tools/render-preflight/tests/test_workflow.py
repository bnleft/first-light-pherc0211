"""Prove that operational failures actually prevent the next command starting."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from render_preflight import LocalSlab


@pytest.mark.parametrize("case,require_dense,expected", [
    ("valid", False, 0), ("zeros", False, 1), ("nonfinite", False, 1),
    ("implicit_nan", False, 1), ("sparse", False, 0), ("sparse", True, 1),
])
def test_workflow_starts_inference_only_after_requested_checks(tmp_path, case, require_dense, expected):
    source = tmp_path / "input.zarr"
    values = np.broadcast_to(np.arange(3, dtype=np.float32)[:, None, None], (3, 2, 4)).copy()
    if case == "zeros":
        values[:] = 0
    if case == "nonfinite":
        values[1, 0, 0] = np.nan
    zarr.create_array(str(source), data=values, chunks=(3, 2, 2), zarr_format=2,
                      fill_value=np.nan if case == "implicit_nan" else 0,
                      config={"write_empty_chunks": True})
    if case in ("implicit_nan", "sparse"):
        LocalSlab(source).chunk_path((0, 0, 1)).unlink()
    report, marker = tmp_path / "qc.json", tmp_path / "inference_started.txt"
    script = Path(__file__).resolve().parents[1] / "examples/preflight_then_infer.py"
    argv = [sys.executable, str(script), str(source), "--report", str(report)]
    if require_dense:
        argv.append("--require-materialized")
    argv.extend(["--", sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('started')", str(marker)])
    result = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == expected, result.stderr
    assert marker.exists() is (expected == 0)
    observed = json.loads(report.read_text())
    checks = observed["operational_checks"]
    assert checks["all_rendered_samples_finite"]["requested"]
    if case == "nonfinite":
        assert observed["ray_counts"]["varying_complete"] == 7
        assert not checks["all_rendered_samples_finite"]["passed"]
    if case == "implicit_nan":
        assert observed["sample_counts"]["nonfinite_materialized"] == 0
        assert observed["resolved_fill_value"] == {"value": "NaN", "is_finite": False}
        assert not checks["all_rendered_samples_finite"]["passed"]
