"""Replay upstream's exact chunk accumulator on our real CT patch schedule.

Only NumPy/Zarr are installed locally. Selected definitions are compiled unchanged
from the named source file; this does not import or exercise the model stack.
No model prediction, end-to-end inference speed, or total RSS claim is made.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import zarr

DEFINITIONS = {
    "Block", "ChunkKey", "_sliding_positions_1d", "iter_blocks",
    "downsample_mask_any", "choose_pyramid_array",
    "compute_nonempty_mask_from_lowres_array", "build_lowres_block_mask",
    "center_crop_layer_indices", "select_layer_indices", "FlatPatchReader",
    "iter_overlapping_chunks", "compute_chunk_contribution_counts",
    "ChunkAccumulator",
}


def source_definitions(path: Path, root: Path):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in DEFINITIONS]
    assert {node.name for node in nodes} == DEFINITIONS
    namespace = dict(globals())
    namespace.update({
        "__name__": "__main__", "LOGGER": logging.getLogger("replay"),
        "DEFAULT_OCCUPANCY_SCAN_LEVEL": "3",
        "open_volume": lambda path, level: zarr.open_array(root / str(level), mode="r"),
    })
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class Group:
    def __init__(self, path):
        self.path = path

    def array_keys(self):
        return [str(level) for level in range(6) if (self.path / str(level) / ".zarray").exists()]

    def __getitem__(self, key):
        return zarr.open_array(self.path / str(key), mode="r")


class DigestStore:
    """Retain only each flushed chunk's hash, never an output-sized image."""
    def __init__(self):
        self.hashes = {}

    def __setitem__(self, index, values):
        key = tuple((item.start, item.stop) for item in index)
        assert key not in self.hashes, f"chunk flushed twice: {key}"
        self.hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()


def replay(source: Path, root: Path, schedule=None):
    ns = source_definitions(source, root)
    array = zarr.open_array(root / "0", mode="r")
    height, width = array.shape[1:]
    occupancy, scale, level = ns["build_lowres_block_mask"](Group(root), height=height, width=width, user_mask=None)
    blocks = ns["iter_blocks"]((height, width), 128, 64, occupancy, scale)
    layer_indices = ns["select_layer_indices"](array.shape[0], layer_start=None, layer_end=None, output_depth=17, direction="forward")
    reader = ns["FlatPatchReader"](input_path=str(root), resolution="0", depth_axis_first=True, height=height, width=width, layer_indices=layer_indices, output_depth=17, preprocessing="divide_255")
    stores = [DigestStore(), DigestStore()]
    class PeakProbe(ns["ChunkAccumulator"]):
        peak_bytes = 0
        peak_chunks = 0

        def _buffers(self, key):
            result = super()._buffers(key)
            self.peak_bytes = max(self.peak_bytes, sum(p.nbytes+w.nbytes for p,w in self.buffers.values()))
            self.peak_chunks = max(self.peak_chunks, len(self.buffers))
            return result

    accumulator = PeakProbe(shape=(height, width), chunk_shape=(128, 128), prob_sum_store=stores[0], weight_sum_store=stores[1], contribution_counts=ns["compute_chunk_contribution_counts"](blocks, chunk_shape=(128, 128)))
    weights = np.ones((128, 128), dtype=np.float32)
    trace, nonempty_flags = [], []
    start = time.perf_counter()
    for i, block in enumerate(blocks):
        raw = reader.read(block.y0, block.x0, 128, 128)
        nonempty = bool(raw.any())
        nonempty_flags.append(nonempty)
        if nonempty:
            # CT-derived probability values test numerical identity of blending.
            # They are explicitly not model predictions or an ink-quality metric.
            probability = raw.astype(np.float32).mean(axis=-1) / np.float32(255)
            accumulator.add_tile(y0=block.y0, x0=block.x0, tile=probability[:block.valid_h, :block.valid_w], tile_weights=weights[:block.valid_h, :block.valid_w])
        elif hasattr(accumulator, "skip_tile"):
            accumulator.skip_tile(y0=block.y0, x0=block.x0, valid_h=block.valid_h, valid_w=block.valid_w)
        trace.append({"block": i, "buffers": len(accumulator.buffers), "buffer_bytes": sum(p.nbytes+w.nbytes for p,w in accumulator.buffers.values())})
    retained = trace[-1]
    retained_keys = sorted((key.row, key.col) for key in accumulator.buffers)
    accumulator.flush_remaining()
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape": list(array.shape), "layers": layer_indices.tolist(),
        "occupancy_level": level, "occupancy_scale": list(scale),
        "occupancy_coverage": float(occupancy.mean()),
        "scheduled": len(blocks), "nonempty": sum(nonempty_flags),
        "raw_empty_skipped": len(blocks)-sum(nonempty_flags),
        "native_buffer_bytes_peak": accumulator.peak_bytes,
        "buffers_peak": accumulator.peak_chunks,
        "peak_measurement": "Sum of array.nbytes retained in accumulator.buffers, measured after every _buffers call; not total process RSS or temporary references after a flush.",
        "before_final_flush": retained,
        "retained_chunk_keys": retained_keys,
        "schedule": [[b.y0, b.x0, b.valid_h, b.valid_w, int(nonempty)] for b, nonempty in zip(blocks, nonempty_flags)],
        "output_chunk_digests": [{repr(key): value for key,value in sorted(store.hashes.items())} for store in stores],
        "wall_seconds": time.perf_counter()-start,
        "trace": trace,
        "limitations": ["Accumulator-only replay using exact upstream definitions.", "CT-derived proxy probabilities; no model forward or ink gain.", "Native NumPy buffer bytes, not total process RSS."]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = replay(args.source, args.data)
    args.report.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key not in {"trace", "output_chunk_digests", "schedule", "retained_chunk_keys"}}, indent=2))


if __name__ == "__main__":
    main()
