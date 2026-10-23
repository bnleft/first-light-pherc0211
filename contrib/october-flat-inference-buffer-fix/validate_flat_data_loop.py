"""Compare the real flat data loop before/after on real CT with a CPU test model.

--mode module imports the complete inference module and requires its supported
dependencies. --mode isolated compiles named definitions unchanged and is an
additional legacy-runtime check, never a substitute for the module-mode run.
Module mode writes and decodes an LZW TIFF. No trained ink checkpoint is used.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
import time
import types
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import zarr


def register_parents(source_root):
    for name in (
        "vesuvius", "vesuvius.data", "vesuvius.utils", "vesuvius.models",
        "vesuvius.models.utilities", "vesuvius.ink_detection",
        "vesuvius.ink_detection.models", "vesuvius.ink_detection.inference",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(source_root.joinpath(*name.split(".")))]
        sys.modules[name] = package


def load_module(source, source_root, mode):
    register_parents(source_root)
    name = "vesuvius.ink_detection.inference.infer"
    if mode == "module":
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    volume_io = importlib.import_module("vesuvius.ink_detection.volume_io")
    names = {
        "Block", "ChunkKey", "_sliding_positions_1d", "iter_blocks",
        "downsample_mask_any", "choose_pyramid_array",
        "compute_nonempty_mask_from_lowres_array", "build_lowres_block_mask",
        "center_crop_layer_indices", "select_layer_indices",
        "FlatPatchReader", "FlatBlockDataset", "normalize_flat_patch",
        "compute_importance_map_2d", "logits_to_probabilities",
        "iter_overlapping_chunks", "compute_chunk_contribution_counts",
        "ChunkAccumulator", "run_block_inference", "iter_probability_tiles",
        "open_temp_zarr_array",
    }
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == names
    module = types.ModuleType(name)
    module.__dict__.update(globals())
    module.__name__ = name
    module.__file__ = str(source)
    module.LOGGER = logging.getLogger("flat_data_loop")
    module.DEFAULT_OCCUPANCY_SCAN_LEVEL = "3"
    module.open_volume = volume_io.open_volume
    module.ZARR_V3 = volume_io.ZARR_V3
    sys.modules[name] = module
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    exec(compile(code, str(source), "exec"), module.__dict__)
    return module


def run_variant(source, source_root, data, mode, batch_size, direction):
    module = load_module(source, source_root, mode)
    volume_io = importlib.import_module("vesuvius.ink_detection.volume_io")
    root = volume_io.open_volume_root(data)
    array = volume_io.select_volume_level(root, "0", source=str(data))
    depth, height, width = array.shape
    occupancy, scale, level = module.build_lowres_block_mask(
        root, height=height, width=width, user_mask=None
    )
    blocks = module.iter_blocks((height, width), 128, 64, occupancy, scale)
    layers = module.select_layer_indices(
        depth, layer_start=None, layer_end=None, output_depth=17, direction=direction
    )
    reader = module.FlatPatchReader(
        input_path=data, resolution="0", depth_axis_first=True,
        height=height, width=width, layer_indices=layers,
        output_depth=17, preprocessing="divide_255",
    )
    dataset = module.FlatBlockDataset(
        reader=reader, blocks=blocks, patch_size=128, preprocessing="divide_255"
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    class CPUContractModel(nn.Module):
        """Deterministic depth-sensitive logits, expressly not an ink model."""
        def __init__(self):
            super().__init__()
            self.input_hash = hashlib.sha256()
            self.images_seen = 0
            self.calls = 0

        def forward(self, image):
            self.input_hash.update(str((tuple(image.shape), str(image.dtype))).encode())
            self.input_hash.update(image.detach().contiguous().numpy().tobytes())
            self.images_seen += len(image)
            self.calls += 1
            weights = torch.arange(1, image.shape[2]+1, dtype=image.dtype).view(1,1,-1,1,1)
            weighted_mean = (image*weights).sum(dim=2)/weights.sum()
            return 6*(weighted_mean-0.4)

    class BufferProbe(module.ChunkAccumulator):
        peak_bytes = 0
        def _buffers(self, key):
            result = super()._buffers(key)
            self.peak_bytes = max(self.peak_bytes, sum(p.nbytes+w.nbytes for p,w in self.buffers.values()))
            return result

    start = time.perf_counter()
    model = CPUContractModel().eval()
    with tempfile.TemporaryDirectory(prefix="flat_loop_arrays_") as temporary:
        stores = [
            module.open_temp_zarr_array(Path(temporary)/name, shape=(height,width), chunks=(128,128))
            for name in ("probability.zarr", "weight.zarr")
        ]
        accumulator = BufferProbe(
            shape=(height,width), chunk_shape=(128,128),
            prob_sum_store=stores[0], weight_sum_store=stores[1],
            contribution_counts=module.compute_chunk_contribution_counts(blocks,chunk_shape=(128,128)),
        )
        module.run_block_inference(
            loader=loader, model=model, accumulator=accumulator,
            weight_map=module.compute_importance_map_2d(patch_size=(128,128),mode="hann").numpy(),
            mask=None, device=torch.device("cpu"), amp_dtype=None,
            tta_axes=(), tta_batch_size=None,
        )
        before_flush = {
            "chunks": len(accumulator.buffers),
            "array_bytes": sum(p.nbytes+w.nbytes for p,w in accumulator.buffers.values()),
        }
        accumulator.flush_remaining()
        blend_hashes = []
        for store in stores:
            digest = hashlib.sha256()
            for y0 in range(0, height, 128):
                for x0 in range(0, width, 128):
                    tile = np.asarray(store[y0:y0+128, x0:x0+128])
                    digest.update(str((tuple(tile.shape), str(tile.dtype))).encode())
                    digest.update(tile.tobytes())
            blend_hashes.append(digest.hexdigest())
        pixels = hashlib.sha256()
        pixel_count = 0
        for tile in module.iter_probability_tiles(*stores,(128,128)):
            pixels.update(str((tuple(tile.shape),str(tile.dtype))).encode())
            pixels.update(tile.tobytes())
            pixel_count += tile.size
        assert pixel_count == height*width
        tiff_verified = False
        tiff_file_sha256 = None
        if mode == "module":
            import tifffile
            output = Path(temporary)/"prediction.tif"
            module.write_output_tiff(*stores,output,(128,128))
            image = tifffile.imread(output)
            assert image.shape == (height,width) and image.dtype == np.uint8
            decoded = hashlib.sha256()
            for y0 in range(0,height,128):
                for x0 in range(0,width,128):
                    tile = image[y0:y0+128,x0:x0+128]
                    decoded.update(str((tuple(tile.shape),str(tile.dtype))).encode())
                    decoded.update(tile.tobytes())
            assert decoded.hexdigest() == pixels.hexdigest()
            tiff_verified = True
            tiff_file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mode": mode, "direction": direction, "batch_size": batch_size,
        "shape_zyx": list(array.shape), "layers": layers.tolist(),
        "scheduled": len(blocks), "images_seen": model.images_seen,
        "raw_empty_skipped": len(blocks)-model.images_seen,
        "model_forward_calls": model.calls,
        "model_input_stream_sha256": model.input_hash.hexdigest(),
        "probability_sum_float32_tile_stream_sha256": blend_hashes[0],
        "weight_sum_float32_tile_stream_sha256": blend_hashes[1],
        "final_uint8_tile_stream_sha256": pixels.hexdigest(),
        "final_pixel_count": pixel_count,
        "tiff_written_and_decoded_pixels_verified": tiff_verified,
        "tiff_file_sha256": tiff_file_sha256,
        "accumulator_buffer_bytes_peak": accumulator.peak_bytes,
        "before_final_flush": before_flush,
        "wall_seconds_including_instrumentation": time.perf_counter()-start,
        "occupancy_level": level,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("module","isolated"), default="module")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--full-input", type=Path, required=True)
    parser.add_argument("--crop-input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pairs = []
    for label, data, batch_size, direction in (
        ("full_forward_batch4",args.full_input,4,"forward"),
        ("full_reverse_batch4",args.full_input,4,"reverse"),
        ("crop_forward_batch1",args.crop_input,1,"forward"),
    ):
        before = run_variant(args.before,args.source_root,data,args.mode,batch_size,direction)
        after = run_variant(args.after,args.source_root,data,args.mode,batch_size,direction)
        assert before["model_input_stream_sha256"] == after["model_input_stream_sha256"]
        assert before["probability_sum_float32_tile_stream_sha256"] == after["probability_sum_float32_tile_stream_sha256"]
        assert before["weight_sum_float32_tile_stream_sha256"] == after["weight_sum_float32_tile_stream_sha256"]
        assert before["final_uint8_tile_stream_sha256"] == after["final_uint8_tile_stream_sha256"]
        assert before["model_forward_calls"] == after["model_forward_calls"]
        assert before["before_final_flush"]["array_bytes"] > 0
        assert after["before_final_flush"]["array_bytes"] == 0
        if args.mode == "module":
            assert before["tiff_file_sha256"] == after["tiff_file_sha256"]
        pairs.append({"case":label,"before":before,"after":after,"input_blend_sums_and_final_pixels_identical":True})
        print(json.dumps({"case":label,"mode":args.mode,"inputs_blend_sums_and_pixels_identical":True,"before_final_flush":before["before_final_flush"],"after_final_flush":after["before_final_flush"]}),flush=True)
    report = {
        "mode":args.mode, "python":sys.version.split()[0],
        "validation_runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch":torch.__version__,"numpy":np.__version__,"zarr":zarr.__version__,
        "num_workers":0,"cpu_threads":1,"cases":pairs,
        "limitations":[
            "Deterministic CPU test network, not a trained ink checkpoint or ink-quality result.",
            "TIFF writing and decoded pixels are checked in module mode only; isolated mode stops at the actual writer's uint8 input tile stream.",
            "Isolated mode compiles selected source definitions unchanged; supported full-module import is a separate gate.",
            "Parent package initializers are bypassed in either mode; no full-project/hosted-CI claim.",
            "Single runs check correctness; instrumented timings do not establish an inference speedup.",
        ],
    }
    args.report.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
