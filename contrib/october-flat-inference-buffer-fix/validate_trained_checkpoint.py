"""Compare actual flat-inference CLI runs on a real crop and pinned ink weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import tifffile
import torch

from run_full_module_tests import prepare_module

CHECKPOINT_SHA256 = "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab"
CHECKPOINT_REVISION = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store_digest(store, tile_shape):
    digest = hashlib.sha256()
    for y0 in range(0, store.shape[0], tile_shape[0]):
        for x0 in range(0, store.shape[1], tile_shape[1]):
            tile = np.asarray(store[y0:y0+tile_shape[0], x0:x0+tile_shape[1]])
            digest.update(str((tile.shape, str(tile.dtype))).encode())
            digest.update(tile.tobytes())
    return digest.hexdigest()


def run_variant(source, source_root, checkpoint, crop):
    prepare_module(SimpleNamespace(source=source, source_root=source_root))
    module = sys.modules["vesuvius.ink_detection.inference.infer"]
    records = []
    contract = {}
    original_load = module.load_flat_inference_state
    original_configure = module.configure_model
    original_run = module.run_block_inference
    original_write = module.write_output_tiff

    def checked_load(model, state):
        result = original_load(model, state)
        assert not result.missing_keys and not result.unexpected_keys
        contract["loaded_state_tensors"] = len(state)
        return result

    def configured(args):
        result = original_configure(args)
        contract.update(patch_size=result.patch_size, input_depth=result.input_depth,
                        preprocessing=result.preprocessing, amp_dtype=str(result.amp_dtype))
        contract["model_parameter_dtypes"] = sorted({str(p.dtype) for p in result.model.parameters()})
        assert contract["model_parameter_dtypes"] == ["torch.float32"]
        return result

    class Probe(module.ChunkAccumulator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.record = {"accumulator_buffer_bytes_peak": 0}
            records.append(self.record)

        def _buffers(self, key):
            result = super()._buffers(key)
            size = sum(p.nbytes+w.nbytes for p, w in self.buffers.values())
            self.record["accumulator_buffer_bytes_peak"] = max(
                size, self.record["accumulator_buffer_bytes_peak"])
            return result

        def flush_remaining(self):
            self.record["before_final_flush"] = {
                "chunks": len(self.buffers),
                "array_bytes": sum(p.nbytes+w.nbytes for p,w in self.buffers.values()),
            }
            return super().flush_remaining()

    def observed_run(**kwargs):
        record = records[-1]
        digest = hashlib.sha256()
        record.update(scheduled=len(kwargs["loader"].dataset), images_seen=0, model_calls=0)
        assert kwargs["device"].type == "cpu"

        def inputs(model, values):
            image = values[0]
            assert image.device.type == "cpu"
            assert image.dtype == torch.float32
            record["model_input_dtype"] = str(image.dtype)
            digest.update(str((tuple(image.shape), str(image.dtype))).encode())
            digest.update(image.detach().contiguous().numpy().tobytes())
            record["images_seen"] += len(image)
            record["model_calls"] += 1

        hook = kwargs["model"].register_forward_pre_hook(inputs)
        try:
            original_run(**kwargs)
        finally:
            hook.remove()
        record["model_input_stream_sha256"] = digest.hexdigest()
        record["raw_empty_skipped"] = record["scheduled"]-record["images_seen"]

    def observed_write(probability, weight, output, tile_shape):
        record = records[-1]
        record["probability_sum_sha256"] = store_digest(probability, tile_shape)
        record["weight_sum_sha256"] = store_digest(weight, tile_shape)
        original_write(probability, weight, output, tile_shape)
        image = tifffile.imread(output)
        assert image.shape == probability.shape and image.dtype == np.uint8
        record.update(output_name=output.name, output_shape=list(image.shape),
                      output_dtype=str(image.dtype), decoded_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                      tiff_file_sha256=sha(output), pixel_min=int(image.min()),
                      pixel_max=int(image.max()), distinct_uint8_values=int(np.unique(image).size))

    module.load_flat_inference_state = checked_load
    module.configure_model = configured
    module.ChunkAccumulator = Probe
    module.run_block_inference = observed_run
    module.write_output_tiff = observed_write
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="flat_trained_checkpoint_") as temporary:
        cli = [str(crop), str(checkpoint), str(Path(temporary)/"prediction.tif"),
               "--workers", "0", "--no-compile", "--batch-size", "4",
               "--overlap", "0.5", "--blend-mode", "hann", "--direction", "both"]
        exit_status = module.main(cli)
        assert exit_status == 0
    assert len(records) == 2
    return {"source_sha256": sha(source), "cli_exit_status": exit_status,
            "contract": contract, "directions": dict(zip(("forward", "reverse"), records)),
            "instrumented_wall_seconds": time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "before", "after", "checkpoint", "crop", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    assert args.checkpoint.stat().st_size == 138360039
    assert sha(args.checkpoint) == CHECKPOINT_SHA256
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    before = run_variant(args.before, args.source_root, args.checkpoint, args.crop)
    after = run_variant(args.after, args.source_root, args.checkpoint, args.crop)
    assert before["contract"] == after["contract"]
    for direction in ("forward", "reverse"):
        old, new = before["directions"][direction], after["directions"][direction]
        for key in ("scheduled", "images_seen", "model_calls", "model_input_stream_sha256",
                    "probability_sum_sha256", "weight_sum_sha256", "decoded_pixels_sha256", "tiff_file_sha256"):
            assert old[key] == new[key], (direction, key)
        assert old["before_final_flush"]["array_bytes"] > 0
        assert new["before_final_flush"]["array_bytes"] == 0
        print(json.dumps({"direction": direction, "inputs_sums_and_TIFF_identical": True,
                          "before_final_flush": old["before_final_flush"],
                          "after_final_flush": new["before_final_flush"]}), flush=True)
    report = {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu",
              "threads": 1, "checkpoint_repository": "scrollprize/ink_9um",
              "checkpoint_file": "hybrid_3d2d-seed42/step-075000.pth",
              "checkpoint_revision": CHECKPOINT_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
              "validation_runner_sha256": sha(Path(__file__)),
              "before": before, "after": after, "all_pair_comparisons_passed": True,
              "limitations": ["Bounded failure-focused real crop; not a full-render trained-model run.",
                              "No ink-quality, legibility, or throughput improvement is established.",
                              "Parent-package initializers bypassed; device selection fixed to CPU.",
                              "Only macOS arm64 CPU checked; no hosted CI or GPU claim."]}
    args.report.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
