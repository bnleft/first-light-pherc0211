"""Recreate the complete bounded real CT crop; do not overwrite a directory."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.fixture, allow_pickle=False) as fixture:
        metadata = json.loads(str(fixture["metadata_json"].item()))
        arrays = {"0": fixture["level0"], "3": fixture["level3"]}
    for level, values in arrays.items():
        assert values.dtype == np.uint8 and values.ndim == 3
        assert list(values.shape) == metadata[f"level{level}_shape"]
        assert hashlib.sha256(values.tobytes()).hexdigest() == metadata[f"level{level}_decoded_sha256"]
    root = zarr.open_group(args.output, mode="w-", zarr_format=2)
    for level, values in arrays.items():
        array = root.create_array(
            level, shape=values.shape, chunks=(28, 128, 128), dtype="u1"
        )
        array[:] = values
        assert hashlib.sha256(np.asarray(array[:]).tobytes()).hexdigest() == metadata[f"level{level}_decoded_sha256"]
    print(json.dumps({"output": str(args.output), "decoded_arrays_match_pinned_real_crop": True, "source_bbox_yx_half_open": metadata["source_bbox_yx_half_open"]}))


if __name__ == "__main__":
    main()
