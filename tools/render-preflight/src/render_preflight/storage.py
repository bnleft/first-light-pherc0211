"""Read local, unsharded Zarr arrays and inspect their physical chunk keys."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import zarr

from .metadata import inspect_axes, read_document


class LocalSlab:
    """A read-only ZYX array; absent keys denote implicit fill, not observations."""

    def __init__(self, path: str | Path, array_key: str | None = "0") -> None:
        self.metadata_pins: dict[Path, str] = {}
        self.root = Path(path).resolve(strict=True)
        if (self.root / ".zarray").exists() or self._is_v3_array(self.root):
            self.path = self.root
        else:
            if array_key is None or Path(array_key).is_absolute() or ".." in Path(array_key).parts:
                raise ValueError("Specify a relative array key, e.g. --array 0")
            self.path = (self.root / array_key).resolve(strict=True)
        self.meta_path = self.path / (".zarray" if (self.path / ".zarray").exists() else "zarr.json")
        metadata = read_document(self.meta_path, self.metadata_pins)
        self.metadata_sha256 = self.metadata_pins[self.meta_path]
        self.format = metadata.get("zarr_format")
        self.array = zarr.open_array(str(self.path), mode="r")
        self.separator, self.prefix = self._key_layout(metadata)
        self.shape = tuple(self.array.shape)
        self.chunks = tuple(self.array.chunks)
        self.dtype = self.array.dtype
        if len(self.shape) != 3 or min(self.shape) < 1 or self.dtype.kind not in "iuf":
            raise ValueError("Expected a nonempty real-valued rank-3 ZYX array")
        self.attributes, self.axis_metadata_sources = inspect_axes(self.root, self.path, metadata, self.metadata_pins)

    def _is_v3_array(self, path: Path) -> bool:
        meta = path / "zarr.json"
        return meta.is_file() and read_document(meta, self.metadata_pins).get("node_type") == "array"

    @staticmethod
    def _key_layout(metadata: dict) -> tuple[str, str]:
        if metadata.get("zarr_format") == 2:
            separator = metadata.get("dimension_separator", ".")
            prefix = ""
        elif metadata.get("zarr_format") == 3:
            if metadata.get("node_type") != "array":
                raise ValueError("Selected Zarr node is not an array")
            if metadata.get("chunk_grid", {}).get("name") != "regular":
                raise ValueError("Only regular, unsharded chunk grids are supported")
            if any(codec.get("name") == "sharding_indexed" for codec in metadata.get("codecs", [])):
                raise ValueError("Sharded arrays are not supported: shard presence is not chunk presence")
            encoding = metadata.get("chunk_key_encoding", {})
            if encoding.get("name") not in ("default", "v2"):
                raise ValueError("Unsupported Zarr chunk key encoding")
            default_separator = "/" if encoding["name"] == "default" else "."
            separator = encoding.get("configuration", {}).get("separator", default_separator)
            prefix = "c" if encoding["name"] == "default" else ""
        else:
            raise ValueError("Expected Zarr format 2 or 3")
        if separator not in ("/", "."):
            raise ValueError("Unsupported chunk key separator")
        return separator, prefix

    def chunk_path(self, indices: tuple[int, int, int]) -> Path:
        components = ([self.prefix] if self.prefix else []) + [str(i) for i in indices]
        return self.path / self.separator.join(components)

    def window(self, bounds: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
        if bounds is None:
            return 0, self.shape[1], 0, self.shape[2]
        y0, y1, x0, x1 = bounds
        if not (0 <= y0 < y1 <= self.shape[1] and 0 <= x0 < x1 <= self.shape[2]):
            raise ValueError("Window must be nonempty and inside the YX array bounds")
        return bounds

    def tiles(self, bounds: tuple[int, int, int, int]) -> Iterator[tuple[int, int, int, int]]:
        y0, y1, x0, x1 = bounds
        cy, cx = self.chunks[1:]
        for iy, ix in itertools.product(range(y0 // cy, (y1 + cy - 1) // cy),
                                        range(x0 // cx, (x1 + cx - 1) // cx)):
            yield max(y0, iy * cy), min(y1, (iy + 1) * cy), max(x0, ix * cx), min(x1, (ix + 1) * cx)

    def materialized_layers(self, y0: int, x0: int) -> tuple[np.ndarray, int, int]:
        present = np.zeros(self.shape[0], dtype=bool)
        stored = missing = 0
        for iz in range((self.shape[0] + self.chunks[0] - 1) // self.chunks[0]):
            path = self.chunk_path((iz, y0 // self.chunks[1], x0 // self.chunks[2]))
            exists = path.is_file()
            stored += int(exists)
            missing += int(not exists)
            present[iz * self.chunks[0]:(iz + 1) * self.chunks[0]] = exists
        return present, stored, missing

    def check_read_size(self, max_read_bytes: int) -> int:
        if max_read_bytes < 1:
            raise ValueError("max_read_bytes must be positive")
        footprint = self.shape[0] * min(self.shape[1], self.chunks[1]) * min(self.shape[2], self.chunks[2])
        largest = max(footprint, math.prod(self.chunks)) * self.dtype.itemsize
        if largest > max_read_bytes:
            raise ValueError(f"Input block requires {largest} bytes, exceeding limit {max_read_bytes}")
        return largest

    def verify_metadata(self) -> None:
        for path, expected in self.metadata_pins.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Input metadata changed during inspection: {path}; scan a stable store")
