"""Pin inspected metadata and validate axes belonging to the selected array."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_document(path: Path, pins: dict[Path, str]) -> dict:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    if path in pins and pins[path] != digest:
        raise ValueError(f"Input metadata changed during inspection: {path}")
    pins[path] = digest
    return document


def read_attributes(directory: Path, pins: dict[Path, str]) -> dict:
    if (directory / ".zattrs").is_file():
        attributes = read_document(directory / ".zattrs", pins)
    elif (directory / "zarr.json").is_file():
        attributes = read_document(directory / "zarr.json", pins).get("attributes", {})
    else:
        attributes = {}
    if not isinstance(attributes, dict):
        raise ValueError(f"Zarr attributes must be an object: {directory}")
    return attributes


def _check_multiscales(attributes: dict, directory: Path, array_path: Path) -> list[str]:
    checked = []
    for namespace in (attributes, attributes.get("ome", {})):
        if not isinstance(namespace, dict):
            raise ValueError("OME metadata must be an object")
        scales = namespace.get("multiscales", [])
        if not isinstance(scales, list):
            raise ValueError("Multiscales metadata must be a list")
        for scale in scales:
            if not isinstance(scale, dict) or not isinstance(scale.get("datasets"), list):
                raise ValueError("Each multiscale must declare its datasets")
            datasets = scale["datasets"]
            if any(not isinstance(d, dict) or not isinstance(d.get("path"), str) for d in datasets):
                raise ValueError("Each multiscale dataset must declare a string path")
            if not any((directory / d["path"]).resolve() == array_path for d in datasets):
                continue
            axes = scale.get("axes")
            if not isinstance(axes, list):
                raise ValueError("Selected multiscale must declare its axes")
            names = [axis if isinstance(axis, str) else axis.get("name") if isinstance(axis, dict) else None
                     for axis in axes]
            if names != ["z", "y", "x"]:
                raise ValueError(f"Expected ZYX multiscale axes, got {names!r}")
            checked.append(str(directory))
    return checked


def inspect_axes(root: Path, array_path: Path, metadata: dict, pins: dict[Path, str]) -> tuple[dict, list[str]]:
    names = metadata.get("dimension_names")
    if names is not None and any(name is not None for name in names) and names != ["z", "y", "x"]:
        raise ValueError(f"Expected ZYX dimension names, got {names!r}")
    directories = {root, array_path}
    # Inspect the immediate containing image group even when an array path was
    # supplied directly; do not let direct paths bypass declared OME axes.
    directory = array_path.parent
    while directory == root or (directory / ".zgroup").is_file() or (directory / "zarr.json").is_file():
        directories.add(directory)
        if directory == root or directory == directory.parent:
            break
        directory = directory.parent
    attributes = {}
    checked = [str(array_path)] if names == ["z", "y", "x"] else []
    for directory in sorted(directories):
        declared = read_attributes(directory, pins)
        if directory == root:
            attributes = declared
        checked.extend(_check_multiscales(declared, directory, array_path))
    return attributes, checked
