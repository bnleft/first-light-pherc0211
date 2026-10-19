"""Fetch only the objects named by a fixed cross-scroll plan (<48 MiB).

The resulting stores are deliberately partial copies. Inspect the named
windows only. A saved manifest can enforce identical responses on repetition.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

METADATA = (".zgroup", ".zattrs", "0/.zarray")


def chunk_keys(dataset: dict, metadata: dict) -> list[str]:
    shape, chunks = dataset["shape"], dataset["chunks"]
    separator = metadata.get("dimension_separator", ".")
    indices = set()
    for y0, y1, x0, x1 in dataset["windows_yx"]:
        if not (0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
            raise ValueError("Planned window lies outside its source array")
        indices.update(itertools.product(range((shape[0] + chunks[0] - 1) // chunks[0]),
                                         range(y0 // chunks[1], (y1 + chunks[1] - 1) // chunks[1]),
                                         range(x0 // chunks[2], (x1 + chunks[2] - 1) // chunks[2])))
    return ["0/" + separator.join(map(str, index)) for index in sorted(indices)]


def read_object(url: str, allow_absent: bool) -> tuple[int, bytes]:
    try:
        with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=20) as response:
            body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("Object exceeds the fixed 4 MiB limit")
        return 200, body
    except HTTPError as error:
        if allow_absent and error.code == 404:
            return 404, b""
        raise


def fetch(plan_path: Path, destination: Path, expected_path: Path | None) -> dict:
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    expected = json.loads(expected_path.read_text()) if expected_path else None
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    if expected and expected["plan_sha256"] != plan_sha:
        raise ValueError("Saved manifest describes a different plan")
    destination.mkdir(parents=True, exist_ok=False)
    records, total, requests = [], 0, 0
    for dataset in plan["datasets"]:
        if not dataset["id"].replace("_", "").isalnum():
            raise ValueError("Dataset id must be a simple folder name")
        store = destination / f"{dataset['id']}.zarr"
        store.mkdir()
        files, metadata = [], None
        keys = list(METADATA)
        for key in keys:
            status, body = read_object(f"{dataset['base_url']}/{key}", key not in METADATA)
            requests += 1
            total += len(body)
            if requests > 64 or total > 48 * 1024 * 1024:
                raise ValueError("Cross-scroll fetch exceeds fixed request/byte limits")
            entry = {"key": key, "status": status, "bytes": len(body),
                     "sha256": hashlib.sha256(body).hexdigest() if status == 200 else None}
            files.append(entry)
            if status == 200:
                target = store / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            if key == "0/.zarray":
                metadata = json.loads(body)
                if (metadata["shape"] != dataset["shape"] or metadata["chunks"] != dataset["chunks"]
                        or metadata["dtype"] != "|u1" or metadata["order"] != "C"
                        or metadata.get("filters") not in (None, [])):
                    raise ValueError("Source layout differs from the fixed uint8/C-order plan")
                keys.extend(chunk_keys(dataset, metadata))
        record = {**dataset, "array_metadata": metadata, "files": files}
        if expected and record != expected["datasets"][len(records)]:
            raise ValueError(f"Source responses changed: {dataset['id']}")
        records.append(record)
    manifest = {"schema": "render_preflight_cross_scroll_fixture_v1", "plan_sha256": plan_sha,
                "retrieved_utc": datetime.now(UTC).isoformat(), "deliberately_partial_copies": True,
                "requests": requests, "total_bytes": total, "datasets": records}
    (destination / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_fixture(root: Path) -> dict:
    manifest = json.loads((root / "fixture_manifest.json").read_text())
    for dataset in manifest["datasets"]:
        store = root / f"{dataset['id']}.zarr"
        for entry in dataset["files"]:
            path = store / entry["key"]
            if entry["status"] == 404:
                if path.exists():
                    raise ValueError(f"Expected absent chunk is now materialized: {path}")
                continue
            raw = path.read_bytes()
            if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Fixture bytes changed: {path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(fetch(args.plan, args.destination, args.verify_manifest), indent=2))
