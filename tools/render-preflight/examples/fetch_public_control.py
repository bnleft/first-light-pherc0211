"""Fetch three fixed 256x256 windows of a public PHerc. 0139 render (<8 MiB).

Only 15 named GETs: 3 metadata objects and 12 full-depth chunks. No listing,
credentials, cloud compute, or model is used. This is a deliberately partial
copy: inspect ONLY WINDOWS, not the rest of the original-shaped array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

BASE = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/"
        "20260317000000-w035_2026031718/surface-volumes/"
        "9.362um-1.2m-113keV-volume-20250728140407.zarr")
WINDOWS = [(2048, 2304, 2048, 2304), (2560, 2816, 2560, 2816), (3072, 3328, 3072, 3328)]
CHUNK_KEYS = [f"0/0/{y}/{x}" for start in (16, 20, 24) for y in (start, start + 1) for x in (start, start + 1)]
KEYS = [".zgroup", ".zattrs", "0/.zarray", *CHUNK_KEYS]


def fetch(destination: Path, expected: Path | None = None) -> dict:
    pins = json.loads(expected.read_text()) if expected else None
    if pins and (pins["base_url"] != BASE or [entry["key"] for entry in pins["files"]] != KEYS):
        raise ValueError("Manifest does not describe these exact public objects")
    destination.mkdir(parents=True, exist_ok=False)
    files, total = [], 0
    for index, key in enumerate(KEYS):
        request = Request(f"{BASE}/{key}", headers={"Accept-Encoding": "identity"})
        with urlopen(request, timeout=20) as response:
            content = response.read(1024 * 1024 + 1)
        total += len(content)
        if len(content) > 1024 * 1024 or total > 8 * 1024 * 1024:
            raise ValueError("Public fixture exceeds fixed download limits")
        entry = {"key": key, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if pins and entry != pins["files"][index]:
            raise ValueError(f"Public object changed: {key}")
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files.append(entry)
    metadata = json.loads((destination / "0/.zarray").read_text())
    if (metadata["shape"] != [28, 5820, 5240] or metadata["chunks"] != [28, 128, 128]
            or metadata["dtype"] != "|u1" or metadata["compressor"] is not None):
        raise ValueError("Expected the original uncompressed uint8 public render layout")
    manifest = {"schema": "render_preflight_public_fixture_v1", "base_url": BASE,
                "retrieved_utc": datetime.now(UTC).isoformat(), "windows_yx": WINDOWS,
                "deliberately_partial_copy": True, "total_bytes": total, "files": files}
    (destination / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--verify-manifest", type=Path, help="Refuse changed bytes against a saved public fixture manifest")
    args = parser.parse_args()
    print(json.dumps(fetch(args.destination, args.verify_manifest), indent=2))
