# Render preflight

A CPU tool for checking surface-conditioned CT renders before ink inference.
It reads **local Zarr v2 or unsharded v3**, writes a JSON summary and optional
per-tile CSV, and needs neither a GPU nor model weights.

This packages the October PHerc. 0211 storage/depth investigations into a
reusable CLI. The new contribution is storage-aware, streaming QA with explicit
denominators and a reproducible real-data validation. The September First
Letters experiment and its award are prior work.

## The problem it addresses

A missing Zarr chunk and an explicitly stored all-zero chunk can return the
same voxel values. They have different provenance. Also, averaging just
positive pixels separately at each depth compares changing pixel populations.
Finally, `argmax` silently assigns every tied or flat profile to its first
maximum, creating artificial depth preferences.

Render preflight reports physical chunk-key coverage separately from stored
zeros, compares layer means on a common set of complete finite rays, and splits
each nonconstant ray's peak weight equally among its exact maxima. Constant
profiles retain their own denominator and contribute no peak weight. Stored
zeros remain in the mean; negative intensities are supported.

Absent keys are valid implicit fill in Zarr, including intentional sparse
zeros. Their absence alone **does not prove lost data**. Intensity maxima do
not identify a papyrus surface or ink. This tool diagnoses storage and render
intensity, without changing the slab, mesh, labels, masks, or inference.

## Use

From this directory, with [uv](https://docs.astral.sh/uv/):

```sh
uv run --locked render-preflight /path/to/segment.zarr --json qc.json --tiles qc.csv
uv run --locked render-preflight /path/to/segment.zarr --window 0 512 1024 1536 --json window.json
```

The source can be a Zarr group (array `0` by default) or an array directory.
Use `--array other/key` to select another array explicitly. The contract is
ZYX: depth, row, column. Declared incompatible axes are refused; absent axis
metadata requires the caller to know the order. Windows are half-open
`Y0 Y1 X0 X1` in the selected array's own coordinates; every depth is retained.

For a render expected to have every chunk stored, finite values and some depth variation:

```sh
uv run --locked render-preflight segment.zarr --json qc.json --require-materialized --require-varying --require-finite
```

Exit codes: `0` means the scan completed and any requested conditions held;
`1` means an explicitly requested storage/variation condition failed; `2`
means invalid input, a decode failure, or another operational error. A completed
scan is **not** an anatomical or ink-quality pass. Use `--require-materialized`
only when a dense store is actually expected. `--require-varying` requires at
least one complete finite ray with depth variation; it is only a basic signal
check. `--require-finite` also checks implicit fill when absent keys are used:
NaN fill must not silently pass a workflow just because stored samples are finite.
Requested conditions and their outcomes are saved in `operational_checks`.
Existing reports are never overwritten. CSV rows are streamed to a
temporary file and published only after the scan succeeds.

An [integration example](examples/preflight_then_infer.py) runs a caller-supplied
inference command only after finite/variation checks pass:

```sh
uv run --locked python examples/preflight_then_infer.py segment.zarr --report qc.json -- python -m your_inference ...
```

Replace the command after `--` with the inference command from your own
environment. A failed check prevents it starting. Sparse finite fill is allowed;
add `--require-materialized` before `--` only for an expected dense store.
The regression suite verifies this with a command that writes a marker file,
including failure cases where depth variation exists but one sample is NaN.

## Outputs and bounded reads

The JSON preserves:

- Expected/materialized chunk-key counts inside the requested window.
- Materialized finite, nonfinite and zero samples, with implicit fill separate.
- Complete, incomplete, constant, all-zero and varying ray counts.
- Finite sample counts and means at each layer, plus equal-population means
  using only rays finite and materialized at **all** depths.
- Tie-weighted maximum histograms and endpoint mass, with the varying-ray
  denominator explicit. Undefined means/fractions are `null`.
- Source metadata hash, declared attributes, array coordinates and layout.
  Declared physical units are copied without independently verifying them.
  OME-Zarr 0.4 and 0.5 axis declarations are checked for the selected array,
  including its containing image group when an array path is supplied directly.

The optional CSV retains each spatial chunk footprint, including empty ones,
and its common-support profile and maximum histogram. The scanner never loads
the whole slab or accumulates the tile records in memory. Each intersecting
chunk is read once. It refuses a decoded chunk or input footprint larger than
`--max-read-mib` (default 64). This bounds input blocks, **not process RSS**:
decoder and NumPy intermediates need additional memory. Oversized inputs are
refused rather than silently sampled.

The source must be stable during the pass. Changes to inspected array/group metadata are detected;
concurrent chunk updates are not snapshotted. Remote stores, ZIP stores,
sharding and rank-4/5 arrays are outside this first version's scope.

## Cross-scroll validation

The release validation covers **three scrolls, nine fixed windows, 294,912 rays
and 16,220,160 samples**. Every window matches direct physical-chunk decoding
and a full scalar ray/peak oracle. Six lossless crop/layout conversions preserve
the results, including compressed v2 depth chunks and v3/OME-Zarr 0.5 uint16
arrays. These conversions are layout tests, not six additional datasets.

The sources are PHerc. 0139 w035 (28-layer, uncompressed, slash keys), Paris4
w046–052 (109-layer, uncompressed, slash keys), and PHerc. 1667 w013
(109-layer, Blosc-compressed, dot keys). The
[fixed plan](examples/cross_scroll_plan.json) declares the windows and prior
exposure. These are convenience windows, not an unbiased survey or anatomical
ground truth. All planned windows are retained.

```sh
uv sync --locked --extra test
uv run --locked --extra test pytest -q
uv run --locked python examples/fetch_cross_scroll.py examples/cross_scroll_plan.json examples/data/cross_scroll --verify-manifest examples/cross_scroll_manifest.json
uv run --locked python examples/validate_cross_scroll.py examples/data/cross_scroll --work examples/data/cross_validation
```

Fetching uses **27 named GETs / 16,170,780 bytes**. The partial local copies
retain their original array shapes: inspect only the planned windows.
The [response manifest](examples/cross_scroll_manifest.json) pins each source
object and the [results](examples/cross_scroll_results.json) retain every window,
the scalar comparison, layout conversions and worker measurements.

For the tested full-depth source chunks, a layer-at-a-time baseline makes
**28× or 109× more chunk GETs and reads the same factor more encoded bytes**.
The counters measure local `LocalStore.get` calls and returned encoded buffers,
not HTTP requests or physical disk traffic. The comparison excludes metadata and is not a general runtime or ink-inference
speedup claim. Mean profiles match exactly. The scalar oracle also checks
**20,466 tied maxima** across these windows; constant profiles have a separate
denominator.

The current suite has **47 passing cases on Python 3.11.8 and 3.12.8**, using the
lockfile's NumPy 2.4.6, Zarr 3.1.6 and numcodecs 0.16.5. Linux/Windows/macOS CI
is defined in the repository workflow; hosted CI has not run before publication.
The initial prototype was also tested with Zarr 3.4.1 as described below.

## Initial single-scroll validation

The fixture consists of three fixed 256 × 256 windows, all 28 layers, from the
[public PHerc. 0139 w035 surface volume](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zattrs).
It is an existing community render, not independently established anatomical
truth. The windows were fixed before inspecting their values. Fetching uses
15 named GETs totaling **5,508,728 bytes**, with no cloud compute or listing.
The resulting local copy deliberately contains only the named windows; do not
treat the rest of that partial copy as an acquired slab.

```sh
uv run --extra test pytest -q
uv run python examples/fetch_public_control.py examples/data/public_control.zarr --verify-manifest examples/public_control_manifest.json
uv run python examples/validate_public_render.py examples/data/public_control.zarr --work examples/data/validation
uv run render-preflight examples/data/public_control.zarr --window 2048 2304 2048 2304 --json example.json --tiles example.csv --require-materialized --require-varying
```

Use new destination/report paths when repeating commands. The pinned
[manifest](examples/public_control_manifest.json) detects upstream byte changes;
[saved validation results](examples/validation_results.json) include all windows
and worker measurements. The original downloaded fixture is ignored by Git.

Initial prototype validation on October 18, 2026 (Chicago), with Python 3.12.8, NumPy 2.5.4,
Zarr 3.4.1 and pytest 9.1.1:

- **27 regression cases passed**, including independent scalar accounting,
  v2/v3 key layouts, compressed corruption, partial/depth chunks, nonfinite
  samples, signed data, tie handling, block limits and read-only inputs.
- **196,608 real rays / 5,505,024 samples** matched an independent reconstruction
  from the original uncompressed chunk bytes. Layer means matched the
  layer-at-a-time baseline exactly in each window.
- Removing one known stored chunk produced exactly 458,752 implicit-fill
  samples and 16,384 incomplete rays. Explicitly storing a zero window
  preserved 65,536 observed all-zero rays and reported zero absent chunks.
- Reversing real depth profiles reversed both the common-support means and
  tie-weighted peak histograms. Original fixture hashes remained unchanged.
- **336 → 12 chunk GETs; 147 → 5.25 MiB returned chunk payloads**, versus a
  layer-at-a-time baseline over the same windows. Full-depth chunks make
  repeated per-layer reads costly. This is **28× less chunk I/O on this layout**,
  not a general 28× speedup claim. Single-run local timings were approximately
  24 ms versus 15–16 ms per window; worker peak RSS was approximately 66 MiB.

## October contribution

The [PHerc. 0211 case study](docs/pherc0211_case_study.md) explains the historical
storage ambiguity motivating the tool; it is separate from the new public-source
validation. The [submission draft](docs/PROGRESS_PRIZE_DRAFT.md) and
[release notes](docs/RELEASE_NOTES.md) state the measured benefit and limits.
Community use, hosted CI, public release and prize submission remain external
milestones. There is no demonstrated surface-placement or ink gain from this tool.

License: MIT. This work extends the diagnostics in `modal_app.py` and the
October joint/per-ray depth investigations in this repository.
