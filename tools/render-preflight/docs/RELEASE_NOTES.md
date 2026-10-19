# Render preflight 0.1.0

Release candidate prepared for review. Public publication is pending.

The CPU-only CLI reads local rank-3 ZYX Zarr v2 and unsharded v3 arrays. It
separates physical chunk-key coverage from stored zeros, computes depth profiles
on common finite support, preserves exact maximum ties, and streams per-tile CSV.
It refuses unsupported layouts and oversized decoded input blocks. JSON and CSV
reports are published without overwriting existing files; failed publication
rolls back reports created by that invocation.

OME-Zarr 0.4 and 0.5 axis metadata are checked for the selected array, including
its containing image group. Inspected metadata hashes are retained and checked
again after the scan. Concurrent chunk mutation is not snapshotted.

Optional materialization, variation and finite-value checks can gate a subsequent
inference command. Nonfinite implicit fill is checked separately from observed
samples. These are operational checks and do not certify geometry or ink quality.

Validation: 47 regression/workflow cases pass on macOS with Python 3.11.8 and
3.12.8 under the lockfile. Three public scroll renders provide nine windows,
294,912 rays and 16,220,160 samples matching direct decoding and a full scalar
oracle. Six lossless layout conversions preserve those metrics. The full-depth
source layouts require 28× or 109× less chunk I/O than a layer-at-a-time baseline.
This is not a universal runtime speedup or an ink/surface accuracy result.

Remote stores, sharding, rank-4/5 arrays, geometry masks and anatomical validation
are outside this version's scope. Community adoption and hosted cross-platform
CI remain unverified until the release is public.

MIT license. Implemented and documented with AI coding assistance; Bryant Le
owns the project and decides publication and submission.
