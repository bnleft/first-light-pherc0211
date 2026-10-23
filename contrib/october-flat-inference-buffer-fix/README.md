# Release flat inference buffers after raw-empty patches

This candidate fixes a reproduced memory-retention problem in the flat ink
inference workflow used by our PHerc0211 work.

**Current publication candidate:** `72845f6e18abdf345fc80f5722253c1ca25d7339`,
based on upstream `e2254ebe0168a232e5ba7b94997c749b4e6a4544`. The
[October 22 validation](CURRENT_UPSTREAM_VALIDATION.md) passes 33 selected CPU
cases, including the six new upstream ensemble cases, and repeats the actual
trained-checkpoint comparison on the real crop. The evidence below records the
original October 20 candidate; current source snapshots, logs, and patch use
`current` in their filenames.

The inference scheduler counts every scheduled patch when deciding when an
output chunk can be flushed. The inference loop skips raw-empty patches without
completing their contributions. A chunk overlapping both empty and nonempty
patches consequently remains in RAM until the final flush.

The patch completes skipped contributions without creating a buffer, running the
model, or adding blend weight. Nonempty model inputs and the order of probability
additions remain unchanged.

## Reproduced result

Input: the complete saved PHerc0211 render from our October 11 run, shape
`28 x 3720 x 8192`, center layers 6–22, 128-pixel patches, stride 64, level-3
occupancy, and 128-pixel output chunks. The downloaded level-0 metadata SHA256
matches our October 12 diagnostic record:
`f477832091320414ca2ee8c33e351e4a3a27efb9b17395b658604cb057fab08d`.

- The schedule has exactly 2,118 patches, matching our saved inference log.
- 52 patches have empty raw input in the selected depth window.
- Before: 44 output buffers / 5.5 MiB remain until the final flush.
- After: no output buffers remain before the final flush.
- Peak arrays retained in the accumulator's buffer dictionary fall from 8.375 MiB
  to 4.875 MiB (41.8%), instrumented on every buffer allocation/access.
- Probability-sum and weight-sum bytes match for all 618 flushed chunks.

Three sequential before/after pairs give the same memory counts and byte
identity. Timing distributions are recorded in [BENCHMARK.json](BENCHMARK.json);
no inference speedup is claimed. The full render is private saved pipeline
output. The bundled 543 KB crop reproduces the retention with 17 scheduled
patches (3 raw-empty): 2 buffers remain before the final flush on the original
code and zero with the patch. This crop has the same peak buffer count in both
variants; the 41.8% peak reduction belongs to the full input.

These are **accumulator replay measurements**, using the actual CT, actual
upstream scheduling/reader/accumulator definitions, and CT-derived proxy
probabilities. They are not a full model run, total process-memory measurement,
inference speedup, or ink-quality result. The absolute saving in this example is
modest. The patch does not establish the size of the saving on other scrolls.

## Candidate and validation

The isolated villa worktree is:

`/private/tmp/vesuvius-render-preflight-20261018/flat-inference-memory`

Changed source:
`vesuvius/src/vesuvius/ink_detection/inference/infer.py`

New regression tests:
`vesuvius/tests/ink_detection/test_flat_inference_empty_chunks.py`

Local candidate commit: `237201a4163680884d72488df1fd7509c0287ef0`, on villa
`23876bef6a10c928abf167e963f9755534cec2dd`. The
[patch](skipped-chunks.patch), before/after source snapshots, new tests, and
upstream MIT license are included. The candidate changes two files.

Eight focused regressions pass against the complete inference module on Python
3.14.6 / PyTorch 2.12.1; seven fail and one passes on the original source.
The tests cover mixed empty/nonempty contributions in either order, all-empty
chunks, batching, partial chunks, and raw-nonempty patches whose normalized
pixels are zero. The changed files pass Ruff's E4/E7/E9/F checks and parse on
Python 3.14. See [focused-tests-before.txt](focused-tests-before.txt) and
[focused-tests-after.txt](focused-tests-after.txt).

An additional real-data check runs the actual reader, normalization, DataLoader,
inference loop, Zarr blending, and uint8 pixel conversion with a deterministic
CPU test network. The full input is checked in both layer directions with batch
size 4; the bounded crop is checked with batch size 1. Model input bytes and
final pixel bytes match before/after in all three cases. The float32 probability
and weight sums also match exactly. Each full-input case covers 30,474,240 final
pixels and releases the same 44 retained buffers. All three pairs produce
byte-identical LZW TIFF files, and their decoded pixels match the writer input.
See [DATA_LOOP_SUPPORTED.json](DATA_LOOP_SUPPORTED.json).

This check imports the complete inference module without extracting definitions.
Unrelated parent-package initializers are bypassed; this is not full-project CI
or a full CLI/checkpoint-loading run. The network is a deterministic CPU test
network, not a trained ink checkpoint. Only macOS arm64 CPU execution has been
verified here. [SETUP.md](SETUP.md) records the approved and executed isolated
setup, exact validation commands, and installed package versions. Historical
legacy-runtime checks are superseded by this supported-runtime evidence.

A separate [trained-checkpoint check](TRAINED_CHECKPOINT_CPU.json) calls the
actual `infer.main()` argument, checkpoint-loading, model-construction,
inference, and TIFF-writing path on the bounded real crop. It uses the published
`scrollprize/ink_9um` seed42 step-075000 checkpoint named by `modal_app.py`, pinned
to the publisher's SHA-256. All 508 state tensors load without missing or
unexpected keys, and the embedded config selects robust-MAD normalization.
Both depth directions preserve model inputs, float32 blend sums, decoded pixels,
and TIFF file bytes. Two retained buffers / 256 KiB become zero in each direction.
The crop's peak buffer storage is unchanged. CPU parameters and inputs are
float32. [CHECKPOINT_PROVENANCE.json](CHECKPOINT_PROVENANCE.json) records the
source; the 138 MB weights remain in scratch and are not bundled.

This trained-model run covers only the failure-focused crop, not the complete
render. The complete-render comparisons above use a test network. Neither check
establishes an ink-quality, legibility, or inference-speed improvement.

The additional model dependencies were approved and installed, preserving the
original 32 package versions. The combined CPU run passes all eight new
regressions and all 19 existing `test_flat_inference.py` cases, including the
spawned worker. The candidate totals **27 passed**; the original totals
**20 passed / 7 failed**, with only the seven expected new regressions failing.
The sandbox permits Unix sockets only inside the task's temporary directory;
probes verify that IP connections, sockets elsewhere, and writes outside approved paths
remain blocked. [SETUP.md](SETUP.md#additional-existing-suite-check) records the
commands and harness configuration; both raw logs and the sandbox probe are included.

No branch has been published and no upstream message or PR has been sent.
The contributor requested an AI-assisted motivation comment, now prepared in
[MOTIVATION_COMMENT.md](MOTIVATION_COMMENT.md) and incorporated into the
[PR draft](UPSTREAM_PR_DRAFT.md). This records the documented workflow and does
not attest to personal human review. Actual
terminal/tool screenshots required by upstream have not been prepared: the
computer-use tool rejected Terminal access with "Computer Use is not allowed
to use the app 'com.apple.Terminal' for safety reasons." The actual test logs
and reproduction commands are included. The candidate is technically validated
for the tested path; community adoption and
October prize eligibility have not been established. The
[code review and upstream overlap check](REVIEW.md) and
[local upstream PR draft](UPSTREAM_PR_DRAFT.md) record the current assessment and
remaining requirements. Neither document is a posted review or a human attestation.

## Standalone real-data reproduction

Use an existing Python environment with NumPy 2 and Zarr 3. The following steps
materialize the complete bounded real crop into a new directory and run the
exact selected definitions from the bundled source snapshots. No Torch or
model checkpoint is required for this accumulator replay.

```sh
python materialize_real_crop.py pherc0211-real-crop.npz /tmp/pherc0211-buffer-crop.zarr
python replay_accumulator.py infer-before.py /tmp/pherc0211-buffer-crop.zarr /tmp/flat-before.json
python replay_accumulator.py infer-after.py /tmp/pherc0211-buffer-crop.zarr /tmp/flat-after.json
```

Compare `output_chunk_digests`: both dictionaries must match. Inspect
`before_final_flush`: the original retains 262,144 bytes and the patched
version retains zero. [VALIDATION.json](VALIDATION.json) states exactly which
checks have run and which remain pending. [SHA256SUMS](SHA256SUMS) pins the
package files; [DATA_MANIFEST.json](DATA_MANIFEST.json) pins all 774 files of the
original downloaded render and both decoded crop arrays.

## Data provenance

The crop is derived from our October 11 PHerc0211 render, produced by our
published spiral-fitting/render workflow from Vesuvius Challenge/ScrollPrize's
PHerc0211 CT scan `20250821151803` at 9.362 micrometres. The source volume is
[available in the public open-data bucket](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/.zattrs).
The package records the original Modal path and crop coordinates. It is a
failure-focused development crop, with no independent holdout or anatomical
accuracy claim. The upstream code license applies to the code snapshots and
patch; it does not replace the source dataset's terms.
