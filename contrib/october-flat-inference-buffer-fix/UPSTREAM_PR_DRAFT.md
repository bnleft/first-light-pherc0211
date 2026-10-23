# Local draft: release flat-inference buffers after skipped patches

Not posted. The motivation and technical evidence below were prepared with AI
assistance. Personal human review is unverified; genuine before/after terminal
screenshots are pending.

**In one sentence:** Flat ink inference can release completed blend buffers even
when some overlapping patches have empty raw input.

**One real example:** Running the actual `infer.main()` path on a bounded
`28 x 384 x 384` crop of our saved PHerc0211 render with the published `ink_9um`
seed42 step-075000 checkpoint, patch 128, overlap 0.5, Hann blending, batch size 4,
and both depth directions schedules 17 patches and skips three raw-empty patches.

**Before:** Two probability/weight buffer pairs (256 KiB) remain in the
accumulator until the final flush because skipped patches never complete their
scheduled contributions.

**After this PR:** No buffers remain before the final flush. Skipped patches
complete their contributions without allocating buffers, running the model, or
adding blend weights. Nonempty predictions are added in the same order.

**Proof:** [Trained-checkpoint real-data comparison](TRAINED_CHECKPOINT_CPU.json),
[baseline regressions](focused-tests-before.txt), and
[candidate regressions](focused-tests-after.txt). Both trained-model directions
preserve model inputs, float32 probability/weight sums, final pixels, and LZW TIFF
file bytes. All eight new regressions pass; seven fail on the original code.
The [complete-render test-network comparison](DATA_LOOP_SUPPORTED.json) additionally
releases 44 retained buffer pairs / 5.5 MiB on 2,118 patches, 52 raw-empty.
Genuine terminal screenshots are pending.

**Why / where this is useful:** Flat ink-inference users with raw-empty patches
overlapping occupied output chunks can release that chunk memory as soon as its
scheduled work finishes. The observed saving is modest; it is not a measurement
of total process memory, inference speed, or ink quality.

- [ ] I personally verified that the example and proof above were produced by this PR on the stated data.

## Motivation

This came from checking the flat-inference path used by our PHerc0211 render
pipeline. We want completed chunks to release their blend buffers even when
neighboring patches are empty. The real-data comparison above reproduces the
retention and verifies unchanged output. The saving is modest, but it addresses
unnecessary memory use in this workflow.

AI assistance was used for this description and the tested implementation.

## Details

<details>
<summary>Reproduction, validation, and limitations</summary>

- Base: `23876bef6a10c928abf167e963f9755534cec2dd`.
- Candidate: `237201a4163680884d72488df1fd7509c0287ef0`.
- Change: 29 source lines in `inference/infer.py`, plus eight regression cases.
- [Exact setup and commands](SETUP.md): Python 3.14.6 / PyTorch 2.12.1, macOS arm64
  CPU. Unrelated parent-package initializers are bypassed; the complete inference
  module is imported without extracting definitions.
- The trained-checkpoint comparison uses the embedded robust-MAD normalization,
  loads all 508 state tensors without missing/unexpected keys, and runs float32
  CPU parameters/inputs. [Checkpoint provenance](CHECKPOINT_PROVENANCE.json)
  pins the public weights; they are not bundled. Historical remote checkpoint
  bytes have not been rehashed.
- Trained-model execution covers only the bounded crop. Full-render comparisons
  use a deterministic CPU test network. No ink-quality or legibility gain is
  claimed. Full-project CI, GPU execution, and other platforms are unverified.
- Combined CPU tests: **27 passed** on the candidate, versus **20 passed /
  7 failed** on the original. All 19 existing cases pass on both versions,
  including the spawned DataLoader worker. Only the seven expected new
  regressions fail on the original. See the
  [original](all-flat-tests-before-cpu.txt) and
  [candidate](all-flat-tests-after-cpu.txt) logs.
- The runner fixes only device selection to CPU and bypasses unrelated parent
  package initializers in the main and spawned processes. Test assertions and
  inference implementations are unchanged. Unix IPC is allowed only within the
  task directory; probes confirm that IP networking and writes outside approved
  paths remain blocked.
- The bundled [real CT crop](DATA_MANIFEST.json) reproduces the same retention
  with 17 scheduled patches, three raw-empty patches, and two retained buffers.
- Three accumulator-only replay pairs give identical float32 sum bytes. Peak
  arrays held in the buffer dictionary fall from 8.375 to 4.875 MiB on the full
  input. This is not total RSS. [Timing distributions](BENCHMARK.json) support no
  inference-speed claim.
- The closest open optimization, [#2003](https://github.com/ScrollPrize/villa/pull/2003),
  changes source caching, backing stores, loader threads, and compile handling;
  its current diff leaves this contribution-counting logic unchanged.

The full saved render is private pipeline output; the bounded real-data fixture,
source snapshots, patch, hashes, and reproduction commands are included locally.
Public evidence links and screenshot attachments must be checked before posting.

</details>
