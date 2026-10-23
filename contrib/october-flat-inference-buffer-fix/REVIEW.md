# Review of the skipped-patch buffer fix

The patch is sound for the tested flat-inference path. This is an automated code
review, not a record that the contributor has personally reviewed or used it.

The invariant is that every scheduled patch completes one contribution for each
output chunk it overlaps. A nonempty patch adds probability and weight before
completing its contribution. A raw-empty patch completes the contribution without
allocating or modifying either array. The shared `_finish_contribution` helper
flushes only after the expected count is reached, and only if a buffer exists.

Handling skipped metadata before a batch's nonempty predictions is safe: any
nonempty prediction in that batch still counts toward the expected total, so its
chunk cannot be flushed prematurely. Nonempty addition order and the raw-input
occupancy flag are preserved. Mask and TTA processing are unchanged.

The eight new regressions check mixed contributions in either order, multiple
and partial chunks, all-empty batches, batch sizes 1/2, and nonempty raw patches
whose normalized pixels are zero. The real-data module checks additionally
compare complete float32 blend sums and actual LZW TIFF bytes in both depth
directions. These checks support the memory-lifetime change without claiming
ink-model accuracy or GPU behavior.

## Current upstream overlap

Checked October 20, 2026 UTC (October 20 in Chicago).

- `origin/main` is now `672339382d4dc71576cacb452bd94bb3dc79fa8d`, seven unrelated
  commits beyond the tested base `23876bef6a10c928abf167e963f9755534cec2dd`.
  The inference, model, data, utility, and ink-test paths, target environment
  definition, contribution rules, and PR template are unchanged. The local
  candidate and its tested source bytes are preserved.
- Targeted GitHub PR searches for `"raw-empty"` and `"flush_remaining"` return
  no matches. A search for `ChunkAccumulator` returns open
  [#2003](https://github.com/ScrollPrize/villa/pull/2003), head
  `2fa7b3e41415561ab33b22bba02f65090b4e96cd`.
- Inspection of #2003's diff confirms it does not change `ChunkAccumulator`,
  `run_block_inference`, or skipped-contribution completion. Its source cache,
  RAM backing stores, and threaded loader address different costs. This is a
  bounded duplicate-work check, not proof that no other work exists.

## Validation and remaining evidence

The user approved the additional `einops`/`timm` scratch dependencies. Fourteen
packages were added, preserving all 32 original package versions. The combined
CPU run passes all eight new regressions and all 19 existing flat-inference
cases, including the spawned DataLoader worker. Candidate: 27 passed. Original:
20 passed / 7 failed, with only the seven expected new regressions failing.
The earlier worker failure was caused by the sandbox also denying Unix IPC;
the final profile permits it only under the task's temporary directory.

The harness imports the full source in the main and spawned processes, binds
normal dotted package attributes for pytest, and fixes the runtime's device
selection hook to CPU. No test assertions or inference functions are rewritten.
The sandbox probe confirms that IP networking, Unix sockets outside the task
directory, and writes outside approved paths remain blocked. Both profiles, the probe,
the test logs, and the extended environment freeze are bundled.

The additional trained-model run calls the actual `infer.main()` path on the
bounded PHerc0211 crop with the public `ink_9um` seed42 step-075000 checkpoint
named by our workflow. Its download matches the publisher's SHA-256; all 508
state tensors load without missing or unexpected keys. The embedded robust-MAD
normalization is used. CPU model parameters and inputs are float32.

Forward and reverse runs preserve model inputs, float32 probability/weight sums,
decoded pixels, and TIFF file bytes. In each direction the original retains two
buffers / 256 KiB before the final flush and the candidate retains none. Peak
buffer storage is unchanged on this small crop. This closes the test-network-only
gap for the bounded reproduction; it does not prove full-render trained-model
behavior, ink quality, legibility, or speed. The historical remote checkpoint
bytes have not been rehashed; repository/file identity matches `modal_app.py`.

[Villa's contribution rules](https://github.com/ScrollPrize/villa/blob/23876bef6a10c928abf167e963f9755534cec2dd/CONTRIBUTING.md)
require actual human use, a motivation, human-written commentary, personal code
review, and genuine before/after terminal/tool screenshots. October 22 browser
captures of the actual saved GitHub test logs are included as
[before](pytest-before-github.jpg) and [after](pytest-after-github.jpg).
Personal human code review remains unattested.

The contributor subsequently authorized drafting the motivation on their behalf.
[MOTIVATION_COMMENT.md](MOTIVATION_COMMENT.md) states the verified workflow and
results, with AI assistance disclosed. It does not attest to personal human review
or claim to satisfy the upstream human-written-commentary requirement.

## October assessment

This is a small, reproducible bug fix tied to an actual saved PHerc0211 workflow.
It has not been released, adopted, or reviewed by maintainers. The measured
absolute memory saving is too small to establish a strong standalone October
prize case. It can be supporting work in an October contribution bundle if it is
useful to the community and the upstream contribution requirements are met.

The [current progress-prize criteria](https://scrollprize.org/prizes#progress-prizes)
favor actual use, demonstrated improvements on real data, and fixes to tools
contributors use themselves; they request bundling multiple small contributions.
The October deadline is October 31, 2026, at 11:59 p.m. Pacific. No prize submission
or award is implied by this review.
