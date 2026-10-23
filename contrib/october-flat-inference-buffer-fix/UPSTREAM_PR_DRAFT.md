**In one sentence:** Flat ink inference can release completed blend buffers when overlapping patches have empty raw input.

**One real example:** A `28 x 384 x 384` crop from our PHerc0211 render, using the published `ink_9um` seed42 step-075000 checkpoint, schedules 17 patches and skips three with empty raw input.

**Before:** Two probability/weight buffer pairs (256 KiB) remain until the final flush in each depth direction because skipped patches never complete their scheduled contributions.

**After this PR:** No buffers remain before the final flush on this crop. Empty patches complete their bookkeeping without allocating arrays or adding blend weights.

**Proof:** The [trained-checkpoint comparison](https://github.com/bnleft/first-light-pherc0211/blob/ab91ab3b76c62c9012722a77b4982417ff2bffa3/contrib/october-flat-inference-buffer-fix/TRAINED_CHECKPOINT_CURRENT.json) produces byte-identical TIFF files before/after in both depth directions. [Current-main validation](https://github.com/bnleft/first-light-pherc0211/blob/ab91ab3b76c62c9012722a77b4982417ff2bffa3/contrib/october-flat-inference-buffer-fix/CURRENT_UPSTREAM_VALIDATION.md) passes 33 selected CPU cases; the baseline fails only seven expected new regressions.

**Why / where this is useful:** Chunks touching empty patches can release their blend arrays as soon as their scheduled work finishes. The observed saving is modest; peak buffer storage on this crop is unchanged.

- [ ] I personally verified that the example and proof above were produced by this PR on the stated data.

## Details

**Motivation:** This came from checking the flat-inference path used by our PHerc0211 render pipeline. We want completed chunks to release their arrays even when neighboring patches are empty. This PR and its validation were prepared with AI assistance.

<details>
<summary>Validation and reproduction</summary>

- Tested base: `e2254ebe0168a232e5ba7b94997c749b4e6a4544`. Candidate: `72845f6e18abdf345fc80f5722253c1ca25d7339`.
- Candidate: 33 passed. Baseline: 26 passed / seven failed. All 25 existing cases in `test_flat_inference.py` and `test_flat_inference_ensemble.py` pass on both. The patch adds eight regression cases covering mixed empty/nonempty ordering, partial chunks, and batching.
- The real-data comparison calls the actual `infer.main()` checkpoint-to-TIFF path with patch 128, overlap 0.5, Hann blending, batch size four, and both depth directions. All 508 checkpoint tensors load. Model input and float32 blend-array bytes match before/after; nonempty predictions are accumulated in the same order.
- [Exact commands and limitations](https://github.com/bnleft/first-light-pherc0211/blob/ab91ab3b76c62c9012722a77b4982417ff2bffa3/contrib/october-flat-inference-buffer-fix/CURRENT_UPSTREAM_VALIDATION.md), [real crop and provenance](https://github.com/bnleft/first-light-pherc0211/blob/ab91ab3b76c62c9012722a77b4982417ff2bffa3/contrib/october-flat-inference-buffer-fix/DATA_MANIFEST.json), and [checkpoint pin](https://github.com/bnleft/first-light-pherc0211/blob/ab91ab3b76c62c9012722a77b4982417ff2bffa3/contrib/october-flat-inference-buffer-fix/CHECKPOINT_PROVENANCE.json) are public.
- Python 3.14.6 / PyTorch 2.12.1 on macOS arm64 CPU. Unrelated parent-package initializers are bypassed and device selection is fixed to CPU. Complete inference modules and unchanged upstream test assertions are used.
- Full-project CI and GPU execution are unverified. Trained-model execution covers the bounded crop. No total-process memory, speed, or ink-quality improvement is claimed.

</details>

<details>
<summary>Before/after screenshots of the saved pytest output</summary>

These browser captures show the actual saved test logs on GitHub.

**Before:** Seven expected new regressions fail on current main.

![Baseline buffer regressions](https://raw.githubusercontent.com/bnleft/first-light-pherc0211/ab91ab3b76c62c9012722a77b4982417ff2bffa3/contrib/october-flat-inference-buffer-fix/pytest-before-github.jpg)

**After:** All 33 selected CPU cases pass with the patch.

![Passing buffer and upstream regressions](https://raw.githubusercontent.com/bnleft/first-light-pherc0211/ab91ab3b76c62c9012722a77b4982417ff2bffa3/contrib/october-flat-inference-buffer-fix/pytest-after-github.jpg)

</details>
