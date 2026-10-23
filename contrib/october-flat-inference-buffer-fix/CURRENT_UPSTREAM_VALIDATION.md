# Current-main validation, October 22, 2026

The fix passes 33 flat-inference CPU cases on current upstream main. The
trained-checkpoint comparison still releases 256 KiB of retained blend arrays
in each depth direction, with byte-identical TIFF output on the real crop.

| Item | Tested revision or result |
| --- | --- |
| Upstream base | `e2254ebe0168a232e5ba7b94997c749b4e6a4544` |
| Candidate | `72845f6e18abdf345fc80f5722253c1ca25d7339` |
| Diff | Two files, 161 insertions and four deletions |
| Baseline tests | 26 passed, seven failed; exit status 1 |
| Candidate tests | 33 passed; exit status 0 |
| Runtime | Python 3.14.6, PyTorch 2.12.1, macOS arm64 CPU |

The selected tests are the unchanged `test_flat_inference.py` (19 cases), the
new upstream `test_flat_inference_ensemble.py` (six cases), and this patch's
`test_flat_inference_empty_chunks.py` (eight cases). All 25 existing upstream
cases pass on both versions. Only seven expected new regressions fail on the
baseline. See the raw [baseline](current-tests-before.txt) and
[candidate](current-tests-after.txt) output.

The new branch applies the original patch to current main without altering its
source or regression logic. It preserves upstream's ensemble checkpoint support
and blocking host-to-MPS copy fix. The earlier October 20 evidence files remain
historical records for the original candidate and base.

## Real-data comparison

[TRAINED_CHECKPOINT_CURRENT.json](TRAINED_CHECKPOINT_CURRENT.json) and its
[raw output](trained-checkpoint-current.stdout.txt) record the actual
`infer.main()` checkpoint-to-TIFF path. The pinned published `ink_9um` checkpoint
loads all 508 state tensors. The input is the bundled PHerc0211 crop, with the
same normalization and settings as [SETUP.md](SETUP.md).

Both depth directions schedule 17 patches, process 14, and skip three with empty
raw input. Before the final flush, two buffer pairs totaling 262,144 bytes remain
on the baseline; none remain on the candidate. Model inputs, float32 probability
and weight sums, decoded pixels, and TIFF file bytes match in each paired run.
Peak accumulator array storage is unchanged at 524,288 bytes on this crop.

This is a modest retention fix. It establishes no total-process memory,
inference-speed, ink-quality, or legibility improvement. Trained-model execution
covers the bounded crop, not the full render. Full-project CI and GPU execution
have not been verified.

## Reproduction

Recreate the frozen CPU environment from
[supported-requirements-with-model-deps.txt](supported-requirements-with-model-deps.txt).
Use [materialize_real_crop.py](materialize_real_crop.py) to materialize the
bundled NPZ and download the pinned checkpoint using [SETUP.md](SETUP.md).
The parent-initializer bypass and CPU selection hook remain unchanged.

```sh
export FLAT_FIX_SCRATCH=/private/tmp/vesuvius-render-preflight-20261018
export FLAT_FIX_BUNDLE=/Users/bryantle/Documents/bnleft/first-light-pherc0211/contrib/october-flat-inference-buffer-fix
export FLAT_FIX_TREE="$FLAT_FIX_SCRATCH/flat-inference-memory"
export FLAT_FIX_SOURCE="$FLAT_FIX_TREE/vesuvius/src"
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$FLAT_FIX_SOURCE"
export TMPDIR="$FLAT_FIX_SCRATCH/flat-worker-ipc"
export TORCH_HOME="$FLAT_FIX_SCRATCH/flat-supported-torch-home"
export TORCHINDUCTOR_CACHE_DIR="$FLAT_FIX_SCRATCH/flat-supported-torch-cache"

# Baseline: expected exit status 1, with seven failures and 26 passes.
/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer-ipc.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/run_full_module_tests.py" --source-root "$FLAT_FIX_SOURCE" --source "$FLAT_FIX_BUNDLE/infer-current-before.py" --tests "$FLAT_FIX_TREE/vesuvius/tests/ink_detection/test_flat_inference.py" "$FLAT_FIX_TREE/vesuvius/tests/ink_detection/test_flat_inference_ensemble.py" "$FLAT_FIX_TREE/vesuvius/tests/ink_detection/test_flat_inference_empty_chunks.py"

# Candidate: expected exit status 0, with 33 passes.
/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer-ipc.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/run_full_module_tests.py" --source-root "$FLAT_FIX_SOURCE" --source "$FLAT_FIX_BUNDLE/infer-current-after.py" --tests "$FLAT_FIX_TREE/vesuvius/tests/ink_detection/test_flat_inference.py" "$FLAT_FIX_TREE/vesuvius/tests/ink_detection/test_flat_inference_ensemble.py" "$FLAT_FIX_TREE/vesuvius/tests/ink_detection/test_flat_inference_empty_chunks.py"

/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer-ipc.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/validate_trained_checkpoint.py" --source-root "$FLAT_FIX_SOURCE" --before "$FLAT_FIX_BUNDLE/infer-current-before.py" --after "$FLAT_FIX_BUNDLE/infer-current-after.py" --checkpoint "$FLAT_FIX_SCRATCH/ink_9um-seed42-step-075000.pth" --crop "$FLAT_FIX_SCRATCH/flat-buffer-evidence/portable-fixture.zarr" --report "$FLAT_FIX_SCRATCH/flat-buffer-evidence/TRAINED_CHECKPOINT_CURRENT.json"
```

The temporary environment from October 20 had been cleared. On October 22 it was
recreated in the same scratch directory with the same 46 pinned packages; no new
dependencies or cloud jobs were added. The original before/after harnesses and
fixture were reused. [SHA256SUMS](SHA256SUMS) pins every bundled file.
