# Approved isolated CPU test environment

The user approved this setup on October 19, 2026, and it has been executed.
It changed only the named scratch environment and cache. It used the installed
Python 3.14 interpreter without installing villa's full models extra or compiling
Volume Cartographer. No cloud jobs were launched. The final validation runs use CPU.

The minimal packages are sufficient to import the flat inference module with
unrelated parent-package initializers bypassed, run the new focused tests, and
run the reader/DataLoader/model-loop/accumulator replay with a deterministic
CPU test network. This is not a full-project installation or CI run.

Approved install commands (executed with scratch-only writes under `fetch.sb`):

```sh
export AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1
export TMPDIR=/private/tmp/vesuvius-render-preflight-20261018
export UV_CACHE_DIR=/private/tmp/vesuvius-render-preflight-20261018/flat-test-uv-cache
uv venv --python /opt/homebrew/opt/python@3.14/bin/python3.14 /private/tmp/vesuvius-render-preflight-20261018/flat-supported-py314
uv pip install --python /private/tmp/vesuvius-render-preflight-20261018/flat-supported-py314/bin/python 'torch==2.12.1' 'numpy==2.4.6' 'zarr==3.1.6' 'numcodecs==0.16.5' 'tifffile>=2024.8.30' 'imagecodecs>=2026.3.6' 'aiohttp>=3.12.15' 'fsspec>=2025.9.0' 'pytest>=8.4.2'
```

Installed runtime: Python 3.14.6, PyTorch 2.12.1, NumPy 2.4.6, Zarr 3.1.6,
imagecodecs 2026.10.10, tifffile 2026.10.15, and pytest 9.1.1 on macOS arm64.
[supported-requirements.txt](supported-requirements.txt) records the original 32
packages. The approved extension below adds 14 packages; its
[full freeze](supported-requirements-with-model-deps.txt) records all 46. No
global interpreter or existing project environment was modified.

The confirmation requirement comes from villa's `AGENTS.md`, section 1.0:

> If dependencies are needed, report the exact minimal install command per target subproject and ask for confirmation.

## Executed validation commands

The candidate passes all eight focused tests; the original fails seven and
passes one. All three real-data comparisons pass, including exact float32 blend
sums, LZW TIFF file bytes, and decoded pixels. Whole-module mode imports the
complete inference module and does not fall back to extracting definitions.

```sh
export FLAT_FIX_SCRATCH=/private/tmp/vesuvius-render-preflight-20261018
export FLAT_FIX_BUNDLE=/Users/bryantle/Documents/bnleft/first-light-pherc0211/contrib/october-flat-inference-buffer-fix
export FLAT_FIX_SOURCE="$FLAT_FIX_SCRATCH/flat-inference-memory/vesuvius/src"
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR="$FLAT_FIX_SCRATCH"
export TORCH_HOME="$FLAT_FIX_SCRATCH/flat-supported-torch-home"
export TORCHINDUCTOR_CACHE_DIR="$FLAT_FIX_SCRATCH/flat-supported-torch-cache"

# Baseline: expected exit status 1, with seven failures and one passing test.
/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/run_full_module_tests.py" --source-root "$FLAT_FIX_SOURCE" --source "$FLAT_FIX_BUNDLE/infer-before.py" --tests "$FLAT_FIX_BUNDLE/test_flat_inference_empty_chunks.py"

# Candidate: expected exit status 0, with eight passing tests.
/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/run_full_module_tests.py" --source-root "$FLAT_FIX_SOURCE" --source "$FLAT_FIX_BUNDLE/infer-after.py" --tests "$FLAT_FIX_BUNDLE/test_flat_inference_empty_chunks.py"

/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/validate_flat_data_loop.py" --mode module --source-root "$FLAT_FIX_SOURCE" --before "$FLAT_FIX_BUNDLE/infer-before.py" --after "$FLAT_FIX_BUNDLE/infer-after.py" --full-input "$FLAT_FIX_SCRATCH/pherc0211-original-render.zarr/segment.zarr" --crop-input "$FLAT_FIX_SCRATCH/flat-buffer-evidence/portable-fixture.zarr" --report "$FLAT_FIX_SCRATCH/flat-buffer-evidence/DATA_LOOP_SUPPORTED_EXACT.json"
```

The data-loop check uses a deterministic CPU test network and verifies the
reader, normalization, batching, model-input stream, inference loop, blend sums,
final uint8 pixels, and (in whole-module mode) actual LZW TIFF writing/decoding.
It does not establish trained-model ink quality, a full CLI/checkpoint-loading
run, full-project CI, GPU behavior, or community adoption.

The runs emit an OpenMP temporary-file warning under the scratch-only sandbox;
pytest also reports that Zarr was imported before assertion rewriting. Neither
warning fails a check. Test output is bundled without suppressing these warnings.

## Additional existing-suite check

The user separately approved the following command on October 19, 2026. It was
executed in the same scratch environment under `fetch.sb`, with
`AGENTS_AGENT_MODE=1`, `AGENTS_ALLOW_INSTALL=1`, and the scratch `TMPDIR` and
`UV_CACHE_DIR` above. The original 32 versions were constrained and preserved.

```sh
uv pip install --python /private/tmp/vesuvius-render-preflight-20261018/flat-supported-py314/bin/python --constraint /private/tmp/vesuvius-render-preflight-20261018/flat-buffer-evidence/supported-requirements.txt 'einops>=0.8.1' 'timm>=1.0.20'
```

This installed einops 0.8.2, timm 1.0.30, and 12 transitive dependencies, including
torchvision 0.27.1. See [the install log](extra-test-deps-install.txt) and
[extended freeze](supported-requirements-with-model-deps.txt).

The first broader attempt exposed missing dotted-package bindings and spawned
process setup in the isolated harness. It also auto-selected the local MPS
device. The final runner imports the complete module in main and spawned
processes, binds package attributes normally, and fixes only the runtime's
device-selection hook to CPU. Tests and inference implementations are unchanged.

The remaining worker restriction was the blanket denial of Unix sockets. The
[final sandbox profile](flat-buffer-ipc.sb) adds only a socket-path allowance
under `flat-worker-ipc`; it retains the IP-network denial and the same approved
write paths (scratch, the evidence bundle, and `/dev/null`). PyTorch's
[shared-memory manager](https://github.com/pytorch/pytorch/blob/v2.12.1/torch/lib/libshm/manager.cpp)
creates a temporary `manager.sock` using
[Unix-domain sockets](https://github.com/pytorch/pytorch/blob/v2.12.1/torch/lib/libshm/socket.h).

Using the exports above, the final combined CPU commands were:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$FLAT_FIX_SOURCE"
export TMPDIR="$FLAT_FIX_SCRATCH/flat-worker-ipc"
mkdir -p "$TMPDIR"

/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer-ipc.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/run_full_module_tests.py" --source-root "$FLAT_FIX_SOURCE" --source "$FLAT_FIX_BUNDLE/infer-before.py" --tests "$FLAT_FIX_SCRATCH/flat-inference-memory/vesuvius/tests/ink_detection/test_flat_inference.py" "$FLAT_FIX_SCRATCH/flat-inference-memory/vesuvius/tests/ink_detection/test_flat_inference_empty_chunks.py"

/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer-ipc.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/run_full_module_tests.py" --source-root "$FLAT_FIX_SOURCE" --source "$FLAT_FIX_SOURCE/vesuvius/ink_detection/inference/infer.py" --tests "$FLAT_FIX_SCRATCH/flat-inference-memory/vesuvius/tests/ink_detection/test_flat_inference.py" "$FLAT_FIX_SCRATCH/flat-inference-memory/vesuvius/tests/ink_detection/test_flat_inference_empty_chunks.py"
```

- Original: **20 passed / 7 failed**, exit status 1.
- Candidate: **27 passed**, exit status 0.
- All eight new regressions pass on the candidate; seven fail on the original.
- All 19 existing cases pass on both, including the spawned-worker test.

Raw logs: [original](all-flat-tests-before-cpu.txt),
[candidate](all-flat-tests-after-cpu.txt). The passing existing cases cover flat
preprocessing, layer selection, occupancy/blending, TTA, TIFF encoding, folder
handling, and CPU checkpoint-to-TIFF lifecycle behavior on test fixtures.

The [sandbox probe](probe_flat_worker_sandbox.py) was run under both profiles:

```sh
/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_SCRATCH/probe_flat_worker_sandbox.py" --expect-ipc blocked
/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer-ipc.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_SCRATCH/probe_flat_worker_sandbox.py" --expect-ipc allowed
```

Both return exit status 0. The [original result](sandbox-probe-original.json)
confirms task Unix IPC was blocked; the [final result](sandbox-probe-local-ipc.json)
confirms a task-local socket round trip works. Both reject TCP connections to
loopback and a reserved external test address, socket binding elsewhere, and a
file-write probe outside the approved paths with `EPERM`. The test assertions,
inference implementation, core package versions, and outside-write restrictions
were preserved. No dependency installation or cloud job was needed for this fix
to the test environment.

## Actual trained-checkpoint comparison

The published checkpoint named by `modal_app.py` was downloaded to scratch from
a pinned Hugging Face revision. The 138,360,039-byte file matches publisher
SHA-256 `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab`.
[CHECKPOINT_PROVENANCE.json](CHECKPOINT_PROVENANCE.json) records the metadata and
URL. This does not prove which exact checkpoint bytes the historical remote run
used; those bytes were not rehashed. The weights are not bundled.

```sh
/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/fetch.sb" curl --fail --location --silent --show-error --max-time 180 --output "$FLAT_FIX_SCRATCH/ink_9um-seed42-step-075000.pth" 'https://huggingface.co/scrollprize/ink_9um/resolve/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/hybrid_3d2d-seed42/step-075000.pth'

/usr/bin/sandbox-exec -f "$FLAT_FIX_SCRATCH/flat-buffer-ipc.sb" "$FLAT_FIX_SCRATCH/flat-supported-py314/bin/python" "$FLAT_FIX_BUNDLE/validate_trained_checkpoint.py" --source-root "$FLAT_FIX_SOURCE" --before "$FLAT_FIX_BUNDLE/infer-before.py" --after "$FLAT_FIX_BUNDLE/infer-after.py" --checkpoint "$FLAT_FIX_SCRATCH/ink_9um-seed42-step-075000.pth" --crop "$FLAT_FIX_SCRATCH/flat-buffer-evidence/portable-fixture.zarr" --report "$FLAT_FIX_SCRATCH/flat-buffer-evidence/TRAINED_CHECKPOINT_CPU.json"
```

The runner verifies the checkpoint hash before calling the actual `infer.main()`
path, fixes only device selection to CPU, and instruments inputs, buffer storage,
and writer inputs. Parent initializers remain bypassed. The actual checkpoint
loader, config normalization, model construction, forward pass, accumulator,
and LZW TIFF writer execute. All 508 state tensors match with no missing or
unexpected keys. The embedded config selects robust-MAD preprocessing, 17 depth
layers, and 128-pixel patches. The report's configured AMP field is float16, but
the upstream CPU path uses float32 model parameters and inputs with CUDA autocast
inactive; their dtypes are asserted and recorded.

Both before/after CLI calls return 0. In both depth directions, model inputs,
float32 sums, decoded pixels, and TIFF bytes are identical. The original retains
two buffer pairs / 256 KiB before final flush; the fix retains zero. The crop's
peak storage is unchanged. See [the report](TRAINED_CHECKPOINT_CPU.json) and
[raw output](trained-checkpoint-cpu.stdout.txt). This is a bounded, failure-focused
real crop, not a full-render trained-model or ink-quality evaluation. Instrumented
wall times do not establish an inference speedup. No additional dependencies or
cloud jobs were needed.
