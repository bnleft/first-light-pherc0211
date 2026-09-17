# Runbook — First Light PHerc. 0211

Goal: run the published First Letters workflow (spiral fit → lasagna flatten →
`vc_render_tifxyz` → `ink_9um` inference) end to end on PHerc. 0211, one of the
two eligible scrolls with every published input and no public ink result, and
write down what worked, what broke, and what it cost.

Template: millerandmuller/first-light-pherc0826 (Aug 2026 progress prize).
Pinned villa commit: `2dcfaf6a08c3bc796fde726c4fa32050c8fc90e7` (main, 2026-09-17).

## Why 0211

Survey on 2026-09-17 (eligible-spiral-dataset `--survey` logic re-checked by
hand against the bucket): 3 of 13 eligible scrolls have a published umbilicus
(0125, 0211, 0826). 0826 is written up. 0211 has the smaller tracks dataset of
the remaining two (7.0 GiB DBM vs 8.9 GiB) and no `segments/` prefix in the
bucket, i.e. no published surfaces or ink.

## Compute

Modal (serverless, billed per second, no idle-pod cost). GPU: A10 24 GB.
Spend cap agreed: $60.

## Steps

| # | Step | Function | Status |
|---|------|----------|--------|
| 0 | Build image, probe binaries | `probe` | done |
| 1 | Fetch tracks + umbilicus (+ lasagna via `pack_pools_fast`) | `fetch_dataset` | done |
| 2 | Render axial slices for the CW/ACW read | `render_slices` | done |
| 3 | Human reads winding sense | (Bryant) | |
| 4 | Pack resident pools (new since the Aug post) | `pack_pools_fast` | done |
| 5 | 1,500-step fits, CW and ACW, z 10000–11000 | `spiral_fit` | done, tied |
| 6 | 30,000-step fits, both senses | `spiral_fit` | done, tied; ACW chosen first |
| 7 | Flatten + render + ink inference (w010–w065) | `render_and_infer` | done, ACW and CW |
| 8 | Control on PHerc0139 w035 + calibration | `control`, `calibrate_control` | done, T=199 |
| 9 | Preregistered readout, writeup, submission | `readout`; README drafted | readout done; verdict + submission = Bryant |

## Walls hit so far

(filled in as they happen; each becomes a section in the writeup and, where
it is a villa defect a human hit on real data, a PR)

## 2026-09-17 — step 0–2 notes

- Image built from `ghcr.io/scrollprize/villa/volume-cartographer:main` + Python 3.14 +
  `uv sync --frozen` in `spiral-fitting`. Probe: torch 2.11.0+cu128, zarr 3.3.0,
  all `vc_*` binaries present **except `vc_obj_uv_lift`** (only used by
  `render_ink.py --strips`, which we do not use). Build ~1 min on Modal.
- `s3fs`/`aiohttp` raise `RuntimeError: ... is not the running loop` at interpreter
  exit on Python 3.14 after the renders are written. Harmless here; noted because it
  makes a successful run end with a traceback.
- Axial slices at level 2 (z 5000 / 9000 / 13000) and level-1 crops around the
  published umbilicus (z 9000 / 10000 / 10500 / 11000): the umbilicus sits in a
  clean, tightly wound core at z 9000–10500; at z 13000 the core is crushed and
  the umbilicus is off the visible centre. Fit window chosen: **z 10000–11000**
  (same width as the Aug run). Winding sense: to be read by Bryant from the crops
  and cross-checked empirically with 1,500-step CW and ACW fits.
- Current villa main only loads the lasagna normal stores through
  `pack_resident_pools.py` sidecars (`lasagna_data.py`: "there is no other loading
  path"). The 18 Aug workflow post and the Aug runbook predate this. Added as a step.

## 2026-09-17 — control run, attempt 1 (failed usefully)

- `ghcr.io/scrollprize/villa/volume-cartographer:main` is built from villa
  `1e3f4c021f4e53bea3867772ed05f51a7e586a9c` (image label
  `org.opencontainers.image.revision`, created 2026-05-13). Its `vc_render_tifxyz`
  rejects `--flip-normals`, which main's source has and the Aug control recipe
  needs. Same defect as villa#1588 ("container images are 3+ months stale"),
  now four months. Fix on our side: rebuild the `vc_runtime` component from the
  pinned commit inside the image (`cmake --preset ci-release-gcc`).
- Our own wrapper masked the failure: `cmd | tee log` returns tee's exit code, so
  inference ran against a store that was never written. Fixed with
  `bash -o pipefail`. Cost of the wasted GPU run: a couple of minutes of A10.

## 2026-09-17 — lasagna stores: do not mirror them into a Modal Volume

- `aws s3 cp --recursive` of one group-2 normal store (~100k chunk files, ~2.6 GB)
  into a Modal Volume progressed at ~2 chunk-rows/min → hours per store. The
  tracks (4 files, 16 GiB) had copied in ~12 min at 33 MB/s, so this is a
  small-file problem, not bandwidth.
- `fit_spiral` reads the normals only through `pack_resident_pools.py` sidecars,
  and the packer packs whatever chunk files exist. New step `pack_pools_fast`:
  pull only the chunk rows for the fit window (±1500 slices) to local disk,
  pack there, write the sidecars (a few large files) + zarr metadata to the volume.
- Stopping the fetch app mid-copy did not lose the tracks: Modal volume writes
  were already visible from the CLI (`modal volume ls`), so `vol.commit()` at the
  end was not what persisted them.

## 2026-09-17 — pack + 1,500-step CW/ACW fits (z 10000–11000)

- `pack_pools_fast`: 32 chunk rows × 3 stores pulled to local disk in ~3 min,
  packed in ~1 min, sidecars verified (500 random voxels each, all match).
  DBM mtime restored so the published crossings cache was accepted
  (`loaded track crossing cache ... 14,360,311 tracks`). Total 4.6 min, CPU only.
- Fits: 741,046 tracks / 34.98 M points in the z-ROI; 5,773 of 25,859 normal
  bricks resident. Both senses converge similarly at 1,500 steps:

| sense | loss @1400 | satisfied tracks | satisfied points | wall (A10) |
|---|---:|---:|---:|---:|
| CW  | 354.3 | 130,843 / 741,046 (17.7%) | 55.9% | 14.0 min |
| ACW | 351.1 | 131,014 / 741,046 (17.7%) | 56.3% | 15.4 min |

  Identical to within noise, as villa#1621 predicts (the metric is periodic in
  the winding). Decision therefore comes from overlaying the fitted sheets on
  the CT slice (`overlay_fits`), plus Bryant's read of the crops.

## 2026-09-17 — 30,000-step CW/ACW fits (z 10000–11000)

| sense | loss @20k | loss @25k | loss @29.8k (track_dt on) | satisfied tracks | satisfied points | wall (A10) |
|---|---:|---:|---:|---:|---:|---:|
| CW  | 136.1 | 137.1 | 212.1 | 131,918 (17.8%) | 52.9% | 61.0 min |
| ACW | 136.0 | 126.4 | 204.4 | 131,561 (17.8%) | 52.9% | 64.9 min |

Still tied. Note the 1,500-step runs took ~14 min because ~5 min of that is
loading (crossings cache, packed track store, resident pools); the 30k runs
amortise it. Winners' 30k on an A6000 took 30 min; A10 is ~2× slower here.
Cost: ~$1.10/h × ~2.1 h ≈ $2.30 for both converged fits.

### Sense selection (recorded before any target inference)

Every automated signal is tied: loss, satisfaction, and mean CT intensity under
the fitted vertices (CW 67–72, ACW 68–71 across three slices; scroll-voxel mean
~108). ACW had the lower loss at every checkpoint from 15k steps on
(144 vs 150, 136 vs 136, 126 vs 137, 204 vs 212), which is weak evidence. We
proceed with **ACW** for the first flatten + render + inference, windings
w010–w065 (the same inner-half scope as the Aug run), and will run CW too if
Bryant's read of the crops disagrees or if the flattened ACW render shows
sheet-switch seams. Both fitted meshes are kept.
