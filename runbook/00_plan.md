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
| 0 | Build image, probe binaries | `probe` | |
| 1 | Fetch tracks + umbilicus + lasagna group-2 stores | `fetch_dataset` | |
| 2 | Render axial slices for the CW/ACW read | `render_slices` | |
| 3 | Human reads winding sense | (Bryant) | |
| 4 | Pack resident pools (new since the Aug post) | `pack_pools` | |
| 5 | 1,500-step fits, CW and ACW, z 10000–11000 | `spiral_fit` | |
| 6 | 30,000-step fit in the confirmed sense | `spiral_fit` | |
| 7 | Flatten + render + ink inference | (to write) | |
| 8 | Control on PHerc0139 w035 | (to write) | |
| 9 | Preregistered readout, writeup, submission | (Bryant + Claude) | |

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
