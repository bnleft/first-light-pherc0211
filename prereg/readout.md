# Preregistered readout — PHerc. 0211, z 10000–11000

Written and committed **before** any ink inference is run on the target
segment. The commit hash of this file at that point is recorded in the README.
Criteria below are not edited after target inference; corrections go in a dated
"Deviation note" section at the end.

## Scope

- Target: PHerc. 0211, volume `20250821151803-9.362um-1.2m-113keV-masked`,
  spiral fit over z 10000–11000, whichever winding sense the converged-fit
  comparison (`overlay_fits` on the 30,000-step CW and ACW runs, plus Bryant's
  read of the axial crops) selects. The selection and its reasoning are
  recorded in `runbook/00_plan.md` before this file is used.
- Model: `scrollprize/ink_9um` `hybrid_3d2d-seed42/step-075000.pth`, run as
  published (`--overlap 0.5 --blend-mode hann --direction both`), no retraining.
- Calibration: the positive control (PHerc0139 w035, `analysis/control/`),
  rendered and inferred with the identical pipeline.

## What counts as ink (decided now)

1. **Threshold.** The median predicted probability over the control's published
   ink-label pixels (`ink_9um/labels/native9-scrollprizeorg-21slices/w035`) is
   the threshold `T`. It is computed once, from the control, and written here
   as a deviation note with its value before target inference.
2. **Candidate.** A 4-connected component of pixels ≥ `T` in *either* direction's
   prediction whose bounding box exceeds 0.5 mm on its longer side
   (≥ 53 px at 9.362 µm/px, full resolution).
3. **Exclusions, applied before looking at shapes:**
   - components touching a rendering boundary (tile seam or the trimmed mesh edge);
   - components that are a straight vertical or horizontal band ≥ 5× longer than wide;
   - components inside a region where the surface volume's middle slice is > 50 % zero.
4. **Verdict wording.** One of exactly:
   - "I didn't see any ink in the window."
   - "I saw candidate strokes but no letterforms."
   - "I saw N letterforms; papyrological review requested."
   No other phrasing, and the sentence is written by Bryant.

## What we will publish regardless of outcome

Renders of the flattened window (forward and reverse), the same-scale
comparison panel (control letterforms | best target candidate | excluded
artifact | plain texture), every count above, the time/cost table, and the
list of walls hit with their villa issue or PR numbers.

## Disclosure

The checkpoint's inference patch is 128 × 128 px = 1.198 mm, larger than the
0.5 × 0.5 mm the prize page recommends for ML-generated images. We claim no
letters from these outputs unless a papyrologist does.

## Deviation note 1 — calibration value (2026-09-17, before target inference)

Computed from the control exactly as specified above (`calibrate_control`,
labels `ink_9um/labels/native9-scrollprizeorg-21slices/w035`, any-depth max,
396,164 label pixels = 1.30 % of the 5820 × 5240 grid):

| direction | median on label | mean on label | p90 off label | p99 off label | frac off-label ≥ median-on |
|---|---:|---:|---:|---:|---:|
| forward | **199** | 185.8 | 122 | 195 | 0.63 % |
| reverse | 68 | 76.7 | 111 | 177 | 58.3 % |

**T = 199** (uint8; ≈ 0.78 probability). The forward direction is the readable
one on the control; the reverse is treated as a second, weaker channel. The
ACW target inference was already running when this was computed, but no target
output had been read; this note is committed before any target output is opened.

## Deviation note 2 — verification steps added after target inference (2026-09-23/24)

Added after the first target readout, as checks on whether the negative result
is an artifact. Criteria above are unchanged; these steps re-apply them.

1. **Sheet-centring diagnostic.** Per 1024-px column block, the mean non-zero
   intensity per layer of the rendered slab; a sheet-centred render peaks in the
   middle. Control: 77 % of blocks peak in the middle third. Original target
   render (ACW): 46 %. So the first readout ran on a surface that is on papyrus
   but not consistently centred on it.
2. **Local re-centring.** Rendered a 64-layer slab from the same flattened mesh,
   re-centred each 256 × 256 tile on its smoothed intensity peak, cut a 28-layer
   stack (median centre layer 32 of 64; p10 14, p90 45; 17 % of tiles at a
   clamp edge). Result: 100 % of blocks peak in the middle third. Inference and
   the readout were re-run on this stack with the same T.
3. **Second checkpoint.** `hybrid_3d2d-seed43` on the original slab.

Neither changes what counts as ink; both are reported alongside the original.
