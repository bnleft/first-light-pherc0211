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
