# Target — PHerc. 0211, ACW fit, windings w010–w065, z 10000–11000

Pipeline: villa `2dcfaf6a` — `fit_spiral` (30,000 steps, ACW) → `render_ink.py` concat of 56
windings + lasagna forward flatten → `vc_render_tifxyz` (28 slices, full res) →
`ink_9um` `hybrid_3d2d-seed42/step-075000.pth`, both directions. 36.1 min on one A10.

Rendered surface: **256640 × 2360 px = 2402.7 mm × 22.1 mm** (2.4 m of unrolled
papyrus, 22 mm tall). Threshold T = 199 (from the control, see `prereg/readout.md`).
Size floor 53 px = 0.5 mm.

| | forward | reverse |
|---|---:|---:|
| mean prediction | 45.4 | 46.9 |
| p99 | 150 | 157 |
| pixels ≥ T | 112,467 (0.019 %) | 161,827 (0.027 %) |
| 4-connected components ≥ T | 3,937 | 6,490 |
| excluded: below 0.5 mm | 3,914 | 6,452 |
| excluded: boundary / band / low coverage | 0 / 0 / 0 | 0 / 0 / 0 |
| **candidates ≥ 0.5 mm** | **23** | **38** |
| largest candidate | 0.95 mm | 0.75 mm |

For scale: on the control, 2 mm crops of real letters have 22–29 % of pixels ≥ T;
the target's largest candidate is a ~1 mm diffuse blob (`same_scale_comparison.png`).
Fraction of pixels ≥ T on the whole target is 0.019 %, against 1.3 % of the control
grid carrying ink labels.

Files: `same_scale_comparison.png` (control letters | target candidates | plain texture, 1 mm bars),
`forward_q.png` / `reverse_q.png` (quarter-scale full strips, 64160 × 590), `*_cand*_2mm.png`
(2 mm crops of the six largest candidates per direction, 2× nearest-neighbour),
`w010-065_flat.00[01].jpg` (render_ink's 1/8-scale CT preview of the flattened surface),
`readout.json` (every number above), `timing.json`.

Verdict: written by Bryant in the top-level README, from the three allowed sentences.
