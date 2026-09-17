# Target — PHerc. 0211, CW fit, windings w010–w065, z 10000–11000

Same pipeline as `../target-ACW-w010-065/`, on the CW fit. Rendered surface
256400 × 2280 px = 2400.4 × 21.3 mm. 18.9 min on one A10
(the volume cache was warm from the ACW run). T = 199.

| | ACW fwd | ACW rev | CW fwd | CW rev |
|---|---:|---:|---:|---:|
| pixels ≥ T (%) | 0.019 | 0.027 | 0.032 | 0.019 |
| components ≥ T | 3,937 | 6,490 | 6,510 | 4,116 |
| candidates ≥ 0.5 mm | **23** | **38** | **42** | **22** |
| largest (mm) | 0.95 | 0.75 | 1.08 | 0.71 |
| excluded (boundary/band/coverage) | 0/0/0 | 0/0/0 | 1/2/0 | 1/1/1 |

Flipping the winding sense flips the surface normal, so CW-forward should
resemble ACW-reverse and vice versa: it does (0.032 % vs 0.027 %; 0.019 % vs 0.019 %).
Same verdict as ACW: diffuse sub-millimetre blobs, no letterforms.
Files: `readout.json`, `timing.json`, `*_cand[1-3]_*_2mm.png`.
