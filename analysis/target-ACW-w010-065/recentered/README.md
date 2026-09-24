# Re-centred slab (verification step, see prereg deviation note 2)

64-layer render of the same flattened ACW mesh, each 256 × 256 tile re-centred on
its smoothed intensity peak, 28 layers kept. Centre layer over tiles: median 32/64,
p10 14, p90 45, 17 % at a clamp edge — i.e. the original 28-layer render was
centred on average but wandered ±~18 layers (±170 µm) along the strip.

| slab | centring: blocks peaking mid-third | fwd: px ≥ T / candidates | rev: px ≥ T / candidates |
|---|---:|---:|---:|
| control, curated w035 mesh (letters present) | 77 % | 1.257 % / **61** | 0.141 % / 9 |
| target ACW, as rendered, seed42 | 46 % | 0.019 % / **23** | 0.027 % / **38** |
| target ACW, as rendered, seed43 | 46 % | 0.015 % / **14** | 0.018 % / **27** |
| target ACW, re-centred per tile, seed42 | 100 % | 0.009 % / **10** | 0.014 % / **20** |
| target CW, as rendered, seed42 | 54 % | 0.032 % / **42** | 0.019 % / **22** |

Wall clock 108.9 min on one A10 (64-layer render ≈ 25 min, re-centre ≈ 15 min,
inference ≈ 2 × 25 min under GPU contention). Files: `readout.json`, `slab_profile.json`,
`recenter.log`, `timing.json`; the stack itself is on the volume under
`ink/ACW_z10000-11000_s30000/fitted_scoped_w010-065_recentered/`.
