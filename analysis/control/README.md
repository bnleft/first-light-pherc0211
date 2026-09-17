# Positive control — PHerc0139 w035 (`20260317000000-w035_2026031718`)

Pipeline check, not a generalisation test: w035 is in the `ink_9um` training set.

- Mesh: `PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz`
  (the 9.362 µm registration; the HF `ink/0139/w035_2026031718` mesh is registered
  to a 2.4 µm volume and renders all-zero against 9.362 µm, per the Aug team).
- Volume: `PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr`, streamed from S3.
- Render: `vc_render_tifxyz --group-idx 0 --scale 1 --num-slices 28 --slice-step 1 --flip-normals`,
  built from villa `2dcfaf6a` (the published `:main` image lacks `--flip-normals`).
  Output 5820×5240×28, matching the team's published surface volume for this segment.
- Inference: `ink_9um/hybrid_3d2d-seed42/step-075000.pth`, `--overlap 0.5 --blend-mode hann --batch-size 4 --direction both`.
  6458 patches at 82.1% occupancy.
- Wall clock: 4.75 min on one A10 (render ≈ 2 min streaming from S3, inference ≈ 1.5 min per direction).
- Result: `control_prediction_q.png` (quarter scale) shows legible letterforms
  (Ρ, ΝΟ, ΝΛ, Τ, Ρ, Ν, ΥϹΙ) on a mottled background. Forward direction is the readable one.

Files: `control_prediction_q.png`, `control_prediction_reverse_q.png` (previews; the
full-res TIFFs are on the Modal volume under `control/out_own_render/`).
