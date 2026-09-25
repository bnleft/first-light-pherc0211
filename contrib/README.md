# Villa contributions (posted 2026-09-24 under bnleft)

- #1588 comment: https://github.com/ScrollPrize/villa/issues/1588#issuecomment-5807394810
- #1716 comment: https://github.com/ScrollPrize/villa/issues/1716#issuecomment-5807394976
- Docs PR: https://github.com/ScrollPrize/villa/pull/1880 (merged 2026-09-25 by pmh47)


Villa's CONTRIBUTING requires a human to have hit the problem on real data and
to write the commentary. Bryant decides what to post and edits the text.

## 1. Comment on villa#1588 (stale container images)

> Still stale as of 2026-09-17: `ghcr.io/scrollprize/villa/volume-cartographer:main`
> carries `org.opencontainers.image.revision=1e3f4c021f4e53bea3867772ed05f51a7e586a9c`
> (created 2026-05-13). Concrete consequence today: its `vc_render_tifxyz` rejects
> `--flip-normals`, which `apps/src/vc_render_tifxyz.cpp` on main has and which
> the PHerc0139/w035 control recipe needs. The `:main` image is also Ubuntu 24.04
> while `scripts/install_build_deps.sh` now targets 26.04 (`flang-21`), so the
> script cannot be used to refresh it in place. Worked around by rebuilding the
> `vc_runtime` component from the pinned commit inside our image
> (`cmake --preset ci-release-gcc -DVC_BUILD_FLATBOI=OFF -DVC_BUILD_UI_TRACER=OFF`).

## 2. Docs PR: spiral-fitting README — resident pools are required

Add under "Scroll specification":

> ### Lasagna inputs must be packed first
> Since <commit>, `fit_spiral.py` reads `normal_x`/`normal_y`/`gradient_magnitude`
> only through resident-pool sidecars (`lasagna_data.py`: "there is no other
> loading path"). Before the first fit on a dataset, run
> `python pack_resident_pools.py <dataset>/lasagna --what normals,grad_mag --normal-group <normal_zarr_group>`
> pointed at the folder holding the `*_nx/*_ny/*_grad_mag.ome.zarr` stores.
> The packer only reads chunk files that exist, so a store that holds only the
> z-rows of your fit window packs fine; rows outside it read as no-data.
> The error you get without this is
> `lasagna normals: resident-pool sidecar '...respool_g2_pair' not found`.

Evidence to attach: our `pack_pools_fast` log (sidecars verified) and a fit log
showing `lasagna normals: resident pool 5,773/25,859 bricks ... loaded`.

## 3. Comment on villa#1716 / #1713 (input_use_tracks default)

Confirm on a second scroll: with the post's overrides verbatim and
`input_use_tracks: true` added, PHerc0211 loads 741,046 tracks / 34.98 M points
in z 10000–11000; without it the fit runs on normals + umbilicus only.

## Not proposing

- The s3fs/aiohttp `RuntimeError: ... is not the running loop` at interpreter
  exit on Python 3.14: upstream (aiobotocore/aiohttp), harmless.
