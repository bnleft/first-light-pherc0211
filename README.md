# First Light — PHerc. 0211

We ran the Vesuvius Challenge's published First Letters workflow end to end on
PHerc. 0211, one of the two eligible scrolls with every published input and no
public ink result. This is what worked, what broke, what it cost, and what we saw.

## First public look

![Flattened inside of PHerc. 0211, windings w010–w065 of slices 10,000–11,000 (1/8-scale CT preview)](analysis/target-ACW-w010-065/w010-065_flat.000.jpg)

56 windings of the inner half of the scroll, z 10,000–11,000, flattened: 2.4 m
of papyrus, 22 mm tall. Pipeline: villa at commit `2dcfaf6a`, spiral fit
(30,000 steps) → lasagna flatten → `vc_render_tifxyz` → `ink_9um` inference,
run as published, with the fixes and workarounds listed below. Before reading
anything into the image, see [`prereg/readout.md`](prereg/readout.md) for what
we committed to count as ink; we wrote the rules before we looked.

## What broke on the way

<!-- TODO(Bryant): keep the ones you actually hit; one line each; link issues/PRs -->

1. **The published VC3D container is four months behind `main`.**
   `ghcr.io/scrollprize/villa/volume-cartographer:main` is built from
   `1e3f4c02` (2026-05-13). Its `vc_render_tifxyz` has no `--flip-normals`,
   which the control recipe needs. Villa #1588 already reports the staleness;
   we add the revision label and the missing flag as concrete evidence. Fix on
   our side: rebuild the `vc_runtime` component from the pinned commit inside
   the image (`modal_app.py`, `vc_image`).
2. **Current `main` will not load the lasagna normal maps without a packing
   step the workflow post never mentions.** `lasagna_data.py` reads normals
   only through `pack_resident_pools.py` sidecars ("there is no other loading
   path"). The 18 Aug post and the Aug runbook predate this. Documented in
   `runbook/00_plan.md`; proposed doc change in `contrib/`.
3. **Mirroring the zarr normal stores into a network volume is hours-slow.**
   ~100k chunk files per store. Because the fitter only needs the sidecars,
   `pack_pools_fast` pulls the fit window's chunk rows to local disk, packs
   there, and stores only the sidecars. 4.6 min total.
4. **Winding sense cannot be read off any fit metric.** Loss, track
   satisfaction and CT brightness under the fitted vertices are identical for
   CW and ACW at 1,500 and 30,000 steps (villa #1621 predicts this). We fitted
   both, rendered both, and report both.
5. *(ours, not villa's)* `cmd | tee log` masked a failed render; inference ran
   on a store that did not exist. `bash -o pipefail` everywhere since.

## What we actually see

![Same-scale comparison: control letterforms, target candidates, plain texture, 1 mm bars](analysis/target-ACW-w010-065/same_scale_comparison.png)

<!-- TODO(Bryant): exactly one of the three sentences from prereg/readout.md §4 -->
**"…"**

How we got there: the identical pipeline first ran on a control segment with
published ink labels (PHerc. 0139/w035, from the model's own training set — a
pipeline check, not a generalisation test). It reproduced legible letterforms
and gave us our threshold: **199** (of 255), the median prediction on labelled
ink. On the ACW target, 112,467 of 605 million pixels crossed that threshold in
the forward direction (0.019 %; the control grid is 1.3 % labelled ink),
forming 23 candidates above the 0.5 mm size floor. No candidate was excluded by
the boundary, band or coverage rules. The largest is a ~1 mm diffuse blob that
looks like neither the control's strokes nor plain texture. Every number and
crop is in [`analysis/target-ACW-w010-065/`](analysis/target-ACW-w010-065/).
<!-- TODO: CW numbers once its readout lands -->

Disclosure: the checkpoint's inference patch is 128 × 128 px = 1.198 mm, larger
than the 0.5 × 0.5 mm the prize page recommends for ML-generated images. We
claim no letters from these outputs.

## Proof

- Criteria committed before target inference: `c7348e7`; calibration value
  appended before any target output was opened: `f29cd93`.
- Control: [`analysis/control/`](analysis/control/).
- Time and cost table below, from `logs/` and the Modal usage page.

### Time and cost

| Step | Where | Wall-clock | Cost (USD) |
|---|---|---:|---:|
| Image builds (3 layers, VC apps from source) | Modal CPU | ~25 min total | <!-- TODO --> |
| Tracks fetch (16 GiB, 4 files) | Modal CPU | ~12 min | |
| Axial-slice renders for the sense read | Modal CPU | ~2 min | |
| `pack_pools_fast` (3 stores, window ±1500) | Modal CPU | 4.6 min | |
| Spiral fit 1,500 steps × 2 senses | A10 | 14.0 + 15.4 min | |
| Spiral fit 30,000 steps × 2 senses | A10 | 61.0 + 64.9 min | |
| Control: render + inference | A10 | 4.75 min | |
| Target ACW: flatten + render + inference (w010–w065) | A10 | 36.1 min | |
| Target CW: same | A10 | <!-- TODO --> | |
| **Total** | | | **<!-- TODO from Modal dashboard -->** (cap: $60) |

Modal bills per second with no idle-pod cost; the Aug team's $56 was ~90 % idle
RunPod time. <!-- TODO(Bryant): paste the dashboard number -->

## Reproduce it yourself

Everything is one file: [`modal_app.py`](modal_app.py). With a Modal account:

```
modal run modal_app.py::probe
modal run modal_app.py::fetch_dataset          # tracks + umbilicus (stop it once the lasagna copy starts, see runbook)
modal run modal_app.py::render_slices          # axial slices for the CW/ACW read
modal run modal_app.py::pack_pools_fast        # lasagna sidecars for z 10000–11000 ±1500
modal run modal_app.py::spiral_fit --sense ACW --steps 30000
modal run modal_app.py::control && modal run modal_app.py::calibrate_control
modal run modal_app.py::render_and_infer --run-tag ACW_z10000-11000_s30000 --winding-min 10 --winding-max 65
modal run modal_app.py::readout --run-tag ACW_z10000-11000_s30000
```

The dated runbook with every wall as we hit it is in [`runbook/`](runbook/).

## How this was built (AI assistance disclosure)

<!-- TODO(Bryant): rewrite in your own words -->
One person plus an LLM coding agent (Claude). The agent wrote the Modal
pipeline, diagnosed the failures, and drafted this page; Bryant chose the
scroll, read the axial slices, approved every spend, and wrote the verdict
sentence. Every claim traces to a log or commit in this repo.

License: MIT.
