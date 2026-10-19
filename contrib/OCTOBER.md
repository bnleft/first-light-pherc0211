# October contribution: render preflight

The concrete deliverable is [Render preflight](../tools/render-preflight/README.md),
a standalone, CPU-only inspector for rendered Zarr slabs. It converts findings
from our October PHerc. 0211 experiments into a tool another participant can run
on their own render before spending time on ink inference.

The contribution addresses three practical failure modes: interpreting absent
chunk fill as an observation, comparing depth means over changing pixel
populations, and assigning tied/constant profiles to an arbitrary peak layer.
It also avoids repeatedly reading a full-depth chunk once per layer.

The real-data validation covers nine fixed windows across public PHerc. 0139,
Paris4 and PHerc. 1667 renders: 294,912 rays / 16,220,160 samples agree with
directly decoded chunk bytes and a full scalar reference. Six lossless layout
conversions preserve the results. Full-depth source chunks require 28× or 109×
fewer diagnostic reads than a layer-at-a-time baseline.
All 47 regression/workflow cases pass on Python 3.11 and 3.12. The package preserves standard Zarr input and
JSON/CSV output; it does not alter the source.

This matches the organizers' request for better diagnostics and reproducible,
chunked workflows in the [current open problems](https://scrollprize.org/2026_open_problems).
The [Progress Prize criteria](https://scrollprize.org/prizes#progress-prizes)
favor a demonstrated practical advantage, documentation and early release.

The [release notes](../tools/render-preflight/docs/RELEASE_NOTES.md),
[historical case study](../tools/render-preflight/docs/pherc0211_case_study.md),
and [submission draft](../tools/render-preflight/docs/PROGRESS_PRIZE_DRAFT.md)
are prepared. Public release, hosted CI and community use remain external
milestones. Submission should accurately distinguish this October tool
from the September PHerc. 0211 workflow already recognized with an award.
The Paris4 estimator remains research work until it demonstrates real-data
accuracy; it is not the evidence for this contribution.

No release, upstream message, prize submission, or October award is recorded
by this local contribution note.
