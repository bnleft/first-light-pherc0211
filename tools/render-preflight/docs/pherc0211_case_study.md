# PHerc. 0211: zeros with different provenance

The historical October 12 depth investigation sampled a 28 × 3,720 × 8,192
rendered window from PHerc. 0211's outer windings. Its 30,474,240 depth profiles
included **21,774,336 profiles of absent-chunk fill** and **8,699,904 profiles
from materialized chunks**. Of the latter, **2,184,788 were stored all-zero
profiles**. A zero read from the slab did not identify which class it belonged to.

That distinction matters when interpreting a weak or negative ink prediction.
**71.45% of profiles were implicit fill**, which is compatible with a deliberately
sparse render and does not itself prove lost data. Treating those fill values as
observed blank papyrus would overstate what was inspected. Conversely, removing
all zeros would also remove genuine values in materialized chunks.

The frozen historical geometry-included population illustrates why denominators
must be explicit: among its 8,232,244 profiles, 789,220 were missing fill,
1,203,843 were stored all-zero and 6,239,181 had a positive uint8 sample.
Positive intensity is not a papyrus or ink classifier. The original analysis
preserved those groups instead of treating an apparently blank raster as evidence
of no ink throughout the window.

The new CLI makes storage provenance, constant/zero profiles and depth maxima
available in a model-independent JSON/CSV report. A user can inspect sparse areas
and choose which input populations need investigation before another inference.
The optional workflow guard checks basic finite/variation conditions; it does not
judge the anatomical validity of a mesh or identify the correct ink depth.

The [historical excerpt](../examples/pherc0211_historical_excerpt.json) contains
the saved scalar counts, source report hash and count identities. It comes from
the October 12 analysis, **not a new Render preflight execution**. The historical
geometry mask is not supported by the new CLI. The full source report remains in
the local research workspace; this excerpt is motivating context rather than an
independently reproducible public benchmark.

The new tool's reproducible validation uses separately pinned public renders from
PHerc. 0139, Paris4 and PHerc. 1667. It claims correct storage/profile accounting
and lower diagnostic I/O, without claiming improved surface placement or recovered
letters. This follow-up is distinct from the September workflow already awarded
a Progress Prize.
