# October 2026 Progress Prize draft

Prepared against the [October form](https://docs.google.com/forms/d/e/1FAIpQLSc4flEfgK2nyjoczz2_U_XrIGMlgrnSknWatLqrFPnbtKfZwg/viewform)
and [current criteria](https://scrollprize.org/prizes#progress-prizes), checked
October 18, 2026 (Chicago). The deadline listed is October 31, 2026 at 11:59pm
Pacific. This document has not been submitted.

## Identity and contribution URL

Name: Bryant Le. Individual project, developed with AI coding assistance.
Contact information, optional Discord identity and acceptance of the terms must
be supplied by Bryant when submitting.

The public contribution URL must point to the actual published release commit
after publication. The repository is
[bnleft/first-light-pherc0211](https://github.com/bnleft/first-light-pherc0211);
the new tool is in `tools/render-preflight`. It is currently a local release
candidate and should not be represented as available on the repository's public
default branch until published.

## Contribution description for the form

Render preflight is a reusable CPU-only diagnostic for surface-conditioned CT
slabs. It accepts local Zarr v2 and unsharded v3 arrays and outputs an auditable
JSON summary and streaming per-tile CSV before ink inference.

The tool addresses ambiguities encountered in our October PHerc. 0211 work:
absent chunks return fill values, which can look identical to stored zeros;
positive-only layer means compare different pixel populations; and first-maximum
selection assigns tied or constant profiles to an arbitrary depth. It separates
storage populations, computes common-support depth means, and splits peak mass
among exact maxima while retaining constant profiles separately. OME-Zarr 0.4/0.5
axis declarations are checked, and input metadata hashes and denominators are
preserved. An integration example blocks a subsequent inference command when
requested basic finite/variation checks fail, including NaN implicit fill.

Validation uses the public PHerc. 0139 w035 render on scan 20250728140407,
Paris4 w046–052 on scan 20260411134726, and PHerc. 1667 w013 on scan
20251217075048. Nine fixed convenience windows contain 294,912 profiles and
16,220,160 samples. Every window matches direct chunk decoding and a full
scalar ray/peak reference. Six lossless layout conversions preserve those
results, exercising compressed v2 depth chunks and v3/OME-Zarr 0.5 uint16
arrays. Forty-seven regression/workflow cases pass under locked dependencies
on Python 3.11 and 3.12. Sources, windows, response hashes and reproduction
commands are included.

The measured practical advantage is 28× or 109× fewer chunk reads and encoded
bytes at the local Zarr store boundary than a layer-at-a-time diagnostic on these
full-depth source chunks, with identical mean profiles. These are not measured
cloud requests or physical disk traffic. This makes storage/depth QA cheaper and exposes inputs
that warrant investigation before model inference. It packages earlier
project-specific diagnostics into a model-independent tool rather than requiring
a contributor to reproduce our cloud workflow.

The historical PHerc. 0211 case study explains why the distinction is useful:
71.45% of profiles in one investigated window were implicit fill, while over two
million additional profiles were stored zeros. That context is explicitly
separate from the new reproducible public benchmark. We do not claim improved
surface placement, new letters, a universal runtime speedup, or community adoption
that has not occurred. The September workflow and its award are prior work;
this submission concerns the October diagnostic tool and validation.

## Evidence to attach or link after publication

- Tool README, integration example and MIT license.
- `examples/cross_scroll_plan.json`, `cross_scroll_manifest.json` and
  `cross_scroll_results.json`.
- Tests, lockfile, release artifacts and their SHA256 manifest.
- Historical case study, clearly labeled as motivating context.
- Hosted CI and user feedback only when there are actual public results to cite.

The current rules favor early availability, actual use and actionable real-data
results. The release candidate demonstrates accounting, workflow behavior and
diagnostic I/O benefits; community use and improved unwrapping/ink output remain
unproven. Those limits should remain in the submission.
