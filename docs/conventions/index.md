# HWO scientific conventions handbook

*Draft contract: the choices marked pending or proposed are open.*

This is the entry point for connecting the suite's libraries without changing
the meaning of a physical quantity, observation or posterior along the way. It
explains the definitions, shows the transformations, and names the evidence
required before a composed scientific workflow is trusted.

Current implementations can disagree in ways that component tests do not
expose; the [limitations page](../evidence/limitations.md) records the reports.
This handbook states the candidate contracts, with the scientific choices
tracked in the [decision register](../profiles/decisions.md). It does not certify current
library behavior or silently adopt every proposed axis, unit or parameter name.

(conventions-how-to-read)=
## How to read this handbook

This handbook holds the shared conventions that every boundary between the
suite's libraries must satisfy: units, signs, frames, time scales, radiometric
measures, the detector experiment, and the reporting law. Each chapter has
definitions, public reference tables, an independent worked example, named
acceptance fixtures, and a pointer to the findings about current
implementations, which the [limitations page](../evidence/limitations.md)
records with owners, stages, status and evidence.

The chapters refer to six adoption stages, in order:

- **The conventions stage**: the conventions and the independent fixtures with
  their tolerances are written down (this handbook).
- **The boundary-anchor stage**: each boundary anchor is implemented and
  passing in the library that owns it.
- **The fixed-campaign stage**: one fixed observing campaign runs end to end on
  those boundaries.
- **The adaptive-choice stage**: adaptive scheduling.
- **The images-and-IFS stage**: the image and integral-field paths.
- **The ensembles-and-references stage**: ensembles and comparisons against
  external reference codes.

A finding's stage is the earliest stage at which its repair is required.

| Reader's question | Start here |
|---|---|
| Why can an orbit look right while brightness or RV is reversed? | [Geometry and time](geometry-time.md) |
| What exactly does a flux, contrast, zodi value or count rate mean? | [Radiometry, detector counts, and spectral measurements](radiometry-detectors.md) |
| Where is a pixel, how does roll act, and what does a PSFlet (the detector image of one lenslet's pupil) contain? | [Optical fields, image coordinates, and IFS products](optics-images.md) |
| What data were actually reported, and what probability model interprets them? | [Measurements, probability, and records](inference-records.md) |
| What does a dust brightness, a zodi level or an exozodi raster mean, and from which observer? | [Dust models: radiance, geometry, and normalization](dust-models.md) |
| What do we build first, and what closes each gate? | The adoption stages above, the integration-stages figure in [the figure atlas](figures.md), and the [limitations page](../evidence/limitations.md) for each chapter |
| Which colormap encodes this quantity, and which hue belongs to this plotted entity? | [Figures and color](figures-and-color.md) |
| Which figures can I reuse or regenerate? | [the figure atlas](figures.md) |
| Where is the source evidence for a current failure? | The [limitations page](../evidence/limitations.md): every finding, its owner, the stage that requires its repair, its status and its evidence |

Every chapter starts with physical meaning, then equations and units, worked
examples, adapter obligations, and acceptance fixtures. Its closing section
links to the chapter's findings on the limitations page, which assigns each to
an owner and a gate. The 81 findings of the first five chapters, and the dust
chapter's further nine, include overlaps, positive anchors and future gaps; they
are not independent bugs.

## Three distinctions to preserve

**A quantity is not its representation.** A relative state, barycentric state,
east/north tangent offset and detector location can describe one system while
using different origins and bases. A time value needs more than a number and
"days". A spectral density needs its integration measure.

**A model is not a measurement.** An intrinsic planet contrast, expected
photoelectron count, extracted estimate and selected detection are different
objects. Each transformation has an owner, assumptions and an error budget.
Posteriors use the acquired record and an assumed response; private simulation
truth remains inaccessible to inference and policy.

**A definition is not verification.** Writing the correct formula does not show
that every consumer implements it. A same-code round trip can preserve a shared
mistake. An independent sign or scale anchor, followed by a producer-to-consumer
test, supplies the missing evidence. Cross-code agreement is benchmarking;
measured-data validation is a separate claim.

## Contract vocabulary and status

Use these labels consistently in documentation, products and reviews:

| Label | Meaning |
|---|---|
| Physical identity | Follows from explicitly stated geometry, probability or model assumptions |
| Inherited design | A choice this handbook inherits rather than proposes, such as TDB for the first physical profile; implementation may still be absent |
| Proposed profile | A coherent recommended choice awaiting the named decision and migration assessment |
| Implemented | Present in a named version of the producer or consumer |
| Verified boundary | Producer and consumer pass the specified independent fixtures at recorded revisions |
| Validated domain | Agreement with appropriate independent measured evidence over a stated domain |
| Unsupported | Rejected before numerical execution; not silently approximated or omitted |

A convention profile is a small, versioned declaration of related meanings.
Suggested names such as `observer-toward-v1` are illustrative until the
conventions stage freezes the identifier and fields. Keep convention versions
separate from package versions, data-schema versions and calibration revisions.
A deprecation or conversion may span several package releases without changing
the underlying physical meaning.

## Decision register

The scientific choices these chapters depend on, with their recommendations,
states, owners and the implementation evidence each requires, are recorded once,
in the [decision register](../profiles/decisions.md). "Pending" there means the
alternatives and recommendation are presented but the contract is not frozen, and
implementation must not assume agreement. The [convention profiles](../profiles/index.md)
state which clauses a named profile groups and whether it is proposed or adopted.

## What travels across a boundary

A lightweight manifest accompanies an exchanged product or response. It records
quantity and unit; coordinate/array order and shape; frame and origin; epoch
format/scale/reference/location; integration measure and support; normalization
and reference planes; covariance and nuisance treatment; response/calibration
revision; acquisition/product/parent identity; convention/schema versions; and
source provenance. Only applicable fields are required; missing required meaning
is a capability error, not a guessed default.

Numerical kernels can retain efficient native units and JAX arrays. Conversion
and interpretation happen once at a named adapter. Domain owners retain their
scientific definitions; hwoutils supplies appropriate shared constants and
conversions. A new universal units or schema framework is not a prerequisite.

(conventions-contributing)=
## Contributing to this handbook

When you repair or add a boundary, ship all of the following with it:

1. A short explanation of the physical observable and its assumptions, linked
   from the producing and consuming libraries' public docs.
2. A reference table listing names, units, signs, axes, normalization, support and
   uncertainty semantics, including supported legacy profiles.
3. A worked dimensional or signed example with an independently calculated
   answer, plus a negative control that exposes the previous ambiguity.
4. An executable example at the public API and a cross-library fixture that does
   not derive its expected value from the implementation being tested.
5. A figure where direction, measure, timing, selection or covariance needs visual
   explanation, with a caption that states assumptions and limits.
6. Migration/release notes when a scientific interpretation changes, and an
   evidence record naming source revisions, data and numerical tolerances.

Write library documentation in the existing Sphinx/MyST-NB system: explanation
pages, compact reference tables and executable MyST tutorials. Link the common
definition in this handbook rather than copying subtly different equations into
every package. This handbook owns the shared definitions and decisions; each
library owns its supported API documentation. Freeze the common contract before
you publish implementation-specific guarantees.

Build figures from editable Python or D2 (the D2 diagram language) source, with
light and dark PNG/SVG/PDF exports. Give them redundant labels and line styles
rather than relying on color alone. Label every numerical demonstration as an
independent teaching example, evidence for a current finding, or a future
acceptance requirement. The existing figures are documented in
[the figure atlas](figures.md).

When a meaning changes, update the profile/version, conversions, affected
fixtures, reference table, worked example and figure together. A parameter
rename alone does not migrate old physical assumptions. Re-run the affected
boundary gates; do not claim the full campaign is certified from one passing
component suite. Keep the evidence behind each finding intact and link its
closure to the repairing commit and acceptance result.

```{toctree}
:maxdepth: 1

geometry-time
optics-images
radiometry-detectors
inference-records
dust-models
figures-and-color
figures
```
