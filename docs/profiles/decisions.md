# Decision register

This register is the single authoritative record of the scientific choices the book's conventions depend on. Each decision has a stable label that clauses, profiles and citations can point to. Operational task tracking for these decisions is kept outside the book and never restates them.

"Pending" means the documentation presents the alternatives and recommendation;
the contract has not been frozen and implementation must not assume agreement.
Decided items are not reopened without new evidence.

**Profile selection and implementation certification are separate gates.** The
conventions stage freezes definitions, mathematical expectations, the migration
rule and the acceptance design. The table's implementation-evidence column is
then discharged from boundary anchors through ensembles and external
references, as the [limitations page](../evidence/limitations.md) assigns. In particular, the
reporting law decision does not require an already calibrated campaign or
adaptive scheduler.

| Decision | State / owner |
|---|---|
| {ref}`Observer basis and node <decision-observer-basis-and-node>` | Pending; orbix + photomancy + scene adapters |
| {ref}`Meaning of dQE <decision-meaning-of-dqe>` | Pending; optixstuff + jaxedith |
| {ref}`Stellar leakage measure <decision-stellar-leakage-measure>` | Pending; optixstuff + yippy + physicaloptix + jaxedith |
| {ref}`Reporting law <decision-reporting-law>` | Pending; measurement adapter + coronalyze + photomancy |
| {ref}`Acquisition experiment <decision-acquisition-experiment>` | Pending; optixstuff + jaxedith + spaceodyssey (campaign library) / planit-py |
| {ref}`Image coordinates and PSFlet origin <decision-image-coordinates-and-psflet-origin>` | Pending by enabled capability; optics/IFS owners |
| {ref}`Evidence and parameter chart <decision-evidence-and-parameter-chart>` | Proposed contract; photomancy |
| {ref}`Record envelope home <decision-record-envelope-home>` | Decided |
| {ref}`Sampled campaign first <decision-sampled-campaign-first>` | Decided; spaceodyssey |
| {ref}`Tolerances before tests <decision-tolerances-before-tests>` | Method required; domain and verification owners set budgets |
| {ref}`One encoding per quantity <decision-one-encoding-per-quantity>` | Proposed; hwostyle + eyepiece |

Other inherited choices include separate truth and model access, a
refit-from-original-prior baseline, causal availability, exactly-once
acquisition accounting, and explicit unsupported capabilities. Point-valued
configuration is sufficient initially; distributions and epistemic intervals
require distinct typed semantics before execution is enabled. A tuple must not
acquire uncertainty meaning by convention alone.

(decision-observer-basis-and-node)=
## Observer basis and node

**Recommendation.** Retain +z toward observer; candidate right-handed dynamics use (north,east,toward), public astrometry uses (east,north). Preserve an explicitly identified legacy orbix profile during migration. External reference: Savransky 2019, section 2.1 (see the geometry chapter).

**State and owner.** Pending; orbix + photomancy + scene adapters

**Required implementation evidence after profile selection.** Signed 3D basis/state/RV/illumination examples; external element transform; saved-posterior migration policy

(decision-meaning-of-dqe)=
## Meaning of dQE

**Recommendation.** Replace the ambiguous meaning of dQE (the detector's quantum-efficiency degradation factor) with an explicit survival fraction (neutral 1), or an explicit fractional loss (neutral 0); never reinterpret old values silently.

**State and owner.** Pending; optixstuff + jaxedith

**Required implementation evidence after profile selection.** Neutral/degraded cases across all adapters; serialized configuration mapping

(decision-stellar-leakage-measure)=
## Stellar leakage measure

**Recommendation.** Define scalar stellar leakage as a density plus a precisely named averaging/aperture rule; every backend exports that meaning.

**State and owner.** Pending; optixstuff + yippy + physicaloptix + jaxedith

**Required implementation evidence after profile selection.** Absolute image-aperture sum at two samplings and on both backends

(decision-reporting-law)=
## Reporting law

**Recommendation.** Choose the first campaign's actual reporting law, including joint position/flux, selection, nulls, nuisance treatment and covariance.

**State and owner.** Pending; measurement adapter + coronalyze + photomancy

**Required implementation evidence after profile selection.** Normalization, detection frequencies, selected distributions, posterior calibration and always-null information limit

(decision-acquisition-experiment)=
## Acquisition experiment

**Recommendation.** Define the first acquisition experiment: actual read frames, parallel paths, rolls, background/reference estimation, live/occupied/charged time.

**State and owner.** Pending; optixstuff + jaxedith + spaceodyssey (campaign library) / planit-py

**Required implementation evidence after profile selection.** Analytic count/variance budget and independently assembled time/resource ledger

(decision-image-coordinates-and-psflet-origin)=
## Image coordinates and PSFlet origin

**Recommendation.** Name optical center, axis order, pixel measure, native/reference-wavelength grid, PSFlet origin, capture losses and covariance scope. Default generated images to geometric center; honor explicit imported calibration metadata.

**State and owner.** Pending by enabled capability; optics/IFS owners

**Required implementation evidence after profile selection.** Odd/even/rectangular grids, signed rotations, refinement, centroid-once and edge-capture anchors

(decision-evidence-and-parameter-chart)=
## Evidence and parameter chart

**Recommendation.** Distinguish component mass, ELBO and evidence; persist an explicit parameter chart; evidence-based claims reject unavailable/invalid normalization.

**State and owner.** Proposed contract; photomancy

**Required implementation evidence after profile selection.** Analytic evidence, compression/serialization and unit-change fixtures; backend capability table

(decision-record-envelope-home)=
## Record envelope home

**Recommendation.** Keep the initial record envelope in spaceodyssey with domain-owned payloads; extract a separate distribution only after two working consumers demonstrate reuse.

**State and owner.** Decided

**Required implementation evidence after profile selection.** Standalone export/import-to-photomancy plus spaceodyssey reuse of the same records, with dependency isolation

(decision-sampled-campaign-first)=
## Sampled campaign first

**Recommendation.** Build the sampled fixed campaign first; do not require unused moments/integration methods on its adapters. Revisit an analytic/integrate engine with the matched external-reference deliverable.

**State and owner.** Decided; spaceodyssey

**Required implementation evidence after profile selection.** One executable sampled path; unsupported engines reject explicitly; later reference engine has its own capability/evidence gate

(decision-tolerances-before-tests)=
## Tolerances before tests

**Recommendation.** Allocate tolerances to observables before tests; use independent primitives, numerical refinement and calibrated Monte Carlo uncertainty; name external reference profiles.

**State and owner.** Method required; domain and verification owners set budgets

**Required implementation evidence after profile selection.** Recorded error allocation, reference/configuration/source identity, positive and deliberately failing controls

(decision-one-encoding-per-quantity)=
## One encoding per quantity

**Recommendation.** Encode each quantity with one colormap role and each plot entity with one palette role; retire map names from figure code.

**State and owner.** Proposed; hwostyle + eyepiece

**Required implementation evidence after profile selection.** The swatch, grayscale-pair and default-free fixtures
