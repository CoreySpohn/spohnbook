# References

Every source the book cites is listed once below, with the clauses that cite it
and the locator each uses. The list is generated from the source catalog,
`docs/_data/handbook.yaml`, at every build; see
{ref}`contributing <contributing-grounding>` for how a source is added.
A citation here shows where a statement comes from. Whether the source supports
the statement as written is a matter for scientific review, which the build does
not perform.

(references-sources)=
## Sources

```{include} _generated/references.md
```

(references-clause-bases)=
## What each clause rests on

Each catalogued clause is listed with its cited sources and locators, the
derivation the book supplies, and the decisions it depends on.

```{include} _generated/clause-bases.md
```

(references-source-gaps)=
## Known source gaps

The statements below are used by the book but have no verified primary source
locator in this edition. They are not reviewed until the gap is closed.

| Clause | Statement | Gap |
|---|---|---|
| {ref}`radiometry-quantities`, {ref}`radiometry-jy-photons`, {ref}`photon-electron-reference-experiment` | 1 Jy = $10^{-26}$ W m$^{-2}$ Hz$^{-1}$ (the jansky is not an SI unit) | No standards document was inspected for this edition. |
| {ref}`radiometry-etc-forecast` | The detection and characterization noise terms of the {ref}`Nemati (2014) <source-nemati2014>` formulation | The 2014 paper's passage was not inspected; the terms were checked against {ref}`Nemati et al. (2023) <source-nemati2023>` and the EXOSIMS source. |
| {ref}`geometry-illumination` | Lambert phase function | Cited to a secondary source; the primary (Sobolev 1975) was not inspected. |
| {ref}`geometry-radial-velocity` | Direction of the line of sight in the literature RV equation | The cited passage does not state whether its line of sight points toward or away from the observer. |
| {ref}`geometry-public-units` | Tangent-plane offsets and $\mu_{\alpha*}$ conventions | The FITS WCS projection paper and the Hipparcos and Gaia catalogue documentation were not inspected; the clause rests on the cited uses and on the first-order expansion. |
| {ref}`optics-coherent-phase` | Noll ordering and normalization of Zernike polynomials | The primary (Noll 1976) was not inspected. |
| {ref}`optics-coherent-phase` | Pairing of the Fourier-transform sign with the time factor | No single primary text states the pairing; the book records it as its own choice. |
| {ref}`optics-roll-rotation` | Position angle measured from north through east | No IAU standard text was inspected; the clause cites two uses. |
| {ref}`optics-ifs-products` | "PSFlet" as the detector image of one lenslet's pupil | The cited sources describe the PSFlet as the lenslet's monochromatic spot; the book's gloss is unsourced. |
| {ref}`records-campaign-causality` | Error inflation under repeated threshold crossing | Cited through a later review; the primary (Armitage, McPherson and Rowe 1969) was not inspected. |

Several choices appear in the chapters as proposals without their own entry in
the [decision register](profiles/decisions.md): the `illumination_angle_rad`
naming and the intrinsic-reflection or apparent-contrast grid label
({ref}`geometry-illumination`), the proposed public units
({ref}`geometry-public-units`), the inherited TDB baseline
({ref}`geometry-time-encoding`), and the ensemble opacity heuristic
({ref}`color-time-ensembles`). They remain proposals.
