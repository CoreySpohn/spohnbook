# Figure atlas

*Draft contract: the choices marked pending or proposed are open.*

The first seven figures explain the contracts independently of the current library
implementations. Each is available in light and dark styles as PNG, SVG and PDF.
The plots come from Python; the quantity-flow and stage diagrams come from D2
(the D2 diagram language). They are teaching figures, not evidence about any
library's current behavior.

## Atlas

| Figure | Question answered | Light vector export | Source |
|---|---|---|---|
| 1. Geometry and RV | Which side is illuminated, and which stellar velocity is positive? | `hwo-conventions-geometry-light.pdf` | `geometry()` in `tools/build_conventions_figures.py` |
| 2. Time coordinates | How do numeric origin, elapsed duration and precision differ? | `hwo-conventions-time-light.pdf` | `time_coordinates()` in `tools/build_conventions_figures.py` |
| 3. Quantity flow | What changes between a source spectrum, photons, electrons and a reported measurement? | `hwo-conventions-pipeline-light.pdf` | `docs/conventions/figures/source/pipeline.d2` |
| 4. Pixels and roll | Where is zero, which way does a rolled source move, and what should refinement conserve? | `hwo-conventions-pixels-light.pdf` | `pixels()` in `tools/build_conventions_figures.py` |
| 5. Reporting and information | What is observed on a nondetection, and which likelihood/information belongs to it? | `hwo-conventions-measurement-light.pdf` | `docs/conventions/figures/source/measurement.d2` |
| 6. Integration stages | Which stages (conventions and boundary anchors) gate the fixed campaign, and which extensions (adaptive choice, images and IFS, and ensembles and external references) can proceed separately? | `hwo-conventions-roadmap-light.pdf` | `docs/conventions/figures/source/roadmap.d2` |
| 7. Color roles | Which colormap encodes which image quantity, and which palette hue and second channel belong to which plotted entity? | `hwo-conventions-color-light.pdf` | `color_roles()` in `tools/build_conventions_figures.py` |

### Figure 1: direction before formula

```{figure} figures/hwo-conventions-geometry-light.svg
:class: only-light
:name: fig-atlas-geometry

Geometry and radial velocity: which side of the planet is illuminated, and which stellar velocity is positive.
```

```{figure} figures/hwo-conventions-geometry-dark.svg
:class: only-dark

Geometry and radial velocity: which side of the planet is illuminated, and which stellar velocity is positive.
```

The observer is to the right. Full/dark labels concern illumination and ignore
occultation and coronagraph visibility. The projected axis does not choose a
north/east basis. The middle panel evaluates two elementary Lambert expressions,
showing why testing only quadrature misses a supplementary-angle error. The RV
panel uses barycentric reflex motion; arrows are schematic and not mass-scaled.

### Figure 2: a number is not an epoch

```{figure} figures/hwo-conventions-time-light.svg
:class: only-light
:name: fig-atlas-time

Time coordinates: how numeric origin, elapsed duration and floating-point precision differ.
```

```{figure} figures/hwo-conventions-time-dark.svg
:class: only-dark

Time coordinates: how numeric origin, elapsed duration and floating-point precision differ.
```

The top panel changes numerical origin on the same TDB scale and at the same
reference position. It is not a UTC/TAI/TDB conversion or a barycentric light-time
correction. The duration example uses the Gregorian calendar beginning in 2000
and compares it with 365.25-day elapsed years. The final panel uses actual NumPy
float32 spacing, including the nearby-relative-epoch improvement; it is not a
claim that all mission computations currently run in float32.

### Figure 3: the physical quantity changes at each boundary

```{figure} figures/hwo-conventions-pipeline-light.svg
:class: only-light
:name: fig-atlas-pipeline

Quantity flow: what changes between a source spectrum, photons, electrons and a reported measurement.
```

```{figure} figures/hwo-conventions-pipeline-dark.svg
:class: only-dark

Quantity flow: what changes between a source spectrum, photons, electrons and a reported measurement.
```

Arrows carry quantities, not imports. The diagram shows the intended composition
and names scientific owners, without asserting their current adapters pass the
contracts. The real assumed-model inference also receives its own model/prior
and response configuration, described in the chapter; no private truth is an
analysis input. Numerical electron rates, random readouts and summaries remain
distinct products even if a scalar implementation combines some calculations.

### Figure 4: coordinate and measure changes are separate

```{figure} figures/hwo-conventions-pixels-light.svg
:class: only-light
:name: fig-atlas-pixels

Pixels and roll: where zero is, which way a rolled source moves, and what refinement should conserve.
```

```{figure} figures/hwo-conventions-pixels-dark.svg
:class: only-dark

Pixels and roll: where zero is, which way a rolled source moves, and what refinement should conserve.
```

The pixel example uses the proposed geometric-center default; imported optical
center metadata can differ. The roll example assumes a declared x-right/y-up sky
chart and identity orientation at zero roll, not a universal east/north mapping.
The flux example has uniform brightness and a fixed square field; each correct
integrated pixel becomes smaller under refinement while the total stays fixed.
The dashed curve illustrates what treating density samples as pixel rates would
do. It is not a new GraterDisk execution.

### Figure 5: the record defines the inference problem

```{figure} figures/hwo-conventions-measurement-light.svg
:class: only-light
:name: fig-atlas-measurement

Reporting and information: what is observed on a nondetection, and which likelihood and information belong to it.
```

```{figure} figures/hwo-conventions-measurement-dark.svg
:class: only-dark

Reporting and information: what is observed on a nondetection, and which likelihood and information belong to it.
```

This is specifically the report-only-on-detection profile. The retained
measurement likelihood is joint with the reporting event. A selected-sample
analysis or forced-photometry experiment requires its own contract. The bottom
information identity assumes the same record is produced under every parameter
value, including any retained response metadata.

### Figure 6: integration follows the dependencies

```{figure} figures/hwo-conventions-roadmap-light.svg
:class: only-light
:name: fig-atlas-roadmap

The six adoption stages (conventions, boundary anchors, fixed campaign, adaptive choice, images and IFS, and ensembles and external references): which gates precede the fixed campaign, and which extensions can proceed separately.
```

```{figure} figures/hwo-conventions-roadmap-dark.svg
:class: only-dark

The six adoption stages (conventions, boundary anchors, fixed campaign, adaptive choice, images and IFS, and ensembles and external references): which gates precede the fixed campaign, and which extensions can proceed separately.
```

The first fixed campaign needs certified conventions and the boundaries it
actually uses. Adaptive, image/IFS and ensemble/external work then have separate
gates. A scalar fixed-schedule ensemble does not require a physical IFS or an
adaptive policy. Individual extended applications inherit the gates for every
capability they use.

## Dust figures

The [dust models chapter](dust-models.md) carries five further figures, built with eyepiece from the analytic fixtures in `tools/dust_reference_cases.py`. They are teaching figures and one negative control; none executes a library dust model.

| Figure | Question answered | Light vector export | Status |
|---|---|---|---|
| 8. Rays and scattering angle | How can the same cloud be viewed from inside and outside, and which angle does the phase function take? | `dust-geometry-light.pdf` | Analytic fixture and schematic |
| 9. Radiance and pixel flux | Why does resolution change a pixel value but not the integral? | `dust-sampling-light.pdf` | Analytic fixture |
| 10. Spectral boundary | Where do wavelength, band, solid angle, zero point and QE enter? | `dust-radiometry-light.pdf` | Synthetic spectrum and schematic |
| 11. Amplitude identifiability | Why can two parameter vectors make the same image? | `dust-identifiability-light.pdf` | Analytic fixture |
| 12. Sign control | What does a negative line-of-sight weight look like on log and signed displays? | `dust-sign-control-light.pdf` | Negative control (deliberately wrong kernel) |

Rebuild them from the repository root with

```sh
python tools/build_dust_figures.py
```

which needs NumPy, Matplotlib, hwostyle, hwoutils and eyepiece 0.4.0 or later, downloads nothing, and writes 30 exports (five figures, light and dark, PNG/SVG/PDF) plus `dust-figure-manifest.json`. The manifest records each figure's question, status, fixture parameters, quantity and normalization; the source hashes; the spohnbook, eyepiece and hwostyle revisions; package versions; and a SHA-256 for every export. Each figure carries an eyepiece provenance stamp naming the script and source hash. Physical radiance is checked for finite, nonnegative values before any logarithmic display; only the sign-control panel bypasses that check, deliberately.

## Explainer diagrams

The chapters open their scientific explanations with physical overview diagrams: the objects, planes and measurements a reader should be able to point at before the first equation. Each comes from one module in `tools/explainers/`, which renders a light and a dark documentation still (PNG for the page, PDF as the vector export), a 16:9 slide still in `talks/stills/`, and, where motion teaches the mechanism, light and dark documentation videos in `docs/_static/explainers/` plus a talk video in `talks/animations/`. Speaker notes are in `talks/notes/`.

| Diagram | Question answered | Home | Source |
|---|---|---|---|
| Observer construction, orbit elements, phase epochs; viewpoint and orbit videos | Where are the angles measured, from which axis and in which direction, and how does the orbit become the observed offset? | [Geometry and time](geometry-time.md) | `tools/explainers/d02_orbit_geometry.py` |
| Radiometric collection, and the four-pixel reference | Which operation turns a density over wavelength, solid angle and area into the count in one pixel? | [Radiometry](radiometry-detectors.md), [reference case](../examples/photon-to-electron-reference.md) | `tools/explainers/d03_radiometric_collection.py` |
| Optical planes and an OPD perturbation; plane-tour video | Where do fields and images live, and which steps keep complex amplitudes versus produce an intensity? | [Optics](optics-images.md) | `tools/explainers/d04_optical_planes.py` |
| Acquisition schedule; accumulation video | Which terms grow with live time, which occur per frame or read, and why do live and elapsed time differ? | [Radiometry](radiometry-detectors.md) | `tools/explainers/d07_detector_acquisition.py` |
| Experiment, record and model; fit and forecast timeline | Which objects exist in the simulated world, which are measured, and which are inferred? | [Measurements and records](inference-records.md), [fit and forecast](../examples/fit-and-forecast.md) | `tools/explainers/d08_experiment_record_model.py` |

They are schematic teaching figures. Where a panel shows a computed quantity, the module's tests recompute every printed number independently of the libraries. Rebuild one module, or all of them, from the repository root with

```sh
python tools/build_explainer_figures.py --only d04
```

Each module writes its own manifest, `docs/conventions/figures/explainer-manifests/<module>.json`, with the caption, alternative text, scientific parameters, source hash, package versions and a SHA-256 for every output.

## Rebuild and provenance

Install the plotting dependencies with `pip install numpy matplotlib hwostyle`
and the renderers `d2` and `rsvg-convert` from their own distributions. From the
repository root, run:

```sh
python tools/build_conventions_figures.py
```

The script `tools/build_conventions_figures.py` in this repository renders the
matplotlib figures and the D2 diagrams in `docs/conventions/figures/source/`
into `docs/conventions/figures/` and writes `conventions-figure-manifest.json`.
It needs NumPy, Matplotlib, hwostyle, D2 with ELK layout, and rsvg-convert. The
command does not download instrument data or execute expensive physical
simulations. It writes all 42 exports, and the manifest records source hashes,
renderer versions, numeric time/precision anchors and the output inventory.
Plain D2 labels permit vector PDF export through rsvg-convert without losing
text to unsupported SVG foreign objects.

The D2 sources live in `docs/conventions/figures/source/`. Source files are the
editable authority. Never repair a rendered figure by painting over a label.

## Contributing figures

Add a figure only when it explains something better than its equations or table
alone. When one is warranted, follow these rules:

- Write the caption first: physical question, assumptions, quantity/units, what
  constitutes agreement, and what the figure cannot establish.
- Keep source and output together with exact input/configuration identity. A
  measured or simulated result must trace to a saved result artifact; an
  illustrative calculation must say so.
- Draw directions, origins and measurement intervals explicitly. Distinguish
  data flow, Python dependencies and prerequisite arrows in the caption.
- Use hwostyle and redundant markers/line styles. Labels must remain readable at
  the intended page/slide size; inspect the actual light and dark renders.
- For images, show physical coordinates, pixel edges and units. For a response,
  show its domain and absolute normalization. For covariance, label both axes and
  state conditioning/marginalization.
- Include residuals or an error/convergence panel in every numerical
  comparison. An attractive normalized image is not absolute-scale evidence.
- Preserve PDF/SVG for publication and PNG for previews. Export variants from one
  source so the scientific content cannot drift between talk and paper versions.
