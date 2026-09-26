# Start here

This book covers the conventions of one suite of libraries for HWO-related
exoplanet analyses and is not a mission requirements document (see the
{ref}`scope <book-scope>`). Pick the library by the question you are asking. Each one installs on its own
from PyPI and has its own documentation; the [map](map.md) shows how they fit.

```{figure} conventions/figures/explainer-d01-overview-light.png
:class: only-light
:name: fig-explainer-d01-overview
:alt: A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

The physical observation, from a distant system to the products an analysis reports. Top row, physical light (solid arrows with filled heads): a host star, and a planet on an orbit in the midplane of an exozodiacal dust disk, seen side on, with the observer to the right along the positive z direction of the proposed observer-toward-positive-Z profile ({ref}`geometry-illumination`); the planet is on the far side, so the observer sees mostly its lit hemisphere. Reflected planet light, scattered dust light and starlight cross an interstellar distance that is not drawn to scale and enter a generic telescope aperture and coronagraph that end on a detector ({ref}`radiometry-response-ownership`), together with local zodiacal light from the dust cloud around the observer, a foreground along the same line of sight (the same scattering process seen from inside the cloud, {doc}`/conventions/dust-models`). Bottom row, data (hollow block arrows): a synthetic raw frame in which planet light, stellar leakage, exozodiacal light and the uniform local zodiacal floor share the same pixels; one record per visit $k$, with its epoch $t_k$, reporting event $D_k$, east and north offsets $(\xi_k,\eta_k)$, their covariance $C_k$ and its calibration revision ({ref}`records-reporting-law`, {ref}`records-covariance`, {ref}`records-identities`); and, in the inferred-orbit sky panel, posterior orbit draws given the positions measured on four visits, drawn with crosshairs of plus and minus one standard deviation, the visit shown in the raw frame ringed. The prior takes eccentricity uniform on [0, 0.6], the argument of periapsis and the mean anomaly at epoch uniform on the circle, and the semimajor axis, inclination and node angle Gaussian about the simulated values (standard deviations of 15 percent, 15 degrees and 15 degrees); six million prior samples are weighted by the Gaussian likelihood of the measured positions and thirty draws are taken by systematic resampling, so the draws gather along the measured arc and fan out on the unobserved one. Sky offsets follow the proposed observer profile of {ref}`geometry-observer-basis` (north, east, toward the observer, with the node angle measured from north toward east); the library outputs, which label the same rotated components right ascension and declination, are mapped into it by exchanging those two components. The inferred-orbit panel, the one sky panel on this map, is shown east to the left and north up, as its compass marks. The raw frame carries no compass: how detector rows and columns map to east and north depends on the roll and on the pending image-coordinate decision ({ref}`optics-roll-rotation`), so its pixels are placed as the sky would appear at zero roll under one possible mapping. The frame amplitudes are schematic, not a brightness ratio. Each outlined name locates the chapter that defines that part: Geometry and time, Dust models, Radiometry, Optical fields, and Measurements, probability, and records. An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

```{figure} conventions/figures/explainer-d01-overview-dark.png
:class: only-dark
:alt: A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

The physical observation, from a distant system to the products an analysis reports. Top row, physical light (solid arrows with filled heads): a host star, and a planet on an orbit in the midplane of an exozodiacal dust disk, seen side on, with the observer to the right along the positive z direction of the proposed observer-toward-positive-Z profile ({ref}`geometry-illumination`); the planet is on the far side, so the observer sees mostly its lit hemisphere. Reflected planet light, scattered dust light and starlight cross an interstellar distance that is not drawn to scale and enter a generic telescope aperture and coronagraph that end on a detector ({ref}`radiometry-response-ownership`), together with local zodiacal light from the dust cloud around the observer, a foreground along the same line of sight (the same scattering process seen from inside the cloud, {doc}`/conventions/dust-models`). Bottom row, data (hollow block arrows): a synthetic raw frame in which planet light, stellar leakage, exozodiacal light and the uniform local zodiacal floor share the same pixels; one record per visit $k$, with its epoch $t_k$, reporting event $D_k$, east and north offsets $(\xi_k,\eta_k)$, their covariance $C_k$ and its calibration revision ({ref}`records-reporting-law`, {ref}`records-covariance`, {ref}`records-identities`); and, in the inferred-orbit sky panel, posterior orbit draws given the positions measured on four visits, drawn with crosshairs of plus and minus one standard deviation, the visit shown in the raw frame ringed. The prior takes eccentricity uniform on [0, 0.6], the argument of periapsis and the mean anomaly at epoch uniform on the circle, and the semimajor axis, inclination and node angle Gaussian about the simulated values (standard deviations of 15 percent, 15 degrees and 15 degrees); six million prior samples are weighted by the Gaussian likelihood of the measured positions and thirty draws are taken by systematic resampling, so the draws gather along the measured arc and fan out on the unobserved one. Sky offsets follow the proposed observer profile of {ref}`geometry-observer-basis` (north, east, toward the observer, with the node angle measured from north toward east); the library outputs, which label the same rotated components right ascension and declination, are mapped into it by exchanging those two components. The inferred-orbit panel, the one sky panel on this map, is shown east to the left and north up, as its compass marks. The raw frame carries no compass: how detector rows and columns map to east and north depends on the roll and on the pending image-coordinate decision ({ref}`optics-roll-rotation`), so its pixels are placed as the sky would appear at zero roll under one possible mapping. The frame amplitudes are schematic, not a brightness ratio. Each outlined name locates the chapter that defines that part: Geometry and time, Dust models, Radiometry, Optical fields, and Measurements, probability, and records. An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

```{figure} conventions/figures/hwo-conventions-pipeline-light.svg
:class: only-light
:name: fig-start-pipeline

One scene through the stack: the quantities that cross each boundary from an astrophysical scene to a reported measurement, and the library that owns each step. The conventions chapters state what each arrow means.
```

```{figure} conventions/figures/hwo-conventions-pipeline-dark.svg
:class: only-dark

One scene through the stack: the quantities that cross each boundary from an astrophysical scene to a reported measurement, and the library that owns each step. The conventions chapters state what each arrow means.
```

| I want ... | Library | Install |
|---|---|---|
| an orbit propagated to positions, velocities and sky-plane offsets | [orbix](https://orbix.readthedocs.io/) | `pip install orbix` |
| a star, planet and disk scene to simulate, fit or plan against | [skyscapes](https://skyscapes.readthedocs.io/) | `pip install skyscapes` |
| zodiacal and exozodiacal surface brightness with stated conventions | [zodi](https://zodi.readthedocs.io/) | `pip install zodi` |
| a coronagraph's throughput, contrast and off-axis response from a yield input package | [yippy](https://yippy.readthedocs.io/) | `pip install yippy` |
| the hardware description (aperture, coronagraph, detector) a simulator reads | [optixstuff](https://optixstuff.readthedocs.io/) | `pip install optixstuff` |
| an exposure time or count-rate budget | [jaxedith](https://jaxedith.readthedocs.io/) | `pip install jaxedith` |
| a simulated coronagraphic image of a scene | [coronagraphoto](https://coronagraphoto.readthedocs.io/) | `pip install coronagraphoto` |
| an integral field spectrograph forward model and extraction | [coronachrome](https://github.com/CoreySpohn/coronachrome) | `pip install coronachrome` |
| point spread functions and diffraction from a physical optics model | [physicaloptix](https://physicaloptix.readthedocs.io/) | `pip install physicaloptix` |
| wavefront error and wavefront-control residuals | [tiptilt](https://tiptilt.readthedocs.io/) | `pip install tiptilt` |
| detection statistics and post-processing on simulated images | [coronalyze](https://coronalyze.readthedocs.io/) | `pip install coronalyze` |
| a Bayesian fit of an orbit, a disk or an atmosphere, or the value of a future observation | [photomancy](https://photomancy.readthedocs.io/) | `pip install photomancy` |
| the decision vocabulary and policies for choosing the next observation | [planit-py](https://github.com/CoreySpohn/planit-py) | `pip install planit-py` |
| an observing campaign with inference in the loop | [spaceodyssey](https://spaceodyssey.readthedocs.io/) | `pip install spaceodyssey` |
| the plot style and plotting primitives the figures use | [hwostyle](https://github.com/HabitableWorldsObservatory/hwostyle), [eyepiece](https://eyepiece.readthedocs.io/) | `pip install hwostyle eyepiece` |
| the shared constants, unit conversions and image transforms | [hwoutils](https://hwoutils.readthedocs.io/) | `pip install hwoutils` |

Two libraries serve the others rather than a science question:
[ineedvalidation](https://ineedvalidation.readthedocs.io/) declares what evidence
each test provides, and [yieldplotlib](https://yieldplotlib.readthedocs.io/)
compares yields against external yield codes.

## Things every library assumes

- Python 3.11 or newer.
- JAX. The numerical libraries are built on JAX and Equinox; install the JAX
  wheel for your platform first if you want a GPU (the CPU wheel comes with the
  libraries). Enable 64-bit precision with
  `jax.config.update("jax_enable_x64", True)` at the top of a script when you
  need it: the libraries honor the global flag uniformly and do not switch
  precision on their own.
- The [conventions](conventions/index.md): units, signs, frames, time scales,
  radiometric measures and the reporting law that every boundary between two
  libraries satisfies. When a result crosses from one library to another, the
  handbook says what it means.

## Read next

- The {ref}`reading routes <book-reading-routes>` for the difference between the
  scientific foundations, the convention profiles, the tutorials and the
  evidence pages.

- The [map](map.md) for what each library owns and what it depends on.
- The [examples](examples/index.md) for code that runs through several
  libraries at once.
- The [library conventions](library-conventions.md) if you are building or
  contributing to one of the libraries.
