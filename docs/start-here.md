# Start here

This book covers the conventions of one suite of libraries for HWO-related
exoplanet analyses and is not a mission requirements document (see the
{ref}`scope <book-scope>`). Pick the library by the question you are asking. Each one installs on its own
from PyPI and has its own documentation; the [map](map.md) shows how they fit.

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
