# Library map

Each library owns one layer of the problem and imports only from the layers
below it. Arrows in the "depends on" column point downward: a scene library
composes orbits, an image simulator composes scenes and hardware, an inference
library fits scenes, and a campaign library composes all of them without
reimplementing any.

The layers, from the bottom up:

- **Foundation**: shared constants, unit conversions, image transforms, and the
  hardware building blocks (apertures, coronagraphs, detectors) that every
  simulator describes its instrument with.
- **Geometry**: Keplerian orbits, their propagation, and observatory geometry.
- **Scene**: the astrophysical scene (stars, planets, disks, zodiacal light) as
  the object that is simulated, fitted and planned against.
- **Data**: coronagraph performance data from the designers' yield input
  packages.
- **Simulation**: image formation through a coronagraph and an integral field
  spectrograph.
- **Wavefront**: wavefront error and wavefront-control residuals, the speckles
  that the images contain.
- **Exposure time**: count rates and integration times.
- **Analysis**: post-processing and detection statistics on simulated images.
- **Inference**: Bayesian fits of orbits, disks and atmospheres, and the value
  of a future observation.
- **Planning** and **orchestration**: the decision vocabulary, policies, and the
  observing campaigns that close the loop from data to the next observation.
- **Visualization**: the plot style and the plotting primitives shared by all of
  the above.

Stages are the declared maturity of each library: pre-alpha (scaffold or
skeleton), alpha (tagged releases and tests in continuous integration), beta
(documentation site, changelog and at least one dependent library).

```{figure} figures/library-graph-light.svg
:class: only-light
:name: fig-map-graph

Every public library in the suite and what it imports. Arrows point from a library to the library it depends on; color separates the science libraries from the visualization libraries. hwoutils (constants and conversions) and hwostyle (the plot style) are imported by nearly everything, so their edges are left out.
```

```{figure} figures/library-graph-dark.svg
:class: only-dark

Every public library in the suite and what it imports. Arrows point from a library to the library it depends on; color separates the science libraries from the visualization libraries. hwoutils (constants and conversions) and hwostyle (the plot style) are imported by nearly everything, so their edges are left out.
```

```{include} _generated/map-table.md
```
