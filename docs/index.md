# The Spohn Book

`spohnbook`: the map of a suite of JAX libraries for exoplanet direct imaging, the
scientific conventions they share, and examples assembled across them.

The libraries cover orbits, astrophysical scenes, optics and coronagraphs,
exposure times, image simulation, post-processing and detection, Bayesian
inference, observation planning and observing campaigns. Each is developed and
released on its own, with its own documentation. This site is where they are
laid out together.

```{figure} figures/library-graph-light.svg
:class: only-light
:name: fig-library-graph

Every public library in the suite and what it imports. Arrows point from a library to the library it depends on; color separates the science libraries from the visualization libraries. hwoutils (constants and conversions) and hwostyle (the plot style) are imported by nearly everything, so their edges are left out.
```

```{figure} figures/library-graph-dark.svg
:class: only-dark

Every public library in the suite and what it imports. Arrows point from a library to the library it depends on; color separates the science libraries from the visualization libraries. hwoutils (constants and conversions) and hwostyle (the plot style) are imported by nearly everything, so their edges are left out.
```

- [Start here](start-here.md) picks the library for the question you are
  asking and lists what every library assumes.
- The [map](map.md) lists every library by layer, what it owns, what it depends
  on, and where its code, documentation and releases live.
- The [conventions](conventions/index.md) are the units, signs, frames, time
  scales, radiometric measures, detector experiment and reporting law that every
  boundary between two libraries must satisfy, with the fixtures that test them.
- The [examples](examples/index.md) run through several libraries at once and
  are executed when this site is built, so a broken boundary between two
  libraries shows up here first.
- The [library conventions](library-conventions.md) describe how a library in
  the suite is built, tested, documented and released.

```{toctree}
:maxdepth: 1
:hidden:

start-here
map
conventions/index
examples/index
library-conventions
```
