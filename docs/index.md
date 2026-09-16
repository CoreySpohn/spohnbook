# spohnbook

The map of a suite of JAX libraries for exoplanet direct imaging, the
scientific conventions they share, and examples assembled across them.

The libraries cover orbits, astrophysical scenes, optics and coronagraphs,
exposure times, image simulation, post-processing and detection, Bayesian
inference, observation planning and observing campaigns. Each is developed and
released on its own, with its own documentation. This site is where they are
laid out together.

- The [map](map.md) lists every library by layer, what it owns, what it depends
  on, and where its code, documentation and releases live.
- The [examples](examples/index.md) run through several libraries at once and
  are executed when this site is built, so a broken boundary between two
  libraries shows up here first.

```{toctree}
:maxdepth: 1
:hidden:

map
examples/index
```
