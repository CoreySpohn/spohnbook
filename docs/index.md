# The Spohn Book

`spohnbook`: the map of a suite of JAX libraries for exoplanet direct imaging, the
scientific conventions they share, and examples assembled across them.

The libraries cover orbits, astrophysical scenes, optics and coronagraphs,
exposure times, image simulation, post-processing and detection, Bayesian
inference, observation planning and observing campaigns. Each is developed and
released on its own, with its own documentation. This site is where they are
laid out together.

(book-scope)=
## Scope

This book records the conventions shared by one suite of open-source libraries
for exoplanet analyses related to the Habitable Worlds Observatory (HWO). It is
written for the people who use the suite, contribute to it, or connect another
code to it. It is not an HWO project document: it states no mission requirement,
carries no mission approval, and makes no determination of compliance with a NASA
standard. Three kinds of statement appear in it and are kept apart. A study
assumption is an input chosen for a particular analysis, such as an exploratory
architecture case, and is cited with its source and version. A proposed convention
is this book's own recommendation, and it stays proposed until the
[decision register](profiles/decisions.md) records a decision. An authoritative
external document, such as a mission document, a standard or a published paper,
is cited where it is used and is not superseded by anything written here.

```{figure} figures/library-graph-light.svg
:class: only-light
:name: fig-library-graph

Every public library in the suite and what it imports. Arrows point from a library to the library it depends on; color separates the science libraries from the visualization libraries. hwoutils (constants and conversions) and hwostyle (the plot style) are imported by nearly everything, so their edges are left out.
```

```{figure} figures/library-graph-dark.svg
:class: only-dark

Every public library in the suite and what it imports. Arrows point from a library to the library it depends on; color separates the science libraries from the visualization libraries. hwoutils (constants and conversions) and hwostyle (the plot style) are imported by nearly everything, so their edges are left out.
```

(book-reading-routes)=
## Reading routes

- **Scientific foundations**: the [conventions chapters](conventions/index.md)
  explain what each quantity means, derive the identities, cite the sources of
  borrowed models and state their limits.
- **Convention profiles and decisions**: the [profiles](profiles/index.md) list
  which meanings a named profile adopts and its status, and the
  [decision register](profiles/decisions.md) is the single record of every open
  and decided scientific choice.
- **Tutorials**: the [examples](examples/index.md) execute the libraries together
  and state their restrictions and evidence scope before the code.
- **Evidence**: the [evidence pages](evidence/index.md) separate a successful
  build, an independent reference case, a known limitation and measured-data
  validation, and say how to reproduce an edition.
- **Publication and contribution**: [contributing](contributing.md) sets the
  grounding and review rules, [citing](citing.md) explains how to cite an
  edition or a clause, and [releases](releases.md) records editions, errata and
  migrations.

## Pages

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
profiles/index
examples/index
evidence/index
contributing
citing
releases
references
library-conventions
```
