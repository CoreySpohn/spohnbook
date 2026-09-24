# Convention profiles

A convention profile is a small, versioned declaration that groups related
clauses of the book under one identifier, so that a producer and a consumer can
state which meanings they exchange. A profile has one of three states. A
**proposed** profile is a coherent recommendation that no recorded decision has
adopted. An **adopted** profile names the entry in the
[decision register](decisions.md) that adopted it. A **deprecated** profile is
kept, unchanged, so that results produced under it can still be interpreted.

A profile's version is separate from the book's edition number, from every
library's package version, and from data-schema and calibration revisions. A
change of meaning produces a new profile version rather than an edit in place, as
{ref}`contributing <contributing-changes>` describes.

The profile state and the evidence about implementations are also separate. A
profile does not become adopted because one of its reference cases passes, and an
adopted profile is not thereby implemented, verified or validated in any library.
The [evidence pages](../evidence/index.md) report each of those states on its own.

(profiles-table)=
## Profiles

```{include} ../_generated/profiles.md
```

(profiles-photon-electron-reference-v1)=
## photon-electron-reference-v1

The only profile defined so far is deliberately narrow. It describes one
reference experiment: a synthetic source with constant photon spectral density,
converted from Jy to photons with the SI Planck constant, collected by a stated
area over a stated interval and exposure, spread over four equally illuminated
pixels, and converted to electrons by one constant quantum efficiency with no
dark current, clock-induced charge or read noise. The
[photon-to-electron reference example](../examples/photon-to-electron-reference.md)
specifies it completely and runs it through the public library interfaces.

The profile is proposed. It does not settle the general acquisition experiment,
the meaning of dQE, the stellar leakage measure, the image-grid convention or the
reporting law, all of which remain pending in the [decision register](decisions.md).
Its name illustrates the naming pattern (a descriptive identifier and a version
suffix); other names used in the chapters, such as `observer-toward-v1`, are
illustrations until a profile is defined here.

```{toctree}
:maxdepth: 1
:hidden:

decisions
```
