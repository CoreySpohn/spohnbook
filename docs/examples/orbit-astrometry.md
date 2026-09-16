---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Orbit astrometry

This page builds a Keplerian orbit with orbix, propagates it over one period,
converts the star-centered positions to sky-plane offsets in milliarcseconds,
and draws the projected orbit with eyepiece under the light hwostyle mode.

```{code-cell} python
import eyepiece as ep
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from orbix.orbit import KeplerianOrbit

jax.config.update("jax_enable_x64", True)
hwostyle.use("light")
plt.rcParams["savefig.dpi"] = 120  # keeps the baked page image small
```

`KeplerianOrbit.from_period` takes the period in days, the eccentricity, the
cosine of the inclination, the longitude of the ascending node, the cosine and
sine of the argument of periastron, and the time of periastron passage, then
converts the period to a semi-major axis through Kepler's third law using the
stellar mass. The planet below orbits a solar-mass star at 10 pc every 600 days.

```{code-cell} python
T_d, e, i_deg, W_deg, w_deg = 600.0, 0.3, 60.0, 30.0, 70.0
tp_d, Ms_kg, dist_pc = 2461000.0, 1.989e30, 10.0

W_rad, w_rad = jnp.deg2rad(W_deg), jnp.deg2rad(w_deg)
orbit = KeplerianOrbit.from_period(
    T_d,
    e,
    jnp.cos(jnp.deg2rad(i_deg)),
    W_rad,
    jnp.cos(w_rad),
    jnp.sin(w_rad),
    tp_d,
    Ms_kg=Ms_kg,
)
print(orbit)
```

`propagate` returns star-centered positions in AU with shape `(K, 3, T)` for
`K` orbits at `T` times. The +z axis points toward the observer: the returned
phase angle is measured from +z, so it equals pi when the planet is directly
behind the star. `position_arcsec` divides the x and y components by the
distance in parsecs and returns them as RA and Dec offsets, so x is the east
offset and y is the north offset, and the conversion to milliarcseconds below
is that same division.

```{code-cell} python
t_jd = tp_d + jnp.linspace(0.0, T_d, 361)
r_AU, phase_rad, dist_AU = orbit.propagate(t_jd=t_jd, Ms_kg=Ms_kg)
east_mas, north_mas = np.asarray(1e3 * r_AU[0, :2] / dist_pc)
print(r_AU.shape, f"max separation {np.hypot(east_mas, north_mas).max():.1f} mas")
```

Periastron is the first sample, because `from_period` places periastron passage
at `tp_d`. The ascending node lies at true anomaly `-w` along the direction `W`
measured from the east axis toward north, and it is the crossing of the sky
plane at which z increases, so the planet moves toward the observer there.

```{code-cell} python
r_node_AU = orbit.a_AU[0] * (1 - e**2) / (1 + e * jnp.cos(-w_rad))
node_dir = jnp.array([jnp.cos(W_rad), jnp.sin(W_rad)])
node_mas = np.asarray(1e3 * r_node_AU / dist_pc * node_dir)
peri_mas = east_mas[0], north_mas[0]
print(f"ascending node at ({node_mas[0]:.1f}, {node_mas[1]:.1f}) mas")
```

`trail` draws the track as a bare line on the axes it is given, and the star,
the line of nodes, and the two marks go on the same axes. East increases to the
left, as on the sky.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
track = ep.trail(np.column_stack([east_mas, north_mas]), ax=ax, depth="none")
ax.axline((0.0, 0.0), tuple(node_mas), color="0.75", lw=0.8, ls="--")
ax.plot(0.0, 0.0, marker="*", ms=11, ls="none", color="0.5")
ax.plot(*peri_mas, marker="D", ms=6, ls="none", color=hwostyle.palette.pink)
ax.plot(*node_mas, marker="^", ms=7, ls="none", color=hwostyle.palette.yellow)
ax.annotate("periastron", peri_mas, xytext=(6, 6), textcoords="offset points")
ax.annotate("ascending node", node_mas, xytext=(6, -12), textcoords="offset points")
ax.set_aspect("equal")
ax.invert_xaxis()
ax.set_xlabel("East offset [mas]")
ax.set_ylabel("North offset [mas]")
```
