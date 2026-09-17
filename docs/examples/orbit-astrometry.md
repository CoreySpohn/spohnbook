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
draws the orbit in space beside its sky-plane projection with eyepiece under
the light hwostyle mode, plots the separation and phase angle over the period,
and animates the planet along both views.

```{code-cell} python
import eyepiece as ep
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
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
is that same division, applied to the z component as well so the three axes
share one scale.

```{code-cell} python
t_jd = tp_d + jnp.linspace(0.0, T_d, 361)
r_AU, phase_rad, dist_AU = orbit.propagate(t_jd=t_jd, Ms_kg=Ms_kg)
east_mas, north_mas, toward_mas = np.asarray(1e3 * r_AU[0] / dist_pc)
print(r_AU.shape, f"max separation {np.hypot(east_mas, north_mas).max():.1f} mas")
```

Periastron is the first sample, because `from_period` places periastron passage
at `tp_d`. The ascending node lies at true anomaly `-w` along the direction `W`
measured from the east axis toward north, and it is the crossing of the sky
plane at which z increases, so the planet moves toward the observer there. Its
passage time follows from Kepler's equation: the true anomaly gives the
eccentric anomaly, the eccentric anomaly gives the mean anomaly, and the mean
anomaly is the fraction of the period elapsed since periastron.

```{code-cell} python
r_node_AU = orbit.a_AU[0] * (1 - e**2) / (1 + e * jnp.cos(-w_rad))
node_dir = jnp.array([jnp.cos(W_rad), jnp.sin(W_rad)])
node_mas = np.asarray(1e3 * r_node_AU / dist_pc * node_dir)
peri_mas = east_mas[0], north_mas[0]

E_node = 2.0 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(-float(w_rad) / 2))
M_node = E_node - e * np.sin(E_node)
t_node_d = (M_node % (2 * np.pi)) / (2 * np.pi) * T_d
print(f"ascending node at ({node_mas[0]:.1f}, {node_mas[1]:.1f}) mas, {t_node_d:.1f} d")
```

`SourceStyles` assigns each named source one color and one marker, so
periastron and the ascending node look the same in every panel below. `Frame`
holds the shared half field of view, 12 lambda/D at 500 nm on a 6 m aperture,
and every sky panel takes its limits from it. The left panel is the orbit in
space: `trail` draws the half nearer the camera solid and the far half dashed,
the gray curve is the projection onto the sky plane, and the arrow is the +z
axis toward the observer. The triad east, north, toward observer is
left-handed, so the 3-D panel plots the west offset on its x axis to keep the
axes right-handed; viewed from +z that puts east on the left, matching the sky
panel on the right, where `trail` draws the projected track as a bare line,
the dashed line is the line of nodes, and east increases to the left as on the
sky.

```{code-cell} python
styles = ep.SourceStyles(["planet", "periastron", "ascending node"])
frame = ep.Frame(half_fov_lod=12.0)
half_mas = 1e3 * frame.extent_arcsec(500.0, 6.0)[1]
west_mas = -east_mas
peri_xyz = west_mas[0], north_mas[0], toward_mas[0]
node_xyz = -node_mas[0], node_mas[1], 0.0

fig = plt.figure(figsize=(9.6, 4.4), layout="constrained")
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
ax3.view_init(elev=22.0, azim=-75.0)
ep.trail(
    np.column_stack([west_mas, north_mas, toward_mas]), ax=ax3, style=styles["planet"]
)
ax3.plot(west_mas, north_mas, 0.0 * toward_mas, color="0.65", lw=0.8)
ax3.quiver(0.0, 0.0, 0.0, 0.0, 0.0, half_mas, color="0.5", arrow_length_ratio=0.12)
ax3.text(0.0, 0.0, 1.08 * half_mas, "to observer (+z)", color="0.5", ha="center")
ax3.plot([0.0], [0.0], [0.0], marker="*", ms=11, ls="none", color="0.5")
for name, xyz in (("periastron", peri_xyz), ("ascending node", node_xyz)):
    ax3.plot([xyz[0]], [xyz[1]], [xyz[2]], ls="none", ms=6, **styles[name])
lim = (-half_mas, half_mas)
ticks = np.arange(-200.0, 201.0, 100.0)
ax3.set(xlim=lim, ylim=lim, zlim=lim, xticks=ticks, yticks=ticks, zticks=ticks)
ax3.set_box_aspect((1, 1, 1))
ax3.grid(False)
ax3.set_xlabel("West offset [mas]")
ax3.set_ylabel("North offset [mas]")
ax3.set_zlabel("Toward observer [mas]")

ax = fig.add_subplot(1, 2, 2)
ep.trail(
    np.column_stack([east_mas, north_mas]), ax=ax, style=styles["planet"], depth="none"
)
ax.axline((0.0, 0.0), tuple(node_mas), color="0.75", lw=0.8, ls="--")
ax.plot(0.0, 0.0, marker="*", ms=11, ls="none", color="0.5")
ax.plot(*peri_mas, ls="none", ms=6, **styles["periastron"])
ax.plot(*node_mas, ls="none", ms=6, **styles["ascending node"])
ax.annotate("periastron", peri_mas, xytext=(6, 6), textcoords="offset points")
ax.annotate("ascending node", node_mas, xytext=(6, -12), textcoords="offset points")
ax.set(xlim=(half_mas, -half_mas), ylim=lim, aspect="equal")
ax.set_xlabel("East offset [mas]")
ax.set_ylabel("North offset [mas]")
```

Left, the orbit in space with the observer axis and its sky-plane projection;
right, the projection alone as the observer sees it, with periastron and the
ascending node carrying the same marker and color in both panels.

The separation is the length of the sky-plane offset. The phase angle is the
one `propagate` returns, measured from the +z observer axis, so it reaches 180
degrees when the planet is directly behind the star; the standard
star-planet-observer angle is 180 degrees minus this value. Both curves carry
the periastron and node marks from the sky panels, placed at the times those
points are passed.

```{code-cell} python
t_d = np.asarray(t_jd - tp_d)
sep_mas = np.hypot(east_mas, north_mas)
phase_deg = np.rad2deg(np.asarray(phase_rad[0]))

fig, (ax_sep, ax_phase) = plt.subplots(
    2, 1, figsize=(4.8, 4.4), sharex=True, layout="constrained"
)
rows = (
    (ax_sep, sep_mas, "Separation [mas]"),
    (ax_phase, phase_deg, "Phase angle [deg]"),
)
for axis, y, label in rows:
    axis.plot(t_d, y, color=styles["planet"]["color"])
    axis.plot(0.0, y[0], ls="none", ms=6, clip_on=False, **styles["periastron"])
    axis.plot(
        t_node_d,
        np.interp(t_node_d, t_d, y),
        ls="none",
        ms=6,
        **styles["ascending node"],
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylabel(label)
ax_sep.annotate(
    "periastron", (0.0, sep_mas[0]), xytext=(6, -12), textcoords="offset points"
)
ax_sep.annotate(
    "ascending node",
    (t_node_d, np.interp(t_node_d, t_d, sep_mas)),
    xytext=(6, 6),
    textcoords="offset points",
)
ax_sep.set_ylim(0.0, None)
ax_phase.set(xlim=(0.0, T_d), ylim=(0.0, 180.0), yticks=(0, 90, 180))
ax_phase.set_xlabel("Days since periastron")
```

Top, the planet-star separation over one period; bottom, the phase angle from
the observer axis, with time running from periastron and the node mark at the
node passage found above.

`animate` binds a figure, a draw function and a frame count, and nothing
renders until `jshtml` embeds the frames as PNGs. The draw function only moves
existing artists: the planet marker, the `fading_track` tail behind it, which
is rebuilt from the previous 40 samples each frame, and the cursor on the
separation curve. The full track stays underneath in gray for context. The sky
panel takes its limits from the same `Frame` as above, the time-axis limits
are fixed before the first frame, and the figure size is fixed, so nothing
drifts between frames. The 36 frames step through one period ten samples at a
time.

```{code-cell} python
xy = np.column_stack([east_mas, north_mas])
tail = 40


def tail_xy(k):
    idx = (10 * k - np.arange(tail, -1, -1)) % 360
    return xy[idx]


def segments(points):
    pts = points.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


fig, (ax_sky, ax_t) = plt.subplots(
    1, 2, figsize=(4.5, 2.6), width_ratios=(1.0, 1.5), layout="constrained"
)
ep.trail(xy, ax=ax_sky, style="0.8", depth="none")
ax_sky.plot(0.0, 0.0, marker="*", ms=8, ls="none", color="0.5")
tail_track = ep.fading_track(tail_xy(0), ax=ax_sky, color=styles["planet"]["color"])
(planet,) = ax_sky.plot(*xy[0], ls="none", ms=5, **styles["planet"])
ax_sky.set(xlim=(half_mas, -half_mas), ylim=lim, aspect="equal")
ax_sky.set_xlabel("East offset [mas]")
ax_sky.set_ylabel("North offset [mas]")

ax_t.plot(t_d, sep_mas, color=styles["planet"]["color"])
cursor = ax_t.axvline(0.0, color="0.75", lw=0.8)
(planet_t,) = ax_t.plot(0.0, sep_mas[0], ls="none", ms=5, **styles["planet"])
ax_t.spines[["top", "right"]].set_visible(False)
ax_t.set(xlim=(0.0, T_d), ylim=(0.0, 1.1 * sep_mas.max()))
ax_t.set_xlabel("Days since periastron")
ax_t.set_ylabel("Separation [mas]")


def draw(fig, k):
    i = 10 * k
    tail_track.artists["collection"].set_segments(segments(tail_xy(k)))
    planet.set_data([xy[i, 0]], [xy[i, 1]])
    cursor.set_xdata([t_d[i], t_d[i]])
    planet_t.set_data([t_d[i]], [sep_mas[i]])


anim = ep.animate(fig, draw, 36, fps=12)
```

```{code-cell} python
HTML(anim.jshtml())
```

The planet moves along its sky-plane track with a tail fading over the
previous 40 samples, while the cursor follows the same instant along the
separation curve.
