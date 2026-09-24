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

# Fit and forecast

This page takes the planet of the orbit astrometry example, observes it with
relative astrometry at five epochs spanning about half a period, fits the
orbit with photomancy, and forecasts where the planet will be at three later
epochs. orbix builds the true orbit and propagates the posterior draws,
photomancy holds the data, the likelihood, the priors and the inference
backends, and eyepiece draws the sky, the posterior and the forecast under
the light hwostyle mode.


| Scope | This page |
|---|---|
| Purpose | Fit relative astrometry with photomancy and forecast future positions |
| Model restrictions | The data are simulated with orbix, and photomancy's forward model uses the same orbix projection and equations, so recovering the truth is a same-code round trip, which can preserve a shared convention error; relative astrometry only; the posterior's parameter chart and evidence normalization follow photomancy's current conventions ({ref}`limitations-records`) |
| Evidence kind | Executable tutorial; no numerical check is enforced, because no independent expected observable exists for this composition |
| Data sources | None; simulated measurements |
| Applicable profile | None adopted; the {ref}`reporting law <decision-reporting-law>` and {ref}`evidence and parameter chart <decision-evidence-and-parameter-chart>` decisions are pending |
| Not evidence of | Scientific correctness of the results shown, or measured-data validation |

```{include} ../_generated/environment.md
```

```{code-cell} python
import eyepiece as ep
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import photomancy as pm
from hwoutils import constants, conversions
from orbix.orbit import KeplerianOrbit
from photomancy.orbit import (
    RelativeAstromData,
    build_orbit_logdensity,
    evaluate_orbit_candidates,
    find_init,
    orbits_from_samples,
    predict_relative_astrometry,
)
from photomancy.posterior import cluster_to_mixture

jax.config.update("jax_enable_x64", True)
hwostyle.use("light")
plt.rcParams["savefig.dpi"] = 120  # keeps the baked page images small
```

The planet orbits a solar-mass star at 10 pc every 600 days with eccentricity
0.3, inclination 60 degrees, node 30 degrees, argument of periastron 70
degrees and periastron at JD 2461000. photomancy has no built-in epoch: the
`times` in its data containers are plain day counts in whatever frame the
caller picks, and the phase parameter it samples is the mean anomaly at day 0
of that frame. Day 0 here is JD 2461160, the middle of the five measurements,
so the true periastron sits at day -160 and the measurements at -140, -70, 0,
70 and 140 days. The same frame goes into orbix, whose `from_period` takes the
periastron time in the caller's day frame as well.

```{code-cell} python
T_d, e, i_deg, W_deg, w_deg = 600.0, 0.3, 60.0, 30.0, 70.0
tp_jd, Ms_kg, dist_pc = 2461000.0, constants.Msun2kg, 10.0
t_ref_jd = 2461160.0
tp_d = tp_jd - t_ref_jd

cos_i = jnp.cos(jnp.deg2rad(i_deg))
W_rad, w_rad = jnp.deg2rad(W_deg), jnp.deg2rad(w_deg)
truth = KeplerianOrbit.from_period(
    T_d, e, cos_i, W_rad, jnp.cos(w_rad), jnp.sin(w_rad), tp_d, Ms_kg=Ms_kg
)
t_obs = jnp.array([-140.0, -70.0, 0.0, 70.0, 140.0])
print(truth, f"periastron at day {tp_d:.0f}")
```

`RelativeAstromData` holds the planet-to-star offsets `ra` and `dec` in
arcseconds with per-axis errors, a per-epoch correlation coefficient and a
`planet_id`; `pad` fills every field to a static length of 64 so that one
compiled likelihood serves any number of epochs, and `is_valid` masks the
padding out of the sum. photomancy's forward model projects the same
`AB` matrices orbix uses and returns the x component as `ra` and the y
component as `dec`, so `ra` is the east offset and `dec` the north offset,
exactly the axes of orbix's `position_arcsec`. The cell checks that agreement
at the five epochs before adding 5 mas of Gaussian noise per axis.

```{code-cell} python
sigma_mas = 5.0
sigma_as = sigma_mas * constants.mas2arcsec
ra_true, dec_true = truth.position_arcsec(t_jd=t_obs, Ms_kg=Ms_kg, dist_pc=dist_pc)
ra_pm, dec_pm = predict_relative_astrometry(
    t_obs,
    truth.a_AU[0],
    e,
    cos_i,
    W_rad,
    jnp.cos(w_rad),
    jnp.sin(w_rad),
    tp_d,
    Ms_kg,
    dist_pc,
)
gap_mas = constants.arcsec2mas * jnp.hypot(ra_pm - ra_true[0], dec_pm - dec_true[0])
print(f"photomancy vs orbix positions agree to {gap_mas.max():.1e} mas")

k_ra, k_dec, k_nuts, k_mix = jax.random.split(jax.random.PRNGKey(7), 4)
ra_obs = ra_true[0] + sigma_as * jax.random.normal(k_ra, t_obs.shape)
dec_obs = dec_true[0] + sigma_as * jax.random.normal(k_dec, t_obs.shape)
data = RelativeAstromData.pad(
    times=t_obs,
    ra=ra_obs,
    dec=dec_obs,
    ra_err=jnp.full(t_obs.shape, sigma_as),
    dec_err=jnp.full(t_obs.shape, sigma_as),
    corr=jnp.zeros(t_obs.shape),
    planet_id=jnp.zeros(t_obs.shape, dtype=int),
)
print(f"{int(data.is_valid.sum())} valid epochs padded to {data.times.shape[0]}")
```

The field of view is 12 lambda/D at 500 nm on a 6 m aperture, held in a
`Frame` so the sky panels below share it, and the inner working angle used
later is 3 lambda/D on the same telescope. `trail` draws the true projected
orbit as a bare line in scenery gray, and `sky_fan` with no tracks draws the
star and the measurements with their error bars. East increases to the left,
as on the sky.

```{code-cell} python
frame = ep.Frame(half_fov_lod=12.0)
half_mas = constants.arcsec2mas * frame.extent_arcsec(500.0, 6.0)[1]
iwa_mas = constants.arcsec2mas * conversions.lambda_d_to_arcsec(3.0, 500.0, 6.0)
scenery = "0.55"
print(f"half field {half_mas:.0f} mas, inner working angle {iwa_mas:.1f} mas")

t_orbit = tp_d + jnp.linspace(0.0, T_d, 361)
ra_orb, dec_orb = truth.position_arcsec(t_jd=t_orbit, Ms_kg=Ms_kg, dist_pc=dist_pc)
true_xy = constants.arcsec2mas * np.column_stack([ra_orb[0], dec_orb[0]])
obs_xy = constants.arcsec2mas * np.column_stack([ra_obs, dec_obs])

fig, ax = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
ep.trail(true_xy, ax=ax, style=scenery, depth="none")
ep.sky_fan([], ax=ax, data=(obs_xy[:, 0], obs_xy[:, 1], sigma_mas))
for (x, y), t in zip(obs_xy, np.asarray(t_obs)):
    r = np.hypot(x, y)
    ax.annotate(
        f"{t:+.0f} d",
        (x, y),
        xytext=(-9.0 * x / r, 9.0 * y / r),
        textcoords="offset points",
        ha="right" if x > 0 else "left",
        va="bottom" if y > 0 else "top",
        color=scenery,
    )
ax.set(xlim=(half_mas, -half_mas), ylim=(-half_mas, half_mas))
ax.set_xlabel("East offset [mas]")
ax.set_ylabel("North offset [mas]")
```

The five measurements with their 5 mas error bars on the true projected orbit,
labeled by their day in the fitting frame; the error bars are small against
the 140 mas semi-major axis, and the measured arc covers the fast half of the
orbit around periastron.

`build_orbit_logdensity` traces photomancy's orbit model once and returns an
`OrbitProblem`: a flat `logdensity` over the unconstrained parameters, the
`to_physical` map back to named elements, and `init_to_z` for seeding. The
model samples log10 of the period in days, uniform on 1 to 4; the eccentricity
under the Kipping (2013) Beta prior; the inclination through a uniform cosine;
the node and the argument of periastron in radians, uniform on 0 to 2 pi; and
the mean anomaly at day 0, `M0`, uniform on 0 to 2 pi. `to_physical` adds the
derived elements: `T` in days, `a` in AU through Kepler's third law with the
stellar mass in kg, and `tp = -M0 T / 2 pi` in days, which lies in the interval
from `-T` to 0 by construction, so a periastron time is only defined modulo
the period. `find_init` runs a Thiele-Innes grid over period, eccentricity
and phase and returns the best start, and the `LaplaceBackend` climbs to the
maximum a posteriori point from there; its `min_eigenvalue` floor of 1 on the
precision keeps a direction the data barely constrain at a bounded variance.

```{code-cell} python
problem = build_orbit_logdensity(Ms_kg, dist_pc, relative_astrom_data=data)
z_init = problem.init_to_z(find_init(data, Ms_kg, dist_pc))
laplace = pm.LaplaceBackend(n_steps=100, min_eigenvalue=1.0).run(
    problem.logdensity, z_init
)
map_phys = problem.to_physical(laplace.mean)
print(problem.param_names)
print(
    {k: round(float(map_phys[k]), 3) for k in ("T", "e", "cos_i", "W", "w_raw", "tp")}
)
```

Half a period of astrometry leaves a long ridge in the period, so the posterior
is not Gaussian in any parameterization and the Laplace covariance only
brackets it. The `NUTSBackend` runs a short No-U-Turn chain from the Laplace
mean through the same `run(logdensity, init, key)` interface and returns a
`SamplePosterior` of equally weighted draws in the unconstrained space;
`to_physical` maps the whole chain to named elements in one `vmap`.

```{code-cell} python
nuts = pm.NUTSBackend(n_warmup=500, n_samples=2000).run(
    problem.logdensity, laplace.mean, k_nuts
)
phys = jax.vmap(problem.to_physical)(nuts.samples)
print(
    nuts.samples.shape,
    f"T = {phys['T'].mean():.0f} +/- {phys['T'].std():.0f} d, "
    f"e = {phys['e'].mean():.2f} +/- {phys['e'].std():.2f}",
)
```

`corner` takes a dict of sample arrays, an ordered list of names, the injected
values as `truths` and a label per name; the units are the ones photomancy
reports, days for the period and the periastron time and radians for the two
angles. The upper triangle is hidden and the diagonal histograms are
normalized to unit area. The two-dimensional cells are returned as
`artists["collection"]`, and the loop at the end moves them onto the
`brand_intensity` colormap, whose floor is the page background, so an empty
cell reads as empty on a light page.

```{code-cell} python
names = ["T", "e", "cos_i", "W", "w_raw", "tp"]
labels = {
    "T": r"$T$ [d]",
    "e": r"$e$",
    "cos_i": r"$\cos i$",
    "W": r"$\Omega$ [rad]",
    "w_raw": r"$\omega$ [rad]",
    "tp": r"$t_p$ [d]",
}
truths = {
    "T": T_d,
    "e": e,
    "cos_i": float(cos_i),
    "W": float(W_rad),
    "w_raw": float(w_rad),
    "tp": tp_d,
}
samples = {k: np.asarray(phys[k]) for k in names}
fig, axes = plt.subplots(6, 6, figsize=(7.5, 7.5), layout="constrained")
corner = ep.corner(samples, names, truths=truths, labels=labels, bins=25, axes=axes)
for mesh in corner.artists["collection"]:
    mesh.set_cmap(hwostyle.cmaps.brand_intensity)
```

The posterior over six elements with the injected values as dashed lines: the
period ridge stretches from about 430 to 720 days and drags the eccentricity,
inclination and argument of periastron along it, while the periastron time is
pinned to about 10 days by the measured arc.

Every twentieth draw of the chain gives 100 orbits, and
`orbits_from_samples` turns their named elements into one batched orbix
`KeplerianOrbit`, so a single `position_arcsec` call propagates every draw
over the 600 days after the first measurement. `sky_fan` fades the tracks by
weight, uniform here because the chain is equally weighted, so the per-track
alpha is set from the count instead; the inner-working-angle disk is the
shaded region, the star and the measurements are drawn in neutral tones, and
the true orbit goes on top in the scenery gray of the first figure. The right
panel propagates all 2000 draws to days 240, 320 and 400, draws each cloud of
predicted positions, and summarizes it with a one-sigma `cov_ellipse` from the
sample mean and covariance; the true positions carry the same gray as the true
orbit.

```{code-cell} python
draws = jax.vmap(problem.to_physical)(nuts.samples[::20])
orbits = orbits_from_samples(draws, Ms_kg)
t_track = t_obs[0] + jnp.linspace(0.0, 600.0, 241)
ra_tr, dec_tr = orbits.position_arcsec(t_jd=t_track, Ms_kg=Ms_kg, dist_pc=dist_pc)
tracks = [
    (constants.arcsec2mas * x, constants.arcsec2mas * y)
    for x, y in zip(np.asarray(ra_tr), np.asarray(dec_tr))
]
alpha_track = min(0.6, 8.0 / len(tracks))

t_future = jnp.array([240.0, 320.0, 400.0])
ra_f, dec_f = orbits_from_samples(phys, Ms_kg).position_arcsec(
    t_jd=t_future, Ms_kg=Ms_kg, dist_pc=dist_pc
)
pred = constants.arcsec2mas * np.stack([np.asarray(ra_f), np.asarray(dec_f)], -1)
ra_ft, dec_ft = truth.position_arcsec(t_jd=t_future, Ms_kg=Ms_kg, dist_pc=dist_pc)
true_future = constants.arcsec2mas * np.column_stack([ra_ft[0], dec_ft[0]])

fig, (ax_fan, ax_fc) = plt.subplots(1, 2, figsize=(9.6, 4.6), layout="constrained")
ep.sky_fan(
    tracks,
    ax=ax_fan,
    iwa=iwa_mas,
    data=(obs_xy[:, 0], obs_xy[:, 1], sigma_mas),
    fan_kw={"color": hwostyle.roles.planet, "alpha": alpha_track / 1.35},
)
ep.trail(true_xy, ax=ax_fan, style=scenery, depth="none", trail_kw={"zorder": 2})

ep.sky_fan([], ax=ax_fc, iwa=iwa_mas, data=(obs_xy[:, 0], obs_xy[:, 1], sigma_mas))
ep.trail(true_xy, ax=ax_fc, style=scenery, depth="none")
for j, t in enumerate(np.asarray(t_future)):
    cloud = pred[::5, j]
    ax_fc.scatter(
        cloud[:, 0], cloud[:, 1], s=3, lw=0, alpha=0.2, color=hwostyle.roles.predicted
    )
    ep.cov_ellipse(
        pred[:, j].mean(0),
        np.cov(pred[:, j].T),
        ax=ax_fc,
        color=hwostyle.roles.predicted,
        label=f"{t:+.0f} d, 1 sigma",
    )
ax_fc.plot(
    true_future[:, 0], true_future[:, 1], ls="none", marker="o", ms=6, color=scenery
)
for axis in (ax_fan, ax_fc):
    axis.set(xlim=(half_mas, -half_mas), ylim=(-half_mas, half_mas))
    axis.set_xlabel("East offset [mas]")
    axis.set_ylabel("North offset [mas]")
```

Left, 100 posterior orbit tracks over the 600 days after the first
measurement, with the measurements, the 51.6 mas inner-working-angle disk and
the true orbit in gray; right, the predicted positions at days 240, 320 and
400 as clouds with their one-sigma moment ellipses and the true positions as
gray dots; the clouds stretch along the orbit as the period ridge takes over,
which the ellipses only summarize.

photomancy's expected information gain scores a candidate epoch by how much
one more astrometric measurement there would shrink the posterior. The
analytic form needs a mixture of Gaussians, so `cluster_to_mixture` groups
the NUTS draws into three weighted Gaussian modes in the unconstrained space,
and `evaluate_orbit_candidates` differentiates the sky position through the
model's constraint maps at every candidate epoch, adds the per-mode Fisher
gain and the between-mode term, and returns the total in nats. The measurement
variance is the same 5 mas per axis as the data, in arcseconds squared.

```{code-cell} python
mixture = cluster_to_mixture(nuts, 3, key=k_mix)
t_cand = jnp.linspace(140.0, 740.0, 121)
eig = evaluate_orbit_candidates(mixture, problem, t_cand, sigma_as**2, Ms_kg, dist_pc)
total = np.asarray(eig["total_eig"])
at_future = np.interp(np.asarray(t_future), np.asarray(t_cand), total)

fig, ax = plt.subplots(figsize=(4.8, 3.2), layout="constrained")
ax.plot(np.asarray(t_cand), total, color=hwostyle.roles.planet)
ax.plot(
    np.asarray(t_future),
    at_future,
    ls="none",
    marker="o",
    ms=6,
    color=hwostyle.roles.predicted,
)
for t, y in zip(np.asarray(t_future), at_future):
    ax.annotate(
        f"{t:+.0f} d",
        (t, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color=hwostyle.roles.predicted,
    )
ax.spines[["top", "right"]].set_visible(False)
ax.set(xlim=(140.0, 740.0), ylim=(0.0, 1.2 * total.max()))
ax.set_xlabel("Candidate epoch [d]")
ax.set_ylabel("Expected information gain [nats]")
print("gain at the forecast epochs [nats]:", np.round(at_future, 2))
```

The expected information gain of one more 5 mas measurement against its
epoch, from the last measurement at day 140 through one nominal period, with
the three forecast epochs of the previous figure marked: the gain climbs
steeply through those epochs as the predicted clouds spread apart, peaks near
day 400 and stays above three nats through the slow half of the orbit.
