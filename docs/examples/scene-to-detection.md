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

# Scene to image to detection

This page chains the scene, orbit, hardware, imaging and post-processing
libraries once, end to end. A solar twin at 10 pc carries one Earth-like
planet (skyscapes and orbix); a 7.2 m telescope with a yield input package
coronagraph images it (optixstuff and yippy); coronagraphoto renders one
exposure per source and then a three-roll sequence; coronalyze turns the
derotated coadd into a signal-to-noise map; and a closing panel checks the
image against the coronagraph's declared contrast. Every figure is drawn with
eyepiece under the light hwostyle mode. Float64 is enabled before any array
exists, because the stellar leakage sits more than ten decades below the star.


```{figure} ../conventions/figures/explainer-d01-overview-tutorial-light.png
:class: only-light
:name: fig-explainer-d01-overview-tutorial
:width: 80%
:alt: The observation map with the scene, the instrument and the raw frame at full strength, a bold key line reading full strength: this tutorial, and every other part faded toward the background. A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

Where this tutorial sits in the physical observation: the scene, the instrument and the raw frame, at full strength, with the rest of the map dimmed ({ref}`full map <fig-explainer-d01-overview>`). An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

```{figure} ../conventions/figures/explainer-d01-overview-tutorial-dark.png
:class: only-dark
:width: 80%
:alt: The observation map with the scene, the instrument and the raw frame at full strength, a bold key line reading full strength: this tutorial, and every other part faded toward the background. A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

Where this tutorial sits in the physical observation: the scene, the instrument and the raw frame, at full strength, with the rest of the map dimmed ({ref}`full map <fig-explainer-d01-overview>`). An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

| Scope | This page |
|---|---|
| Purpose | Chain the scene, orbit, hardware, imaging and post-processing libraries once, end to end |
| Model restrictions | The epoch is the ascending node crossing, where the observer-axis angle and the illumination angle coincide ({ref}`geometry-illumination`), because the current composition inverts the phase curve elsewhere (`geometry-observer-axis-angles-enter-illumination` on the [limitations page](../evidence/limitations.md)). This restriction means the page cannot test the phase convention. The star is a `Star` with an explicit diameter because `FlatStar` has none (`examples-flatstar-diameter`) |
| Evidence kind | Executable tutorial. The closing comparison reads the rendered image against the same coronagraph tables that produced it, so it is a consistency view, not an independent check |
| Data sources | The `eac1_optimal_order_6_1d` yield input package ({ref}`reproducing-inputs`) |
| Applicable profile | None adopted; the {ref}`observer basis <decision-observer-basis-and-node>`, {ref}`stellar leakage <decision-stellar-leakage-measure>` and {ref}`image coordinate <decision-image-coordinates-and-psflet-origin>` decisions are pending |
| Not evidence of | Scientific correctness of the results shown, or measured-data validation |

```{include} ../_generated/environment.md
```

```{code-cell} python
import coronagraphoto as cp
import coronalyze as cl
import eyepiece as ep
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optixstuff as ox
import yippy
from coronagraphoto.simulation import pre_coro_bin_processing
from hwoutils.constants import G, Msun2kg, d2s, pc2m, rad2arcsec
from hwoutils.conversions import (
    arcsec_to_lambda_d,
    lambda_d_to_arcsec,
    mag_to_flux_jy,
)
from hwoutils.transforms import rotate_image
from matplotlib.patches import Circle, Rectangle
from orbix.equations.orbit import mean_motion, period_n
from orbix.kepler import get_grid_solver
from orbix.orbit import KeplerianOrbit
from skyscapes.background import AYOZodi
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import Planet, Scene, Star, System

jax.config.update("jax_enable_x64", True)
hwostyle.use("light")
plt.rcParams["savefig.dpi"] = 120  # keeps the baked page images small
yippy.logger.setLevel("ERROR")  # the loader reports every cache lookup at INFO
```

## The coronagraph sets the scale

`fetch_yip` downloads the `eac1_optimal_order_6_1d` yield input package on
first use (about 30 MB), unpacks it into the cache and returns the directory;
`EqxCoronagraph` loads its tables into a JAX module. The package is a 1 micron
design (0.9 to 1.1 micron band) for a 7.2 m circumscribed aperture, tabulated
at 0.25 lambda/D per pixel on a 256 pixel grid, and it reports its working
angles in lambda/D. Every angle on this page is converted with that diameter
and wavelength, where one lambda/D is 28.6 mas, and the outer working angle
is the edge of the tabulated field.

```{code-cell} python
yip_dir = yippy.fetch_yip("eac1_optimal_order_6_1d")
coro = yippy.EqxCoronagraph(yip_dir)
D_m, wl_nm, bw_nm = 7.2, 1000.0, 200.0
lod_arcsec = float(lambda_d_to_arcsec(1.0, wl_nm, D_m))
iwa_arcsec, owa_arcsec = coro.IWA * lod_arcsec, coro.OWA * lod_arcsec
print(
    f"{coro.psf_shape[0]} px at {coro.pixel_scale_lod} lambda/D, "
    f"one lambda/D = {1e3 * lod_arcsec:.1f} mas at {wl_nm:.0f} nm on {D_m} m"
)
```

## The scene

The star is a `Star` with a flat 57.5 Jy flux table across the band (4.5 AB
magnitudes, a solar twin at 10 pc) and the Sun's angular diameter at that
distance. `FlatStar` carries no diameter, and `star_rate` reads
`star.diameter_arcsec` to select the finite-size stellar leakage map, so the
flat-spectrum class cannot pass through the imager. The planet has one Earth
radius on a circular 1 AU orbit inclined 45 degrees with its ascending node 30
degrees from the x axis, wrapped in a `Planet` with a grey Lambertian model of
geometric albedo 0.3. orbix's frame puts x along the RA offset, y along the Dec
offset and z toward the observer, and `propagate` returns the phase angle
measured from that +z axis; the skyscapes Lambertian model applies the Lambert
phase function to that angle as returned, so the contrast it produces is
largest when the planet is nearest the observer. The epoch is the ascending
node crossing (`M0_rad = 0` with `w_rad = 0` at `t0_d`), where the planet lies
in the sky plane, the angle is 90 degrees and the phase function is 1/pi
whichever side of the sky plane it is measured from. `System` takes the scalar
Kepler solver as a static field, and `Scene` adds the AYO zodiacal background,
22 magnitudes per square arcsecond at V with the Leinert color correction.

```{code-cell} python
t0 = 2461000.5
dist_pc, Ms_kg = 10.0, Msun2kg
flux_jy = float(mag_to_flux_jy(4.5))
diam_arcsec = 2.0 * 6.957e8 / (dist_pc * pc2m) * rad2arcsec
star = Star(
    Ms_kg=Ms_kg,
    dist_pc=dist_pc,
    wavelengths_nm=jnp.linspace(800.0, 1200.0, 5),
    times_jd=t0 + jnp.linspace(-400.0, 400.0, 5),
    flux_density_jy=jnp.full((5, 5), flux_jy),
    diameter_arcsec=diam_arcsec,
)
orbit = KeplerianOrbit(
    a_AU=1.0,
    e=0.0,
    W_rad=jnp.deg2rad(30.0),
    i_rad=jnp.deg2rad(45.0),
    w_rad=0.0,
    M0_rad=0.0,
    t0_d=t0,
)
planet = Planet(
    Rp_Rearth=jnp.array([1.0]),
    Mp_Mearth=jnp.array([1.0]),
    orbit=orbit,
    physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
)
solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
system = System(star=star, planets=(planet,), trig_solver=solver)
zodi = AYOZodi(jnp.linspace(500.0, 1300.0, 17), surface_brightness_mag=22.0)
scene = Scene(system=system, zodi=zodi)

epoch = jnp.array([t0])
ra_as, dec_as = np.asarray(planet.position_arcsec(solver, epoch, star=star))[:, 0, 0]
sep_lod = float(arcsec_to_lambda_d(np.hypot(ra_as, dec_as), wl_nm, D_m))
contrast = float(planet.contrast(solver, wl_nm, epoch, star=star)[0, 0])
print(
    f"planet at ({1e3 * ra_as:.1f}, {1e3 * dec_as:.1f}) mas, "
    f"{sep_lod:.2f} lambda/D, contrast {contrast:.2e}"
)
```

The detector below is 72 pixels of 11.5 mas, so its half field is 14.5
lambda/D; the sky panel uses a `Frame` of 35 lambda/D converted to arcseconds
at the observing wavelength, wide enough to hold the outer working angle, and
`extent_arcsec` gives the detector's pixel-edge footprint in the same units.
`trail` draws the projected orbit over one period as a bare line
(`depth="none"`), the marker is the planet at the epoch in the same
`SourceStyles` entry, the star is a scenery marker at the origin, and the
working angles are `Circle` patches in the scenery gray, because eyepiece has
no ring primitive; the helper restores the axes limits, since a patch would
otherwise autoscale an image panel out to the outer ring. The x axis is the
RA offset increasing to the right, which is the orientation of the readout
array below, so the sky panel and the image panels share one frame.

```{code-cell} python
n_pix, pix_mas = 72, 11.5
pix_arcsec = 1e-3 * pix_mas
extent = ep.extent_arcsec(n_pix, pix_mas)
sky_extent = ep.Frame(half_fov_lod=35.0).extent_arcsec(wl_nm, D_m)
styles = ep.SourceStyles(["planet"])
T_d = float(period_n(mean_motion(orbit.a_AU, G * Ms_kg))[0])
t_orbit = t0 + jnp.linspace(0.0, T_d, 181)
ra_orbit, dec_orbit = orbit.position_arcsec(
    solver, t_orbit, Ms_kg=Ms_kg, dist_pc=dist_pc
)
track = np.column_stack([np.asarray(ra_orbit)[0], np.asarray(dec_orbit)[0]])


def working_angle_rings(ax):
    """Draw the IWA and OWA as scenery-rank circles without moving the limits."""
    limits = ax.get_xlim(), ax.get_ylim()
    for radius in (iwa_arcsec, owa_arcsec):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, color="0.6", lw=0.8))
    ax.set(xlim=limits[0], ylim=limits[1])


fig, ax = plt.subplots(figsize=(4.8, 4.6), layout="constrained")
ax.set(xlim=sky_extent[:2], ylim=sky_extent[2:], aspect="equal")
ep.trail(track, ax=ax, style=styles["planet"], depth="none")
ax.plot(0.0, 0.0, marker="*", ms=11, ls="none", color="0.5")
ax.plot(ra_as, dec_as, ls="none", ms=7, **styles["planet"])
ax.annotate(
    "planet at the epoch", (ra_as, dec_as), xytext=(8, 4), textcoords="offset points"
)
working_angle_rings(ax)
leader = {"arrowstyle": "-", "color": "0.6", "lw": 0.6}
ax.annotate(
    "IWA",
    (-0.7 * iwa_arcsec, -0.7 * iwa_arcsec),
    xytext=(-0.3, -0.3),
    ha="center",
    color="0.45",
    arrowprops=leader,
)
ax.annotate(
    "OWA",
    (0.0, owa_arcsec),
    xytext=(0, -12),
    textcoords="offset points",
    ha="center",
    color="0.45",
)
ax.add_patch(
    Rectangle(
        (extent[0], extent[2]),
        extent[1] - extent[0],
        extent[3] - extent[2],
        fill=False,
        color="0.8",
        lw=0.8,
    )
)
ax.annotate(
    "detector",
    (extent[1], extent[3]),
    xytext=(-4, -12),
    textcoords="offset points",
    ha="right",
    color="0.6",
)
ax.set_xlabel("RA offset [arcsec]")
ax.set_ylabel("Dec offset [arcsec]")
```

Figure 1: the scene at the epoch, with the planet's projected orbit, its
position at the node crossing between the inner and outer working angles, and
the footprint of the detector used for every image below.

## The optical path

`OpticalPath` bundles a primary, an ordered tuple of attenuating elements, a
coronagraph and a detector. The package header gives the obscured fraction of
the aperture area (0.175); `SimplePrimary` takes the linear obscuration, so
the square root is passed and the collecting area comes out at 33.6 m^2. One
`ConstantThroughput` of 0.4 stands in for the whole optical train.
`YippyCoronagraph` wraps the `EqxCoronagraph` built above and serves the
sampling-explicit image contract from its tables. `IdealDetector` has 11.5
mas pixels and a quantum efficiency of 0.9, with no dark current or read
noise. The path reports the working angles of its coronagraph.

yippy defines raw contrast as the stellar flux over the planet flux inside the
same 0.7 lambda/D photometric aperture. This design carries no wavefront error,
so for a point source the tabulated `raw_contrast` sits at the table floor of
1e-20 across the whole dark zone. The star here has a finite diameter of
0.032 lambda/D, which does leak, and the curve below evaluates the same
definition for that diameter from the `stellar_intens` map and the off-axis
PSFs on the native grid, out to 28 lambda/D so the aperture stays inside the
tabulated field. `schematic` draws the Lyot-coronagraph rail with the focal
plane highlighted, the plane the rest of the page lives in, and
`plot_contrast_curve` shades the regions outside the working angles.

```{code-cell} python
path = ox.OpticalPath(
    primary=ox.SimplePrimary(
        diameter_m=D_m, obscuration=float(np.sqrt(coro.frac_obscured))
    ),
    attenuating_elements=(ox.ConstantThroughput(throughput=0.4, name="optics"),),
    coronagraph=ox.YippyCoronagraph(backend=coro),
    detector=ox.IdealDetector(
        pixel_scale_arcsec=pix_arcsec, shape=(n_pix, n_pix), quantum_efficiency=0.9
    ),
)
print(path)
print(
    f"IWA {path.coronagraph.IWA:.2f} lambda/D ({1e3 * iwa_arcsec:.0f} mas), "
    f"OWA {path.coronagraph.OWA:.1f} lambda/D ({1e3 * owa_arcsec:.0f} mas)"
)

diam_lod = float(arcsec_to_lambda_d(diam_arcsec, wl_nm, D_m))
sep_curve = np.linspace(1.0, 28.0, 55)
pix_native = coro.pixel_scale_lod
star_native = np.asarray(coro.stellar_intens(diam_lod))
psf_native = np.asarray(
    coro.create_psfs(jnp.asarray(sep_curve), jnp.zeros(sep_curve.size))
)
yy, xx = np.mgrid[: coro.psf_shape[0], : coro.psf_shape[1]]
c_native = (coro.psf_shape[0] - 1) / 2.0
in_aperture = (
    np.hypot(xx - c_native - sep_curve[:, None, None] / pix_native, yy - c_native)
    <= 0.7 / pix_native
)
raw_star = (star_native * in_aperture).sum(axis=(1, 2)) / (
    psf_native * in_aperture
).sum(axis=(1, 2))
raw_point = np.asarray(coro.raw_contrast(sep_curve))

fig, (ax_rail, ax_c) = plt.subplots(
    1, 2, figsize=(9.6, 3.6), width_ratios=(1.15, 1.0), layout="constrained"
)
ep.schematic("coronagraph", ax=ax_rail, highlight="focal")
curve = ep.plot_contrast_curve(
    sep_curve,
    raw_star,
    ax=ax_c,
    iwa=coro.IWA,
    owa=coro.OWA,
    floors=[(sep_curve, raw_point, "point source")],
    xlabel=r"$r$ [$\lambda/D$]",
    ylabel="raw contrast",
)
ax_c.plot(
    sep_lod, contrast, marker="o", ms=6, ls="none", color=plt.rcParams["text.color"]
)
ax_c.annotate(
    "Earth twin", (sep_lod, contrast), xytext=(8, -3), textcoords="offset points"
)
ax_c.text(
    10.0, 3e-15, "star of 0.032 lambda/D", color=curve.artists["line"].get_color()
)
ax_c.text(
    10.0,
    4e-20,
    "point source (table floor)",
    color=curve.artists["lines"][0].get_color(),
)
ax_c.set(xlim=(0.0, 34.0), ylim=(1e-21, 1e-8))
ax_c.spines[["top", "right"]].set_visible(False)
```

Figure 2: the coronagraph rail with the focal plane highlighted (left) and the
raw contrast of the design (right), at the table floor for a point source and
between 1e-14 and 1e-17 across the dark zone for the finite-diameter star,
with the Earth twin's contrast five decades above the leakage at its
separation.

## The image

coronagraphoto's per-source functions take the source, the path, a PRNG key and
keyword-only observation parameters. Each `*_rate` function returns the
noiseless photon rate per detector pixel: the source flux in photons per
second per square meter per nanometer, times the bin width, the collecting
area and the system throughput, redistributed by the coronagraph's tables.
The tables are resampled from the native 0.25 lambda/D grid to the detector
grid at the bin center, where the detector's arcsecond pixel scale is converted
to lambda/D with the primary diameter; the resample samples the native map at
the detector pixel centers and scales by the pixel-area ratio. Each
`*_readout` function draws Poisson photons over the exposure and a binomial
quantum-efficiency selection, so a readout is a float64 array of integer
photo-electron counts per pixel. The star map is the finite-diameter leakage
table, the planet PSF is placed at its sky position rotated by minus the
telescope position angle, and the zodi is its surface brightness per square
arcsecond times the pixel area, attenuated by the sky transmission map. The
readout's column index grows with the RA offset and its row index with the
Dec offset, the star sits at the geometric center, and under `origin="lower"`
the RA offset increases to the right. `zodi_readout` requires the ecliptic
latitude and solar longitude as keywords, which the AYO model accepts and
ignores. The exposure is 300 s at position angle zero. `system_readout` sums
the same three draws from one split key.

```{code-cell} python
t_exp_s = 300.0
obs = {"start_time_jd": t0, "wavelength_nm": wl_nm, "bin_width_nm": bw_nm}
key_star, key_planet, key_zodi = jax.random.split(jax.random.PRNGKey(0), 3)
star_e = cp.star_readout(star, path, key_star, exposure_time_s=t_exp_s, **obs)
planet_e = cp.planet_readout(
    planet,
    path,
    key_planet,
    exposure_time_s=t_exp_s,
    telescope_pa_deg=0.0,
    star=star,
    trig_solver=solver,
    **obs,
)
zodi_e = cp.zodi_readout(
    zodi,
    path,
    key_zodi,
    exposure_time_s=t_exp_s,
    ecliptic_lat_deg=0.0,
    solar_lon_deg=135.0,
    **obs,
)
total_e = star_e + planet_e + zodi_e
print(
    f"electrons: star {float(star_e.sum()):.0f}, planet {float(planet_e.sum()):.0f} "
    f"(peak {float(planet_e.max()):.0f}), zodi {float(zodi_e.mean()):.1f} per pixel"
)

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
ep.compare_row(
    [star_e, planet_e, total_e],
    ["Star", "Planet", "Star + planet + zodi"],
    axes=axes,
    norm="log",
    vmin=1.0,
    extent=extent,
    cmap=hwostyle.cmaps.readouts,
    cbar_label="electrons per pixel",
)
for ax in axes:
    ax.set_xlabel("RA offset [arcsec]")
axes[0].set_ylabel("Dec offset [arcsec]")
working_angle_rings(axes[2])
axes[2].plot(ra_as, dec_as, marker="o", ms=11, mfc="none", mec="0.6", ls="none")
```

Figure 3: the three per-source readouts of one 300 s exposure under one log
norm floored at one electron; the star panel is empty to within a few
electrons because the design nulls the whole star to about 1e-12 of its light,
the planet is a compact core of about a hundred electrons, and the total panel
is the zodi's Poisson field with that core just outside the inner working
angle ring, at the planet's true position.

## Detection

coronalyze 1.1.1 has no container for an observing sequence, so the sequence
is a plain `(3, 72, 72)` array of `system_readout` frames at position angles
of -15, 0 and +15 degrees, 300 s each and 300 s apart, with the angles kept
beside it; the bookkeeping is done by hand here. coronagraphoto rotates the
source positions by `ccw_rotation_matrix(-telescope_pa_deg)`, so a positive
angle turns the scene clockwise on the readout, and `rotate_image` from
hwoutils turns an image counter-clockwise for a positive angle, so each frame
is derotated by its own position angle. The printed peak pixels of the
noiseless planet rate show both halves of that: the planet moves clockwise
with the angle and returns to one pixel after derotation. The star contributes
about one electron per frame, so no PSF subtraction is applied; the coadd is
the sum of the three derotated frames and its noise is the zodi's photon
noise. `snr_map` evaluates the Mawet et al. (2014) statistic at every pixel,
the aperture flux there against the mean and standard deviation of the
non-overlapping apertures on the same circle with the small-sample penalty,
and returns NaN inside one FWHM of the center; the aperture is one lambda/D
across, which is the FWHM handed to it in pixels.

```{code-cell} python
rolls_deg = np.array([-15.0, 0.0, 15.0])
center = (n_pix - 1) / 2.0
planet_yx = (center + dec_as / pix_arcsec, center + ra_as / pix_arcsec)
for pa in rolls_deg:
    rate = cp.planet_rate(
        planet, path, telescope_pa_deg=pa, star=star, trig_solver=solver, **obs
    )
    iy, ix = np.unravel_index(np.argmax(rate), rate.shape)
    jy, jx = np.unravel_index(np.argmax(rotate_image(rate, pa)), rate.shape)
    print(
        f"PA {pa:+.0f} deg: planet peak at row {iy}, column {ix}; "
        f"derotated to row {jy}, column {jx}"
    )
print(f"planet expected at row {planet_yx[0]:.1f}, column {planet_yx[1]:.1f}")

keys = jax.random.split(jax.random.PRNGKey(1), len(rolls_deg))
frames = np.stack(
    [
        np.asarray(
            cp.system_readout(
                scene,
                path,
                key,
                start_time_jd=t0 + k * t_exp_s / d2s,
                exposure_time_s=t_exp_s,
                wavelength_nm=wl_nm,
                bin_width_nm=bw_nm,
                telescope_pa_deg=pa,
                ecliptic_lat_deg=0.0,
                solar_lon_deg=135.0,
            )
        )
        for k, (pa, key) in enumerate(zip(rolls_deg, keys))
    ]
)
derotated = np.stack(
    [np.asarray(rotate_image(jnp.asarray(f), pa)) for f, pa in zip(frames, rolls_deg)]
)
coadd = derotated.sum(axis=0)

fwhm_px = lod_arcsec / pix_arcsec
snr_map = np.asarray(cl.snr_map(jnp.asarray(coadd), fwhm=fwhm_px))
snr_planet = float(cl.snr(jnp.asarray(coadd), jnp.array([planet_yx]), fwhm=fwhm_px)[0])
best_yx = np.unravel_index(np.nanargmax(snr_map), snr_map.shape)
best_xy = ((best_yx[1] - center) * pix_arcsec, (best_yx[0] - center) * pix_arcsec)
print(
    f"SNR {snr_planet:.1f} at the planet; brightest SNR {np.nanmax(snr_map):.1f} "
    f"at row {best_yx[0]}, column {best_yx[1]}"
)

fig, (ax_img, ax_snr) = plt.subplots(1, 2, figsize=(9.4, 4.0), layout="constrained")
ep.imshow_log(
    coadd,
    ax=ax_img,
    extent=extent,
    vmin=1.0,
    cmap=hwostyle.cmaps.readouts,
    cbar_label="electrons per pixel",
)
ep.imshow_diverging(snr_map, ax=ax_snr, extent=extent, cmap="PuOr", cbar_label="SNR")
for ax in (ax_img, ax_snr):
    working_angle_rings(ax)
    ax.plot(ra_as, dec_as, marker="o", ms=13, mfc="none", mec="0.6", ls="none")
    ax.set_xlabel("RA offset [arcsec]")
ax_img.set_ylabel("Dec offset [arcsec]")
ax_img.set_title("Derotated coadd, 3 x 300 s")
ax_snr.plot(*best_xy, marker="x", ms=8, ls="none", color=plt.rcParams["text.color"])
ax_snr.annotate(
    f"SNR {np.nanmax(snr_map):.0f}", best_xy, xytext=(8, 6), textcoords="offset points"
)
ax_snr.set_title("Mawet SNR map")
```

Figure 4: the derotated coadd of the three rolls (left), its corners dimmer
where the rotated frames do not overlap, and its signal-to-noise map (right),
with the planet's true position as an open ring on both and the brightest
pixel of the map marked; the detection lands on the ring, and the map is
undefined inside one FWHM of the star.

## The image against the declared contrast

To compare the coadd with the raw contrast curve of Figure 2 in one set of
units, the coadd is put through the same aperture definition: a hard 0.7
lambda/D disk of detector pixels is summed at every position with coronalyze's
`flux_map`, and the sum is divided by the electrons a unit-contrast planet
would put in that aperture, the star's electrons through the path times the
core throughput at that radius. That turns the coadd into an aperture-contrast
map, masked inside the inner working angle where the throughput vanishes. The
star's own expectation on the detector goes through the same conversion, so
its azimuthal mean is the leakage as rendered by the resample, to be read
against the curve evaluated on the native tables. `radial_profile_plot`
computes each azimuthal mean with hwoutils and draws it in the next palette
color; `plot_contrast_curve` adds the table curve and the working-angle
shading on the same axes.

```{code-cell} python
qe = path.detector.quantum_efficiency
n_star_e = (
    float(
        pre_coro_bin_processing(star.spec_flux_density(wl_nm, t0), wl_nm, bw_nm, path)
    )
    * qe
    * len(rolls_deg)
    * t_exp_s
)
pix_lod = float(arcsec_to_lambda_d(pix_arcsec, wl_nm, D_m))
kernel = cl.make_aperture_kernel(radius=0.7 / pix_lod, soft=False)
r_lod = np.asarray(cl.radial_distance((n_pix, n_pix))) * pix_lod
thr_map = np.asarray(coro.throughput(jnp.asarray(r_lod.ravel()))).reshape(r_lod.shape)
unit_planet_e = n_star_e * np.maximum(thr_map, 1e-6)
star_model_e = (
    np.asarray(cp.star_rate(star, path, **obs)) * qe * len(rolls_deg) * t_exp_s
)


def to_contrast(image_e):
    """Aperture electrons over those of a unit-contrast planet, outside the IWA."""
    aperture_e = np.asarray(cl.flux_map(jnp.asarray(image_e), kernel))
    return np.where(r_lod >= coro.IWA, aperture_e / unit_planet_e, np.nan)


contrast_map = to_contrast(coadd)
star_contrast_map = to_contrast(star_model_e)

fig, (ax_map, ax_prof) = plt.subplots(
    1, 2, figsize=(9.8, 3.9), width_ratios=(1.0, 1.3), layout="constrained"
)
fig.get_layout_engine().set(wspace=0.08)
ep.imshow_log(
    contrast_map,
    ax=ax_map,
    extent=extent,
    cmap=hwostyle.cmaps.readouts,
    colorbar="figure",
    cbar_label="aperture contrast",
)
working_angle_rings(ax_map)
ax_map.set_xlabel("RA offset [arcsec]")
ax_map.set_ylabel("Dec offset [arcsec]")
image_prof = ep.radial_profile_plot(
    contrast_map,
    pix_lod,
    ax=ax_prof,
    log=True,
    ylabel="azimuthal mean of the aperture contrast",
)
star_prof = ep.radial_profile_plot(star_contrast_map, pix_lod, ax=ax_prof)
table = ep.plot_contrast_curve(
    sep_curve, raw_star, ax=ax_prof, iwa=coro.IWA, owa=coro.OWA
)
ax_prof.plot(
    sep_lod, contrast, marker="o", ms=6, ls="none", color=plt.rcParams["text.color"]
)
ax_prof.annotate(
    "Earth twin", (sep_lod, contrast), xytext=(8, -12), textcoords="offset points"
)
ax_prof.text(
    7.0, 8e-10, "coadd: zodi and planet", color=image_prof.artists["line"].get_color()
)
ax_prof.text(
    9.0, 5e-15, "star on the detector", color=star_prof.artists["line"].get_color()
)
ax_prof.text(
    9.0, 1e-17, "star from the tables", color=table.artists["line"].get_color()
)
ax_prof.set(xlim=(0.0, 15.0), ylim=(1e-19, 1e-8))
ax_prof.spines[["top", "right"]].set_visible(False)
```

Figure 5: the coadd as an aperture-contrast map (left) and its azimuthal mean
(right) against the star's leakage rendered on the detector and the raw
contrast evaluated from the native tables; the two leakage curves agree to
within the pixelization, the coadd sits on the zodi plateau near 3e-10 with
the planet's contrast just below it, and the coronagraph's own floor lies
five decades lower, so at 1 micron this design is limited by the zodiacal
light and not by starlight.
