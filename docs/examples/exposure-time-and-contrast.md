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

# Exposure time and contrast

This page reads a coronagraph design from a yield input package with yippy,
draws the two curves that summarize it, wraps it in the hardware description
that jaxedith consumes through optixstuff, and computes how long a 7.2 m
telescope with this coronagraph needs to detect an Earth twin at 10 pc as a
function of where the planet sits. Every figure is drawn with eyepiece under
the light hwostyle mode. Float64 is enabled before any array exists, because
the contrast values on this page span twenty decades. yippy logs each loading
step at info level; the page silences everything below an error.

```{code-cell} python
import eyepiece as ep
import hwostyle
import jax
import jax.numpy as jnp
import jaxedith
import matplotlib.pyplot as plt
import numpy as np
import optixstuff as ox
import yippy
from hwoutils import constants as const
from hwoutils.conversions import (
    au_to_arcsec,
    jy_to_photons_per_nm_per_m2,
    lambda_d_to_arcsec,
    mag_to_flux_jy,
)
from matplotlib.patches import Circle

jax.config.update("jax_enable_x64", True)
hwostyle.use("light")
plt.rcParams["savefig.dpi"] = 120  # keeps the baked page images small
yippy.logger.setLevel("ERROR")
GRAY = "#999999"  # scenery: reference lines, apertures, floors
```

`fetch_yip` downloads the package the first time and returns the cached
directory after that. `Coronagraph` reads its FITS tables: off-axis PSFs at 40
radial offsets (this package is radially symmetric), stellar intensity maps at
12 stellar angular diameters, and a sky transmission map. The header carries
the telescope this design was made for, a 7.2 m circumscribed aperture with
17.5 percent of the circumscribed area not collecting light, at a design
wavelength of 1000 nm over a 900 to 1100 nm band, sampled at 0.25 lambda/D per
pixel. Performance curves are computed at load time for a point-source star,
and this design nulls a point source completely: its point-source stellar map
peaks below 1e-28 per pixel, so the raw contrast sits at yippy's 1e-20
numerical floor everywhere. The leakage that matters is the star's finite
disk. The Sun at 10 pc is 0.93 mas across, 0.032 lambda/D at this wavelength,
and the nearest tabulated diameter is 0.0316 lambda/D.
`compute_all_performance_curves(stellar_diam=...)` recomputes the raw
contrast and the core mean intensity for that disk in place (nothing is
written to disk), and `EqxCoronagraph` then wraps the tables as JAX
interpolators.

```{code-cell} python
yip_path = yippy.fetch_yip("eac1_optimal_order_6_1d")
yip = yippy.Coronagraph(yip_path)
diameter_m = yip.header.diameter.to_value("m")
wavelength_nm = yip.header.lambda0.to_value("nm")
dlambda_nm = yip.header.maxlam.to_value("nm") - yip.header.minlam.to_value("nm")
lod_arcsec = float(lambda_d_to_arcsec(1.0, wavelength_nm, diameter_m))

R_SUN_M = 6.957e8  # IAU 2015 nominal solar radius
T_SUN_K = 5772.0  # IAU 2015 nominal solar effective temperature
dist_pc = 10.0
dist_m = dist_pc * const.pc2m
star_diam_lod = 2.0 * R_SUN_M / dist_m * const.rad2arcsec / lod_arcsec
diams = yip.stellar_intens.diams
k_star = int(np.argmin(np.abs(diams.value - star_diam_lod)))
yip.compute_all_performance_curves(stellar_diam=diams[k_star], save_to_fits=False)
coro = yippy.EqxCoronagraph(yippy_coro=yip)
print(
    f"D = {diameter_m} m, lambda = {wavelength_nm:.0f} nm, "
    f"band {dlambda_nm:.0f} nm, 1 lambda/D = {lod_arcsec * 1e3:.2f} mas\n"
    f"star {star_diam_lod:.4f} lambda/D across, tabulated at "
    f"{float(diams[k_star].value):.4f}; IWA {coro.IWA:.2f}, OWA {coro.OWA:.1f} "
    f"lambda/D, {coro.pixel_scale_lod} lambda/D per pixel"
)
```

The raw contrast at a separation is the stellar flux divided by the planet
flux inside the same 0.7 lambda/D radius aperture, each measured on the
tables at that offset; the core throughput is the fraction of the planet's
light inside that aperture. yippy defines the inner working angle as the
separation where the throughput reaches half its maximum, and the outer
working angle as the half-width of the PSF image. `plot_contrast_curve`
shades both working angles out to the axes edge and draws each `floors` entry
as a dashed reference line. The floor here is `noise_floor_exosims`, which is
`max(raw contrast, 1e-10) / 30`: a post-processing gain of 30 over a residual
that is never trusted below 1e-10, so for this design it sits at 3.3e-12
across the whole dark zone, above the raw contrast. The other floor yippy
exposes, `noise_floor_ayo`, is the core mean intensity over 30 and is a
per-pixel quantity rather than a contrast; it reappears in the last figure as
a count rate. Both curves stop at 29.5 lambda/D, the last tabulated offset
inside the image.

```{code-cell} python
sep = np.linspace(0.3, 29.5, 400)
contrast = np.asarray(coro.raw_contrast(sep))
throughput = np.asarray(coro.throughput(sep))
floor = np.asarray(coro.noise_floor_exosims(sep))
sep_label = r"separation [$\lambda/D$]"

fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained")
ep.plot_contrast_curve(
    sep,
    contrast,
    ax=ax_c,
    iwa=coro.IWA,
    owa=coro.OWA,
    floors=[(sep, floor, "floor")],
    color=hwostyle.roles.star,
    floor_kw={"color": GRAY},
    xlabel=sep_label,
    ylabel="raw contrast",
)
ax_c.set_xlim(0, 34)
ax_c.text(29.5, contrast[-1], " raw", color=hwostyle.roles.star, va="center")
ax_c.text(29.5, floor[-1], " floor", color=GRAY, va="center")
ep.plot_radial(
    sep,
    throughput,
    ax=ax_t,
    color=hwostyle.roles.planet,
    xlabel=sep_label,
    ylabel="core throughput",
)
half_max = throughput.max() / 2
ax_t.plot([0, coro.IWA], [half_max, half_max], color=GRAY, lw=1)
for angle, name in ((coro.IWA, "IWA"), (coro.OWA, "OWA")):
    ax_t.axvline(angle, color=GRAY, lw=1)
    ax_t.text(
        angle, 0.98, name, transform=ax_t.get_xaxis_transform(), ha="center", va="top"
    )
ax_t.set_xlim(0, 34)
ax_t.set_ylim(0, 0.7)
for ax in (ax_c, ax_t):
    ax.spines[["top", "right"]].set_visible(False)
```

Figure 1: the raw contrast of the design for a 0.0316 lambda/D star with the
1e-10 / 30 post-processing floor (left) and the core throughput in a 0.7
lambda/D aperture with the half-maximum rule that defines the 1.57 lambda/D
inner working angle (right); the shaded regions are outside the working
angles.

`create_psfs` synthesizes the off-axis PSF at any position from the tabulated
offsets, and returns each image as the fraction of that source's photons
landing in each pixel, so a PSF the coronagraph does not touch sums to one.
The three separations below sit on pixel centers; each panel is a 33 pixel
window centered on the nominal planet position, so its axes read relative to
that position, and `compare_row` draws the three windows under one shared log
norm. The gray circle is the 0.7 lambda/D aperture centered on the PSF peak,
which is where yippy places it when it measures the throughput, and the gray
star in the first panel marks the star. Inside the inner working angle the
coronagraph passes 31 percent of the planet light and pushes the peak
outward; at 12 and 28 lambda/D the PSF is the same core, with 0.99 and 0.97
of the light in the image and 0.60 in the aperture.

```{code-cell} python
seps_psf = (1.25, 12.0, 28.0)
half = 16  # pixels on either side of the nominal position: 4 lambda/D
pix = coro.pixel_scale_lod
cx, cy = int(coro.center_x), int(coro.center_y)
psfs = np.asarray(coro.create_psfs(jnp.asarray(seps_psf), jnp.zeros(3)))
crops = [
    psf[cy - half : cy + half + 1, ix - half : ix + half + 1]
    for psf, ix in zip(psfs, [cx + round(r / pix) for r in seps_psf], strict=True)
]
window = ep.extent_lod_from_pixels(2 * half + 1, pix)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), layout="constrained")
ep.compare_row(
    crops,
    [rf"$r$ = {r} $\lambda/D$" for r in seps_psf],
    axes=axes,
    norm="log",
    vmin=1e-7,
    extent=window,
    cbar_label="fraction of planet photons per pixel",
)
for ax, crop in zip(axes, crops, strict=True):
    iy, ix = np.unravel_index(crop.argmax(), crop.shape)
    ax.add_patch(
        Circle(((ix - half) * pix, (iy - half) * pix), 0.7, fill=False, color=GRAY)
    )
    ax.set_xlabel(r"$x - r$ [$\lambda/D$]")
axes[0].set_ylabel(r"$y$ [$\lambda/D$]")
axes[0].plot(-seps_psf[0], 0.0, marker="*", color=GRAY, ms=9)
print("fraction of planet light in each image:", psfs.sum(axis=(1, 2)).round(3))
```

Figure 2: the off-axis PSF just inside the inner working angle, in the dark
zone, and near the outer working angle, each in a window centered on the
nominal planet position under one log norm, with the 0.7 lambda/D throughput
aperture on the PSF peak.

`OpticalPath` is the hardware object jaxedith reads. Its `primary` supplies
the diameter that turns lambda/D into arcseconds and the collecting area;
`SimplePrimary` computes the area as the circumscribed disk times
`shape_factor`, which here removes the 17.5 percent the package header
declares as not collecting. `system_throughput` is the product of the
`attenuating_elements`, and it is the only throughput jaxedith applies to the
photon rates; the detector's `quantum_efficiency` is read only by the thermal
term, so the QE enters the chain as its own `ConstantThroughput`. The
`coronagraph` is the yippy object behind the `YippyCoronagraph` adapter, which
serves the throughput, core area, core mean intensity, and occulter
transmission curves at any separation. The `IdealDetector` fields the
calculator reads are the plate scale, which sets how many pixels the aperture
covers, the dark current in electrons per pixel per second, the read noise in
electrons per pixel per read, and the clock-induced charge in electrons per
pixel per frame; the plate scale below is Nyquist at the design wavelength,
and the read noise is zero, as for a photon-counting detector, so the read
time never enters.

```{code-cell} python
optical_path = ox.OpticalPath(
    primary=ox.SimplePrimary(
        diameter_m=diameter_m, obscuration=0.0, shape_factor=1.0 - coro.frac_obscured
    ),
    attenuating_elements=(
        ox.ConstantThroughput(throughput=0.5, name="optics"),
        ox.ConstantThroughput(throughput=0.9, name="detector QE"),
    ),
    coronagraph=ox.YippyCoronagraph(backend=coro),
    detector=ox.IdealDetector(
        pixel_scale_arcsec=0.5 * lod_arcsec,
        shape=(1024, 1024),
        quantum_efficiency=0.9,
        dark_current_rate_e_per_s=3e-5,
        read_noise_e=0.0,
        clock_induced_charge_rate_e_per_frame=1.3e-3,
    ),
)
print(optical_path)
```

`ETCScene` holds the astrophysics as dimensionless ratios against a flux
zero point `F0` in photons per second per square meter per nanometer. `F0` is
the AB zero point at the design wavelength, the same choice jaxedith's
system-mode wrappers make. `Fs_over_F0` is the star's flux over that zero
point: a 5772 K blackbody with the solar radius at 10 pc, which comes out at AB
magnitude 4.5 at 1000 nm. `Fp_over_Fs` is the planet's contrast: a Lambert
sphere of Earth's radius at 1 AU seen at quadrature, where the phase function
is 1 / pi, with a geometric albedo of 0.2, which gives 1.2e-10. `Fzodi` is the
local zodiacal surface brightness in units of `F0` per square arcsecond,
taken from jaxedith's own `zodi_fn_ayo`, which evaluates the AYO default of 22
mag per square arcsecond at V with a color correction to the requested
wavelength; that callable reads the wavelength from an `ExposureConfig` and
ignores its other two arguments. `Fexozodi` is the exozodiacal surface
brightness at 1 AU from the star, set here to three times the local zodi; the
calculator scales it to the planet's projected separation as `1 / (dist_pc x
sep_arcsec)^2`, so a scene is built per separation with `sep_arcsec` matching
`separation_lod`. The planet contrast stays fixed as the separation sweeps,
so the curves isolate the instrument's response.

```{code-cell} python
F0 = float(jy_to_photons_per_nm_per_m2(mag_to_flux_jy(0.0), wavelength_nm))
nu_hz = const.c / (wavelength_nm * const.nm2m)
planck_nu = 2 * const.h * nu_hz**3 / const.c**2
planck_nu /= np.expm1(const.h * nu_hz / (const.k_B * T_SUN_K))
star_flux_jy = np.pi * planck_nu * (R_SUN_M / dist_m) ** 2 / const.Jy
Fs_over_F0 = star_flux_jy / float(mag_to_flux_jy(0.0))
albedo = 0.2
Fp_over_Fs = albedo / np.pi * const.Rearth2AU**2
exposure = ox.ExposureConfig(
    start_time_jd=jnp.asarray(0.0),
    exposure_time_s=jnp.asarray(0.0),
    central_wavelength_nm=jnp.asarray(wavelength_nm),
    bin_width_nm=jnp.asarray(dlambda_nm),
    position_angle_deg=jnp.asarray(0.0),
)
Fzodi = float(jaxedith.zodi_fn_ayo(None, exposure, None))
n_zodi = 3.0
sep_earth = float(au_to_arcsec(1.0, dist_pc)) / lod_arcsec


def scene_at(sep_lod):
    """The Earth-twin scene with the exozodi evaluated at this separation."""
    return jaxedith.ETCScene(
        F0=F0,
        Fs_over_F0=Fs_over_F0,
        Fp_over_Fs=Fp_over_Fs,
        Fzodi=Fzodi,
        Fexozodi=n_zodi * Fzodi,
        dist_pc=dist_pc,
        sep_arcsec=sep_lod * lod_arcsec,
    )


print(
    f"F0 = {F0:.3e} ph/s/m^2/nm, star AB mag {-2.5 * np.log10(Fs_over_F0):.2f}, "
    f"planet contrast {Fp_over_Fs:.2e} (dMag {-2.5 * np.log10(Fp_over_Fs):.2f}),\n"
    f"zodi {Fzodi:.2e} F0/arcsec^2 ({-2.5 * np.log10(Fzodi):.2f} mag/arcsec^2), "
    f"exozodi {n_zodi * Fzodi:.2e} F0/arcsec^2 at 1 AU, "
    f"Earth twin at {sep_earth:.2f} lambda/D"
)
```

Every rate jaxedith returns is a count rate integrated over the photometric
aperture: the flux chain `F0 x ratio x area x system_throughput x bandwidth`
times the coronagraph factor for that term, which its docstrings write as
electrons per second. `exptime_ayo` solves the AYO equation
`t = SNR^2 (Cp + 2 Cb) / (Cp^2 - (SNR Cnf)^2)`, where `Cp` is the planet
rate, `Cb` the sum of the background rates, the factor 2 is the AYO
assumption that the background is measured twice, and `Cnf` is a noise-floor
rate that the SNR multiplies: the stellar leakage divided by `ppfact` (30
here) and by the pixel area. `exptime_exosims_det` solves
`t = SNR^2 Cb / (Cp^2 - (SNR Csp)^2)` with `Csp = leakage x ppfact`, so the
same post-processing gain is `1 / 30` in that variant. Both are pure JAX, so
`jax.vmap` sweeps them over separation. The budget beside the curves is the
per-term rates at the Earth twin's separation from the public per-term
functions, ordered as the calculator sums them: eyepiece has no bar
primitive, so the budget is a matplotlib dot plot in palette colors.

```{code-cell} python
snr, ppf = 7.0, 30.0


def t_ayo(sep_lod):
    return jaxedith.exptime_ayo(
        optical_path,
        scene_at(sep_lod),
        wavelength_nm,
        sep_lod,
        dlambda_nm,
        snr,
        ppfact=ppf,
    )


def t_det(sep_lod):
    return jaxedith.exptime_exosims_det(
        optical_path,
        scene_at(sep_lod),
        wavelength_nm,
        sep_lod,
        dlambda_nm,
        snr,
        ppfact=1.0 / ppf,
    )


def rates_at(sep_lod):
    """The per-term count rates the calculator sums, in the order it sums them."""
    scene = scene_at(sep_lod)
    args = (optical_path, wavelength_nm, sep_lod, dlambda_nm, F0)
    planet = jaxedith.planet_signal(*args, Fs_over_F0, Fp_over_Fs)
    leakage = jaxedith.stellar_leakage(*args, Fs_over_F0)
    zodi = jaxedith.zodi_background(*args, Fzodi)
    exozodi = jaxedith.exozodi_background(
        *args, scene.Fexozodi, scene.dist_pc, scene.sep_arcsec
    )
    detector = jaxedith.detector_noise(
        optical_path, wavelength_nm, sep_lod, planet + leakage + zodi + exozodi
    )
    floor = jaxedith.stellar_noise_floor(*args, Fs_over_F0, ppfact=ppf)
    return {
        "planet": planet,
        "leakage": leakage,
        "zodi": zodi,
        "exozodi": exozodi,
        "detector": detector,
        "floor": floor,
    }


seps = jnp.linspace(0.3, 29.5, 400)
t_ayo_s = np.asarray(jax.vmap(t_ayo)(seps))
t_det_s = np.asarray(jax.vmap(t_det)(seps))
budget = {name: float(v) for name, v in rates_at(sep_earth).items()}
t_earth = float(t_ayo(sep_earth))
COLORS = {
    "planet": hwostyle.roles.planet,
    "leakage": hwostyle.roles.star,
    "zodi": hwostyle.palette.green,
    "exozodi": hwostyle.roles.disk,
    "detector": "#4A4A4A",
    "floor": GRAY,
}

fig, (ax_t, ax_b) = plt.subplots(
    1, 2, figsize=(8.8, 3.6), layout="constrained", width_ratios=[1.3, 1]
)
ayo = ep.plot_contrast_curve(
    seps, t_ayo_s, ax=ax_t, iwa=coro.IWA, owa=coro.OWA, xlabel=sep_label
)
exo = ep.plot_contrast_curve(seps, t_det_s, ax=ax_t)
ax_t.set_ylabel(f"exposure time to SNR {snr:.0f} [s]")
ax_t.set_xlim(0, 34)
ax_t.set_ylim(1e2, 1e7)
ax_t.text(29.8, t_ayo_s[-1], "AYO", color=ayo.artists["line"].get_color(), va="center")
ax_t.text(
    29.8, t_det_s[-1], "EXOSIMS", color=exo.artists["line"].get_color(), va="center"
)
ax_t.plot(sep_earth, t_earth, marker="o", color=plt.rcParams["text.color"])
ax_t.annotate(
    f"Earth twin, 1 AU at 10 pc: {t_earth / 60:.0f} min",
    (sep_earth, t_earth),
    (sep_earth + 1.5, t_earth * 12),
    arrowprops={"arrowstyle": "-", "color": GRAY},
)
names = list(budget)
for i, name in enumerate(names):
    ax_b.plot([1e-8, budget[name]], [i, i], color=GRAY, lw=1)
    ax_b.plot(budget[name], i, marker="o", color=COLORS[name])
    ax_b.text(budget[name] * 1.6, i, f"{budget[name]:.2g}", va="center")
ax_b.set_xscale("log")
ax_b.set_xlim(1e-8, 30)
ax_b.set_yticks(
    range(len(names)), [name.replace("floor", "noise floor") for name in names]
)
ax_b.invert_yaxis()
ax_b.set_xlabel(f"count rate at {sep_earth:.1f} " + r"$\lambda/D$ [e/s]")
for ax in (ax_t, ax_b):
    ax.spines[["top", "right"]].set_visible(False)
```

Figure 3: the exposure time to SNR 7 on the Earth twin as a function of its
separation under the AYO and EXOSIMS detection equations, with the Earth
twin's own separation marked (left), and the per-term count rates at that
separation, where the exozodi and the local zodi outweigh the planet and the
nulled star contributes nothing measurable (right).

The exposure-time curve is the count-rate curves seen through the equation.
The planet rate is the throughput curve of figure 1 times a constant, so it
collapses inside the inner working angle and is nearly flat beyond 3 lambda/D.
The
leakage rate is the core mean intensity, the azimuthal mean of the stellar map
on the left in fraction of stellar photons per pixel, times the aperture area
in square lambda/D and the flux chain; jaxedith applies no pixel-area
conversion in this term, and does apply one in the noise-floor term, which is
why the floor rate sits close to the leakage rate here instead of 30 times
below it. Both fall by three decades across the dark zone, the shape of the
contrast curve. The zodi rate follows the occulter transmission, which is flat
beyond 1 lambda/D, and the exozodi rate is the same transmission divided by
the squared separation, so it is what the exposure time keeps gaining from as
the planet moves out. The detector rate is the pixel count in the aperture
times the dark current plus a clock-induced-charge term that scales with the
total photon rate through a photon-counting frame-rate constant of 6.73
inherited from pyEDITH, so it tracks the exozodi.

```{code-cell} python
rates = {name: np.asarray(v) for name, v in jax.vmap(rates_at)(seps).items()}
star_map = np.asarray(coro.stellar_intens(float(diams[k_star].value)))

fig, (ax_m, ax_r) = plt.subplots(
    1, 2, figsize=(9.0, 3.8), layout="constrained", width_ratios=[1, 1.35]
)
ep.imshow_log(
    star_map,
    ax=ax_m,
    extent=ep.extent_lod_from_pixels(star_map.shape[0], pix),
    vmin=1e-19,
    cbar_label="fraction of stellar photons per pixel",
)
ep.label_lod(ax_m)
ax_m.set_title(rf"stellar leakage, {float(diams[k_star].value):.3f} $\lambda/D$ star")
nudge = {"planet": 2.0, "zodi": 0.5, "exozodi": 1.5, "detector": 0.65, "leakage": 1.0}
for name, factor in nudge.items():
    ep.plot_radial(seps, rates[name], ax=ax_r, color=COLORS[name], log=True)
    ax_r.text(29.8, rates[name][-1] * factor, name, color=COLORS[name], va="center")
ep.plot_radial(seps, rates["floor"], ax=ax_r, color=GRAY, line_kw={"ls": "--"})
ax_r.text(29.8, rates["floor"][-1] * 0.4, "noise floor", color=GRAY, va="center")
ax_r.set_xlim(0, 36)
ax_r.set_ylim(1e-10, 3)
ax_r.set_xlabel(sep_label)
ax_r.set_ylabel("count rate [e/s]")
ax_r.spines[["top", "right"]].set_visible(False)
```

Figure 4: the stellar leakage map for the 0.0316 lambda/D star (left) and the
per-term count rates against separation (right): the planet rate follows the
throughput curve, the leakage and noise-floor rates follow the contrast
curve, and the exozodi rate falls as the inverse square of the separation.
