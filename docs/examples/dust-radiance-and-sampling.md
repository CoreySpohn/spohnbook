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

# Dust radiance, rays and sampling

This page works through the reference cases of [the dust models chapter](../conventions/dust-models.md) with the analytic fixtures in `tools/dust_reference_cases.py`. Nothing here runs a library dust model or downloads data: every expected value is derived from geometry or algebra, so the page shows what a correct kernel must reproduce rather than what any current implementation does. Constants and unit conversions come from hwoutils; figures are drawn with eyepiece under the light hwostyle mode.

The fixture module lives in the repository's `tools/` directory, so the page finds the repository root from wherever it runs (the docs build executes it from `docs/examples/`).

```{code-cell} python
import importlib.util
import math
from pathlib import Path

import eyepiece as ep
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from hwoutils.constants import arcsec2rad, c, h, um2nm

root = next(
    (
        p
        for p in (Path.cwd(), *Path.cwd().parents)
        if (p / "tools" / "dust_reference_cases.py").exists()
    ),
    None,
)
if root is None:
    raise FileNotFoundError("run this page from inside the spohnbook repository")
spec = importlib.util.spec_from_file_location(
    "dust_reference_cases", root / "tools" / "dust_reference_cases.py"
)
drc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drc)

hwostyle.use("light")
plt.rcParams["savefig.dpi"] = 120  # keeps the baked page images small
```

## Only the positive half-ray counts

A ray is $\boldsymbol x(s)=\boldsymbol x_{\rm obs}+s\hat{\boldsymbol n}$ with $s\ge0$. Take a sphere of radius 2 AU at the origin with constant photon emissivity $j=3$ photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$ per AU. The radiance along a ray is $j$ times the length of the half-ray inside the sphere, which the quadratic $|\boldsymbol x_{\rm obs}+s\hat{\boldsymbol n}|^2=R^2$ gives exactly.

```{code-cell} python
radius, emissivity = 2.0, 3.0
cases = [
    ("inside, toward +x", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 3.0),
    ("inside, toward -x", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 9.0),
    ("outside, toward -x", (5.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 12.0),
    ("outside, toward +x", (5.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0),
    ("grazing, toward +y", (2.0, -3.0, 0.0), (0.0, 1.0, 0.0), 0.0),
]
print(f"{'ray':<20}{'path [AU]':>10}{'radiance':>10}{'full line':>11}")
for name, obs, direc, expected in cases:
    path = drc.sphere_path_length(obs, direc, radius)
    value = drc.sphere_ray_radiance(obs, direc, radius, emissivity)
    chord = drc.sphere_chord_length(obs, direc, radius)
    assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
    print(f"{name:<20}{path:>10.3f}{value:>10.3f}{chord:>11.3f}")
```

The last column integrates the whole line, both signs of $s$. It returns 4 AU for both inside rays and for the outside observer looking away, the error a kernel makes when it ignores which way the observer is looking. Accumulating along the ray shows where each observer collects its light:

```{code-cell} python
s = np.linspace(0.0, 8.0, 401)
fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
disk = hwostyle.roles.disk
for obs, ls, label in (((1.0, 0.0, 0.0), "-", "inside"), ((5.0, 0.0, 0.0), "--", "outside")):
    curve = drc.cumulative_sphere_radiance(obs, (-1.0, 0.0, 0.0), radius, emissivity, s)
    ax.plot(s, curve, color=disk, ls=ls)
    ax.text(s[-1], curve[-1] + 0.3, f"{label} observer: {curve[-1]:g}", ha="right")
ax.set(xlabel="distance along the ray, s [AU]", ylim=(0, 13.5))
ax.set_ylabel("cumulative radiance\n[photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$]")
ax.spines[["top", "right"]].set_visible(False)
plt.show()
```

## Scattering angle and illumination angle

The phase function takes the scattering angle $\Theta$ between the incident propagation direction (star to grain) and the direction toward the observer, $-\hat{\boldsymbol n}$. For an observer far along $+x$ looking back along $-x$:

```{code-cell} python
direction = (-1.0, 0.0, 0.0)
for label, grain in (("between star and observer", (1.0, 0.0, 0.0)),
                     ("beside the star", (0.0, 1.0, 0.0)),
                     ("behind the star", (-1.0, 0.0, 0.0))):
    theta = math.degrees(drc.scattering_angle(grain, direction))
    alpha = math.degrees(drc.illumination_angle(grain, direction))
    print(f"{label:<26} Theta = {theta:5.1f} deg   alpha = {alpha:5.1f} deg")
```

A grain between the star and the observer scatters forward ($\Theta=0$) while a planet at the same place would show its dark side ($\alpha=\pi$). Passing $\alpha$ where $\Theta$ is expected swaps forward and back scattering.

## Radiance versus pixel flux

A pixel-integrated flux is the radiance integrated over the pixel's solid angle. The fixture integrates a Gaussian clump of peak radiance 7 photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$ and width 0.25 arcsec exactly over each pixel of a fixed 2 arcsec field, using the small-angle pixel solid angle.

```{code-cell} python
peak, sigma, field = 7.0, 0.25, 2.0
grids = (24, 48, 96)
maps = [drc.gaussian_patch_pixel_flux(peak, sigma, field, n) for n in grids]
for n, img in zip(grids, maps):
    assert np.all(np.isfinite(img)) and img.min() >= 0.0  # before any log display
    naive = drc.sampled_radiance_sum(1.0, n)  # unweighted sum of unit samples
    print(f"{n:3d} px/side: peak pixel {img.max():.4e}, sum {img.sum():.10e}, "
          f"unweighted-sum growth x{naive / 24**2:.0f}")

half = field / 2
res = ep.compare_row(
    maps,
    titles=[f"{n} x {n}" for n in grids],
    norm="log",
    vmax=maps[0].max(),
    vmin=1e-3 * maps[0].max(),  # three decades below the coarsest peak
    extent=(-half, half, -half, half),
    cbar_label="photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ per pixel",
    panel_size=2.6,
)
for ax in res.axes:
    ep.label_arcsec(ax)
plt.show()
```

The three maps share one logarithmic norm, so the colors themselves show each pixel carrying a quarter of the light at every refinement, while the printed sums agree to the last digit shown. Summing radiance samples without the solid-angle weight instead grows by four per refinement: that is the per-pixel contract failure the radiometry chapter records for sampled disk maps.

## A physical radiance anchor: the S10 unit

Leinert et al. (1998, section 8.3) quote zodiacal brightness in S10 units, with 1 S10 $=1.28\times10^{-8}$ W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$ at 500 nm. Converting that energy radiance to photon units per nm and per arcsec$^2$ uses the photon energy at 500 nm, the $\mu$m-to-nm density change, and the arcsec$^2$ solid angle:

```{code-cell} python
s10_w_m2_sr_um = 1.28e-8
lam_m = 500e-9
per_nm = s10_w_m2_sr_um / um2nm  # W m^-2 sr^-1 nm^-1
photon_sr = per_nm / (h * c / lam_m)  # photon s^-1 m^-2 sr^-1 nm^-1
photon_arcsec2 = photon_sr * arcsec2rad**2
print(f"1 S10 at 500 nm = {photon_sr:.4e} photon s^-1 m^-2 nm^-1 sr^-1")
print(f"               = {photon_arcsec2:.4e} photon s^-1 m^-2 nm^-1 arcsec^-2")
print(f"ecliptic pole, 60 S10 (Leinert Table 16): {60 * photon_arcsec2:.4e} per arcsec^2")
```

Each of the three factors appears exactly once. Converting at 500 nm is correct for a monochromatic value; a band value integrates the photon density over the passband instead, as the next section shows.

## Converting to photons inside the band

For an energy spectrum $B_\lambda\propto\lambda^k$, converting at every wavelength and then integrating over the band is exact; integrating energy first and converting once at the band center is an approximation.

```{code-cell} python
b0, lam0 = 1.0e-6, 550.0
print(f"{'k':>5}" + "".join(f"{f'{int(f * 100)}% band':>12}" for f in (0.1, 0.2, 0.5)))
for k in (0.0, -4.0, 2.0):
    row = []
    for frac in (0.1, 0.2, 0.5):
        lo, hi = lam0 * (1 - frac / 2), lam0 * (1 + frac / 2)
        exact = drc.power_law_photon_band_integral(b0, lam0, k, lo, hi)
        center = drc.center_conversion_band_integral(b0, lam0, k, lo, hi)
        rel = round(100 * (center / exact - 1), 6) + 0.0  # no negative zero
        row.append(f"{rel:+11.3f}%")
    print(f"{k:5.1f}" + " ".join(row))
```

The approximation is exact only for $k=0$, where the photon density is linear in wavelength. Its error grows with bandwidth and spectral slope, so a zodi or exozodi color law needs the in-band conversion, not a center-wavelength shortcut.

## Distance: what changes and what does not

A face-on disk of uniform radiance and physical radius 3 AU around a star of photon luminosity $4\times10^{44}$ photon s$^{-1}$ nm$^{-1}$, seen from 10 pc and from 20 pc:

```{code-cell} python
lum = 4.0e44
near = drc.distant_disk_observables(7.0, 3.0, 10.0, lum)
far = drc.distant_disk_observables(7.0, 3.0, 20.0, lum)
for key in ("radiance", "solid_angle_sr", "disk_flux", "star_flux", "contrast"):
    print(f"{key:<15} 10 pc {near[key]:.4e}   20 pc {far[key]:.4e}   ratio {far[key] / near[key]:.4f}")

d_omega = (0.01 * arcsec2rad) ** 2  # one 10 mas pixel
ratio = drc.pixel_host_contrast(7.0, d_omega, 20.0, lum) / drc.pixel_host_contrast(7.0, d_omega, 10.0, lum)
print(f"contrast of a fixed 10 mas pixel, 20 pc / 10 pc: {ratio:.4f}")
```

Radiance along a physical sightline and the whole-disk contrast do not change; the solid angle and both fluxes drop by four. A fixed angular pixel, however, covers four times more of the disk at 20 pc, so its contrast rises by four. The distance dependence lives in the physical pixel area $D^2\Delta\Omega$.

## Two parameters, one constraint

When a model's brightness is a dust level times a free spectral amplitude, the data see only the product:

```{code-cell} python
morph = np.array([[0.0, 1.0], [2.0, 0.5]])
a = drc.product_amplitude_image(2.0, 0.15, morph)
b = drc.product_amplitude_image(1.0, 0.30, morph)
jac = drc.product_amplitude_jacobian(2.0, 0.15, morph)
print("max |a - b| =", np.abs(a - b).max())
print("column norms of the Jacobian:", np.linalg.norm(jac, axis=0))
print("rank of the Fisher matrix J^T J:", np.linalg.matrix_rank(jac.T @ jac))
```

Both derivatives are nonzero, so a gradient-based fit runs, but the Fisher matrix has rank one: only nzodis times albedo is identified. Calibrate one factor, fit the declared product, or state the prior that separates them.

## Phase-function normalization

A Henyey-Greenstein phase function integrates to one over $4\pi$ sr for any asymmetry parameter; a mixture does only when its weights sum to one.

```{code-cell} python
mu, w = np.polynomial.legendre.leggauss(4000)
for g in (-0.5, 0.0, 0.3, 0.9):
    total = 2 * math.pi * np.sum(w * drc.henyey_greenstein(mu, g))
    print(f"g = {g:+.1f}: integral over 4 pi sr = {total:.12f}")
mixture = drc.henyey_greenstein_mixture(mu, (0.7, 0.3), (0.8, -0.2))
print(f"mixture (0.7, 0.3): {2 * math.pi * np.sum(w * mixture):.12f}")
try:
    drc.henyey_greenstein_mixture(mu, (0.7, 0.7), (0.8, -0.2))
except ValueError as err:
    print("weights (0.7, 0.7) rejected:", err)
```
