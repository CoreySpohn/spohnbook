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

# PSF and speckles

This page builds a circular telescope pupil with physicaloptix, propagates it
to an Airy point spread function, puts a charge-6 vortex coronagraph in the
optical path, linearizes that path in a Fourier wavefront basis, and drives the
resulting speckle field with a drifting wavefront through tiptilt. Every image
is drawn with eyepiece under the light hwostyle mode. Float64 is enabled before
any array exists, because the coronagraph null sits ten decades below the star.


| Scope | This page |
|---|---|
| Purpose | Propagate a circular pupil to an Airy pattern and a vortex coronagraph with physicaloptix, linearize the path, and drive a speckle field with tiptilt |
| Model restrictions | Scalar Fraunhofer propagation on a half-pixel-offset grid; the drift uses `tiptilt.TabulatedSpeckleField` because the direct call fails at the recorded versions (`examples-tiptilt-speckle-call` on the [limitations page](../evidence/limitations.md)) |
| Evidence kind | Executable tutorial, with one bound check: the energy on the finite focal grid does not exceed the pupil energy |
| Data sources | None |
| Applicable profile | None adopted; the {ref}`image coordinates and PSFlet origin decision <decision-image-coordinates-and-psflet-origin>` is pending |
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
import physicaloptix as po
import tiptilt
from IPython.display import HTML

jax.config.update("jax_enable_x64", True)
hwostyle.use("light")
plt.rcParams["savefig.dpi"] = 120  # keeps the baked page images small
```

physicaloptix propagates in a dimensionless core. A pupil `Grid` spans one
pupil diameter, so its coordinates are in units of D; a focal `Grid` has a
pixel scale in lambda/D; both are half-pixel offset, so no sample sits at
r = 0. A `Field` is complex data tagged with the grid and plane it lives in.
The aperture below is a gray-pixel disk, each pupil sample holding the fraction
of its cell inside the circle (8 x 8 subsamples). `Fraunhofer` is the
pupil-to-focal continuous Fourier transform; it evaluates the Nyquist sampling
ratio of its kernel at construction and warns when that ratio drops below 1.

```{code-cell} python
npup, nfoc, pixscale_lod = 96, 128, 0.25
pupil_grid = po.Grid.pupil(npup)
focal_grid = po.Grid.focal(nfoc, pixscale_lod)


def gray_disk(grid, radius, subsamples=8):
    """Fraction of each pupil cell inside a circle of the given radius."""
    offsets = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * grid.dx
    inside = np.zeros((grid.npix, grid.npix))
    for dy in offsets:
        for dx in offsets:
            x, y = np.meshgrid(grid.coords + dx, grid.coords + dy)
            inside += np.hypot(x, y) <= radius
    return inside / subsamples**2


aperture = gray_disk(pupil_grid, 0.5)
entrance = po.Field(
    data=jnp.asarray(aperture, dtype=complex),
    grid=pupil_grid,
    plane=po.PlaneKind.PUPIL,
)
telescope = po.Fraunhofer(pupil_grid, focal_grid)
airy = telescope(entrance).intensity()
peak = float(airy.max())
print(
    f"sampling ratio {telescope.sampling_parameter:.2f}, "
    f"pupil energy {float(entrance.energy()):.4f}, "
    f"energy on the focal grid {float(airy.sum()) * focal_grid.weights:.4f}"
)
# Bound check: a finite focal grid cannot hold more energy than the pupil sent.
assert float(airy.sum()) * focal_grid.weights <= float(entrance.energy()) * (1 + 1e-9)
```

`extent_lod` turns a vector of pixel-center coordinates into the pixel-edge
extent that `imshow` needs, and it works for the pupil coordinates in D just as
it does for focal coordinates in lambda/D. `compare_row` draws its panels under
one shared norm, and a single-panel call with `norm="linear"` is the way to draw
a transmission map; `imshow_log` is for an intensity that spans decades. The
Airy PSF is divided by its own peak on this grid.

```{code-cell} python
pupil_extent = ep.extent_lod(pupil_grid.coords)
focal_extent = ep.extent_lod(focal_grid.coords)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), layout="constrained")
ep.compare_row(
    [aperture],
    ["Entrance pupil"],
    axes=axes[0],
    norm="linear",
    extent=pupil_extent,
    cmap=hwostyle.cmaps.mask,
    cbar_label="transmission",
)
axes[0].set_xlabel("x [D]")
axes[0].set_ylabel("y [D]")
ep.imshow_log(
    airy / peak,
    ax=axes[1],
    extent=focal_extent,
    vmin=1e-7,
    cbar_label="intensity / peak",
)
axes[1].set_title("Airy PSF")
ep.label_lod(axes[1])
```

Figure 1: the gray-pixel entrance pupil (left) and its Airy PSF over plus or
minus 16 lambda/D (right), normalized to the peak on a log scale spanning seven
decades.

```{figure} ../conventions/figures/explainer-d04-optical-planes-light.png
:class: only-light
:alt: Side-view diagram of an unfolded optical train with five planes from left to right: entrance pupil, DM plane, focal plane, Lyot plane and image plane, each a dashed line across a gray beam. Starlight enters from the left; a relay lens pair between the entrance pupil and the DM plane and a single lens in each later gap are marked relay and FT. The beam is parallel at the pupils and focused at the focal and image planes, and it narrows at the Lyot stop. Below the beam the hardware is named: aperture, deformable mirror, vortex phase mask, Lyot stop and detector. Below that a row of square thumbnails, grouped as what the element applies, what the light carries and what the detector records, shows a filled disk of pupil transmission, vertical OPD fringes, a six-fold phase pattern with a cyclic key, a bright ring of field amplitude just outside a solid pupil-edge circle and a dashed stop circle, an image with two speckles either side of a dark center, and, after an arrow, simulated detector counts. A bracket under the first four marks the complex field, a mark between the fourth and fifth reads |E| squared taken here, and a second bracket marks the intensity and counts.

An illustrative vortex coronagraph train, unfolded in side view with its mirrors drawn as transmissive elements; it is not a prescription for any mission. Each dashed line is a plane, a location along the beam; the glyph on it is the hardware element there; the framed thumbnail below is a sampled array of the quantity on that plane. The first three thumbnails show what an element applies, the next two what the light carries, and the last what the detector records. The beam envelope is schematic geometry, not to scale, and is not evidence of coronagraph performance. The thumbnails come from one propagated physicaloptix example: a 64 by 64 pupil grid, a 48 by 48 focal grid at 0.5 lambda/D, 550 nm, a charge-6 vortex (its phase winds six times through 2 pi around the axis; Mawet et al. 2005, ApJ 633, 1191; Foo et al. 2005, Opt. Lett. 30, 3308) and a Lyot stop, the pupil stop in the relayed pupil, of 0.8 D. The DM plane carries a 1 nm cosine ripple of 6 cycles per D standing in for a static wavefront error. The vortex phase is evaluated from its closed form, because the model does not output its internal focal-plane field. The pupil-plane thumbnails span 2 D: the path keeps only a D-wide array, which holds about 4 percent of the entrance energy at the Lyot plane, so the Lyot thumbnail evaluates the same vortex operator on a wider grid to show the light moved outside the pupil edge. Image intensities are relative to the unocculted on-axis peak $|\mathcal{F}[A](0)|^2$, not to the brightest sample of the half-pixel-offset grid. Between the entrance pupil and the image plane the model carries a complex field: each element multiplies it and each gap marked FT is a Fourier transform. The relay from the entrance pupil to the DM plane is an identity in the model, which applies the DM to the entrance array, and the model reaches the Lyot plane with an inverse transform, so it does not invert the pupil as a physical relay would. An OPD W enters as exp(+i 2 pi W/lambda) under the proposed coherent profile ({ref}`optics-coherent-phase`). The OPD is a path difference, not a mirror surface height: at normal reflection a surface displacement h gives an OPD of 2h ({ref}`Hecht 2017, Sec. 9.4.2, eq. 9.44 <source-hecht2017>`). The intensity forms only at the image plane, and the detector applies its own boundary contract ({ref}`optics-signed-intensity`): here 1e7 photons per second in a pixel at the unocculted peak, a 1 s exposure, quantum efficiency 1, a Gaussian approximation to shot noise, 1 electron of read noise, and detector pixels equal to the focal samples. The thumbnails show each array with x right and y up; whether that is the view looking downstream or upstream, and how a relay's pupil inversion is recorded, belong to the pending {ref}`image coordinates and PSFlet origin decision <decision-image-coordinates-and-psflet-origin>`. The picture is an example implementation of the proposed profile. A backend built on response tables stores only a final-plane response and does not model the intermediate planes.
```

```{figure} ../conventions/figures/explainer-d04-optical-planes-dark.png
:class: only-dark
:alt: Side-view diagram of an unfolded optical train with five planes from left to right: entrance pupil, DM plane, focal plane, Lyot plane and image plane, each a dashed line across a gray beam. Starlight enters from the left; a relay lens pair between the entrance pupil and the DM plane and a single lens in each later gap are marked relay and FT. The beam is parallel at the pupils and focused at the focal and image planes, and it narrows at the Lyot stop. Below the beam the hardware is named: aperture, deformable mirror, vortex phase mask, Lyot stop and detector. Below that a row of square thumbnails, grouped as what the element applies, what the light carries and what the detector records, shows a filled disk of pupil transmission, vertical OPD fringes, a six-fold phase pattern with a cyclic key, a bright ring of field amplitude just outside a solid pupil-edge circle and a dashed stop circle, an image with two speckles either side of a dark center, and, after an arrow, simulated detector counts. A bracket under the first four marks the complex field, a mark between the fourth and fifth reads |E| squared taken here, and a second bracket marks the intensity and counts.

An illustrative vortex coronagraph train, unfolded in side view with its mirrors drawn as transmissive elements; it is not a prescription for any mission. Each dashed line is a plane, a location along the beam; the glyph on it is the hardware element there; the framed thumbnail below is a sampled array of the quantity on that plane. The first three thumbnails show what an element applies, the next two what the light carries, and the last what the detector records. The beam envelope is schematic geometry, not to scale, and is not evidence of coronagraph performance. The thumbnails come from one propagated physicaloptix example: a 64 by 64 pupil grid, a 48 by 48 focal grid at 0.5 lambda/D, 550 nm, a charge-6 vortex (its phase winds six times through 2 pi around the axis; Mawet et al. 2005, ApJ 633, 1191; Foo et al. 2005, Opt. Lett. 30, 3308) and a Lyot stop, the pupil stop in the relayed pupil, of 0.8 D. The DM plane carries a 1 nm cosine ripple of 6 cycles per D standing in for a static wavefront error. The vortex phase is evaluated from its closed form, because the model does not output its internal focal-plane field. The pupil-plane thumbnails span 2 D: the path keeps only a D-wide array, which holds about 4 percent of the entrance energy at the Lyot plane, so the Lyot thumbnail evaluates the same vortex operator on a wider grid to show the light moved outside the pupil edge. Image intensities are relative to the unocculted on-axis peak $|\mathcal{F}[A](0)|^2$, not to the brightest sample of the half-pixel-offset grid. Between the entrance pupil and the image plane the model carries a complex field: each element multiplies it and each gap marked FT is a Fourier transform. The relay from the entrance pupil to the DM plane is an identity in the model, which applies the DM to the entrance array, and the model reaches the Lyot plane with an inverse transform, so it does not invert the pupil as a physical relay would. An OPD W enters as exp(+i 2 pi W/lambda) under the proposed coherent profile ({ref}`optics-coherent-phase`). The OPD is a path difference, not a mirror surface height: at normal reflection a surface displacement h gives an OPD of 2h ({ref}`Hecht 2017, Sec. 9.4.2, eq. 9.44 <source-hecht2017>`). The intensity forms only at the image plane, and the detector applies its own boundary contract ({ref}`optics-signed-intensity`): here 1e7 photons per second in a pixel at the unocculted peak, a 1 s exposure, quantum efficiency 1, a Gaussian approximation to shot noise, 1 electron of read noise, and detector pixels equal to the focal samples. The thumbnails show each array with x right and y up; whether that is the view looking downstream or upstream, and how a relay's pupil inversion is recorded, belong to the pending {ref}`image coordinates and PSFlet origin decision <decision-image-coordinates-and-psflet-origin>`. The picture is an example implementation of the proposed profile. A backend built on response tables stores only a final-plane response and does not model the intermediate planes.
```

The coronagraph is an `OpticalPath` of named `Stage`s, checked for plane
consistency when it is built. `MultiScaleVortex` takes the pupil field through
a ladder of progressively finer focal grids, applies the charge-6 phase ramp on
each, and returns the Lyot-plane pupil field. `SampledOptic` multiplies that
field by an 80 percent Lyot stop sampled on the same pupil grid, and refuses a
field on any other grid rather than resampling. A second `Fraunhofer` forms the
image, and `propagate` folds the field through every stage. Contrast on this
page means intensity divided by the Airy peak of the unocculted telescope on
the same focal grid. The gray-pixel edge is what makes the null deep: a binary
disk on this 96-sample grid leaks about 3e-7 at its brightest pixel, the
gray-pixel disk about 1e-9.

```{code-cell} python
lyot_stop = gray_disk(pupil_grid, 0.4)
path = po.OpticalPath(
    stages=(
        po.Stage(name="vortex", op=po.MultiScaleVortex.build(charge=6, npup=npup)),
        po.Stage(
            name="lyot",
            op=po.SampledOptic(
                transmission=jnp.asarray(lyot_stop),
                grid=pupil_grid,
                plane=po.PlaneKind.PUPIL,
            ),
        ),
        po.Stage(name="camera", op=po.Fraunhofer(pupil_grid, focal_grid)),
    )
)
coro_field, _ = path.propagate(entrance)
coro = coro_field.intensity() / peak
print(
    f"brightest vortex pixel {float(coro.max()):.1e}, "
    f"starlight through the Lyot stop "
    f"{float(coro_field.energy()) / float(entrance.energy()):.1e}"
)
```

`compare_row` builds one `LogNorm` from both images and hands the same object
to both panels, so the two are directly comparable rather than merely scaled to
look alike; `vmin` pins its floor. `radial_profile_plot` computes the azimuthal
mean of an image with `hwoutils.radial.radial_profile` and draws each call in
the next palette color, so the two curves land in different colors without a
color argument.

```{code-cell} python
fig, axes = plt.subplots(
    1, 3, figsize=(11.0, 3.5), layout="constrained", width_ratios=[1, 1, 1.15]
)
ep.compare_row(
    [airy / peak, coro],
    ["Airy PSF", "Charge-6 vortex"],
    axes=axes[:2],
    norm="log",
    extent=focal_extent,
    vmin=1e-12,
    cbar_label="contrast",
)
for ax in axes[:2]:
    ep.label_lod(ax)
ep.radial_profile_plot(airy / peak, pixscale_lod, ax=axes[2], nbins=64, log=True)
ep.radial_profile_plot(coro, pixscale_lod, ax=axes[2], nbins=64, log=True)
axes[2].set_ylabel("azimuthal mean contrast")
axes[2].set_xlim(0, 16)
axes[2].set_ylim(1e-12, 1.5)
axes[2].spines[["top", "right"]].set_visible(False)
axes[2].text(10.5, 3e-4, "Airy", color=hwostyle.palette.cyan)
axes[2].text(10.5, 3e-12, "vortex", color=hwostyle.palette.pink)
```

Figure 2: the Airy PSF and the charge-6 vortex image under one shared log norm
(left, middle) and their azimuthal mean contrast profiles (right); the vortex
holds the residual starlight below 1e-9 everywhere in the field.

`show_field` draws a complex field as its real and imaginary parts over its
amplitude and phase. The real, imaginary, and amplitude panels are each
rescaled to their own power of ten, recorded in the panel title, so a field
this faint stays readable; the phase is masked wherever the amplitude has no
support.

```{code-cell} python
fig = plt.figure(figsize=(8.0, 6.6), layout="constrained")
view = ep.show_field(
    np.asarray(coro_field.data) / np.sqrt(peak), fig=fig, extent=focal_extent
)
for cbar in view.artists["cbar"][:3]:
    cbar.set_label("field / sqrt(peak)")
for ax in view.axes[1, :]:
    ax.set_xlabel(r"$x$ [$\lambda/D$]")
for ax in view.axes[:, 0]:
    ax.set_ylabel(r"$y$ [$\lambda/D$]")
```

Figure 3: the complex focal field behind the vortex and Lyot stop, in units of
the square root of the Airy peak, with each panel's power-of-ten scale in its
title.

`fourier_dm_basis` builds the cosine and sine wavefront modes that a deformable
mirror with 24 actuators across the pupil controls between 2 and 10 cycles per
aperture, each normalized to 1 nm rms over the aperture; a mode at k cycles per
aperture puts a speckle pair at k lambda/D. `linearize` propagates every mode
through the path once and returns `E_nom`, the nominal focal field, and `G`,
the complex sensitivity of the focal field to each mode per nanometer of
coefficient: the first-order model `E(eps) = E_nom + G eps`. A Gaussian draw of
the coefficients with 2 nm total rms gives one wavefront error, and
`linearity_residual` propagates that same wavefront exactly and reports the
relative error of the linear model.

```{code-cell} python
wavelength_nm = 550.0
basis = po.fourier_dm_basis(pupil_grid, n_actuators=24, k_min=2.0, k_max=10.0)
lin = path.linearize(entrance, basis, wavelength_nm=wavelength_nm)
total_rms_nm = 2.0
sigma_nm = total_rms_nm / np.sqrt(lin.n_modes)
eps_nm = sigma_nm * jax.random.normal(jax.random.PRNGKey(0), (lin.n_modes,))
opd_nm = np.asarray(jnp.tensordot(eps_nm, basis.B, axes=1)) * aperture
e_perturbed = lin.e_nom + jnp.tensordot(eps_nm, lin.G, axes=1)
nominal = np.asarray(jnp.abs(lin.e_nom) ** 2 / peak)
perturbed = np.asarray(jnp.abs(e_perturbed) ** 2 / peak)
rel_err = po.linearity_residual(path, entrance, basis, lin, eps_nm)
print(
    f"{lin.n_modes} modes, G shape {lin.G.shape}, "
    f"wavefront rms {np.sqrt((opd_nm[aperture > 0.5] ** 2).mean()):.2f} nm, "
    f"linear-model relative error {rel_err:.1e}"
)
```

`triptych` draws A and B under one shared log norm and their difference on a
symmetric diverging norm. Both images are floored at 1e-11 before the call, so
the shared norm spans the speckles rather than the numerical floor of the
nominal image. The wavefront error that made the speckles sits in the same
figure, drawn with `imshow_diverging` and the signed `opd` colormap, so the
cause in the pupil and its effect in the image are read together.

```{code-cell} python
fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), layout="constrained")
trip = ep.triptych(
    np.maximum(nominal, 1e-11),
    np.maximum(perturbed, 1e-11),
    mode="residual",
    titles=("Nominal", "Perturbed", "Perturbed - nominal"),
    axes=[axes[0, 0], axes[0, 1], axes[1, 1]],
    extent=focal_extent,
)
trip.artists["cbar"][0].set_label("contrast")
trip.artists["cbar"][1].set_label("contrast difference")
for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
    ep.label_lod(ax)
ep.imshow_diverging(
    opd_nm,
    ax=axes[1, 0],
    extent=pupil_extent,
    cmap="BrBG",  # the opd role of the conventions; hwostyle 1.5.0 still maps opd to RdBu_r
    cbar_label="OPD [nm]",
)
axes[1, 0].set_title("Wavefront error")
axes[1, 0].set_xlabel("x [D]")
axes[1, 0].set_ylabel("y [D]")
```

Figure 4: the nominal and perturbed coronagraph images under one log norm
(top), the 2 nm rms wavefront error that perturbed them (bottom left), and the
contrast difference (bottom right); the speckles fill the 2 to 10 lambda/D
annulus that the wavefront basis spans.

The drift is an explicit trajectory. Each mode coefficient follows a
first-order autoregressive process, `eps[k + 1] = rho * eps[k] + sqrt(1 -
rho**2) * sigma * xi` with `rho = exp(-dt / tau)`, one frame per minute and a
ten-minute decorrelation time, so the field boils slowly while its rms stays at
2 nm. `tiptilt.TabulatedSpeckleField` takes the `(E_nom, G)` pair from
physicaloptix together with the tabulated trajectory, and `realize` returns, at
any elapsed time, the contrast delta `2 Re(conj(E_nom) G eps) + |G eps|**2`
divided by the normalization, which is the Airy peak here; adding the nominal
contrast gives each frame. `animate` binds a figure, a draw function that only
mutates existing artists, and a frame count, and `jshtml` renders the frames
into a self-contained player. The norm is pinned to the brightest frame so the
colors mean the same thing in every frame.

```{code-cell} python
n_frames, dt_s, tau_s = 24, 60.0, 600.0
rho = float(np.exp(-dt_s / tau_s))
keys = jax.random.split(jax.random.PRNGKey(1), n_frames)
trajectory = [eps_nm]
for key in keys[1:]:
    kick = sigma_nm * np.sqrt(1.0 - rho**2) * jax.random.normal(key, (lin.n_modes,))
    trajectory.append(rho * trajectory[-1] + kick)
times_s = dt_s * jnp.arange(n_frames)
drift = tiptilt.TabulatedSpeckleField(
    lin.e_nom,
    lin.G,
    times_s,
    jnp.stack(trajectory),
    peak,
    pixel_scale_lod=pixscale_lod,
    coherent=True,
)
frames = [
    nominal + np.asarray(drift.realize(wavelength_nm=wavelength_nm, time_s=t))
    for t in times_s
]

fig, ax = plt.subplots(figsize=(4.5, 4.0), layout="constrained")
speckles = ep.imshow_log(
    frames[0],
    ax=ax,
    extent=focal_extent,
    vmin=1e-10,
    vmax=max(frame.max() for frame in frames),
    cbar_label="contrast",
)
ep.label_lod(ax)


def draw(fig, k):
    speckles.update(frames[k])
    ax.set_title(f"t = {k * dt_s / 60:.0f} min")


anim = ep.animate(fig, draw, n_frames, fps=8)
HTML(anim.jshtml(dpi=100))
```

Figure 5: 24 frames of the boiling speckle field, one per minute of a
wavefront drifting with a ten-minute decorrelation time, at a fixed contrast
scale.
