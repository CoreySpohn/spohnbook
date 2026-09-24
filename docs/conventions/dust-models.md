# Dust models: radiance, geometry, and normalization

*Draft contract: the choices marked pending or proposed are open.*

Local zodiacal light and exozodiacal light are the same physical process seen from different places: starlight scattered once by an optically thin cloud of grains. One radiance formulation can describe both. Sharing the formulation does not make the models interchangeable. A tabulated Solar-system brightness, a cheap yield-scaling law, a parametric debris-disk density, and an imported simulated image answer different questions, carry different calibrations, and support different operations. This chapter defines the shared quantities and geometry, then states what each model profile can and cannot do.

The chapter reuses the vocabulary of [the radiometry chapter](radiometry-detectors.md) for photon quantities and spectral measures, of [the geometry chapter](geometry-time.md) for angles and the observer axis, and of [the inference chapter](inference-records.md) for parameters and records. Status labels follow [the handbook](index.md): a *physical identity* follows from stated assumptions; a *proposed profile* awaits a named decision; *implemented* means present in a named library version; *verified boundary* and *validated domain* require the evidence those labels name. Every numerical example below is an independent teaching fixture computed from stated geometry. None is evidence about a library's current behavior.

## Quantities and measures

A dust model outputs radiance. Everything after that (pixel flux, contrast, counts) is an instrument or adapter step with its own owner.

| Symbol | Meaning | Unit or measure |
|---|---|---|
| $I_\lambda$ | Photon spectral radiance (surface brightness) along one sightline | photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$, or explicitly per arcsec$^2$ |
| $B_\lambda$ | Energy spectral radiance | W m$^{-2}$ nm$^{-1}$ sr$^{-1}$ (or per $\mu$m, stated) |
| $j_\lambda$ | Photon emissivity per unit path length | $I_\lambda$ units per AU (or per m, stated) |
| $\alpha_{\rm sca}$ | Scattering cross-section density, $n\,\sigma_{\rm sca}$ summed over grains | AU$^{-1}$ or m$^{-1}$ |
| $\Phi_{{\rm inc},\lambda}$ | Incident stellar photon flux density at the grain | photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ |
| $p_\lambda(\Theta)$ | Scattering phase function, $\int_{4\pi}p\,d\Omega=1$ | sr$^{-1}$ |
| $\Phi_{\lambda,p}$ | Pixel-integrated photon flux density, $\int_{\Omega_p}I_\lambda\,d\Omega$ | photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ per pixel |
| $c_p$ | Host-relative contrast of one pixel, $\Phi_{\lambda,p}/\Phi_{\star,\lambda}$ | dimensionless, host star as the named denominator |

Three measure changes cause most dust-scale errors:

- **Energy versus photons.** $I_\lambda = B_\lambda\,\lambda/(hc)$ at each wavelength. Convert inside a band integral, not once at the band center ({ref}`the radiometry figure <fig-dust-radiometry>`).
- **Per meter versus per nanometer.** A density per m of wavelength is $10^9$ times a density per nm of the same spectrum. The quantity table names one; an adapter converts once.
- **Per steradian versus per arcsec$^2$.** One arcsec$^2$ is $(\pi/648000)^2\approx2.35\times10^{-11}$ sr; use the hwoutils conversion rather than a typed constant.

The exchange default proposed in the radiometry chapter applies here: **exchange physical photon radiance**; legacy ratios (brightness divided by a zero-magnitude flux, or magnitudes per solid angle) are derived in the receiving adapter with their zero point named.

## Rays, frames and angles

```{figure} figures/dust-geometry-light.svg
:class: only-light
:name: fig-dust-geometry

Ray geometry for one constant-emissivity sphere (radius 2 AU, emissivity 3 photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$ AU$^{-1}$), an analytic fixture, not a dust model. (a) Each fixture ray on its own track: only the portion with $s\ge0$ inside the sphere contributes, so an observer inside sees 1 or 3 AU of path depending on direction, an outside observer sees the full 4 AU chord or nothing, and a grazing ray sees nothing. (b) The scattering angle $\Theta$ is measured between the incident propagation direction (star to grain) and the direction toward the observer, $-\hat n$; the planetary illumination angle at the same point is $\alpha=\pi-\Theta$. (c) Cumulative radiance along the $-x$ rays of the inside and outside observers.
```

```{figure} figures/dust-geometry-dark.svg
:class: only-dark

Ray geometry for one constant-emissivity sphere (radius 2 AU, emissivity 3 photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$ AU$^{-1}$), an analytic fixture, not a dust model. (a) Each fixture ray on its own track: only the portion with $s\ge0$ inside the sphere contributes. (b) The scattering angle $\Theta$ and the illumination angle $\alpha=\pi-\Theta$. (c) Cumulative radiance along the $-x$ rays of the inside and outside observers.
```

**Ray (physical identity once the frame is named).** A sightline is

$$
\boldsymbol x(s)=\boldsymbol x_{\rm obs}+s\,\hat{\boldsymbol n},\qquad s\ge0,
$$

with $\hat{\boldsymbol n}$ a unit vector pointing from the observer into the scene. Coordinates are Cartesian and star-centered, with lengths in AU unless stated. Light scattered toward the observer propagates along $-\hat{\boldsymbol n}$. A kernel receives explicit numerical rays and illumination; ephemerides and target geometry remain the caller's (or orbix's) responsibility. The workspace-wide observer-axis basis is still **pending** ([the geometry chapter](geometry-time.md), observer basis and node decision); a dust record names its frame and basis rather than assuming one.

**Scattering angle (physical identity).** For a star at the origin, light arriving at a grain at $\boldsymbol x$ propagates along $\hat{\boldsymbol k}_{\rm in}=\boldsymbol x/|\boldsymbol x|$. The scattering (deflection) angle is

$$
\cos\Theta=\hat{\boldsymbol k}_{\rm in}\cdot(-\hat{\boldsymbol n}),\qquad \Theta\in[0,\pi],
$$

so $\Theta=0$ is forward scattering (the grain lies between the star and the observer) and $\Theta=\pi$ is backscattering. The planetary illumination angle at the same point is its supplement, $\alpha=\pi-\Theta$: full phase $\alpha=0$ is backscattering $\Theta=\pi$ (Cahoy et al. 2010, ApJ 724, 189, section 3.2, [doi:10.1088/0004-637X/724/1/189](https://doi.org/10.1088/0004-637X/724/1/189); Hedman & Stark 2015, ApJ 811, 67, section 1, [doi:10.1088/0004-637X/811/1/67](https://doi.org/10.1088/0004-637X/811/1/67)). Some sources call $\Theta$ a "phase angle" (Henyey & Greenstein 1941 define their $\alpha$ as the deviation from the forward direction; ExoVista calls $\theta$ the "scattering phase angle"). Record the angle by its definition, never by that name. Passing an illumination angle into a phase function that expects $\Theta$ inverts forward and back scattering; the geometry chapter tracks the same supplement error for planets.

**Inclination and near side (proposed profile).** Inclination $i\in[0,\pi]$, with $i=0$ face-on. Under the observer-toward-$+z$ profile, the *near side* is the half of the disk with $z>0$; its grains have $\cos\Theta=z/|\boldsymbol x|>0$ and so scatter forward. For a thin layer, the path weight is $h/|\cos i|$, positive on both sides of $i=\pi/2$; $i$ and $\pi-i$ exchange which half is near. A weight that keeps the sign of $\cos i$ produces negative radiance above 90 degrees ({ref}`the sign-control figure <fig-dust-sign>`). Whether $i>\pi/2$ denotes retrograde rotation (the angular-momentum convention) is part of the pending observer basis decision; the path weight is positive under every choice.

## The single-scattering integral

**Physical identity under stated assumptions.** For an optically thin cloud illuminated by a point star,

$$
I_\lambda(\hat{\boldsymbol n},\boldsymbol x_{\rm obs})
=\int_{s_{\rm in}}^{s_{\rm out}}
\alpha_{{\rm sca},\lambda}(\boldsymbol x)\,
\Phi_{{\rm inc},\lambda}(\boldsymbol x)\,
p_\lambda\big(\Theta(\boldsymbol x)\big)\,ds,
\qquad
\Phi_{{\rm inc},\lambda}(\boldsymbol x)=\frac{L_{\lambda}}{4\pi|\boldsymbol x|^2}.
$$

Here $L_\lambda$ is the intrinsic stellar photon luminosity (photon s$^{-1}$ nm$^{-1}$), so illumination falls off from the star, not from the observer. There is no extra inverse-square factor in observer distance: radiance is conserved along a ray in free space. Units check: AU$^{-1}\times$ photon s$^{-1}$ m$^{-2}$ nm$^{-1}\times$ sr$^{-1}\times$ AU gives photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$.

Keep five factors separate, because each has a different owner and uncertainty:

1. **Density** $n(\boldsymbol x)$: the spatial law (for example GRaTeR, below).
2. **Cross section and albedo**: $\alpha_{\rm sca}=n\,\sigma_{\rm sca}$, equivalently single-scattering albedo times extinction density; a size distribution integrates $Q_{\rm sca}\pi a^2$.
3. **Phase function** $p_\lambda(\Theta)$, normalized to one over $4\pi$ sr.
4. **Illumination** $\Phi_{\rm inc}$: stellar luminosity and the star-grain distance.
5. **Quadrature**: the ray support $[s_{\rm in},s_{\rm out}]$ and the nodes inside it.

The zodiacal-light reference form is Kelsall et al. (1998, ApJ 508, 44), section 4.1, equation 1 ([doi:10.1086/306380](https://doi.org/10.1086/306380)): a sum over cloud components of $\int n_c\,[A_{c,\lambda}F^\odot_\lambda\Phi_\lambda(\Theta)+(1-A_{c,\lambda})E\,B_\lambda(T)K_\lambda(T)]\,ds$, with the phase function normalized so its integral over $4\pi$ sr is one (their equation 2). The first term is the scattering integral above in energy units; the second is thermal emission. This chapter **excludes** thermal emission, multiple scattering, extinction along the ray, finite stellar angular size and polarization. Each needs its own law and tests; thermal emission in particular needs a temperature and emissivity law, not a rescaled scattering amplitude.

**Phase functions.** The Henyey-Greenstein (HG) form,

$$
p_{\rm HG}(\Theta;g)=\frac{1}{4\pi}\,\frac{1-g^2}{(1+g^2-2g\cos\Theta)^{3/2}},
$$

integrates to one over $4\pi$ sr for every $|g|<1$. Henyey & Greenstein (1941, ApJ 93, 70, equation 2, [doi:10.1086/144246](https://doi.org/10.1086/144246)) write the same shape multiplied by the spherical albedo $\gamma$, so their function integrates to $\gamma$; do not carry an albedo in both the phase function and $\alpha_{\rm sca}$. A mixture $\sum_iw_ip_{\rm HG}(\Theta;g_i)$ is normalized only when $\sum_iw_i=1$, as in ExoVista's three-component fit (Stark et al. 2022, AJ 163, 105, section 2.3, equation 7, [doi:10.3847/1538-3881/ac45f5](https://doi.org/10.3847/1538-3881/ac45f5)). An unnormalized weight vector silently rescales the dust amplitude.

## Ray support and convergence

The integral runs over the part of the half-ray that lies inside the model's support. For a support bounded by an outer surface and an optional inner exclusion (a sublimation radius or inner edge), the ray parameters where $\boldsymbol x(s)$ crosses each surface give zero, one or two intervals:

- **Observer outside, looking in:** $s_{\rm in}>0$ at entry, $s_{\rm out}$ at exit.
- **Observer inside:** $s_{\rm in}=0$; the path runs to the first exit in the direction of $\hat{\boldsymbol n}$ only.
- **Looking away, or grazing:** both crossings at $s<0$, or a tangent point: the path length is zero, not negative.
- **Inner exclusion:** a ray through the exclusion splits into two intervals; the excluded chord contributes nothing.

Every interval length is nonnegative, so every path weight is nonnegative, so a physical radiance is nonnegative. A kernel that integrates the whole line (both signs of $s$), or that inherits the sign of a direction cosine, fails the sphere anchors below.

**Convergence has two independent axes.** Refining quadrature nodes inside a fixed support cannot recover material outside it. Check support expansion (outer radius, vertical extent, inner edge) and node refinement separately, then together, following the three-level rule of the radiometry chapter. Report local errors (one sightline, one aperture) as well as the image total; a Gaussian vertical profile truncated at a fixed height loses a column fraction that is not the same as the fractional intensity loss along an inclined ray.

## Model profiles

The profiles below are distinct model choices, not fidelity levels of one model. Changing the observer of a synthetic cloud does not turn a debris-disk density law into a calibrated Solar-system cloud, and an empirical table does not become a three-dimensional model by being fitted.

### Leinert tabulated approximation

- **What it is:** the empirical zodiacal-light brightness of Leinert et al. (1998, A&AS 127, 1, [doi:10.1051/aas:1998105](https://doi.org/10.1051/aas:1998105)): $I_{\rm ZL}=f_R\,I(\lambda-\lambda_\odot,\beta)\,f_{\rm abs}\,f_{\rm co}\,f_{\rm SP}$ (section 8.1, equation 14), with the helioecliptic table at 500 nm (Tables 16 and 17; 1 S10 = $1.28\times10^{-8}$ W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$ at 500 nm, section 8.3), color factors (section 8.4, Table 19) and a heliocentric distance factor $I(R)/I(1\,{\rm AU})=R^{-2.3\pm0.1}$ (section 8.2, equation 15).
- **Inputs and outputs:** ecliptic longitude from the Sun, ecliptic latitude, wavelength, epoch through the geometry; returns radiance along one sightline from near Earth.
- **Calibration and domain:** measured sky brightness from 1 AU observers, over the tabulated elongations and latitudes; the distance factor rests on Helios data for its stated elongation range. The table's color and distance factors are separate approximations with their own domains.
- **Observer and time:** Earth-orbit observers; time enters through Sun-relative geometry. An out-of-domain policy (reject, clamp, or extrapolate) must be explicit on each axis.
- **Unsupported:** an arbitrary observer position, three-dimensional density, or grain optics. A physical model fitted to these tables is calibrated, not validated by them.
- **Status:** implemented (zodi, and skyscapes `LeinertZodi`); open color findings are in the radiometry chapter's coverage table.

### Analytic exozodi scaling law

- **What it is:** a surface-brightness level at a reference radius scaled to other radii and stars, as in Stark et al. (2014, ApJ 795, 122, appendix C, equations C1 to C4, [doi:10.1088/0004-637X/795/2/122](https://doi.org/10.1088/0004-637X/795/2/122)): one zodi has V surface brightness 22 mag arcsec$^{-2}$ at the Earth-equivalent insolation distance (EEID), $r_{\rm EEID}=1\,{\rm AU}\sqrt{L_\star/L_\odot}$.
- **Inputs and outputs:** zodi level, stellar V luminosity, radius, band color factor; returns a scalar brightness or a flux ratio per arcsec$^2$ at that radius.
- **Calibration and domain:** a convention anchored to the Solar-system cloud viewed from outside; fast and reproducible for yield calculations; no spatial structure.
- **Observer and time:** a distant observer; geometry enters only through a radius and an inclination factor.
- **Unsupported:** morphology, forward-scattering asymmetry at a given position angle, and any value already evaluated at radius $r$ receiving a second $r^{-2}$ or inclination factor.
- **Status:** implemented (zodi `exozodi_flux_ratio_v`, `jez0`, `scale_jez`); consumer dialects differ, see the boundary checklist.

### Parametric scattered-light disk

- **What it is:** a density law integrated along lines of sight with an HG phase function. The GRaTeR law of Augereau et al. (1999, A&A 348, 557, section 3.1, [arXiv:astro-ph/9906429](https://arxiv.org/abs/astro-ph/9906429)) is $n(r,z)=n_0R(r)Z(r,z)$ with $R(r)\propto[(r/r_c)^{-2\alpha_{\rm in}}+(r/r_c)^{-2\alpha_{\rm out}}]^{-1/2}$, $Z=\exp[-(|z|/\zeta(r))^\gamma]$ and $\zeta(r)=\zeta_0(r/r_0)^\beta$ (unnumbered displayed equations; cite the section).
- **Inputs and outputs:** density shape, phase function, orientation, stellar illumination, sightlines; returns radiance per sightline, then pixel flux after angular integration.
- **Calibration and domain:** optically thin; the amplitude is phenomenological until tied to a declared reference profile (see "one zodi" below).
- **Observer and time:** a distant observer in current implementations; an embedded observer requires the ray support above, not a smaller distance.
- **Unsupported:** absolute Solar-system calibration, thermal emission, and optically thick disks.
- **Status:** implemented as skyscapes `GraterDisk` and `ExovistaParametricDisk`; their pixel measure, inclination domain, amplitude identifiability and support truncation are open (coverage table below).

### Imported ExoVista raster

- **What it is:** a precomputed image cube from ExoVista (Stark et al. 2022, section 2.4): "disk contrast per pixel (flux of disk per pixel divided by stellar flux)", with a default 2 mas pixel scale. Its normalization is a Solar-system twin at 60 degrees inclination with V surface brightness 22 mag arcsec$^{-2}$ at 1 AU and 90 degrees scattering angle (section 2.3).
- **Inputs and outputs:** file, native pixel scale, wavelengths; returns pixel-integrated host contrast on the native grid.
- **Calibration and domain:** the generator's calibration at generation time; geometry, orientation and distance are baked into the image.
- **Observer and time:** none; the viewpoint is fixed.
- **Unsupported:** a different viewpoint, inclination or distance; a two-dimensional image cannot acquire them. Resampling to another grid must conserve pixel-integrated flux and record edge losses.
- **Status:** implemented as skyscapes `ExovistaDisk`; flux-conserving resampling and native-sampling provenance are open.

### Exploratory arbitrary-observer cloud

- **What it is:** a proposed kernel that takes explicit rays, illumination, density and phase function, so one synthetic cloud can be viewed from inside and outside.
- **Status:** **proposed**, not implemented; no package owns it. It is a bounded experiment whose acceptance needs the sphere anchors, independent quadrature and distant-observer limits below. Reproducing Earth-view tables alone does not validate dependence on observer radius.

## What "one zodi" means

"Zodi" names at least two families of definitions:

- **Surface-brightness based.** A reference Solar-system brightness at a reference radius and view: Roberge et al. (2012, PASP 124, 799, section 1.2, [doi:10.1086/667218](https://doi.org/10.1086/667218)) and Stark et al. (2014, appendix C) use about 22 mag arcsec$^{-2}$ in V at the EEID; ExoVista anchors 22 mag arcsec$^{-2}$ at 1 AU and 90 degrees scattering angle for a 60-degree view.
- **Surface-density based.** A face-on geometrical optical depth: Kennedy et al. (2015, ApJS 216, 23, section 2.2.3, equation 3, [doi:10.1088/0067-0049/216/2/23](https://doi.org/10.1088/0067-0049/216/2/23)) set $\Sigma_m=z\,\Sigma_{m,0}(r/r_0)^{-\alpha}$ with $\Sigma_{m,0}=7.12\times10^{-8}$ at $r_0=\sqrt{L_\star/L_\odot}$ AU; Ertel et al. (2020, AJ 159, 177, section 3.2, [doi:10.3847/1538-3881/ab7817](https://doi.org/10.3847/1538-3881/ab7817)) call this "a unit of vertical geometrical optical depth (surface density)", independent of passband.

These do not convert into each other without grain optics, a phase function and a viewing geometry. **Proposed profile:** a dust amplitude always travels with a specification that names

| Field | Example values |
|---|---|
| Definition family | surface brightness, or face-on optical depth |
| Reference star | the Sun; or the host, through $L_\star$ and $M_V$ |
| Reference radius | 1 AU, or the EEID |
| Passband or spectrum | V band, a named filter, or a spectral law |
| Viewing geometry | inclination and scattering angle of the reference |
| Calibration identity | model, table or generator revision |

A value evaluated along a physical sightline (a line-of-sight brightness) is distinct from a *legacy radius-scaled amplitude* (a reference value times $(r_{\rm ref}/r)^2$ and an inclination factor). Consumers that hold empirically different definitions keep them as distinct parameters; forcing them into one `nzodis` field hides a unit change.

## Distance and contrast

For the same star, cloud and view in the distant-observer limit, three identities hold (physical identities; fixtures below):

1. **Radiance along corresponding physical sightlines is independent of observer distance $D$.** A sightline through the same physical point of the cloud has the same $I_\lambda$ at $D$ and $2D$.
2. **Observed integrated flux scales as $D^{-2}$.** The cloud subtends solid angle $\propto D^{-2}$; $F=\int I\,d\Omega$.
3. **Host-relative contrast of the whole cloud is independent of $D$,** because the star also dims as $D^{-2}$.

For one pixel of solid angle $\Delta\Omega$,

$$
c_p=\frac{I_\lambda\,\Delta\Omega}{L_\lambda/(4\pi D^2)}=\frac{4\pi\,I_\lambda\,(D^2\Delta\Omega)}{L_\lambda}.
$$

The physical area $D^2\Delta\Omega$ carries the distance dependence: a fixed angular pixel covers four times more cloud at twice the distance. Multiplying a sampled map by the angular pixel area alone therefore fixes the per-pixel measure (below) but does not establish an absolute dust normalization. Comparing the same angular pixel at two distances compares two different physical sightlines.

```{figure} figures/dust-sampling-light.svg
:class: only-light
:name: fig-dust-sampling

Radiance versus pixel flux, an analytic fixture: a Gaussian clump of peak photon radiance 7 photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$ and width 0.25 arcsec, integrated exactly over each pixel of a fixed 2 arcsec field. (a) to (c) share one logarithmic norm: each refinement divides the peak pixel by about four while the printed sum stays fixed. (d) The sum of pixel fluxes is invariant; the sum of radiance samples without a solid-angle weight grows as the square of the pixels per side. Small-angle pixel solid angles.
```

```{figure} figures/dust-sampling-dark.svg
:class: only-dark

Radiance versus pixel flux, an analytic fixture: a Gaussian clump integrated exactly over each pixel of a fixed 2 arcsec field. (a) to (c) share one logarithmic norm; the pixel values drop as the grid refines while the sum stays fixed. (d) The sum of pixel fluxes is invariant; the unweighted sum of radiance samples grows as the square of the pixels per side.
```

**Proposed contract (from the radiometry chapter):** rendered scene pixels carry integrated flux or integrated host-relative contrast; a density-valued array names its measure; refinement preserves the integral.

## Spectral and radiometric boundary

```{figure} figures/dust-radiometry-light.svg
:class: only-light
:name: fig-dust-radiometry

Where the spectral and angular measures enter, a synthetic example. (a) Photon spectral radiance of three power-law energy spectra $B_\lambda\propto\lambda^k$ normalized at 550 nm, with a 500 to 600 nm box passband. (b) Relative error of integrating energy over the band and converting to photons once at the band center, against converting at every wavelength; exact only for $k=0$, where the photon density is linear in wavelength. (c) The order in which band response, pixel solid angle, collecting area and optics, and QE each enter once. A zero-point ratio is a receiving-adapter product, not the exchange quantity. No consumer implementation is represented.
```

```{figure} figures/dust-radiometry-dark.svg
:class: only-dark

Where the spectral and angular measures enter, a synthetic example: photon spectral radiance of three power-law energy spectra, the error of converting once at band center, and the order in which band, solid angle, area and optics, and QE each enter once.
```

Band integration follows the radiometry chapter: $Q_b=\int R(\lambda)\,I_\lambda\,d\lambda$ in photon units, then the angular integral, then the response. The dust-specific additions are small: a Leinert color factor has an energy-density basis and needs one $\lambda/\lambda_{\rm ref}$ factor to become a photon-density ratio; an exozodi reference brightness in a flux-ratio dialect needs its zero point named; a band equivalent width appears once, never also as a multiplier on an already integrated quantity.

## Parameters and identifiability

```{figure} figures/dust-identifiability-light.svg
:class: only-light
:name: fig-dust-identifiability

Amplitude identifiability, an analytic fixture. A normalized ring morphology multiplied by (a) nzodis = 2 and albedo = 0.15, and (b) nzodis = 1 and albedo = 0.3, shares one linear norm in arbitrary units. (c) Their difference is exactly zero everywhere; the signed display is pinned to a symmetric range rather than rescaled to the empty field. (d) Gaussian likelihood contours ($\Delta\chi^2=1,4,9$) for per-pixel noise chosen so the product is measured to $\pm0.02$: the data constrain only nzodis times albedo, along the dashed hyperbola.
```

```{figure} figures/dust-identifiability-dark.svg
:class: only-dark

Amplitude identifiability, an analytic fixture: two parameter vectors with the same product give identical images, and the likelihood constrains only the product.
```

A fit parameter is useful only if the data can distinguish it. Differentiability is not identifiability: in the figure, both partial derivatives are nonzero, yet the Fisher matrix for (nzodis, albedo) has rank one.

- **Amplitude products.** When brightness depends on $n_{\rm zodis}\times A$ (a dust level times a free spectral amplitude or albedo), only the product is constrained. Calibrate one factor, fit the declared product, or state the external constraint and the prior dependence.
- **Phase-function mixtures.** Normalize HG weights to one; otherwise the weights trade against the amplitude.
- **Orientation.** A disk midplane and planet orbits have independent orientations unless a coplanarity constraint or a common-frame transform is declared; changing midplane parameters does not rotate planets.
- **Distance.** Stellar distance has one owner (the star). A disk that also carries its own distance field creates a duplicate leaf that can go stale when distance is fitted.
- **Support parameters.** A sharp inner or outer edge moves the integration boundary; check derivatives with respect to it explicitly.

## Consumer boundary checklist

An adapter that passes dust brightness to an exposure-time calculator, an image simulator or a yield code states each of the following:

- **Measure:** sampled spectral density or band-integrated value; photon or energy; per sr or per arcsec$^2$; per pixel or per area.
- **Photon conversion:** inside the band integral, once.
- **Zero point:** the exact reference (AB, a Vega spectrum and passband, or none) whenever a ratio or magnitude crosses the boundary.
- **Bandwidth:** applied once.
- **Radius:** true circumstellar radius versus projected separation.
- **Sky position:** signed sky coordinates with a named basis, not an unsigned separation.
- **Axes:** wavelength, ray and epoch axes explicit; a scalar keyword does not silently receive an array.
- **Exposure averaging:** evaluated at one epoch or averaged over the exposure, stated.
- **No second correction:** an evaluated line-of-sight brightness receives no further radius or inclination scaling.

## Records and caches

Dust enters records under [the inference chapter](inference-records.md). **Proposed:** each star carries a versioned dust specification (definition family, reference fields, model profile and parameters, calibration identity, convention version). A cached rate or completeness value includes that identity in its key, and detection-band and characterization-band evaluations are computed separately rather than reusing a detection-band brightness. A thermal-emission color normalization (apparent-flux basis, representative star, distance and target ordering) is an **open validation case**: it is a source-derived risk, not a certified numerical defect.

## Reference cases and evidence

The fixtures below are implemented in `tools/dust_reference_cases.py` and tested in `tests/test_dust_reference_cases.py` of this repository. They verify the fixture mathematics only. Production tests of a dust model belong beside that model's code and must not derive their expected values from the model under test.

| Fixture | Setup | Expected (independently derived) | What a wrong kernel shows |
|---|---|---|---|
| **Positive half-ray sphere** | Sphere radius 2 AU at the origin, emissivity $j=3$ photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$ AU$^{-1}$ | Observer $(1,0,0)$: toward $+x$ path 1 AU, $I=3$; toward $-x$ path 3 AU, $I=9$. Observer $(5,0,0)$: toward $-x$ path 4 AU, $I=12$; toward $+x$ $I=0$. Observer $(2,-3,0)$ toward $+y$: tangent, $I=0$ | Full-line integration gives 4 AU from inside; a sign error in $\boldsymbol x_{\rm obs}\cdot\hat{\boldsymbol n}$ fails five anchors |
| **Scattering-angle sign** | Observer on $+x$ looking $-x$ | Grain at $(1,0,0)$: $\Theta=0$; at $(-1,0,0)$: $\Theta=\pi$; $\alpha=\pi-\Theta$ | Using $+\hat{\boldsymbol n}$ as the outgoing direction swaps forward and back |
| **Radiance to pixels** | Uniform 7 photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$ over 2 arcsec, grids 24, 48, 96 | Total $I\,(2\,{\rm arcsec})^2$ in sr on every grid; each pixel quarters per doubling | Unweighted sums grow by four per doubling |
| **Distance doubling** | Face-on uniform disk, radius 3 AU, $D=10$ and 20 pc | Radiance unchanged; solid angle and flux divided by four; contrast unchanged; fixed-angle pixel contrast multiplied by four | Treating a fixed angular pixel as a fixed sightline |
| **Amplitude product** | (2, 0.15) and (1, 0.30) | Identical images; Fisher rank one | A test of gradients alone passes |
| **Phase normalization** | HG with $g\in\{-0.5,0,0.3,0.9\}$; mixture weights (0.7, 0.3) | $\int_{4\pi}p\,d\Omega=1$ | Weights (0.7, 0.7) are rejected |
| **Inclination sign control** | Thin slab, 120 degrees | Radiance $2hj$ inside the projected ellipse on both sides of 90 degrees | Keeping the sign of $\cos i$ gives $-2hj$ |
| **Band conversion** | $B_\lambda\propto\lambda^k$ over 500 to 600 nm | Closed-form photon integral agrees with quadrature | Center conversion is exact only for $k=0$ |

Tolerances for these fixtures are float64 fixture tolerances (relative and absolute $10^{-12}$), not production accuracy requirements. Production acceptance additionally needs the three-level support and node ladders above, recorded source revisions, and positive and deliberately failing controls, as the tolerances before tests decision in [the handbook](index.md) requires.

```{figure} figures/dust-sign-control-light.svg
:class: only-light
:name: fig-dust-sign

A negative control: a thin uniform slab (radius 4 AU, thickness 0.1 AU, emissivity 5 per AU) at 120 degrees inclination. (a) The correct weight $h/|\cos i|$ on a logarithmic display. (b) The deliberately wrong weight $h/\cos i$ on the same display: a log display floors negative pixels, so a negative map looks like empty sky. (c) The same wrong map on a signed display exposes it. Physical radiance is checked for finite, nonnegative values before any logarithmic display.
```

```{figure} figures/dust-sign-control-dark.svg
:class: only-dark

A negative control: the correct and sign-inheriting path weights for a thin slab at 120 degrees inclination, on logarithmic and signed displays.
```

## Pending decisions

These join the [decision register](index.md); none is decided by this chapter.

- **Dust exchange profile:** physical photon radiance with named measure and frame as the boundary quantity; legacy ratios derived by receivers.
- **One-zodi specification:** the required fields above, and which definition family each consumer uses.
- **Inclination and near side:** the proposed domain $[0,\pi]$ with positive weights, and the angular-momentum sense of $i>\pi/2$, tied to the observer basis decision.
- **Arbitrary-observer kernel ownership:** a bounded experiment first; public ownership, and the constants policy for a NumPy-only dust core that cannot import JAX-dependent helpers, are decided from its evidence.

## Coverage and gates

| Finding | Contract owner | Gates and acceptance fixture |
|---|---|---|
| parametric disks return negative radiance above 90 degrees inclination because path weights keep the sign of the direction cosine | skyscapes | conventions / boundary anchors; Inclination sign control, Positive half-ray sphere |
| parametric disks return sampled brightness under a per-pixel flux contract | skyscapes, coronagraphoto | conventions / boundary anchors / images and IFS; Radiance to pixels (shared with the radiometry chapter) |
| dust level times free spectral amplitude is exactly degenerate | skyscapes, photomancy | conventions / fixed campaign; Amplitude product |
| fixed vertical integration bound truncates the flared outer column | skyscapes | boundary anchors; support-expansion ladder |
| disk wrappers duplicate stellar distance as their own field | skyscapes | boundary anchors; Distance doubling with distance owned by the star |
| composite disks lack the pixel scale an image renderer requires, and equal extents do not guarantee equal shapes | skyscapes, coronagraphoto | boundary anchors / images and IFS; composite-to-renderer check |
| imported rasters have no flux-conserving resampling | skyscapes | images and IFS; Radiance to pixels on a non-integer resample |
| exozodi reference radius, luminosity scaling and radius convention differ between consumers | zodi, jaxedith, EXOSIMS and pyEDITH adapters | conventions / boundary anchors / ensembles and external references; one-zodi specification round trip |
| dust identity is absent from cached rates, and detection-band brightness is reused for characterization | EXOSIMS adapters | fixed campaign / adaptive choice; cold and warm cache checks |
| thermal-emission color normalization | zodi, consumer adapters | ensembles and external references; open validation case |
