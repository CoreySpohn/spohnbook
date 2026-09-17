# Optical fields, image coordinates, and IFS products

*Draft contract: the choices marked pending or proposed are open.*

An image needs more than a shape and a pixel scale. A consumer must know where its optical origin lies, which physical direction each axis represents, what measure each value carries, and which optical losses have already been applied. A complex field additionally needs a phase convention. A spectrum extracted from that image needs an axis order and a covariance that describes the actual estimator.

This chapter proposes a common profile under the **observer basis and node decision** (geometry basis) and the **image coordinates and PSFlet origin decision** (optical grids, pixel meaning, and IFS products). It does **not** claim every package already implements it. The current implementations disagree in the ways the coverage table at the end of this chapter lists. Imported instrument data keep their authoritative calibration metadata; adapters transform that data explicitly into the common profile.

## Symbols and product metadata

| Symbol | Meaning |
|---|---|
| `(E,N)` | East and north tangent-plane angular offsets; east includes the right-ascension cosine factor |
| `(x,y)` | Physical coordinates in a declared image frame |
| `(r,c)` | Array row and column indices; an array is indexed `image[r,c]` |
| `(c_x,c_y)` | Optical origin in zero-based column and row coordinates, possibly fractional |
| `s_x,s_y` | Physical angular spacing per column and row |
| `theta` | Telescope roll: positive rotation of detector axes relative to the sky chart |
| `lambda, lambda_ref, D` | Evaluated wavelength, grid reference wavelength, and declared aperture diameter |
| `u` | Dimensionless focal coordinate, with either native or reference-wavelength units |
| `E_0, delta_E` | Nominal complex optical field and its coherent perturbation |
| `W` | Optical path difference (OPD), a signed length |
| `I, f` | Intensity or rate density, and its integral over a pixel |
| `(a,w)` | Lenslet/channel index and wavelength-bin index |
| `K` | Covariance with units equal to the product of its two measured quantities |

At a boundary, carry the coordinate frame and handedness, optical center, pixel spacing and edges, angular-grid reference, wavelength bins, and value units. Also identify the optical reference plane, throughput already included, and calibration revision. Labels such as "contrast," "normalized PSF," or "lambda/D grid" alone are insufficient.

## Physical directions and stored pixels

For the proposed display-independent image profile, columns increase with physical x and rows increase with physical y:

$$
x=(c-c_x)s_x,\qquad y=(r-c_y)s_y.
$$

Display it with `origin="lower"` to see x rightward and y upward. Array storage itself has no "up" or "down." A viewer using an upper origin changes the appearance, not the stored physical mapping. Coordinate tuples use `(x,y)`; array positions and shape use `(row,column)` and `(ny,nx)`. Names such as `center_yx` must preserve that order.

New centered grids default to the geometric optical origin

$$
c_x=(n_x-1)/2,\qquad c_y=(n_y-1)/2.
$$

A 4-by-4 image therefore has pixel centers numbered 0, 1, 2, 3 and origin `(1.5,1.5)`. At unit spacing its physical centers are -1.5, -0.5, +0.5, +1.5; zero lies between pixels. A 5-by-5 image has origin `(2,2)` and a sample at zero. "Half-pixel centered" does not imply that every odd-sized grid avoids zero.

The edges of column c are `(c-c_x-1/2)s_x` and `(c-c_x+1/2)s_x`; rows follow the same rule. Plot extents describe **edges**, not extrema of pixel-center coordinates. An imported PSF with calibrated center 128 in a 256-pixel image retains that center until explicitly transformed. A full-array flip reflects about 127.5, so it is not automatically a physical reflection about that imported optical axis. Cropping, rebinning, and rotation must update the center, including half-pixel shifts when source and target dimensions have different parity. A producer that crops internally, such as PathCoronagraph (the physicaloptix class that wraps a live optical-path propagation as a coronagraph model), must follow the same rule.

```{figure} figures/hwo-conventions-pixels-light.svg
:class: only-light
:name: fig-pixels

Pixel centers, orientation, and pixel integrals. The 4-by-4 grid has centers 0 through 3 and optical origin 1.5. Physical x points right and y up. Positive telescope roll moves the sky clockwise in detector coordinates; positive active image rotation moves content counterclockwise. Coarser pixels collect more of a fixed density because their area increases.
```

```{figure} figures/hwo-conventions-pixels-dark.svg
:class: only-dark

Pixel centers, orientation, and pixel integrals. The 4-by-4 grid has centers 0 through 3 and optical origin 1.5. Physical x points right and y up. Positive telescope roll moves the sky clockwise in detector coordinates; positive active image rotation moves content counterclockwise. Coarser pixels collect more of a fixed density because their area increases.
```

## East/north, telescope roll, and active image rotation

The observer basis and node decision should record the matrix taking `(E,N)` into the chosen sky chart. For the worked examples here, choose

$$
\mathbf{s}=(x_{\rm sky},y_{\rm sky})=(E,N),\qquad
R(\alpha)=
\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{pmatrix}.
$$

Positive telescope roll rotates the detector basis, so a fixed sky source has detector coordinates

$$
\mathbf{d}=R(-\theta)\mathbf{s}.
$$

Any calibrated zero-point orientation or reflection belongs in an explicit additional matrix. Astronomical position angle measured east of north is also distinct: in this example chart a source at separation rho and position angle psi has `(E,N)=(rho sin psi,rho cos psi)`. Parallactic angle is another named quantity, not a synonym for telescope roll. A conventional north-up, east-left plot can reverse its horizontal display axis without changing these stored numbers.

For an exact 4-by-4 example at spacing s, a source at `(x,y)=(1.5s,0.5s)` occupies `(r,c)=(2,3)`. At telescope roll +90 degrees it becomes `(0.5s,-1.5s)` and occupies `(0,2)`. An **active** +90-degree image rotation instead sends the same physical point to `(-0.5s,1.5s)`, at `(3,1)`. On a 5-by-5 grid, `(x,y)=(s,0)` starts at `(2,3)` and reaches `(1,2)` after +90-degree telescope roll. These expected pixels follow geometry, independently of a rotation helper.

A multi-roll detector sequence, such as a FrameSet (the coronalyze class that carries a time series of detector frames with their roll angles), cannot simply be coadded at fixed array indices to obtain a sky-aligned image. Its product must declare whether frames remain in detector coordinates or have been aligned. Alignment must transport the signal, center, validity mask, and noise model consistently; interpolation can introduce pixel covariance.

## Native and fixed angular grids

A native focal coordinate means

$$
u_\lambda=\frac{\alpha D}{\lambda},
$$

where alpha is an angle in radians. A grid with fixed native spacing `du` therefore has angular spacing `du lambda/D`, which changes with wavelength. A reference-wavelength grid instead stores `u_ref=alpha D/lambda_ref` and has fixed angular spacing `du_ref lambda_ref/D` across its wavelength planes.

On the fixed angular grid, a fixed sky source stays at the same location while diffraction structure grows with wavelength. Converting its stored scale again using each evaluated wavelength would magnify it a second time. A grid record must distinguish these cases and identify D: outer, inscribed, or another explicitly calibrated aperture diameter cannot be silently substituted. A field exported as bare arrays plus `pixel_scale_lod` loses this information unless the unit convention accompanies it.

## Density, integrated pixels, and photometric normalization

A sampled density and an integrated pixel are different objects:

$$
f_{r,c}=\int_{\mathrm{pixel}(r,c)}I(x,y)\,dx\,dy.
$$

For constant density 10 photons per second per square angular unit, a square pixel of width 0.25 contains 0.625 photons per second. Width 0.5 contains 2.5; four fine pixels sum to that same value. A PSF sampled only at its centers approximates this integral by density times area, with an error set by unresolved structure.

Interpolation followed by an area ratio is a center-sampling approximation, not a general conservative integrator or antialiasing filter. Pixel-overlap rebinning conserves integrated flux over a fully covered footprint under a piecewise-constant pixel model; it does not reconstruct unresolved optical structure. Rotation, coarse sampling, finite support, and clipping require separate checks. Coherent field amplitudes must be propagated or interpolated as fields before squaring; summing complex values as if they were photon counts has different mathematics.

State the normalization denominator. Fraction of incident source photons per pixel, density per `(lambda/D)^2`, peak-referenced contrast, and unit-sum shape templates are not interchangeable. A mean density multiplied by aperture area yields an aperture flux fraction; a mean **per-pixel** value requires the number of pixels instead. Specify the aperture too: an azimuthal annulus and an off-axis PSF core average different parts of a non-axisymmetric stellar leakage map.

## Coherent phase, OPD, and the adjoint measure

The proposed coherent profile retains the consistent internal pairing

$$
E\mapsto E\exp(+i2\pi W/\lambda),\qquad
\mathcal{F}[E](u)=\int E(x)\exp(-i2\pi ux)\,dx,
$$

with the corresponding two-dimensional transform. Positive OPD adds positive phase, and a positive x phase ramp displaces the focal response toward positive x. With a time factor `exp(-i omega t)`, this is the usual positive accumulated propagation phase. A first-order OPD sensitivity carries `+i2pi/lambda`; the interference term is `2 Re(conj(E_0) delta_E)`. Conjugating a field can preserve its intensity while reversing phase-sensitive predictions, so intensity-only agreement cannot certify this convention.

Zernike products must declare indexing, sine/cosine signs, pupil orientation, and normalization aperture. The present physicaloptix basis uses one-based Noll ordering and discrete RMS over its sampled circular aperture. Transferring those coefficients onto a segmented or obscured pupil does not automatically preserve RMS. Record both basis units and coefficient units in `W=sum(c_k B_k)`.

DM **mechanical surface displacement** and OPD are separate lengths. At normal reflection, displacement h in the path-increasing direction produces OPD 2h. The actuator normal and reflection geometry determine the sign and any oblique-incidence conversion. Internal tiptilt bases already describe OPD; multiplying them all by two would be another error. Perform the mechanical conversion once, at the hardware boundary.

An adjoint is defined by an inner product, not by conjugation alone. For uniform grids, the physical inner products include pupil and focal cell areas. On a fixed angular chromatic grid, let `s=lambda_ref/lambda`; the forward transform scales its coordinates and amplitude by s. Its adjoint under the **stored fixed-grid area** must use that same measure. A backward routine integrating over native coordinates `s u` introduces an additional `s^2` area factor unless its amplitude conversion compensates for it.

physicaloptix's current broadband test checks an adjoint identity using the wavelength-native focal measure. That identity can pass while the fixed-grid identity differs by `s^2`. The image coordinates and PSFlet origin decision must distinguish those operators before backward propagation is used as a physical return path. Forward-only energy agreement does not settle this choice.

## Signed intensity changes and absolute light

A coherent perturbation produces the signed change

$$
\Delta I=|E_0+\delta E|^2-|E_0|^2
=2\Re(E_0^*\delta E)+|\delta E|^2.
$$

It can be negative while the absolute intensity remains nonnegative. For real `E_0=0.2` and `delta_E=-0.1`, the nominal intensity is 0.04, the change is -0.03, and the final intensity is 0.01. Clipping the change before adding its floor destroys interference.

Compose the matched nominal floor and signed change before photon sampling. Their wavelength, grid, optical plane, incident-flux denominator, and throughput must match. An independently supplied static intensity map is not automatically compatible with an arbitrary complex nominal field. On a dimensionless focal grid, multiplying intensity density by cell area and dividing by incident field energy gives a per-pixel flux fraction. Apply subsequent photon-to-electron and detector operations according to their own boundary contract. A signed delta is not an independently sampleable Poisson source.

## Lenslet collection, PSFlet placement, and extracted spectra

An IFS first integrates entrance-plane flux over **physical lenslet cells**, then distributes each lenslet's spectral flux over detector pixels. The entrance optical center, lenslet-grid origin, and detector trace origin are separate calibrated quantities. For nearest-neighbor pitch p, square cells have area `p^2`; regular hexagonal cells have area `sqrt(3)p^2/2`. Hex-packed centers paired with square collection cells do not partition the same physical plane.

The proposed template profile stores the pixel-integrated shape of each PSFlet (the detector image of one lenslet's pupil) as a function of offsets **from its centroid**, plus a separate centroid correction relative to the geometric dispersion trace. Recenter a physical template before attaching that correction. An alternative geometric-offset format is possible, but must be a distinct declared format: adding a measured centroid to an already displaced template shifts it twice.

Keep three effects separate: physical optical transmission, finite numerical template support, and detector-edge capture. Masking a half-clipped PSFlet and renormalizing its surviving half to unity restores photons that never hit the detector. Capture metadata must affect the operator or its acceptance certificate; a provenance field ignored by the consumer is not an applied loss. Test finite support by enlarging it, and apply real physical losses exactly once at their declared reference plane.

Use explicit bin edges for bin-integrated input flux and its dispersion footprint. A density per wavelength needs integration over the bin; a bin-integrated rate must not be multiplied by the width again. Pixel-integrated PSFlet tables, spatial collection quadrature, and interpolation each require their own refinement evidence.

The proposed axis profile is:

| Product | Axes or flattened index |
|---|---|
| Entrance cube | `(wavelength, row, column)` |
| Lenslet flux | `(channel, wavelength)` |
| Flattened lenslet flux | `k = channel * n_wavelength + wavelength` |
| Detector image | `(row,column)`; flat index `row * n_columns + column` |
| Full spectral covariance | `K[(channel,wavelength),(channel,wavelength)]` |
| Selected within-channel blocks | `(selected_channel,wavelength,wavelength)` |

coronachrome's current renderer and extractor agree on this flattening. Selected within-channel covariance blocks omit cross-channel blocks; they cannot be treated as a full block-diagonal covariance when overlapping traces correlate neighboring lenslets. State whether uncertainties are conditional on known backgrounds or marginal over uncertain speckles and calibration. Also record whether the estimated amplitude means incident flux, transmitted flux, rate, or accumulated counts, and whether regularization changes the estimator.

## Independent acceptance fixtures and adoption stages

A shared **Asymmetric grid fixture** should supply independently calculated expected pixels, energies, and covariance entries. Include odd and even grids, a non-geometric imported center, signed x/y tilts, three wavelengths, a destructive-interference case, and two overlapping lenslets. Check the same declared source through both optical producers and every consumer. Agreement between two paths calling the same conversion is useful regression evidence but cannot establish the conversion's correctness. Deliberate axis swaps, conjugations, omitted area factors, and duplicated centroid shifts must make the fixture fail.

Use the **conventions stage** for decisions and vocabulary, the **boundary-anchor stage** for boundary repairs/import, the **fixed-campaign stage** for a fixed campaign, the **adaptive-choice stage** for adaptive use, the **images-and-IFS stage** for image/IFS products, and the **ensembles-and-references stage** for ensembles and external comparisons. The table assigns each finding to implementation owners and the stage that needs their contract established.

| Finding | Contract owner | Adoption gate |
|---|---|---|
| Signed coherent residual is clipped before adding its compatible floor | coronagraphoto; physicaloptix and optixstuff (floor metadata) | boundary anchors: composition before coherent imaging (images and IFS); **Asymmetric grid fixture** |
| The scalar stellar-leakage seam still mixes per-pixel intensity with density | optixstuff (scalar contract); yippy and jaxedith (adapters) | boundary anchors: absolute leakage anchor before the fixed campaign |
| YIP centers are not honored uniformly across emission, lookup and symmetry | yippy; optixstuff (center transport) | boundary anchors: imported-center authority; images and IFS: astrometry; **Asymmetric grid fixture** |
| Chromatic Fraunhofer backward is an adjoint under a different measure | physicaloptix | conventions: the measure under the image coordinates and PSFlet origin decision; boundary anchors: backward operator before images and IFS |
| Chromatic speckle products lose the distinction between native and reference-wavelength grids | physicaloptix and optixstuff; coronagraphoto | boundary anchors: wavelength-grid metadata before images and IFS |
| Maintenance export still defaults the focal sampling to 0.25 | tiptilt (maintenance export) | boundary anchors: derive sampling before controlled residuals (images and IFS) |
| Target-sampled image APIs still promise conservation that center interpolation does not supply | hwoutils; optixstuff and coronagraphoto (consumers) | conventions: pixel meaning under the image coordinates and PSFlet origin decision; images and IFS: integration convergence |
| IFS origin and lenslet indexing are not the common geometric-center convention | coronachrome | conventions: origins under the observer basis and node decision and the image coordinates and PSFlet origin decision; images and IFS: entrance-to-detector anchor |
| Hexagonal lenslet centers are paired with square collection cells | coronachrome | images and IFS: physical cell geometry or explicit hex restriction |
| PSFlet normalization erases window and detector-edge losses | coronachrome; physicaloptix (PSFlet pack emitter) | boundary anchors: loss ownership; images and IFS: capture and sensitivity |
| Physical PSFlet packs double-apply an aberration centroid | physicaloptix (PSFlet pack emitter); coronachrome | boundary anchors: centroid-relative format; images and IFS: signed placement; **Asymmetric grid fixture** |
| Alternate science pixel scale changes template location but not template sampling | coronalyze (template provider) | boundary anchors: grid compatibility; images and IFS: flux recovery |
| Rectangular crops with mixed parity move the PathCoronagraph origin | physicaloptix (PathCoronagraph) | boundary anchors: mixed-parity grid support or restriction; **Asymmetric grid fixture** |
| Telescope roll is applied by the simulator but ignored by the coadd detection arms | coronalyze (FrameSet builder) | boundary anchors: frame declaration; images and IFS: roll-aware detection; **Asymmetric grid fixture** |
| DM APIs use OPD coefficients while several hardware names suggest surface displacement | tiptilt; hardware and STOP (structural-thermal-optical performance) adapters | conventions: OPD vocabulary; boundary anchors: conversion; ensembles and external references: hardware comparison |
| Coherent sign and rotation conventions have useful consistent anchors | physicaloptix and hwoutils; external adapters | conventions: preserve coherent profile; boundary anchors: signed fixtures (**Asymmetric grid fixture**); ensembles and external references |
| The Eqx pixel-scale name is repaired, but legacy image entry points remain unsafe | yippy and optixstuff (public APIs) | boundary anchors: sampling-explicit methods and legacy restrictions |
| IFS flattening is internally consistent, but returned covariance is only selected within-spaxel blocks | coronachrome; spaceodyssey (campaign library) product and inference adapters | images and IFS: covariance contract before adaptive spectral use (adaptive choice); **Asymmetric grid fixture** |
| External optical parity and STOP orientation corrections live in scripts, not a shared adapter contract | physicaloptix (external adapters); the STOP replay adapter | boundary anchors: frame manifest; ensembles and external references: complex-field comparison |

A scalar campaign at the fixed-campaign stage need not wait for every capability of the images-and-IFS stage. It must declare that boundary and avoid claiming image-level evidence. Any adaptive policy or ensemble that later consumes an optical product inherits that product's unresolved convention and numerical-error limits.
