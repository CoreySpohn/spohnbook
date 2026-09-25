# Optical fields, image coordinates, and IFS products

*Draft contract: the choices marked pending or proposed are open.*

An image needs more than a shape and a pixel scale. A consumer must know where its optical origin lies, which physical direction each axis represents, what measure each value carries, and which optical losses have already been applied. A complex field additionally needs a phase convention. A spectrum extracted from that image needs an axis order and a covariance that describes the actual estimator.

This chapter proposes a common profile under the **observer basis and node decision** (geometry basis) and the **image coordinates and PSFlet origin decision** (optical grids, pixel meaning, and IFS products). It does **not** claim every package already implements it. Reports that current implementations disagree with it are recorded on the {ref}`limitations page <limitations-optics>`. Imported instrument data keep their authoritative calibration metadata; adapters transform that data explicitly into the common profile.

```{figure} figures/explainer-d04-optical-planes-light.png
:class: only-light
:name: fig-explainer-d04-optical-planes
:alt: Side-view diagram of an unfolded optical train with five planes from left to right: entrance pupil, DM plane, focal plane, Lyot plane and image plane, each a dashed line across a gray beam. Starlight enters from the left; a relay lens pair between the entrance pupil and the DM plane and a single lens in each later gap are marked relay and FT. The beam is parallel at the pupils and focused at the focal and image planes, and it narrows at the Lyot stop. Below the beam the hardware is named: aperture, deformable mirror, vortex phase mask, Lyot stop and detector. Below that a row of square thumbnails, grouped as what the element applies, what the light carries and what the detector records, shows a filled disk of pupil transmission, vertical OPD fringes, a six-fold phase pattern with a cyclic key, a bright ring of field amplitude just outside a solid pupil-edge circle and a dashed stop circle, an image with two speckles either side of a dark center, and, after an arrow, simulated detector counts. A bracket under the first four marks the complex field, a mark between the fourth and fifth reads |E| squared taken here, and a second bracket marks the intensity and counts.

An illustrative vortex coronagraph train, unfolded in side view with its mirrors drawn as transmissive elements; it is not a prescription for any mission. Each dashed line is a plane, a location along the beam; the glyph on it is the hardware element there; the framed thumbnail below is a sampled array of the quantity on that plane. The first three thumbnails show what an element applies, the next two what the light carries, and the last what the detector records. The beam envelope is schematic geometry, not to scale, and is not evidence of coronagraph performance. The thumbnails come from one propagated physicaloptix example: a 64 by 64 pupil grid, a 48 by 48 focal grid at 0.5 lambda/D, 550 nm, a charge-6 vortex (its phase winds six times through 2 pi around the axis; Mawet et al. 2005, ApJ 633, 1191; Foo et al. 2005, Opt. Lett. 30, 3308) and a Lyot stop, the pupil stop in the relayed pupil, of 0.8 D. The DM plane carries a 1 nm cosine ripple of 6 cycles per D standing in for a static wavefront error. The vortex phase is evaluated from its closed form, because the model does not output its internal focal-plane field. The pupil-plane thumbnails span 2 D: the path keeps only a D-wide array, which holds about 4 percent of the entrance energy at the Lyot plane, so the Lyot thumbnail evaluates the same vortex operator on a wider grid to show the light moved outside the pupil edge. Image intensities are relative to the unocculted on-axis peak $|\mathcal{F}[A](0)|^2$, not to the brightest sample of the half-pixel-offset grid. Between the entrance pupil and the image plane the model carries a complex field: each element multiplies it and each gap marked FT is a Fourier transform. The relay from the entrance pupil to the DM plane is an identity in the model, which applies the DM to the entrance array, and the model reaches the Lyot plane with an inverse transform, so it does not invert the pupil as a physical relay would. An OPD W enters as exp(+i 2 pi W/lambda) under the proposed coherent profile ({ref}`optics-coherent-phase`). The OPD is a path difference, not a mirror surface height: at normal reflection a surface displacement h gives an OPD of 2h ({ref}`Hecht 2017, Sec. 9.4.2, eq. 9.44 <source-hecht2017>`). The intensity forms only at the image plane, and the detector applies its own boundary contract ({ref}`optics-signed-intensity`): here 1e7 photons per second in a pixel at the unocculted peak, a 1 s exposure, quantum efficiency 1, a Gaussian approximation to shot noise, 1 electron of read noise, and detector pixels equal to the focal samples. The thumbnails show each array with x right and y up; whether that is the view looking downstream or upstream, and how a relay's pupil inversion is recorded, belong to the pending {ref}`image coordinates and PSFlet origin decision <decision-image-coordinates-and-psflet-origin>`. The picture is an example implementation of the proposed profile. A backend built on response tables stores only a final-plane response and does not model the intermediate planes.
```

```{figure} figures/explainer-d04-optical-planes-dark.png
:class: only-dark
:alt: Side-view diagram of an unfolded optical train with five planes from left to right: entrance pupil, DM plane, focal plane, Lyot plane and image plane, each a dashed line across a gray beam. Starlight enters from the left; a relay lens pair between the entrance pupil and the DM plane and a single lens in each later gap are marked relay and FT. The beam is parallel at the pupils and focused at the focal and image planes, and it narrows at the Lyot stop. Below the beam the hardware is named: aperture, deformable mirror, vortex phase mask, Lyot stop and detector. Below that a row of square thumbnails, grouped as what the element applies, what the light carries and what the detector records, shows a filled disk of pupil transmission, vertical OPD fringes, a six-fold phase pattern with a cyclic key, a bright ring of field amplitude just outside a solid pupil-edge circle and a dashed stop circle, an image with two speckles either side of a dark center, and, after an arrow, simulated detector counts. A bracket under the first four marks the complex field, a mark between the fourth and fifth reads |E| squared taken here, and a second bracket marks the intensity and counts.

An illustrative vortex coronagraph train, unfolded in side view with its mirrors drawn as transmissive elements; it is not a prescription for any mission. Each dashed line is a plane, a location along the beam; the glyph on it is the hardware element there; the framed thumbnail below is a sampled array of the quantity on that plane. The first three thumbnails show what an element applies, the next two what the light carries, and the last what the detector records. The beam envelope is schematic geometry, not to scale, and is not evidence of coronagraph performance. The thumbnails come from one propagated physicaloptix example: a 64 by 64 pupil grid, a 48 by 48 focal grid at 0.5 lambda/D, 550 nm, a charge-6 vortex (its phase winds six times through 2 pi around the axis; Mawet et al. 2005, ApJ 633, 1191; Foo et al. 2005, Opt. Lett. 30, 3308) and a Lyot stop, the pupil stop in the relayed pupil, of 0.8 D. The DM plane carries a 1 nm cosine ripple of 6 cycles per D standing in for a static wavefront error. The vortex phase is evaluated from its closed form, because the model does not output its internal focal-plane field. The pupil-plane thumbnails span 2 D: the path keeps only a D-wide array, which holds about 4 percent of the entrance energy at the Lyot plane, so the Lyot thumbnail evaluates the same vortex operator on a wider grid to show the light moved outside the pupil edge. Image intensities are relative to the unocculted on-axis peak $|\mathcal{F}[A](0)|^2$, not to the brightest sample of the half-pixel-offset grid. Between the entrance pupil and the image plane the model carries a complex field: each element multiplies it and each gap marked FT is a Fourier transform. The relay from the entrance pupil to the DM plane is an identity in the model, which applies the DM to the entrance array, and the model reaches the Lyot plane with an inverse transform, so it does not invert the pupil as a physical relay would. An OPD W enters as exp(+i 2 pi W/lambda) under the proposed coherent profile ({ref}`optics-coherent-phase`). The OPD is a path difference, not a mirror surface height: at normal reflection a surface displacement h gives an OPD of 2h ({ref}`Hecht 2017, Sec. 9.4.2, eq. 9.44 <source-hecht2017>`). The intensity forms only at the image plane, and the detector applies its own boundary contract ({ref}`optics-signed-intensity`): here 1e7 photons per second in a pixel at the unocculted peak, a 1 s exposure, quantum efficiency 1, a Gaussian approximation to shot noise, 1 electron of read noise, and detector pixels equal to the focal samples. The thumbnails show each array with x right and y up; whether that is the view looking downstream or upstream, and how a relay's pupil inversion is recorded, belong to the pending {ref}`image coordinates and PSFlet origin decision <decision-image-coordinates-and-psflet-origin>`. The picture is an example implementation of the proposed profile. A backend built on response tables stores only a final-plane response and does not model the intermediate planes.
```

(optics-product-metadata)=
## Symbols and product metadata

| Symbol | Meaning |
|---|---|
| `(E,N)` | East and north tangent-plane angular offsets; east includes the right-ascension cosine factor ({ref}`Brandt et al. 2021, Sec. 2 <source-brandt2021>`) |
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

(optics-pixel-directions)=
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

(optics-roll-rotation)=
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

Any calibrated zero-point orientation or reflection belongs in an explicit additional matrix. Astronomical position angle, measured from north through east ({ref}`Brandt et al. 2021, Sec. 4.2 <source-brandt2021>`; {ref}`Perryman 2018, Sec. 2.1 <source-perryman2018>`), is also distinct: in this example chart a source at separation rho and position angle psi has `(E,N)=(rho sin psi,rho cos psi)`. Parallactic angle is another named quantity, not a synonym for telescope roll. A conventional north-up, east-left plot can reverse its horizontal display axis without changing these stored numbers.

For an exact 4-by-4 example at spacing s, a source at `(x,y)=(1.5s,0.5s)` occupies `(r,c)=(2,3)`. At telescope roll +90 degrees it becomes `(0.5s,-1.5s)` and occupies `(0,2)`. An **active** +90-degree image rotation instead sends the same physical point to `(-0.5s,1.5s)`, at `(3,1)`. On a 5-by-5 grid, `(x,y)=(s,0)` starts at `(2,3)` and reaches `(1,2)` after +90-degree telescope roll. These expected pixels follow geometry, independently of a rotation helper.

A multi-roll detector sequence, such as a FrameSet (the coronalyze class that carries a time series of detector frames with their roll angles), cannot simply be coadded at fixed array indices to obtain a sky-aligned image. Its product must declare whether frames remain in detector coordinates or have been aligned. Alignment must transport the signal, center, validity mask, and noise model consistently; interpolation can introduce pixel covariance.

(optics-angular-grids)=
## Native and fixed angular grids

A native focal coordinate means

$$
u_\lambda=\frac{\alpha D}{\lambda},
$$

where alpha is an angle in radians. A grid with fixed native spacing `du` therefore has angular spacing `du lambda/D`, which changes with wavelength. A reference-wavelength grid instead stores `u_ref=alpha D/lambda_ref` and has fixed angular spacing `du_ref lambda_ref/D` across its wavelength planes.

On the fixed angular grid, a fixed sky source stays at the same location while diffraction structure grows with wavelength. Converting its stored scale again using each evaluated wavelength would magnify it a second time. A grid record must distinguish these cases and identify D: outer, inscribed, or another explicitly calibrated aperture diameter cannot be silently substituted. A field exported as bare arrays plus `pixel_scale_lod` loses this information unless the unit convention accompanies it.

(optics-pixel-measure)=
## Density, integrated pixels, and photometric normalization

A sampled density and an integrated pixel are different objects:

$$
f_{r,c}=\int_{\mathrm{pixel}(r,c)}I(x,y)\,dx\,dy.
$$

For constant density 10 photons per second per square angular unit, a square pixel of width 0.25 contains 0.625 photons per second. Width 0.5 contains 2.5; four fine pixels sum to that same value. A PSF sampled only at its centers approximates this integral by density times area, with an error set by unresolved structure.

Interpolation followed by an area ratio is a center-sampling approximation, not a general conservative integrator or antialiasing filter. Pixel-overlap rebinning conserves integrated flux over a fully covered footprint under a piecewise-constant pixel model; it does not reconstruct unresolved optical structure. Rotation, coarse sampling, finite support, and clipping require separate checks. Coherent field amplitudes must be propagated or interpolated as fields before squaring; summing complex values as if they were photon counts has different mathematics.

State the normalization denominator. Fraction of incident source photons per pixel, density per `(lambda/D)^2`, peak-referenced contrast, and unit-sum shape templates are not interchangeable. A mean density multiplied by aperture area yields an aperture flux fraction; a mean **per-pixel** value requires the number of pixels instead. Specify the aperture too: an azimuthal annulus and an off-axis PSF core average different parts of a non-axisymmetric stellar leakage map.

(optics-coherent-phase)=
## Coherent phase, OPD, and the adjoint measure

The proposed coherent profile retains the consistent internal pairing

$$
E\mapsto E\exp(+i2\pi W/\lambda),\qquad
\mathcal{F}[E](u)=\int E(x)\exp(-i2\pi ux)\,dx,
$$

with the corresponding two-dimensional transform. Positive OPD adds positive phase, and a positive x phase ramp displaces the focal response toward positive x. With a time factor `exp(-i omega t)` ({ref}`Goodman 2015, Sec. 3.8.3, eq. 3.8-16 <source-goodman2015>`), this is the usual positive accumulated propagation phase. Published texts pair the transform sign and the time factor both ways, so this pairing is the book's own choice rather than a borrowed one. A first-order OPD sensitivity carries `+i2pi/lambda`; the interference term is `2 Re(conj(E_0) delta_E)`. Conjugating a field can preserve its intensity while reversing phase-sensitive predictions, so intensity-only agreement cannot certify this convention.

Zernike products must declare indexing, sine/cosine signs, pupil orientation, and normalization aperture. The {ref}`limitations page <limitations-optics>` records the ordering and normalization of the current Zernike basis. Transferring those coefficients onto a segmented or obscured pupil does not automatically preserve RMS. Record both basis units and coefficient units in `W=sum(c_k B_k)`.

DM **mechanical surface displacement** and OPD are separate lengths. At normal reflection, displacement h in the path-increasing direction produces OPD 2h ({ref}`Hecht 2017, Sec. 9.4.2, eq. 9.44 <source-hecht2017>`). The actuator normal and reflection geometry determine the sign and any oblique-incidence conversion. A basis that already describes OPD must not be doubled again. Perform the mechanical conversion once, at the hardware boundary.

An adjoint is defined by an inner product, not by conjugation alone. For uniform grids, the physical inner products include pupil and focal cell areas. On a fixed angular chromatic grid, let `s=lambda_ref/lambda`; the forward transform scales its coordinates and amplitude by s. Its adjoint under the **stored fixed-grid area** must use that same measure. A backward routine integrating over native coordinates `s u` introduces an additional `s^2` area factor unless its amplitude conversion compensates for it.

A test of the adjoint identity under the wavelength-native focal measure can pass while the fixed-grid identity differs by `s^2`; the {ref}`limitations page <limitations-optics>` records the current case. The image coordinates and PSFlet origin decision must distinguish those operators before backward propagation is used as a physical return path. Forward-only energy agreement does not settle this choice.

(optics-signed-intensity)=
## Signed intensity changes and absolute light

A coherent perturbation produces the signed change

$$
\Delta I=|E_0+\delta E|^2-|E_0|^2
=2\Re(E_0^*\delta E)+|\delta E|^2.
$$

It can be negative while the absolute intensity remains nonnegative. For real `E_0=0.2` and `delta_E=-0.1`, the nominal intensity is 0.04, the change is -0.03, and the final intensity is 0.01. Clipping the change before adding its floor destroys interference.

```{figure} figures/explainer-d04-opd-perturbation-light.png
:class: only-light
:name: fig-explainer-d04-opd-perturbation
:alt: A thin optical-train rail with the DM plane and the image plane highlighted and joined by lines to the panels below. Left: a pattern of alternating brown and teal spots over a circular aperture, the DM command as OPD in nanometers, with a small equation under it showing it as minus a vertical-fringe cosine map plus a horizontal-fringe sine map. Middle: two log-scaled images relative to the unocculted peak; the nominal image has two speckles on the horizontal axis and the perturbed image has two speckles on the vertical axis instead, with a star glyph at the center of each. Right: the difference on a blue-white-red scale, with blue spots labeled less than zero at the old speckle positions and red spots labeled greater than zero at the new ones.

A deformable-mirror command changes the image through the field, so the intensity change is signed ({ref}`optics-signed-intensity`). The nominal state carries the 1 nm cosine ripple of 6 cycles per D along x from the train figure, which puts a speckle pair at plus and minus 6 lambda/D on the x axis. The command (left, as OPD in nm over the aperture, spelled out below it as minus the x cosine plus the y sine) subtracts that ripple and adds a 1 nm sine ripple of 6 cycles per D along y. The perturbed image loses the x pair and gains a y pair. The difference (right) is negative where the command cancels the existing field and positive where it adds light; clipping it at zero before adding the nominal image would erase the cancellation. Both images share one log norm relative to the unocculted on-axis peak, the difference uses a symmetric norm about zero, and the star glyph marks the on-axis star behind the vortex. The rail above marks the DM plane over the command and the image plane over the three image panels. Same propagated physicaloptix path and parameters as the train figure, noiseless; an illustration of the stated model, not a performance claim.
```

```{figure} figures/explainer-d04-opd-perturbation-dark.png
:class: only-dark
:alt: A thin optical-train rail with the DM plane and the image plane highlighted and joined by lines to the panels below. Left: a pattern of alternating brown and teal spots over a circular aperture, the DM command as OPD in nanometers, with a small equation under it showing it as minus a vertical-fringe cosine map plus a horizontal-fringe sine map. Middle: two log-scaled images relative to the unocculted peak; the nominal image has two speckles on the horizontal axis and the perturbed image has two speckles on the vertical axis instead, with a star glyph at the center of each. Right: the difference on a blue-white-red scale, with blue spots labeled less than zero at the old speckle positions and red spots labeled greater than zero at the new ones.

A deformable-mirror command changes the image through the field, so the intensity change is signed ({ref}`optics-signed-intensity`). The nominal state carries the 1 nm cosine ripple of 6 cycles per D along x from the train figure, which puts a speckle pair at plus and minus 6 lambda/D on the x axis. The command (left, as OPD in nm over the aperture, spelled out below it as minus the x cosine plus the y sine) subtracts that ripple and adds a 1 nm sine ripple of 6 cycles per D along y. The perturbed image loses the x pair and gains a y pair. The difference (right) is negative where the command cancels the existing field and positive where it adds light; clipping it at zero before adding the nominal image would erase the cancellation. Both images share one log norm relative to the unocculted on-axis peak, the difference uses a symmetric norm about zero, and the star glyph marks the on-axis star behind the vortex. The rail above marks the DM plane over the command and the image plane over the three image panels. Same propagated physicaloptix path and parameters as the train figure, noiseless; an illustration of the stated model, not a performance claim.
```

```{raw} html
<video class="only-light" controls loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Animation of the optical-train diagram above three panels: the total OPD at the DM plane with the command value c, the phase of the image-plane field, and the intensity change at the image plane. First the highlight moves from the entrance pupil through the DM plane, focal plane, Lyot plane and image plane to the detector, with a title stating what each plane carries. Then c steps from 0 to -2 nm: the OPD fringes fade to zero and return with reversed colors, the two speckles vanish and return, the intensity change turns blue at the speckles and then back to white, and the speckle phase changes from red to blue."><source src="../_static/explainers/explainer-d04-plane-tour-light.mp4" type="video/mp4"></video>
<video class="only-dark" controls loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Animation of the optical-train diagram above three panels: the total OPD at the DM plane with the command value c, the phase of the image-plane field, and the intensity change at the image plane. First the highlight moves from the entrance pupil through the DM plane, focal plane, Lyot plane and image plane to the detector, with a title stating what each plane carries. Then c steps from 0 to -2 nm: the OPD fringes fade to zero and return with reversed colors, the two speckles vanish and return, the intensity change turns blue at the speckles and then back to white, and the speckle phase changes from red to blue."><source src="../_static/explainers/explainer-d04-plane-tour-dark.mp4" type="video/mp4"></video>
```

**Animation.** The train figure, highlighted one plane at a time as the title names the quantity each plane carries, followed by a model-parameter sequence that is not a time series. In the second part a deformable-mirror command c scales a cosine ripple opposing the static 1 nm ripple, from c = 0 to c = -2 nm; the left panel shows the total OPD and the value of c. At c = -1 nm the total OPD vanishes, the speckle pair disappears and the intensity change there is negative. At c = -2 nm the OPD has reversed sign: the intensity is back to its nominal value within 4 percent of the speckle peak (the remainder is interference with the small leakage of this sampled vortex model) while the field phase at the speckles has changed by pi, from +pi/2 to -pi/2, so intensity alone barely tells the two fields apart ({ref}`optics-coherent-phase`, {ref}`optics-signed-intensity`). Phase is drawn only where the intensity exceeds 3e-7 of the unocculted on-axis peak. Every color scale is fixed across the frames: OPD within plus and minus 1 nm, phase over one cycle, intensity from 1e-10 to 1e-4 of the unocculted peak, and the change within plus and minus its largest value over the sequence. Same propagated physicaloptix path as the train figure; the train itself is schematic and not to scale.

Compose the matched nominal floor and signed change before photon sampling. Their wavelength, grid, optical plane, incident-flux denominator, and throughput must match. An independently supplied static intensity map is not automatically compatible with an arbitrary complex nominal field. On a dimensionless focal grid, multiplying intensity density by cell area and dividing by incident field energy gives a per-pixel flux fraction. Apply subsequent photon-to-electron and detector operations according to their own boundary contract. A signed delta is not an independently sampleable Poisson source.

(optics-ifs-products)=
## Lenslet collection, PSFlet placement, and extracted spectra

An IFS first integrates entrance-plane flux over **physical lenslet cells**, then distributes each lenslet's spectral flux over detector pixels ({ref}`Rizzo et al. 2017, Sec. 2.2 <source-rizzo2017>`; {ref}`Brandt et al. 2017, Sec. 4 <source-brandt2017>`). The entrance optical center, lenslet-grid origin, and detector trace origin are separate calibrated quantities. For nearest-neighbor pitch p, square cells have area `p^2`; regular hexagonal cells have area `sqrt(3)p^2/2`. Hex-packed centers paired with square collection cells do not partition the same physical plane.

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

Selected within-channel covariance blocks, the form a chi-squared extraction returns for each lenslet ({ref}`Brandt et al. 2017, Sec. 5.2 <source-brandt2017>`), omit cross-channel blocks; they cannot be treated as a full block-diagonal covariance when overlapping traces correlate neighboring lenslets. State whether uncertainties are conditional on known backgrounds or marginal over uncertain speckles and calibration. Also record whether the estimated amplitude means incident flux, transmitted flux, rate, or accumulated counts, and whether regularization changes the estimator.

(optics-fixtures)=
## Independent acceptance fixtures and adoption stages

A shared **Asymmetric grid fixture** should supply independently calculated expected pixels, energies, and covariance entries. Include odd and even grids, a non-geometric imported center, signed x/y tilts, three wavelengths, a destructive-interference case, and two overlapping lenslets. Check the same declared source through both optical producers and every consumer. Agreement between two paths calling the same conversion is useful regression evidence but cannot establish the conversion's correctness. Deliberate axis swaps, conjugations, omitted area factors, and duplicated centroid shifts must make the fixture fail.

Every finding about the current implementations that bears on this chapter, with its contract owner, adoption gate, status and evidence, is recorded under {ref}`limitations-optics` on the limitations page. A scalar campaign at the fixed-campaign stage need not wait for every capability of the images-and-IFS stage. It must declare that boundary and avoid claiming image-level evidence. Any adaptive policy or ensemble that later consumes an optical product inherits that product's unresolved convention and numerical-error limits.
