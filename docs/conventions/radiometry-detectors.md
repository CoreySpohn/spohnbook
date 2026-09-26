# Radiometry, detector counts, and spectral measurements

*Draft contract: the choices marked pending or proposed are open.*

A radiometric quantity is defined by what it counts, where it is evaluated, and the measure over which it is a density. A number labeled "flux" or "contrast" supplies too little information to connect a scene, an exposure-time calculator, an image, and a retrieved spectrum.

```{figure} figures/explainer-d01-overview-radiometry-light.png
:class: only-light
:name: fig-explainer-d01-overview-radiometry
:width: 80%
:alt: The observation map with the light arriving at the aperture and the counts in the detector pixels at full strength, its Radiometry tag drawn bold and keyed as this chapter, and every other part faded toward the background. A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

Where this chapter sits in the physical observation: the light arriving at the aperture and the counts in the detector pixels, at full strength, with the rest of the map dimmed ({ref}`full map <fig-explainer-d01-overview>`). An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

```{figure} figures/explainer-d01-overview-radiometry-dark.png
:class: only-dark
:width: 80%
:alt: The observation map with the light arriving at the aperture and the counts in the detector pixels at full strength, its Radiometry tag drawn bold and keyed as this chapter, and every other part faded toward the background. A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

Where this chapter sits in the physical observation: the light arriving at the aperture and the counts in the detector pixels, at full strength, with the rest of the map dimmed ({ref}`full map <fig-explainer-d01-overview>`). An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

This chapter proposes the contracts for those connections. Physical identities below are established; proposed API choices remain **draft contracts**, not descriptions of completed repairs. Reports that current implementations disagree with these contracts are recorded, with the acceptance gate for each, on the {ref}`limitations page <limitations-radiometry>`. Agreement with another code is a cross-code benchmark, not validation against measured hardware.

```{figure} figures/hwo-conventions-pipeline-light.svg
:class: only-light
:name: fig-pipeline

Transformations from source light to a reported measurement and its assumed-model interpretation. Arrows carry scientific quantities, not Python imports. The boxes name intended owners; they do not claim the current implementations already satisfy the contracts. Optical losses, QE and reporting each have a distinct boundary.
```

```{figure} figures/hwo-conventions-pipeline-dark.svg
:class: only-dark

Transformations from source light to a reported measurement and its assumed-model interpretation. Arrows carry scientific quantities, not Python imports. The boxes name intended owners; they do not claim the current implementations already satisfy the contracts. Optical losses, QE and reporting each have a distinct boundary.
```

```{figure} figures/explainer-d03-radiometric-collection-light.png
:class: only-light
:name: fig-explainer-d03-radiometric-collection
:alt: Top: a side-view diagram. At left, a hatched green extended source with a yellow star in front of it and a dashed gray square marking a pixel's sky cell. Two yellow rays leave the star, pass scale breaks, and continue as parallel starlight to a vertical aperture bar labeled area A, where dotted gray lines from the sky cell meet at an angle labeled Omega p. A lens and an optics plate labeled T opt of lambda focus the rays onto a column of detector pixels labeled QE q p of lambda; a gray spread profile beside the column shades the part falling in the outlined pixel p, labeled P p of lambda. At right, the photon rate arriving in p from the point source, a note that the integral of I lambda P p over solid angle replaces Phi lambda P p for the extended source, and a data arrow labeled times q p, integral over wavelength and integral over live time leading to the electron count mu e p. Italic tags under the aperture, optics and detector say which factor holds each loss. Bottom: a face-on 7 by 7 pixel window with a spread point-source image at one wavelength and pixel p outlined; normalized curves of Phi lambda, T opt, P p and q p against wavelength with their product shaded between the band edges; and expected electrons in p rising linearly during a shaded live interval labeled t live.

What an aperture and a pixel collect ({ref}`radiometry-response-ownership`, {ref}`radiometry-pixel-brightness`). Top, side view with light traveling to the right: a point source with photon flux density $\Phi_\lambda$ at the aperture and an extended source with brightness $I_\lambda$ feed an aperture of area $A$; the optics transmit $T_{\rm opt}(\lambda)$; the image of the point source spreads over several pixels, and $P_p(\lambda)$ is the share that lands in pixel $p$; the detector converts photons to electrons with QE $q_p(\lambda)$. For the extended source, $\int I_\lambda P_p\,d\Omega$ replaces $\Phi_\lambda P_p$; the dashed sky cell $\Omega_p$ is the idealized footprint of pixel $p$, and the spread image also brings light from outside it. The tags under each element state where a loss is assigned, so that no loss enters two factors. Bottom, the three integrals that turn densities into one pixel's expected count $\mu_{e,p}$: over solid angle, where the face-on detector shows an illustrative Airy image at one wavelength on a logarithmic scale; over wavelength, where $T_{\rm opt}$, $P_p$ and $q_p$ multiply $\Phi_\lambda$ at each wavelength before the band from $\lambda_b^-$ to $\lambda_b^+$ is integrated; and over the live interval $t_{\rm live}$, while the pixel is exposing rather than being read out, with constant rates assumed. Curves are normalized and show shapes only. The picture illustrates the chapter's expected-count integral; it is an original schematic, not an instrument prescription, and the ownership assignments are the chapter's proposed ledger.
```

```{figure} figures/explainer-d03-radiometric-collection-dark.png
:class: only-dark
:alt: Top: a side-view diagram. At left, a hatched green extended source with a yellow star in front of it and a dashed gray square marking a pixel's sky cell. Two yellow rays leave the star, pass scale breaks, and continue as parallel starlight to a vertical aperture bar labeled area A, where dotted gray lines from the sky cell meet at an angle labeled Omega p. A lens and an optics plate labeled T opt of lambda focus the rays onto a column of detector pixels labeled QE q p of lambda; a gray spread profile beside the column shades the part falling in the outlined pixel p, labeled P p of lambda. At right, the photon rate arriving in p from the point source, a note that the integral of I lambda P p over solid angle replaces Phi lambda P p for the extended source, and a data arrow labeled times q p, integral over wavelength and integral over live time leading to the electron count mu e p. Italic tags under the aperture, optics and detector say which factor holds each loss. Bottom: a face-on 7 by 7 pixel window with a spread point-source image at one wavelength and pixel p outlined; normalized curves of Phi lambda, T opt, P p and q p against wavelength with their product shaded between the band edges; and expected electrons in p rising linearly during a shaded live interval labeled t live.

What an aperture and a pixel collect ({ref}`radiometry-response-ownership`, {ref}`radiometry-pixel-brightness`). Top, side view with light traveling to the right: a point source with photon flux density $\Phi_\lambda$ at the aperture and an extended source with brightness $I_\lambda$ feed an aperture of area $A$; the optics transmit $T_{\rm opt}(\lambda)$; the image of the point source spreads over several pixels, and $P_p(\lambda)$ is the share that lands in pixel $p$; the detector converts photons to electrons with QE $q_p(\lambda)$. For the extended source, $\int I_\lambda P_p\,d\Omega$ replaces $\Phi_\lambda P_p$; the dashed sky cell $\Omega_p$ is the idealized footprint of pixel $p$, and the spread image also brings light from outside it. The tags under each element state where a loss is assigned, so that no loss enters two factors. Bottom, the three integrals that turn densities into one pixel's expected count $\mu_{e,p}$: over solid angle, where the face-on detector shows an illustrative Airy image at one wavelength on a logarithmic scale; over wavelength, where $T_{\rm opt}$, $P_p$ and $q_p$ multiply $\Phi_\lambda$ at each wavelength before the band from $\lambda_b^-$ to $\lambda_b^+$ is integrated; and over the live interval $t_{\rm live}$, while the pixel is exposing rather than being read out, with constant rates assumed. Curves are normalized and show shapes only. The picture illustrates the chapter's expected-count integral; it is an original schematic, not an instrument prescription, and the ownership assignments are the chapter's proposed ledger.
```

(radiometry-quantities)=
## Quantities and measures

| Symbol | Meaning | Unit or measure |
|---|---|---|
| $F_\nu$ | Energy flux density at the receiving aperture | W m$^{-2}$ Hz$^{-1}$; 1 Jy = $10^{-26}$ in these units |
| $F_{\lambda,\mathrm m}$ | Energy flux density per wavelength in meters | W m$^{-2}$ m$^{-1}$ |
| $\Phi_{\lambda,\mathrm{nm}}$ | Photon flux density per wavelength in nanometers | photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ |
| $I_{\lambda}$ | Extended-source photon brightness | photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$, or explicitly per arcsec$^2$ |
| $Q_b$ | Photon rate integrated over spectral bin $b$ | photon s$^{-1}$ at a stated plane |
| $c_\lambda$, $c_{pb}$ | Spectral contrast; contrast integrated over a pixel and bin | Dimensionless ratios with a named denominator |
| $A$, $T$, $q$ | Collecting area, optical transmission, effective QE | m$^2$, dimensionless, electron/photon |
| $\mu_p$, $\Sigma$ | Expected detector counts and their covariance | electron, electron$^2$ |
| $t_f$, $t_r$, $t_{\rm live}$ | Frame integration, read duration, accumulated live time | s; these are distinct quantities |

An array's shape does not settle its measure. A spectral vector can contain densities or integrals. A two-dimensional image can contain sampled brightness or pixel-integrated flux. Its metadata must decide which.

(radiometry-jy-photons)=
### From Jy to photons

Use a single physical spectrum when changing spectral coordinates:

$$
F_\nu\,|d\nu|=F_{\lambda,\mathrm m}\,d\lambda_{\mathrm m},
\qquad
F_{\lambda,\mathrm m}=F_\nu\frac{c}{\lambda_{\mathrm m}^2}.
$$

Converting the wavelength measure and dividing by photon energy gives

$$
\Phi_{\lambda,\mathrm{nm}}
=10^{-9}\frac{F_{\lambda,\mathrm m}}{hc/\lambda_{\mathrm m}}
=\frac{10^{-9}F_\nu}{h\lambda_{\mathrm m}}.
$$

With wavelength entered numerically in nm, the resulting numerical photon density is

$$
\Phi_{\lambda,\mathrm{nm}}
=\frac{F_{\rm Jy}\,10^{-26}}{h\lambda_{\rm nm}}.
$$

For **1 Jy at 700 nm**, the exact SI value $h=6.62607015\times10^{-34}$ J s ({ref}`BIPM 2019, Sec. 2.2 and Table 1 <source-bipm2019>`) gives **21,559.8597091736 photon s$^{-1}$ m$^{-2}$ nm$^{-1}$**. This independently calculated SI anchor is the expected value of the {ref}`photon-to-electron reference case <photon-electron-reference-experiment>`, whose evidence reports the library result for each build. It is not a band-integrated rate. A flat $F_\nu$ spectrum has photon density proportional to $1/\lambda$, so even a flat-Jy source requires spectral integration.

For a bin with edges $\lambda_b^-,\lambda_b^+$,

$$
Q_b=\int_{\lambda_b^-}^{\lambda_b^+}r_\lambda\,d\lambda.
$$

Here $r_\lambda$ already includes whichever collecting area and response belong before the declared output plane. Summing $Q_b$ combines bins; multiplying it by another bin width is incorrect. Conversely, summing densities without quadrature weights does not compute a flux.

(radiometry-pixel-brightness)=
### From brightness to pixels

For an extended source, integrate over solid angle as well:

$$
\Phi_{\lambda,p}=\int_{\Omega_p}I_\lambda(\boldsymbol\theta)\,d\Omega.
$$

```{figure} figures/explainer-d03-collection-where-light.png
:class: only-light
:name: fig-explainer-d03-collection-where
:alt: Two side-view rows and a face-on panel. Top row, titled point source, A Phi lambda T opt P p of lambda and theta q p in p: a yellow star labeled point source Phi lambda at theta, in photon per second per square meter per nanometer, sends two parallel yellow rays, cut by scale breaks and labeled parallel starlight, to a vertical aperture bar labeled aperture A. A lens focuses them through a plate labeled optics T opt of lambda onto a column of seven detector pixels labeled detector QE q p of lambda, whose center pixel p is outlined. A gray profile beside the column peaks at pixel p, and its part inside p is shaded more heavily; the label reads image spread, share in p of a source at theta, P p of lambda and theta. Bottom row, titled extended source, integral of I lambda P p of lambda and theta d Omega replaces Phi lambda P p: a dotted green patch labeled extended source I lambda, in photon per second per square meter per nanometer per steradian, with a dashed sky cell of p, marked idealized. A lightly shaded cone labeled Omega p narrows from the cell to its vertex at the lens center, where an arc spans it; dotted gray lines from the cell edges cross at that vertex and end on the edges of pixel p. A green dot labeled outside Omega p sits above the cell; its green ray, with a small head, passes through the lens center and ends on the pixel just below p. The gray profile beside the column now peaks on that neighboring pixel, and its tail inside p is shaded more heavily, labeled its image reaches into p. A badge reads schematic, not to scale. Right: a face-on 7 by 7 pixel window titled face-on, P p of lambda and theta in p, with a spread point-source image at one wavelength on a log color scale and the center pixel p outlined.

Where each factor of one pixel's count lives ({ref}`radiometry-pixel-brightness`, {ref}`radiometry-response-ownership`). Side views with light traveling to the right, one row per kind of source. Top, a point source at direction $\boldsymbol{\theta}$ with photon flux density $\Phi_\lambda$ sends parallel starlight into an aperture of area $A$; the optics transmit $T_{\rm opt}(\lambda)$; the image spreads over several pixels, and $P_p(\lambda,\boldsymbol{\theta})$ is the share in pixel $p$ of a point source at direction $\boldsymbol{\theta}$; the detector converts photons to electrons with QE $q_p(\lambda)$, so the electron rate per unit wavelength in $p$ is $A\Phi_\lambda T_{\rm opt}P_p(\lambda,\boldsymbol{\theta})q_p$. Bottom, an extended source with brightness $I_\lambda(\boldsymbol{\theta})$. Straight lines through the lens center carry the dashed sky cell onto pixel $p$; the shaded cone with its vertex at the lens center is the solid angle $\Omega_p$ of that cell. This idealized footprint is the cell of $\Phi_{\lambda,p}=\int_{\Omega_p}I_\lambda\,d\Omega$. A direction one pixel pitch from the cell center, half a pitch beyond its edge, has its image centered on the neighboring pixel, on the opposite side of the axis because the image is inverted, and the spread of that image still reaches into $p$; light from inside the cell likewise spreads out of $p$. The pixel therefore receives $\int I_\lambda P_p(\lambda,\boldsymbol{\theta})\,d\Omega$ over all directions, which replaces $\Phi_\lambda P_p$ in the expected-count integral. The part of each profile beside the pixel column that falls inside $p$ is shaded more heavily. Right, pixel $p$ face-on at one wavelength: an illustrative Airy image of an unobscured circular aperture, integrated over pixels one $\lambda/D$ wide and shown on a logarithmic scale; the profiles at left are cuts through the same Airy pattern before pixel integration. The next figure, {ref}`the spectral integral for pixel p <fig-explainer-d03-collection-wavelength>`, opens with this face-on panel as its left panel. An original schematic of the chapter's expected-count integral, not to scale and not an instrument prescription.
```

```{figure} figures/explainer-d03-collection-where-dark.png
:class: only-dark
:alt: Two side-view rows and a face-on panel. Top row, titled point source, A Phi lambda T opt P p of lambda and theta q p in p: a yellow star labeled point source Phi lambda at theta, in photon per second per square meter per nanometer, sends two parallel yellow rays, cut by scale breaks and labeled parallel starlight, to a vertical aperture bar labeled aperture A. A lens focuses them through a plate labeled optics T opt of lambda onto a column of seven detector pixels labeled detector QE q p of lambda, whose center pixel p is outlined. A gray profile beside the column peaks at pixel p, and its part inside p is shaded more heavily; the label reads image spread, share in p of a source at theta, P p of lambda and theta. Bottom row, titled extended source, integral of I lambda P p of lambda and theta d Omega replaces Phi lambda P p: a dotted green patch labeled extended source I lambda, in photon per second per square meter per nanometer per steradian, with a dashed sky cell of p, marked idealized. A lightly shaded cone labeled Omega p narrows from the cell to its vertex at the lens center, where an arc spans it; dotted gray lines from the cell edges cross at that vertex and end on the edges of pixel p. A green dot labeled outside Omega p sits above the cell; its green ray, with a small head, passes through the lens center and ends on the pixel just below p. The gray profile beside the column now peaks on that neighboring pixel, and its tail inside p is shaded more heavily, labeled its image reaches into p. A badge reads schematic, not to scale. Right: a face-on 7 by 7 pixel window titled face-on, P p of lambda and theta in p, with a spread point-source image at one wavelength on a log color scale and the center pixel p outlined.

Where each factor of one pixel's count lives ({ref}`radiometry-pixel-brightness`, {ref}`radiometry-response-ownership`). Side views with light traveling to the right, one row per kind of source. Top, a point source at direction $\boldsymbol{\theta}$ with photon flux density $\Phi_\lambda$ sends parallel starlight into an aperture of area $A$; the optics transmit $T_{\rm opt}(\lambda)$; the image spreads over several pixels, and $P_p(\lambda,\boldsymbol{\theta})$ is the share in pixel $p$ of a point source at direction $\boldsymbol{\theta}$; the detector converts photons to electrons with QE $q_p(\lambda)$, so the electron rate per unit wavelength in $p$ is $A\Phi_\lambda T_{\rm opt}P_p(\lambda,\boldsymbol{\theta})q_p$. Bottom, an extended source with brightness $I_\lambda(\boldsymbol{\theta})$. Straight lines through the lens center carry the dashed sky cell onto pixel $p$; the shaded cone with its vertex at the lens center is the solid angle $\Omega_p$ of that cell. This idealized footprint is the cell of $\Phi_{\lambda,p}=\int_{\Omega_p}I_\lambda\,d\Omega$. A direction one pixel pitch from the cell center, half a pitch beyond its edge, has its image centered on the neighboring pixel, on the opposite side of the axis because the image is inverted, and the spread of that image still reaches into $p$; light from inside the cell likewise spreads out of $p$. The pixel therefore receives $\int I_\lambda P_p(\lambda,\boldsymbol{\theta})\,d\Omega$ over all directions, which replaces $\Phi_\lambda P_p$ in the expected-count integral. The part of each profile beside the pixel column that falls inside $p$ is shaded more heavily. Right, pixel $p$ face-on at one wavelength: an illustrative Airy image of an unobscured circular aperture, integrated over pixels one $\lambda/D$ wide and shown on a logarithmic scale; the profiles at left are cuts through the same Airy pattern before pixel integration. The next figure, {ref}`the spectral integral for pixel p <fig-explainer-d03-collection-wavelength>`, opens with this face-on panel as its left panel. An original schematic of the chapter's expected-count integral, not to scale and not an instrument prescription.
```

A uniform field in photon units per arcsec$^2$ contributes its brightness times the pixel area in arcsec$^2$. For a distorted grid, use the local solid-angle Jacobian rather than assuming a square pixel. On a grid in $\boldsymbol u=\boldsymbol\theta/(\lambda/D)$, $d\Omega=(\lambda/D)^2d^2u$ when angles are small and in radians (the tangent-plane approximation). State whether that grid uses the current wavelength or a fixed reference wavelength.

**Proposed contract:** rendered scene pixels carry integrated flux or integrated host-relative contrast. Density-valued intermediate arrays explicitly name their measure. Regridding preserves their physical integral, subject to declared edge loss and numerical error. Sampling more rays must not increase a disk's total brightness. A resolution ladder, the same model evaluated at successively finer grids, diagnoses a model that breaks this contract without establishing its absolute calibration; the {ref}`limitations page <limitations-radiometry>` records such a report. Scalar coronagraph leakage likewise needs a coordinated per-pixel versus per-$(\lambda/D)^2$ decision across implementors.

(radiometry-contrast-zodi)=
## Contrast, magnitudes, and zodiacal light

At one wavelength, define host-relative planet contrast as $c_\lambda=\Phi_{p,\lambda}/\Phi_{\star,\lambda}$ at a common reference plane. A band contrast under response $R$ is instead

$$
c_b=\frac{\int_b R(\lambda)\Phi_{p,\lambda}\,d\lambda}
{\int_b R(\lambda)\Phi_{\star,\lambda}\,d\lambda}.
$$

It is generally not the unweighted mean of $c_\lambda$. Host-star flux, a zero-magnitude reference, an unocculted PSF peak, and total incident stellar flux are different denominators. Record the denominator and any aperture definition beside every ratio.

```{figure} figures/explainer-d12-contrast-denominators-light.png
:class: only-light
:name: fig-explainer-d12-contrast-denominators
:alt: Two log-scaled images side by side above a four-row table, sharing one colorbar, the fraction of the star's incident photons per pixel, from 1e-19 to 1e-11. Left: the off-axis PSF of a planet at 3.22 lambda/D to the right of a plus sign that marks the star, a bright core inside a dashed circle keyed 4 with diffraction rings around it; a note gives the aperture radius 0.7 lambda/D, its area 1.54 square lambda/D or 24.6 pixels, and that it holds 0.58 of the planet light. Right: the stellar leakage of a 0.0316 lambda/D star, dark at the star with faint rings, and the same dashed circle at the planet position; notes give the raw contrast in that aperture, 2.4e-14, keyed 4, and, keyed 2, that without the mask the star's peak pixel is 0.0390. Key 3 sits at the top end of the colorbar. The table lists four denominators with the numerator and denominator each uses and the value it gives for the planet: 1, host-star flux, 1.0e-10; 2, unocculted PSF peak, 9.4e-11, because the off-axis peak is 94 percent of the unocculted peak; 3, total incident stellar flux, 3.7e-12 in the peak 0.25 lambda/D pixel; 4, an aperture sum, 1.0e-10. A line under the table reads: rows 1 and 4 agree by construction; compare ratios only when their denominators match.

One planet, four denominators ({ref}`radiometry-contrast-zodi`). Left, the off-axis PSF of an example planet with band contrast $c_b=\Phi_p/\Phi_\star=10^{-10}$ at 3.22 $\lambda/D$ (92 mas at 1000 nm on the 7.2 m circumscribed aperture), a tabulated offset of the eac1_optimal_order_6_1d yield input package, so the image is the package's own PSF with no interpolation. The package's PSFs are averages over its 0.9 to 1.1 $\mu$m band, evaluated at five wavelengths, so every ratio here is a band ratio, not a monochromatic $c_\lambda$. Right, the stellar leakage of the same coronagraph for a star 0.0316 $\lambda/D$ across, the tabulated diameter nearest the Sun at 10 pc; this design nulls a point-source star. Both images are in the package's unit, the fraction of the star's photons incident on the collecting area that lands in each 0.25 $\lambda/D$ pixel (the planet image is its PSF times $10^{-10}$), drawn as raw pixels on one log scale with x right and y up from the star; how these axes map to the sky belongs to the pending {ref}`image coordinates and PSFlet origin decision <decision-image-coordinates-and-psflet-origin>`. The dashed circle is the 0.7 $\lambda/D$ photometric aperture, 1.54 $(\lambda/D)^2$ or 24.6 pixels, at the planet position. The table gives, for the same planet, the ratio each denominator returns. (1) Host-star flux: the host-relative ratio this section defines at one reference plane before the coronagraph, the flux ratio of {ref}`Nemati et al. (2023, Sec. 1, text at eq. 1) <source-nemati2023>`; it appears in neither image. (2) The unocculted PSF peak: the planet's peak pixel over the star's peak pixel without the focal-plane mask, a one-pixel form of the normalized intensity of {ref}`Nemati et al. (2023, Sec. 1.1, eq. 4) <source-nemati2023>` applied to the planet. The package carries no unocculted PSF, so its off-axis PSF at 20.5 $\lambda/D$, where the core throughput has leveled off, stands in; it is sampled at the planet's pixel phase, not at an on-axis star's. The ratio falls below $10^{-10}$ because the planet's peak is 94 percent of that unocculted peak. (3) The total incident stellar flux: the planet's peak pixel read directly in the image unit. That value scales with the pixel area: it is $3.7\times10^{-12}$ in one 0.25 $\lambda/D$ pixel, or $5.9\times10^{-11}$ per $(\lambda/D)^2$, which is the per-pixel versus per-$(\lambda/D)^2$ choice of {ref}`radiometry-pixel-brightness`. (4) An aperture sum, the yield-package normalization: the planet's sum inside the aperture divided by the same sum for a source with planet flux equal to the star's, which is the core throughput, 0.58. This is yippy's raw contrast and the contrast of {ref}`Nemati et al. (2023, Sec. 1.1, eq. 3) <source-nemati2023>`; for the leakage it gives $2.4\times10^{-14}$, so inside this aperture the planet-to-leakage ratio is the band contrast divided by the raw contrast. Row 4 returns $c_b$ exactly only for a planet whose spectrum has the star's shape, so that $c_\lambda$ is constant across the band; otherwise it weights $c_\lambda$ by the throughput-weighted stellar spectrum. Rows 1 and 4 agree by construction of that normalization, not because they are the same quantity: the first is a flux ratio at a reference plane, the second a ratio of image sums over one named aperture. This section lists four denominators, host-star flux, a zero-magnitude reference, an unocculted PSF peak and total incident stellar flux; the figure draws three of them, omits the zero-magnitude reference, and adds the aperture sum, whose aperture {ref}`optics-pixel-measure` requires to be named. The values come from yippy's throughput, core_area and raw_contrast tables and its off-axis PSFs; the definitions are identities, the planet is an example, and the scalar leakage measure every backend exports is the pending {ref}`stellar leakage measure decision <decision-stellar-leakage-measure>`.
```

```{figure} figures/explainer-d12-contrast-denominators-dark.png
:class: only-dark
:alt: Two log-scaled images side by side above a four-row table, sharing one colorbar, the fraction of the star's incident photons per pixel, from 1e-19 to 1e-11. Left: the off-axis PSF of a planet at 3.22 lambda/D to the right of a plus sign that marks the star, a bright core inside a dashed circle keyed 4 with diffraction rings around it; a note gives the aperture radius 0.7 lambda/D, its area 1.54 square lambda/D or 24.6 pixels, and that it holds 0.58 of the planet light. Right: the stellar leakage of a 0.0316 lambda/D star, dark at the star with faint rings, and the same dashed circle at the planet position; notes give the raw contrast in that aperture, 2.4e-14, keyed 4, and, keyed 2, that without the mask the star's peak pixel is 0.0390. Key 3 sits at the top end of the colorbar. The table lists four denominators with the numerator and denominator each uses and the value it gives for the planet: 1, host-star flux, 1.0e-10; 2, unocculted PSF peak, 9.4e-11, because the off-axis peak is 94 percent of the unocculted peak; 3, total incident stellar flux, 3.7e-12 in the peak 0.25 lambda/D pixel; 4, an aperture sum, 1.0e-10. A line under the table reads: rows 1 and 4 agree by construction; compare ratios only when their denominators match.

One planet, four denominators ({ref}`radiometry-contrast-zodi`). Left, the off-axis PSF of an example planet with band contrast $c_b=\Phi_p/\Phi_\star=10^{-10}$ at 3.22 $\lambda/D$ (92 mas at 1000 nm on the 7.2 m circumscribed aperture), a tabulated offset of the eac1_optimal_order_6_1d yield input package, so the image is the package's own PSF with no interpolation. The package's PSFs are averages over its 0.9 to 1.1 $\mu$m band, evaluated at five wavelengths, so every ratio here is a band ratio, not a monochromatic $c_\lambda$. Right, the stellar leakage of the same coronagraph for a star 0.0316 $\lambda/D$ across, the tabulated diameter nearest the Sun at 10 pc; this design nulls a point-source star. Both images are in the package's unit, the fraction of the star's photons incident on the collecting area that lands in each 0.25 $\lambda/D$ pixel (the planet image is its PSF times $10^{-10}$), drawn as raw pixels on one log scale with x right and y up from the star; how these axes map to the sky belongs to the pending {ref}`image coordinates and PSFlet origin decision <decision-image-coordinates-and-psflet-origin>`. The dashed circle is the 0.7 $\lambda/D$ photometric aperture, 1.54 $(\lambda/D)^2$ or 24.6 pixels, at the planet position. The table gives, for the same planet, the ratio each denominator returns. (1) Host-star flux: the host-relative ratio this section defines at one reference plane before the coronagraph, the flux ratio of {ref}`Nemati et al. (2023, Sec. 1, text at eq. 1) <source-nemati2023>`; it appears in neither image. (2) The unocculted PSF peak: the planet's peak pixel over the star's peak pixel without the focal-plane mask, a one-pixel form of the normalized intensity of {ref}`Nemati et al. (2023, Sec. 1.1, eq. 4) <source-nemati2023>` applied to the planet. The package carries no unocculted PSF, so its off-axis PSF at 20.5 $\lambda/D$, where the core throughput has leveled off, stands in; it is sampled at the planet's pixel phase, not at an on-axis star's. The ratio falls below $10^{-10}$ because the planet's peak is 94 percent of that unocculted peak. (3) The total incident stellar flux: the planet's peak pixel read directly in the image unit. That value scales with the pixel area: it is $3.7\times10^{-12}$ in one 0.25 $\lambda/D$ pixel, or $5.9\times10^{-11}$ per $(\lambda/D)^2$, which is the per-pixel versus per-$(\lambda/D)^2$ choice of {ref}`radiometry-pixel-brightness`. (4) An aperture sum, the yield-package normalization: the planet's sum inside the aperture divided by the same sum for a source with planet flux equal to the star's, which is the core throughput, 0.58. This is yippy's raw contrast and the contrast of {ref}`Nemati et al. (2023, Sec. 1.1, eq. 3) <source-nemati2023>`; for the leakage it gives $2.4\times10^{-14}$, so inside this aperture the planet-to-leakage ratio is the band contrast divided by the raw contrast. Row 4 returns $c_b$ exactly only for a planet whose spectrum has the star's shape, so that $c_\lambda$ is constant across the band; otherwise it weights $c_\lambda$ by the throughput-weighted stellar spectrum. Rows 1 and 4 agree by construction of that normalization, not because they are the same quantity: the first is a flux ratio at a reference plane, the second a ratio of image sums over one named aperture. This section lists four denominators, host-star flux, a zero-magnitude reference, an unocculted PSF peak and total incident stellar flux; the figure draws three of them, omits the zero-magnitude reference, and adds the aperture sum, whose aperture {ref}`optics-pixel-measure` requires to be named. The values come from yippy's throughput, core_area and raw_contrast tables and its off-axis PSFs; the definitions are identities, the planet is an example, and the scalar leakage measure every backend exports is the pending {ref}`stellar leakage measure decision <decision-stellar-leakage-measure>`.
```

AB magnitudes refer to $F_\nu$: $m_{\rm AB}=-2.5\log_{10}f_\nu-48.60$ with $f_\nu$ in erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ ({ref}`Oke 1974, Sec. I <source-oke1974>`), so the zero point is **3630.780547701013 Jy**, not the rounded 3631 Jy. hwoutils stores this value to float64 precision. A Vega magnitude requires a specified reference spectrum and passband. Similarly named flux fields carry different measures in different codes, such as a Vega-spectrum **band integral** in one and a **per-nm photon density** in another. The {ref}`limitations page <limitations-radiometry>` lists the reported cases, and adapters must translate them before assigning one field to another.

Zodi has three useful exchange forms: physical photon brightness, magnitude per solid angle, and brightness divided by a specified zero-magnitude flux. **Proposed boundary default:** exchange physical photon brightness; derive legacy ratios only in the receiving adapter. One "zodi" additionally needs a reference star, radius, wavelength/passband, inclination model, and normalization definition. An exozodi value already evaluated at radius $r$ must not receive another $r^{-2}$ factor. Projected star-planet separation is not generally the true dust radius. [The dust models chapter](dust-models.md) defines the dust radiance itself: the line-of-sight integral, the model profiles and their domains, the one-zodi specification fields, and the distance and pixel-measure fixtures. This chapter keeps the radiometric conversions they rely on.

Spectral color factors must name their density basis. A Table 19 $I_\lambda$ energy ratio from {ref}`Leinert et al. (1998, Sec. 8.4.1, Table 19) <source-leinert1998>` becomes a photon-density ratio after one $\lambda/\lambda_{\rm ref}$ factor, or an $I_\nu$ ratio after one squared factor. Combining those routes repeats a Jacobian. The {ref}`limitations page <limitations-radiometry>` records reports of both kinds of color error in current zodiacal-light paths; ratios measured in such reports are fixture results, not universal correction factors. Local-zodi records also carry geometry and epoch, a table/anchor revision, and an explicit out-of-domain policy for each axis. Clamping on one axis does not imply clamping on another, and a vector of epochs must not be reduced to its first element.

(radiometry-response-ownership)=
## Reference planes and response ownership

For a point source, a general expected photoelectron count in detector pixel $p$ is

$$
\mu_{e,p}=\int_{\rm live}dt\int d\lambda\;
A\Phi_\lambda(t)\,T_{\rm opt}(\lambda,t)\,
P_p(\lambda,t)\,q_p(\lambda,t).
$$

```{figure} figures/explainer-d03-collection-wavelength-light.png
:class: only-light
:name: fig-explainer-d03-collection-wavelength
:alt: Left: a face-on 7 by 7 pixel window with a spread point-source image at one wavelength on a log color scale and the center pixel p outlined. Right: normalized curves of photon density, optical transmission, dotted P p and dashed QE against wavelength, with their product, labeled Phi lambda T opt P p q p, shaded between the band edges lambda b minus and lambda b plus and labeled integral d lambda.

The spatial response and the spectral integral for pixel $p$ ({ref}`radiometry-response-ownership`, {ref}`radiometry-spectral-covariance`). Left, the face-on detector at one wavelength: an illustrative Airy image of a point source spreads over neighboring pixels, and $P_p$ is the share in the outlined pixel. Right, the photon density $\Phi_\lambda$, optical transmission $T_{\rm opt}(\lambda)$, spatial response $P_p(\lambda)$ (the share falls as the image grows with wavelength) and QE $q_p(\lambda)$ multiply at each wavelength, and only then is the band from $\lambda_b^-$ to $\lambda_b^+$ integrated (shaded). Curves are normalized and show shapes only. Schematic of the chapter's expected-count integral, not an instrument prescription.
```

```{figure} figures/explainer-d03-collection-wavelength-dark.png
:class: only-dark
:alt: Left: a face-on 7 by 7 pixel window with a spread point-source image at one wavelength on a log color scale and the center pixel p outlined. Right: normalized curves of photon density, optical transmission, dotted P p and dashed QE against wavelength, with their product, labeled Phi lambda T opt P p q p, shaded between the band edges lambda b minus and lambda b plus and labeled integral d lambda.

The spatial response and the spectral integral for pixel $p$ ({ref}`radiometry-response-ownership`, {ref}`radiometry-spectral-covariance`). Left, the face-on detector at one wavelength: an illustrative Airy image of a point source spreads over neighboring pixels, and $P_p$ is the share in the outlined pixel. Right, the photon density $\Phi_\lambda$, optical transmission $T_{\rm opt}(\lambda)$, spatial response $P_p(\lambda)$ (the share falls as the image grows with wavelength) and QE $q_p(\lambda)$ multiply at each wavelength, and only then is the band from $\lambda_b^-$ to $\lambda_b^+$ integrated (shaded). Curves are normalized and show shapes only. Schematic of the chapter's expected-count integral, not an instrument prescription.
```

$P_p$ is the integrated spatial/dispersion response into that pixel. It can include coronagraph attenuation and finite capture; its sum need not equal one. Extended sources add the angular integral. A finite star integrates the response over its angular brightness distribution; replacing that star by a point source is a model approximation, not a units conversion.

**Proposed ownership ledger:**

| Contribution | Owner and rule |
|---|---|
| Gross/effective aperture area | optixstuff primary and package adapter declare whether obscuration, gaps, and support structures are already included |
| Mirror/filter/contamination losses | Optical elements declare input/output planes and their transmission |
| Coronagraph response | yippy/physicaloptix adapters declare incident-flux denominator, stellar diameter, aperture and sampling |
| IFS transmission and finite capture | coronachrome response declares spectral/spatial support and physical losses |
| Photon-to-electron conversion | Detector response supplies effective QE exactly once, before wavelength information is discarded |
| Detector noise and calibration | Readout/acquisition adapter owns frames, noise draws, mean subtraction, and covariance |

No loss may appear in both effective collecting area and a normalized response without an explicit compensating convention. This requires a normalization certificate for each data package, not a generic area multiplier.

The proposed shared `optical_throughput` excludes QE. An explicitly named electron response may include it. Embedding QE in optical elements is a legacy workaround, not a portable shared path; the {ref}`limitations page <limitations-radiometry>` records where current paths omit the detector conversion.

**Open degradation choice (the meaning of dQE decision):** prefer an explicitly named survival fraction $\eta_{\rm deg}\in[0,1]$, neutral value 1, included once in $q_{\rm eff}=q_0\eta_{\rm deg}$. This is a proposal. A fractional-loss parameter with neutral value 0 is another coherent convention. The {ref}`limitations page <limitations-radiometry>` records reports that existing `dqe` parameters (the detector's quantum-efficiency degradation factor) have incompatible neutral values and algebra. Renaming or migrating them requires an explicit decision; multiplying by a default of zero is not a repair.

(radiometry-detector-counts)=
## Detector expectation, variance, and cadence

```{figure} figures/explainer-d07-acquisition-schedule-light.png
:class: only-light
:name: fig-explainer-d07-acquisition-schedule
:alt: Schematic in three bands. Top: planet light and background light enter one of the four aperture pixels, drawn as a charge well holding filled dots for photoelectrons and open dots for dark charge. An arrow clocks the charge out past a diamond marking clock-induced charge to a read amplifier marked with a small Gaussian curve for read noise, giving one frame value. Under the well a hatched bar reads grows with live time; under the clocking and read a solid bar reads once per frame, at its read. Middle: a time axis with hatched 10 s live intervals and solid read bars for frames 1, 2 and 10, frames 3 to 9 omitted behind a break; the first read bar is labeled read, 0.05 s, and frame 2 is outlined as the frame shown in detail. Under each read a diamond and a Gaussian curve mark one CIC draw and one read-noise draw, and a data arrow carries each frame value to a box reading E_p equals the sum of 10 frame values. A bold line states live time 10 x 10 s = 100 s and elapsed time 10 x (10 s + 0.05 s) = 100.5 s. Bottom: a table of means and variances summed over the four aperture pixels: planet photoelectrons 1000 and 1000, background photoelectrons 4000 and 4000, dark charge 4 and 4, all growing with live time; clock-induced charge 0.8 and 0.8, growing with frames; read noise 0 and 160, growing with reads as variance only; total 5004.8 electrons and 5164.8 electrons squared, and an SNR of 13.915 after subtracting the known background, dark and CIC means.

One pixel's acquisition under the simple detector model of {ref}`radiometry-detector-counts`: independent Poisson photoelectrons, dark charge and clock-induced charge (CIC), and independent Gaussian read noise, with one read per frame, unit gain, and no saturation, gain fluctuation, correlated reads or charge-transfer effects. Photoelectrons and dark charge accumulate during the live intervals (hatched), so their means and variances scale with the live time $t_{\rm live}$; CIC and read noise enter once per frame at its read (solid), so theirs scale with the frame count $n_f$, and read noise adds variance only. The schedule and the budget are the {ref}`worked count example <radiometry-count-example>`, a four-pixel aperture observed in ten 10 s frames, each followed by a 0.05 s nonoverlapping read, which occupy 100.5 s of elapsed time for 100 s of live time. The drawn pixel is one of the four; $E_p$ is one pixel's accumulated output, and the table sums it over the four pixels, with the source rates stated for the whole aperture and the detector terms per pixel. Frames 3 to 9 are omitted and the reads are drawn wider than to scale. The table lists expected values and numerical variances, not a random draw; the SNR is rounded to three decimals (13.9147 in the chapter). The terms match the analog-mode noise model of {ref}`Nemati et al. (2023, Sec. 4.1, eq. 32; Sec. 4.5) <source-nemati2023>` with unit gain and no excess-noise factor. The expectation and variance are identities under these assumptions; the read cadence is the example's stated schedule, and the {ref}`acquisition experiment decision <decision-acquisition-experiment>` that will fix read cadence is pending. The drawing does not describe nondestructive sampling or a particular detector architecture.
```

```{figure} figures/explainer-d07-acquisition-schedule-dark.png
:class: only-dark
:alt: Schematic in three bands. Top: planet light and background light enter one of the four aperture pixels, drawn as a charge well holding filled dots for photoelectrons and open dots for dark charge. An arrow clocks the charge out past a diamond marking clock-induced charge to a read amplifier marked with a small Gaussian curve for read noise, giving one frame value. Under the well a hatched bar reads grows with live time; under the clocking and read a solid bar reads once per frame, at its read. Middle: a time axis with hatched 10 s live intervals and solid read bars for frames 1, 2 and 10, frames 3 to 9 omitted behind a break; the first read bar is labeled read, 0.05 s, and frame 2 is outlined as the frame shown in detail. Under each read a diamond and a Gaussian curve mark one CIC draw and one read-noise draw, and a data arrow carries each frame value to a box reading E_p equals the sum of 10 frame values. A bold line states live time 10 x 10 s = 100 s and elapsed time 10 x (10 s + 0.05 s) = 100.5 s. Bottom: a table of means and variances summed over the four aperture pixels: planet photoelectrons 1000 and 1000, background photoelectrons 4000 and 4000, dark charge 4 and 4, all growing with live time; clock-induced charge 0.8 and 0.8, growing with frames; read noise 0 and 160, growing with reads as variance only; total 5004.8 electrons and 5164.8 electrons squared, and an SNR of 13.915 after subtracting the known background, dark and CIC means.

One pixel's acquisition under the simple detector model of {ref}`radiometry-detector-counts`: independent Poisson photoelectrons, dark charge and clock-induced charge (CIC), and independent Gaussian read noise, with one read per frame, unit gain, and no saturation, gain fluctuation, correlated reads or charge-transfer effects. Photoelectrons and dark charge accumulate during the live intervals (hatched), so their means and variances scale with the live time $t_{\rm live}$; CIC and read noise enter once per frame at its read (solid), so theirs scale with the frame count $n_f$, and read noise adds variance only. The schedule and the budget are the {ref}`worked count example <radiometry-count-example>`, a four-pixel aperture observed in ten 10 s frames, each followed by a 0.05 s nonoverlapping read, which occupy 100.5 s of elapsed time for 100 s of live time. The drawn pixel is one of the four; $E_p$ is one pixel's accumulated output, and the table sums it over the four pixels, with the source rates stated for the whole aperture and the detector terms per pixel. Frames 3 to 9 are omitted and the reads are drawn wider than to scale. The table lists expected values and numerical variances, not a random draw; the SNR is rounded to three decimals (13.9147 in the chapter). The terms match the analog-mode noise model of {ref}`Nemati et al. (2023, Sec. 4.1, eq. 32; Sec. 4.5) <source-nemati2023>` with unit gain and no excess-noise factor. The expectation and variance are identities under these assumptions; the read cadence is the example's stated schedule, and the {ref}`acquisition experiment decision <decision-acquisition-experiment>` that will fix read cadence is pending. The drawing does not describe nondestructive sampling or a particular detector architecture.
```

For independent Poisson photoelectrons, dark current $d$ in electron/pixel/s, Poisson CIC (clock-induced charge) $c$ in electron/pixel/frame, and independent Gaussian read noise $\sigma_r$ in electron RMS/pixel/read, one pixel's accumulated output has

$$
\mathbb E[E_p]=\mu_{e,p}+d\,t_{\rm live}+c\,n_f,
$$

$$
\operatorname{Var}(E_p)
=\mu_{e,p}+d\,t_{\rm live}+c\,n_f+\sigma_r^2n_f.
$$

These use numerical electron counts: each Poisson mean contributes the same numerical variance, with units electron$^2$. The same terms appear in the analog-mode noise model of {ref}`Nemati et al. (2023, Sec. 4.1, eq. 32; Sec. 4.5) <source-nemati2023>` with unit gain and no excess-noise factor. The model assumes no gain fluctuations, correlated reads, saturation, or charge-transfer effects. Additional detector physics needs its own expectation and covariance, not a silently reused formula. For gain $g$ in electron/ADU, calibrated ADU variance is electron variance divided by $g^2$ ({ref}`Janesick et al. 1987, Sec. 3.2 <source-janesick1987>`); a bias offset and its uncertainty are separate.

Count frames from the actual acquisition schedule. A read duration contributes to wall-clock occupancy when it prevents integration; it is not automatically the denominator of read-noise variance per live second. Partial frames, destructive versus nondestructive reads, and reference exposures must be explicit. Parallel optical paths, spectral bins, and telescope rolls are separate counts.

Subtracting a perfectly known background mean removes its expectation, not its shot noise. An independently measured reference adds its own covariance after scaling. Draw source-independent detector noise once per physical readout, not once per source or spectral bin. A readout that returns source electrons only leaves the detector-only noise to its caller, and a detector's deterministic variance and its stochastic readout must describe the same configured noise; the {ref}`limitations page <limitations-radiometry>` records reports of both. "Ideal" does not make photon shot noise deterministic.

(radiometry-count-example)=
### Worked count example

This is a mathematical fixture, **not evidence from the present pipeline**. An aperture contains four pixels, receives planet/background rates of 20/80 photon/s, and has constant QE .5. Observe for 100 live seconds in ten 10-second frames. Each pixel has dark current .01 electron/s, CIC .02 electron/frame, and read noise 2 electron RMS/read.

| Quantity | Calculation | Result |
|---|---|---:|
| Planet photoelectrons | $20\times.5\times100$ | 1000 |
| Background photoelectrons | $80\times.5\times100$ | 4000 |
| Dark mean and numerical variance | $4\times.01\times100$ | 4 |
| CIC mean and numerical variance | $4\times.02\times10$ | .8 |
| Read variance | $4\times2^2\times10$ | 160 electron$^2$ |

The raw mean is 5004.8 electrons; variance is 5164.8 electron$^2$. Subtracting exactly known background, dark, and CIC means leaves expected planet signal 1000 and SNR $1000/\sqrt{5164.8}=13.9147$. A measured reference would change the variance. If every frame is followed by a .05-second nonoverlapping read, this specified schedule occupies 100.5 seconds before other overheads. Replacing the 10-second frame interval with .05 seconds in the noise formula describes another experiment.

```{raw} html
<video class="only-light" controls loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Animation on fixed axes with four panels. Upper left, a gauge for the four pixels fills during each live interval and empties at each read. Upper right, charge in the four pixels against elapsed time from 0 to 100.5 s: a pink sawtooth of expected charge rising to about 500 electrons in every 10 s live interval and dropping to zero at each read, over a gray simulated trace drawn up to the current time. Lower left, a stack of frame-value blocks grows by one block per read. Lower right, the recorded sum against elapsed time: a pink expected staircase rising 500.48 electrons per read inside a pink one-standard-deviation band, and a gray simulated staircase with square markers at the reads, labeled as one simulated realization selected as representative. The top title states elapsed time as live time plus the number of reads times 0.05 s, ending at 100.5 s = 100 s + 10 reads x 0.05 s; the bottom title gives the recorded sum, 4969.0 electrons at the end, and the expected sum, 5004.8 plus or minus 71.9 electrons."><source src="../_static/explainers/explainer-d07-acquisition-count-light.mp4" type="video/mp4"></video>
<video class="only-dark" controls loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Animation on fixed axes with four panels. Upper left, a gauge for the four pixels fills during each live interval and empties at each read. Upper right, charge in the four pixels against elapsed time from 0 to 100.5 s: a pink sawtooth of expected charge rising to about 500 electrons in every 10 s live interval and dropping to zero at each read, over a gray simulated trace drawn up to the current time. Lower left, a stack of frame-value blocks grows by one block per read. Lower right, the recorded sum against elapsed time: a pink expected staircase rising 500.48 electrons per read inside a pink one-standard-deviation band, and a gray simulated staircase with square markers at the reads, labeled as one simulated realization selected as representative. The top title states elapsed time as live time plus the number of reads times 0.05 s, ending at 100.5 s = 100 s + 10 reads x 0.05 s; the bottom title gives the recorded sum, 4969.0 electrons at the end, and the expected sum, 5004.8 plus or minus 71.9 electrons."><source src="../_static/explainers/explainer-d07-acquisition-count-dark.mp4" type="video/mp4"></video>
```

**Animation.** One simulated realization of the {ref}`worked count example <radiometry-count-example>`, summed over its four-pixel aperture, as elapsed time runs through ten 10 s live intervals and their 0.05 s reads. The realization was selected as representative: of random seeds 0 to 399, seed 16 was chosen because its total ends 0.5 standard deviations below the expected total and no frame value lies more than 1.1 standard deviations from its mean. Top: the charge held in the four pixels rises during each live interval at the expected 50.04 e/s (photoelectrons plus dark charge) and empties at each read. Bottom: the recorded sum grows by one frame value per read; a frame value is the frame's charge plus one Poisson draw of clock-induced charge and one Gaussian read-noise draw. Pink curves are expectations under the model of {ref}`radiometry-detector-counts`, known before observing, and the pink band is one standard deviation of the recorded sum, which grows as the square root of the number of reads; the gray trace and squares are the realization. The top title keeps the identity elapsed time = live time + reads x 0.05 s, so ten frames occupy 100.5 s of elapsed time for 100 s of live time. Photoelectrons and dark charge are simulated as one Poisson arrival process, and every draw is independent; the animation illustrates the stated model and is not a detector simulation.

(radiometry-etc-forecast)=
## ETC forecasts and the reported experiment

Let $C_p$ be expected planet electron rate, $V$ the variance rate for the chosen estimator, and $C_{sp}$ a systematic residual amplitude in electron/s. Under the particular independent-noise plus time-coherent-floor model ({ref}`Nemati et al. 2023, Sec. 3.3, eqs. 21-23 <source-nemati2023>`),

$$
\mathrm{SNR}(t)=\frac{C_pt}{\sqrt{Vt+(C_{sp}t)^2}},
\qquad
t=\frac{S^2V}{C_p^2-S^2C_{sp}^2}.
$$

The latter requires a positive denominator. Its interpretation depends on $V$, reference subtraction, and whether $t$ is per roll or accumulated exposure. A floor is not automatically a random process with this covariance law.

EXOSIMS detection in the {ref}`Nemati (2014) <source-nemati2014>` formulation excludes planet shot noise from its background variance expression; characterization adds $\mathrm{ENF}^2C_{p0}$, where ENF (excess noise factor) scales the variance and $C_{p0}$ is the planet photoelectron rate before the photon-counting-efficiency and charge-transfer factors. Its expected signal multiplies $C_{p0}$ by those two factors. The jaxedith detection equation follows that detection convention; **missing $C_p$ in detection background is not itself a defect**. Its simplified characterization $+C_p$ agrees with the corresponding upstream term only under matching assumptions. The {ref}`limitations page <limitations-radiometry>` records the checked implementation and its version.

An SNR forecast does not specify false-alarm probability, completeness, or a nondetection likelihood. Records need the statistic, null/alternative model, reporting/selection rule, and response support. Likewise, `ppfact` cannot mean both residual fraction and reciprocal improvement: store the chosen physical meaning and convert explicitly. A zero-exozodi scene and an exozodi scene with no systematic floor are distinct assumptions.

(radiometry-spectral-covariance)=
## Spectral response and covariance

For an IFS, wavelength-dependent QE must act before spectral contributions mix in a detector pixel. Generally, $\sum_b q_bQ_b\ne q_{\rm center}\sum_bQ_b$. A constant-QE demonstration is a named restricted case, not a solution for arbitrary QE curves; the {ref}`limitations page <limitations-radiometry>` records the current scalar-QE readout.

For bin-integrated incident photon rates $\boldsymbol z$ and a response $H$ that includes electron conversion once,

$$
\boldsymbol\mu_e=tH\boldsymbol z+\boldsymbol b.
$$

In a fixed-response Gaussian approximation with covariance $\Sigma_e$, unregularized full-rank GLS (generalized least squares) gives ({ref}`Riley et al. 2006, Sec. 27.6, eqs. 27.98-27.99 <source-riley2006>`)

$$
\operatorname{Cov}(\hat{\boldsymbol z})
=\left[t^2H^\mathsf T\Sigma_e^{-1}H\right]^{-1}.
$$

Its off-diagonal terms matter. Marginalizing nuisance parameters, imposing priors, or using regularization changes the estimator and uncertainty interpretation. Declare whether an exported covariance is full, a selected block, conditional, or marginalized. Converting measurements by a known linear calibration $K$ requires $\Sigma' = K\Sigma K^\mathsf T$; uncertain stellar calibration introduces additional shared uncertainty.

Intrinsic contrast evaluated at channel centers is not automatically the forward model of extracted lenslet-bin flux. The retrieval forward must include bin integration and the applicable response, or an explicitly justified calibrated equivalent. Preserving covariance and response across the adapter from extraction to retrieval is an acceptance requirement; the {ref}`limitations page <limitations-radiometry>` records the current gap.

(radiometry-exchange-fixtures)=
## Exchange contract and acceptance fixtures

The proposed minimal product metadata are below. [The handbook](index.md) owns the shared decisions: **the stellar leakage measure decision** covers scalar leakage/apertures, **the acquisition experiment decision** covers read cadence and the observing experiment, **the image coordinates and PSFlet origin decision** covers grid/density/IFS profiles, **the reporting law decision** covers the reporting law, and **the tolerances before tests decision** covers reference fixtures and tolerances. This chapter supplies their radiometric consequences; it does not independently adopt them.

| Field group | Required meaning | Primary owner |
|---|---|---|
| Quantity | Photon/energy/electron/ADU, unit, density measure or integral, ratio denominator | Producing domain; hwoutils conversions |
| Spectral/spatial coordinates | Bin edges, quadrature, pixel edges/solid angles, angular reference wavelength | Scene and instrument adapters |
| Response | Input/output planes, area convention, included losses/QE, finite-star model, revision and support | optixstuff and response producer |
| Acquisition | Live intervals, frame/read schedule, roll/path identity, occupied time | Acquisition adapter and spaceodyssey (campaign library) |
| Statistics | Expectation, covariance units/scope, calibrated means, reference/nuisance identity, reporting rule | Detector, extraction, likelihood owners |

Before accepting integrated counts, run **three levels** in both wavelength and time: for example $N,2N,4N$ quadrature intervals over the same physical band and live interval. Refine one axis at a time, then check the combined setting. Compare total counts and local observables such as line flux and aperture leakage. For smooth inputs estimate observed order and residual error; at sharp throughput edges or occultations report convergence without inventing a formal order. Carry the resulting numerical-error band at the chosen setting. A single-point bandwidth/time approximation is acceptable only within that demonstrated domain.

Shared fixtures should include: **SI Jy anchor and colored spectra**, the SI Jy anchor and two colored spectra; **Uniform brightness and disk on three grids**, uniform brightness and a disk on three spatial grids; **Known area-loss-QE and stellar diameters**, known area/loss/QE and two stellar diameters; **Count and readout experiment**, the count/readout experiment above; **Zodi reference geometry and epoch**, explicit zodi reference, radius, inclination and epoch; **Spectral and time refinement**, spectral/time refinement with a line and changing geometry; **Correlated bins with known posterior**, two correlated spectral bins with a known Gaussian posterior; **Detection, estimation and reporting experiments**, separate detection-null, source-inclusive estimation, and reporting experiments. Expected values come from primitives independently of the implementation under test. Monte Carlo tolerances derive from sample count and a stated rejection probability.

## Coverage and gates

Every finding about the current implementations that bears on this chapter, with its contract owner, gates, acceptance fixture, status and evidence, is recorded under {ref}`limitations-radiometry` on the limitations page. A listed extension gate applies when that capability enters; it need not block a deliberately narrower observing campaign. Acceptance belongs to the owner of the supported capability; an isolated component check does not close a producer-to-consumer boundary.
