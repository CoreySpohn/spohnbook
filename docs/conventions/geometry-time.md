# Geometry and time: from a physical orbit to a reported observation

*Draft contract: the choices marked pending or proposed are open.*

An orbit predicts a relationship between bodies. An observation adds an observer, a coordinate system, a clock, and a reporting rule. Most failures at this boundary leave the orbit looking plausible: its separation stays positive, its period remains reasonable, and a synthetic fit can recover parameters generated with the same mistake. Signed positions, illumination and velocity distinguish the physical interpretations.

This chapter separates three kinds of statements. **Physical identities** follow from the stated geometry or dynamical approximation. **Proposed choices** define a candidate public contract and remain pending at the conventions stage gate. Reports about **current implementations** are kept out of these definitions and recorded, with their evidence and status, on the {ref}`limitations page <limitations-geometry>`.

(geometry-illumination)=
## Start with the observer

Put the star at the origin and a distant observer to its right. A planet between the star and observer shows its unilluminated hemisphere. A planet on the far side shows its illuminated hemisphere. This says nothing about whether the star or coronagraph hides the planet: illumination and visibility are separate calculations.

```{figure} figures/hwo-conventions-geometry-light.svg
:class: only-light
:name: fig-geometry

The proposed observer-toward-positive-Z profile, shown in side view. The vertical direction is an unnamed projected coordinate, so the drawing does not choose a north/east ordering. The near-side planet is dark and the far-side planet is full. The illumination angle is measured at the planet; the observer-axis angle is measured at the star. Velocity arrows establish the recession sign independently of either brightness or apparent position.
```

```{figure} figures/hwo-conventions-geometry-dark.svg
:class: only-dark

The proposed observer-toward-positive-Z profile, shown in side view. The vertical direction is an unnamed projected coordinate, so the drawing does not choose a north/east ordering. The near-side planet is dark and the far-side planet is full. The illumination angle is measured at the planet; the observer-axis angle is measured at the star. Velocity arrows establish the recession sign independently of either brightness or apparent position.
```

| Symbol | Meaning in this chapter |
|---|---|
| $\mathbf r=\mathbf r_p-\mathbf r_\star$, $r=\lVert\mathbf r\rVert$ | Planet position relative to its host star and its magnitude |
| $\hat{\mathbf o}$ | Unit direction from star toward observer |
| $Z=\mathbf r\cdot\hat{\mathbf o}$, $\rho$ | Relative line-of-sight coordinate and projected physical separation |
| $\alpha$, $\beta_{\rm axis}$ | Illumination angle and observer-axis angle; both in $[0,\pi]$ |
| $\xi,\eta$ | East and north tangent-plane angular offsets |
| $a,e,i,\Omega,\omega_p$ | Relative-orbit elements; the angular basis is defined below |
| $M,\nu$ | Mean and true anomaly |
| $t_0,M_0,t_p$ | Element reference epoch, mean anomaly there, and periastron epoch |
| $\mu=G(M_\star+M_p)$, $n=\sqrt{\mu/a^3}$ | Two-body gravitational parameter and mean motion |
| $v_{r,\star}$ | Stellar radial velocity, positive in recession |

For a point-source star and distant observer,

$$
\cos\alpha=-\frac{Z}{r},\qquad
\beta_{\rm axis}=\operatorname{atan2}(\rho,Z),\qquad
\alpha=\pi-\beta_{\rm axis}.
$$

Consequently, a planet on the $+Z$ axis ($\rho=0$) has $\alpha=\pi$, while one on the $-Z$ axis has $\alpha=0$. A general near-side planet with $Z>0$ has $\pi/2<\alpha\leq\pi$; the sign of Z alone does not imply exact conjunction. For a Lambert sphere ({ref}`Perryman 2018, Sec. 6.15.1, eqs. 6.93 and 6.95 <source-perryman2018>`),

$$
\Phi(\alpha)=\frac{\sin\alpha+(\pi-\alpha)\cos\alpha}{\pi},\qquad
\frac{F_p}{F_\star}=A_g\left(\frac{R_p}{r}\right)^2\Phi(\alpha).
$$

The limiting phase-function values are zero, $1/\pi$, and one at new phase, quadrature, and full phase. At finite observer distance, use the actual planet-to-observer vector ({ref}`Savransky et al. 2019, Sec. 2.1, eq. 3 <source-savransky2019>`): $\cos\alpha=(-\mathbf r)\cdot(\mathbf R_{\rm obs}-\mathbf r)/(r\lVert\mathbf R_{\rm obs}-\mathbf r\rVert)$. The supplement identity is the distant-observer limit.

**Proposed choice:** physical-model inputs carry `illumination_angle_rad`; an observer-axis angle has a different name. A grid must say whether it stores intrinsic reflection or apparent contrast. The latter includes $r^{-2}$ and cannot generally be indexed by phase alone: an eccentric face-on orbit with $e=0.5$ has identical phase at periapsis and apoapsis but a 9:1 brightness ratio. The {ref}`recorded limitations <limitations-geometry>` include reports that some current models and imported grids take the observer-axis angle in place of this illumination angle.

(geometry-observer-basis)=
## Choose a basis before interpreting orbital angles

**Candidate for the observer basis and node decision, pending the conventions stage:** retain $+Z$ toward the observer and use a right-handed dynamical basis

$$
(\hat{\mathbf X},\hat{\mathbf Y},\hat{\mathbf Z})
=(\mathrm{north},\mathrm{east},\mathrm{toward\ observer}).
$$

Public tangent-plane vectors are ordered **(east, north)**, so $(\xi,\eta)\simeq(Y/d,X/d)$ in radians. This output order is not the dynamical axis order. The triad `(east, north, toward observer)` is left-handed; an array with those labels cannot silently inherit a right-handed cross-product or orbital-rotation formula.

For the candidate basis, define the ascending node as the crossing with $\dot Z>0$. Measure $\Omega$ from north toward east and $\omega_p$ from that node toward planet periapsis in the direction of motion. Inclination is the angle between angular momentum and $+Z$. With active, right-handed rotations acting on column vectors,

$$
\mathbf r=R_Z(\Omega)R_X(i)R_Z(\omega_p)
\begin{bmatrix}r\cos\nu\\r\sin\nu\\0\end{bmatrix}.
$$

This defines the node explicitly; external packages may use another observer direction or ascending-node meaning. An east-left plot is a display choice and changes none of these definitions.

(geometry-savransky-profile)=
### External reference profile: Savransky 2019

The direct-imaging community's statement of this geometry is section 2.1 of {ref}`Savransky et al. (2019) <source-savransky2019>`, Proc. SPIE 11117, with its Fig. 1: an inertial frame $\mathcal I=(O,\hat{\mathbf e}_1,\hat{\mathbf e}_2,\hat{\mathbf e}_3)$ with the star at $O$, the plane of the sky in $\hat{\mathbf e}_1$-$\hat{\mathbf e}_2$ and the observer along $+\hat{\mathbf e}_3$ "by convention"; a perifocal frame $\mathcal P=(O,\hat{\mathbf e},\hat{\mathbf q},\hat{\mathbf h})$ with $\hat{\mathbf e}$ toward periastron and $\hat{\mathbf h}$ along the orbital angular momentum; $\mathcal P$ a 3-1-3 $(\Omega,I,\omega)$ rotation from $\mathcal I$; $\theta=\nu+\omega$ the argument of latitude; $\mathbf s$ the projection of $\mathbf r_{P/O}$ onto the sky plane; and the phase angle $\cos\beta=-z/r$ in the distant-observer limit (its equations 4 and 6).

| Savransky 2019 | This chapter |
|---|---|
| observer along $+\hat{\mathbf e}_3$ | $+Z$ toward the observer |
| $\hat{\mathbf e}_1,\hat{\mathbf e}_2$, unnamed in the frame definition | $(\hat{\mathbf X},\hat{\mathbf Y})=(\mathrm{north},\mathrm{east})$ |
| 3-1-3 $(\Omega,I,\omega)$ rotation from $\mathcal I$ to $\mathcal P$ | $\mathbf r=R_Z(\Omega)R_X(i)R_Z(\omega_p)[r\cos\nu,\ r\sin\nu,\ 0]^{\mathsf T}$ |
| ascending node implied by the 3-1-3 construction: the crossing moving toward $+\hat{\mathbf e}_3$ | the crossing with $\dot Z>0$ |
| phase angle $\beta$, $\cos\beta=-z/r$ | illumination angle $\alpha$, $\cos\alpha=(-\mathbf r)\cdot\hat{\mathbf o}/r$ |
| $s=\lVert\mathbf r_{P/O}-(\mathbf r_{P/O}\cdot\hat{\mathbf e}_3)\hat{\mathbf e}_3\rVert$ | projected separation; $(\xi,\eta)\simeq(Y/d,X/d)$ |

The observer direction, rotation order, node, phase angle and projected separation agree exactly. In that table the phase angle $\beta$ maps to this chapter's $\alpha$; the paper's own $\alpha$ (its equation 2) is the angular separation, a different quantity. The ascending-node row is implied by the rotation and the paper's equation 6 rather than stated in words. One sentence differs: the paper states (section 2.1, the paragraph after its equation 2) that with the observer on $+\hat{\mathbf e}_3$ and $\hat{\mathbf e}_1$ chosen north, $\hat{\mathbf e}_2$ "becomes West-pointing", with east only for an observer on $-\hat{\mathbf e}_3$. For a right-handed frame viewed from $+\hat{\mathbf e}_3$ with $\hat{\mathbf e}_1$ up, $\hat{\mathbf e}_2$ lies to the viewer's left, which on the sky with north up is east, and $\mathrm{north}\times\mathrm{east}$ points toward the observer. The two statements cannot both hold. This chapter keeps $\hat{\mathbf Y}=\mathrm{east}$ and does not inherit the sentence; **Three-axis basis transform** must certify the handedness before the observer basis and node profile is accepted.

**Migration rule:** preserve the existing orbix profile under a distinct identifier until its meaning and adapters are certified. Its current first/second projected components are labeled RA/Dec; the candidate above must not replace those labels in place. Transform Cartesian states and covariance through a declared basis matrix, then derive or transform elements with independent anchors. Do not assume that swapping two output labels alone is a valid migration of fitted elements, angular momentum, and RV. For an orthogonal transform $Q$, polar vectors transform as $Q\mathbf r$; if $\det Q=-1$, cross products have the additional determinant factor. A reflection is not a rotation.

(geometry-origin-mass-phase)=
## Keep origin, mass and orbital phase together

Barycentric states become host-relative states by subtracting both position and velocity at the same physical epoch:

$$
\mathbf r=\mathbf r_p-\mathbf r_\star,\qquad
\mathbf v=\mathbf v_p-\mathbf v_\star.
$$

For an isolated Newtonian two-body system, relative motion uses total mass, $\mu=G(M_\star+M_p)$, and $P=2\pi\sqrt{a^3/\mu}$ ({ref}`Lovis and Fischer 2010, Sec. 2.1, eqs. 1 and 12 <source-lovis2010>`). A stellar-mass-only kernel is a separately declared approximation. At $M_p/M_\star=0.1$, it makes the period 4.88% too long; this cannot be repaired by changing the reflex amplitude.

An osculating element set is the Keplerian orbit tangent to a specified state under a specified gravitational model at a specified epoch. Imported angles from one state cannot safely be combined with header $a,e$ from another state or constant set. An N-body ephemeris also need not follow that osculating Kepler orbit indefinitely. Certify reconstruction at the epoch first, then measure later model divergence separately.

For constant $n$, all time arguments in the following identities use one coordinate and unit:

$$
M(t)=M_0+n(t-t_0)=n(t-t_p),\qquad
t_p=t_0-\frac{M_0}{n}.
$$

$t_p$ is defined modulo a period when only a wrapped mean anomaly is supplied. A fractional periastron parameter $\tau$ additionally requires its reference epoch: $M(t)=2\pi[(t-t_{\rm ref})/P-\tau]$. Reanchoring an ensemble requires each draw's own $n$; applying the nominal mean motion to every draw changes the prior's meaning.

At circular or exactly face-on states, some element angles are non-unique. Acceptance checks compare reconstructed Cartesian states and observables. They must not invent a small physical inclination to avoid a singular derivative; the {ref}`limitations page <limitations-geometry>` records a report of an implementation that does.

(geometry-radial-velocity)=
## Derive radial velocity from motion, then choose its units

**Proposed observable convention:** positive RV means recession. After separating systemic motion and other velocity corrections, in the proposed observer profile

$$
v_{r,\star}=-\dot Z_\star,
\qquad
\mathbf r_\star=-\frac{M_p}{M_\star+M_p}\mathbf r,
\qquad
v_{r,\star}=\frac{M_p}{M_\star+M_p}\dot Z.
$$

These reflex relations assume two bodies about their common barycenter. From the candidate orbital rotation,

$$
v_{r,\star}=K_\star[\cos(\nu+\omega_p)+e\cos\omega_p],\qquad
K_\star=\frac{M_p}{M_\star+M_p}\frac{na\sin i}{\sqrt{1-e^2}}.
$$

The literature writes this curve with the star's argument of periastron ({ref}`Lovis and Fischer 2010, Sec. 2.1, eqs. 10-12 <source-lovis2010>`), so mapping it onto this profile needs the body and line-of-sight conventions stated here. If the same node convention and orbital phase are retained, the stellar periapsis argument is $\omega_\star=\omega_p+\pi$. Rewriting the bracket with $\omega_\star$ introduces a minus sign. A familiar RV formula using an unlabeled `omega` is therefore insufficient evidence of compatibility. Its body, node, observer direction and velocity sign must travel together. These equations model kinematic reflex RV; precision spectroscopic redshifts and barycentric corrections require additional declared terms.

(geometry-rv-worked-example)=
### Worked signed example

Take a circular relative orbit with $a=1$ AU, $i=90$ deg, $\Omega=\omega_p=M_0=0$, and $t=t_0$. In the candidate basis, the planet is north of its star: $(X,Y,Z)=(a,0,0)$. It moves toward the observer, $\dot Z=na>0$. The star moves away, $\dot Z_\star<0$, so its recession RV is **positive**. An Earth/Sun mass pair gives about **+0.08946 m/s**. The illumination is quadrature, $\Phi=1/\pi$.

A quarter period later the planet is at $+Z$: it is dark and the stellar reflex RV is zero. Three quarters of a period later it is at $-Z$: it is full and the RV is again zero. Brightness cannot determine the sign of RV by itself. The {ref}`recorded limitations <limitations-geometry>` include reports of an implementation with the opposite stellar RV sign at this epoch and of AU/day values reaching a container that declares m/s.

(geometry-public-units)=
## Public quantities and uncertainty transforms

The following **proposed public units** separate exchanged observations from efficient dynamical kernels. Every field still carries its quantity meaning; a unit suffix alone cannot define a frame.

| Quantity | Proposed exchange representation |
|---|---|
| Relative/barycentric state | AU and AU/day; explicit origin, basis, epoch, and gravitational model |
| Mass and gravitational parameter | kg; AU$^3$/day$^2$ for the declared orbital kernel |
| Orbital and illumination angles | radians; periodic angles wrapped to $[0,2\pi)$, $i,\alpha\in[0,\pi]$ |
| Absolute sky coordinates | RA/Dec in degrees, with sky frame and reference epoch |
| Relative astrometry | $(\xi,\eta)=(\mathrm{east},\mathrm{north})$ in arcsec; declared tangent point |
| RV and velocity uncertainty/jitter | m/s, positive recession; named observed body and correction state |
| Catalog proper motion / PM anomaly | $(\mu_{\alpha *},\mu_\delta)$ in mas per Julian year; covariance in squared units |
| Exposure duration / absolute epoch | SI seconds / explicit format, scale, reference and timestamp-location metadata |

An AU/day velocity converts to m/s by 149 597 870 700 m per au ({ref}`IAU 2012 Resolution B2 <source-iau2012b2>`) over 86400 s per day, 1731456.84, the ratio of the hwoutils constants `AU2m / d2s`. Apply that factor to uncertainties and jitter too, and its square to variance. Proposed public proper motion converts to an arcsec/day kernel by $1/(1000\times365.25)$, using the Julian year of 365.25 days ({ref}`Rots et al. 2015, Sec. 4.2 <source-rots2015>`).

For a small field around fixed declination $\delta_0$, $\xi\simeq\Delta\mathrm{RA}\cos\delta_0$ and $\eta\simeq\Delta\mathrm{Dec}$. At $\delta_0=60$ deg, a 0.2-arcsec RA longitude difference is a 0.1-arcsec east displacement. This local approximation is not a general large-field coordinate transform.

For any differentiable conversion $\mathbf y=f(\mathbf x)$, first-order covariance propagation is $C_y=JC_xJ^T$, with $J=\partial f/\partial\mathbf x$. In the fixed-tangent-point example, $J=\operatorname{diag}(\cos\delta_0,1)$: east variance scales by $\cos^2\delta_0$, and east-north covariance by $\cos\delta_0$. If the tangent point is uncertain, include it and its cross-covariances in the state rather than holding it fixed silently. Proper motion uses $\mu_{\alpha *}=\dot{\mathrm{RA}}\cos\delta$ ({ref}`Brandt et al. 2021, Sec. 2 <source-brandt2021>`; {ref}`Perryman et al. 2014, Appendix A <source-perryman2014>`); catalog epoch propagation also needs parallax, radial velocity and perspective terms when relevant to the required accuracy.

Detector roll is another coordinate transform. A detector-to-sky map must declare axis order, row direction, center, rotation sense and distortion. Raw frames taken at unequal rolls are not aligned measurements of the same sky pixels. Transform positions and uncertainties, and account for covariance introduced by resampling before coadding. [Optical fields, image coordinates, and IFS products](optics-images.md) owns the pixel contract; this chapter owns its east/north destination.

(geometry-time-encoding)=
## A time value answers several different questions

```{figure} figures/hwo-conventions-time-light.svg
:class: only-light
:name: fig-time

Time origin, numerical format, time scale, timestamp location, and elapsed duration are separate decisions. Changing JD to MJD shifts the numerical origin on the same scale. Converting TAI to UTC changes the coordinate assigned to the same physical instant. A light-time correction additionally changes the reference location/event convention; it is not supplied by either operation.
```

```{figure} figures/hwo-conventions-time-dark.svg
:class: only-dark

Time origin, numerical format, time scale, timestamp location, and elapsed duration are separate decisions. Changing JD to MJD shifts the numerical origin on the same scale. Converting TAI to UTC changes the coordinate assigned to the same physical instant. A light-time correction additionally changes the reference location/event convention; it is not supplied by either operation.
```

An exchanged epoch needs its **format** (JD, MJD, or relative days), **scale** (for example TAI, UTC or TDB), **reference epoch** for relative values, and **timestamp location/correction** (for example reception at a named observatory or a specified barycentric correction). The FITS time representation separates the same components ({ref}`Rots et al. 2015, Sec. 2 <source-rots2015>`). An epoch also needs an acquisition meaning: start, end, midpoint, or a defined effective epoch. Those labels remain distinct from when data become available to an inference engine.

**Inherited choice:** the first physical profile uses TDB, as defined by {ref}`IAU 2006 Resolution B3 <source-iau2006b3>`. That is a design baseline, not evidence that current adapters implement it. **Proposed completion at the conventions stage:** retain original timestamp metadata; convert at the adapter; evaluate orbits in TDB days relative to an explicit high-precision reference; charge exposure durations in SI seconds on a continuous clock. This does not identify TAI elapsed seconds with TDB coordinate intervals or prescribe barycentric light-time correction for every simulated image. Each measurement model declares the required event convention and accuracy.

On one scale, $\mathrm{MJD}=\mathrm{JD}-2400000.5$ ({ref}`IAU 1997 Resolution B1 <source-iau1997b1>`). A scale conversion is different: at numeric MJD 60000, default UTC and TAI represent instants 37 seconds apart, because UTC - TAI = -37 s from 2017 January 1 ({ref}`IERS Bulletin C 72 <source-iers2026>`). Rebuilding a `Time` from `.jd` without its scale can therefore change the event. A float32 JD near J2000 has six-hour spacing; use adequate precision and a nearby reference or split epoch before converting to numerical arrays ({ref}`Rots et al. 2015, Sec. 3 <source-rots2015>`).

An elapsed Julian year is exactly 365.25 days of 86400 SI seconds ({ref}`Rots et al. 2015, Sec. 4.2 <source-rots2015>`). Calendar decimal years instead interpolate within the actual calendar year. From calendar 2000.0 to 2001.0 is 366 days, so treating one elapsed Julian year as that interval introduces 0.75 day. The {ref}`limitations page <limitations-geometry>` records how one external scene format encodes durations.

For an exposure interval, evaluate the time-dependent prediction over that interval when orbital motion or response changes matter. A midpoint approximation requires an error bound; a photon-weighted effective epoch requires its weighting rule. Mission day zero is operational metadata, not permission to replace an imported orbit's $t_0$ or $t_p$. Keepout configuration likewise belongs to the named platform: one context cannot silently alternate between 0-degree and 45-degree solar exclusion defaults.

(geometry-fixtures)=
## Named acceptance fixtures

These proposed fixtures become executable only after the conventions stage selects the profile. Expected answers come from primitives or independent state geometry, not the producer's own conversion. Passing them verifies the boundary mathematics; it is not measured-data validation.

| Fixture | Required discriminator |
|---|---|
| **Phase at three positions** | Equal-radius near, quadrature and far positions give $\Phi=0,1/\pi,1$; phase labels survive grid import. |
| **Eccentric periastron and apoastron contrast** | Face-on $e=0.5$ peri/apo samples share phase but have a 9:1 contrast ratio. |
| **Barycentric state origin** | Nonzero stellar barycentric position and velocity; reconstruct a complete eccentric relative state and quantify later Kepler/N-body divergence separately. |
| **Signed stellar radial velocity** | Finite-difference the stellar Cartesian position and verify $v_r=-\dot Z_\star$ in m/s, with a non-negligible-mass case and omega-body round-trip. |
| **Three-axis basis transform** | Transform all three basis vectors and an inclined eccentric orbit; certify node, handedness and signed east/north outputs. |
| **Astrometric covariance transform** | Declination 60 deg, correlated errors, proper-motion unit conversion and uncertain reference propagation. |
| **One event in every time encoding** | One event in JD/MJD/relative-day and UTC/TAI/TDB encodings; preserve one-second differences and 10-day elapsed-year sampling across a leap year. |
| **Two-roll source recovery** | One asymmetric source at two unequal rolls returns one signed sky location with certified flux and uncertainty. |
| **Ensemble epoch and precision** | Distinct semi-major axes retain their own phase advance; storage precision preserves supported threshold decisions. |

## Coverage and disposition

Every finding about the current implementations that bears on this chapter, with its owner, the stage that requires its repair, its status and its evidence, is recorded under {ref}`limitations-geometry` on the limitations page. Listed owners and gates are proposed, not completed work. The adaptive-choice stage inherits the certified geometry and time profile; it must not introduce a second convention for predictions used by policy.
