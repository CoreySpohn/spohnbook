# Radiometry, detector counts, and spectral measurements

*Draft contract: the choices marked pending or proposed are open.*

A radiometric quantity is defined by what it counts, where it is evaluated, and the measure over which it is a density. A number labeled "flux" or "contrast" supplies too little information to connect a scene, an exposure-time calculator, an image, and a retrieved spectrum.

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

AB magnitudes refer to $F_\nu$: $m_{\rm AB}=-2.5\log_{10}f_\nu-48.60$ with $f_\nu$ in erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ ({ref}`Oke 1974, Sec. I <source-oke1974>`), so the zero point is **3630.780547701013 Jy**, not the rounded 3631 Jy. hwoutils stores this value to float64 precision. A Vega magnitude requires a specified reference spectrum and passband. Similarly named flux fields carry different measures in different codes, such as a Vega-spectrum **band integral** in one and a **per-nm photon density** in another. The {ref}`limitations page <limitations-radiometry>` lists the reported cases, and adapters must translate them before assigning one field to another.

Zodi has three useful exchange forms: physical photon brightness, magnitude per solid angle, and brightness divided by a specified zero-magnitude flux. **Proposed boundary default:** exchange physical photon brightness; derive legacy ratios only in the receiving adapter. One "zodi" additionally needs a reference star, radius, wavelength/passband, inclination model, and normalization definition. An exozodi value already evaluated at radius $r$ must not receive another $r^{-2}$ factor. Projected star-planet separation is not generally the true dust radius.

Spectral color factors must name their density basis. A Table 19 $I_\lambda$ energy ratio from {ref}`Leinert et al. (1998, Sec. 8.4.1, Table 19) <source-leinert1998>` becomes a photon-density ratio after one $\lambda/\lambda_{\rm ref}$ factor, or an $I_\nu$ ratio after one squared factor. Combining those routes repeats a Jacobian. The {ref}`limitations page <limitations-radiometry>` records reports of both kinds of color error in current zodiacal-light paths; ratios measured in such reports are fixture results, not universal correction factors. Local-zodi records also carry geometry and epoch, a table/anchor revision, and an explicit out-of-domain policy for each axis. Clamping on one axis does not imply clamping on another, and a vector of epochs must not be reduced to its first element.

(radiometry-response-ownership)=
## Reference planes and response ownership

For a point source, a general expected photoelectron count in detector pixel $p$ is

$$
\mu_{e,p}=\int_{\rm live}dt\int d\lambda\;
A\Phi_\lambda(t)\,T_{\rm opt}(\lambda,t)\,
P_p(\lambda,t)\,q_p(\lambda,t).
$$

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
