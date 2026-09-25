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

# Photon-to-electron reference case

This page follows one boundary of the suite from its source to its evidence. A
source defined in janskys is converted to photons by hwoutils and turned into an
expected electron variance by an optixstuff detector, and the result is compared
with an answer calculated independently from the SI value of the Planck constant and the definition of the jansky. The same
comparison runs as the reference case `jy-to-ideal-detector-shot-variance` in the
repository's tests, and the [evidence page](../evidence/index.md) reports its
result for this build.

| Scope | This page |
|---|---|
| Purpose | Trace one conversion chain from a cited definition to a tested library result |
| Model restrictions | Synthetic source with constant photon spectral density over 1 nm; four equally illuminated pixels; constant QE; no dark current, clock-induced charge or read noise; deterministic variance only |
| Evidence kind | Code verification against an independent analytic anchor, with two negative controls |
| Data sources | None: the only inputs are the SI value of the Planck constant and the definition of the jansky |
| Applicable profile | {ref}`photon-electron-reference-v1 <profiles-photon-electron-reference-v1>`, proposed |
| Not evidence of | Stochastic readout, band integration of a real spectrum, coronagraph throughput, nonzero detector noise, dQE, an exposure time calculator, or measured-data validation |

```{include} ../_generated/environment.md
```

(photon-electron-reference-experiment)=
## The reference experiment

```{figure} ../conventions/figures/explainer-d03-four-pixel-reference-light.png
:class: only-light
:name: fig-explainer-d03-four-pixel-reference
:alt: Left: a rectangular spectrum 1 nm wide centered at 700 nm, labeled with photon density Phi lambda nm equal to 21,559.86 photon per second per square meter per nanometer, 1 Jy at 700 nm. Right: an aperture labeled A equals 1 square meter and t equals 1 second; times A times delta lambda gives 21,559.86 photon per second, sent to a 2 by 2 pixel grid; times t over 4 gives each pixel 5,389.96 photon in 1 second, an equal split by construction, not a point-spread function. A data arrow leads to a column titled expected electrons: summed electrons equal the summed Poisson variance numerically, 0.00 for q 0.0, 10,779.93 for q 0.5 and 21,559.86 for q 1.0, with a note that there is no dark current, clock-induced charge or read noise.

The reference experiment drawn with its actual parameters ({ref}`photon-electron-reference-experiment`). A synthetic source whose photon spectral density $\Phi_{\lambda,\mathrm{nm}}$ equals that of 1 Jy at 700 nm is held constant over a 1 nm rectangular interval; 1 m$^2$ collects it for 1 s, and the photons are split equally over four pixels by construction, not by a point-spread function. With constant QE $q$ and no dark current, clock-induced charge or read noise, the summed expected electron count, and numerically the summed Poisson variance in electron$^2$, is $q$ times the photon count, shown for the three tested values of $q$. The density comes from the exact SI value of the Planck constant ({ref}`BIPM 2019, Sec. 2.2 and Table 1 <source-bipm2019>`) and the definition of the jansky; numbers are rounded to two decimals. This is a code-verification fixture against an analytic anchor, not validation, and it covers no real spectrum, optical loss, coronagraph or stochastic readout.
```

```{figure} ../conventions/figures/explainer-d03-four-pixel-reference-dark.png
:class: only-dark
:alt: Left: a rectangular spectrum 1 nm wide centered at 700 nm, labeled with photon density Phi lambda nm equal to 21,559.86 photon per second per square meter per nanometer, 1 Jy at 700 nm. Right: an aperture labeled A equals 1 square meter and t equals 1 second; times A times delta lambda gives 21,559.86 photon per second, sent to a 2 by 2 pixel grid; times t over 4 gives each pixel 5,389.96 photon in 1 second, an equal split by construction, not a point-spread function. A data arrow leads to a column titled expected electrons: summed electrons equal the summed Poisson variance numerically, 0.00 for q 0.0, 10,779.93 for q 0.5 and 21,559.86 for q 1.0, with a note that there is no dark current, clock-induced charge or read noise.

The reference experiment drawn with its actual parameters ({ref}`photon-electron-reference-experiment`). A synthetic source whose photon spectral density $\Phi_{\lambda,\mathrm{nm}}$ equals that of 1 Jy at 700 nm is held constant over a 1 nm rectangular interval; 1 m$^2$ collects it for 1 s, and the photons are split equally over four pixels by construction, not by a point-spread function. With constant QE $q$ and no dark current, clock-induced charge or read noise, the summed expected electron count, and numerically the summed Poisson variance in electron$^2$, is $q$ times the photon count, shown for the three tested values of $q$. The density comes from the exact SI value of the Planck constant ({ref}`BIPM 2019, Sec. 2.2 and Table 1 <source-bipm2019>`) and the definition of the jansky; numbers are rounded to two decimals. This is a code-verification fixture against an analytic anchor, not validation, and it covers no real spectrum, optical loss, coronagraph or stochastic readout.
```

```{figure} ../conventions/figures/explainer-d07-reference-restriction-light.png
:class: only-light
:name: fig-explainer-d07-reference-restriction
:alt: Schematic. A synthetic source labeled with the photon spectral flux density of 1 Jy at 700 nm, constant over 1 nm, sends a ray into one of four pixels, drawn as a charge well with filled dots for photoelectrons; labels say the dark charge is 0 here, the clocking adds no clock-induced charge and the read adds no read noise, and the chain ends in a box reading expected variance only. Brackets read one 1 s exposure and computed, not sampled. A table below gives the photon spectral flux density 21 559.86 photon per second per square meter per nanometer, 21 559.86 photons on the four pixels, 10 779.93 expected photoelectrons for a quantum efficiency of 0.5, a summed Poisson variance of 10 779.93 electrons squared, and zero for dark charge, clock-induced charge and read noise.

The same pixel annotated for the restricted {ref}`photon-to-electron reference case <photon-electron-reference-experiment>`: a synthetic source with the photon spectral flux density of 1 Jy at 700 nm, held constant over 1 nm, collected on 1 m$^2$ for one 1 s exposure and spread over four pixels with constant quantum efficiency. Dark current, clock-induced charge and read noise are zero, and the case compares a computed expected variance with the library's, not a sampled readout, so the clocking and read operations add nothing. For independent Poisson photoelectrons the summed variance equals the summed mean ({ref}`radiometry-detector-counts`); the photon density follows from the exact SI Planck constant ({ref}`BIPM 2019, Sec. 2.2 and Table 1 <source-bipm2019>`). The picture is a schematic of that one restricted experiment; it is not evidence of a stochastic readout, nonzero detector noise or the band integration of a real spectrum.
```

```{figure} ../conventions/figures/explainer-d07-reference-restriction-dark.png
:class: only-dark
:alt: Schematic. A synthetic source labeled with the photon spectral flux density of 1 Jy at 700 nm, constant over 1 nm, sends a ray into one of four pixels, drawn as a charge well with filled dots for photoelectrons; labels say the dark charge is 0 here, the clocking adds no clock-induced charge and the read adds no read noise, and the chain ends in a box reading expected variance only. Brackets read one 1 s exposure and computed, not sampled. A table below gives the photon spectral flux density 21 559.86 photon per second per square meter per nanometer, 21 559.86 photons on the four pixels, 10 779.93 expected photoelectrons for a quantum efficiency of 0.5, a summed Poisson variance of 10 779.93 electrons squared, and zero for dark charge, clock-induced charge and read noise.

The same pixel annotated for the restricted {ref}`photon-to-electron reference case <photon-electron-reference-experiment>`: a synthetic source with the photon spectral flux density of 1 Jy at 700 nm, held constant over 1 nm, collected on 1 m$^2$ for one 1 s exposure and spread over four pixels with constant quantum efficiency. Dark current, clock-induced charge and read noise are zero, and the case compares a computed expected variance with the library's, not a sampled readout, so the clocking and read operations add nothing. For independent Poisson photoelectrons the summed variance equals the summed mean ({ref}`radiometry-detector-counts`); the photon density follows from the exact SI Planck constant ({ref}`BIPM 2019, Sec. 2.2 and Table 1 <source-bipm2019>`). The picture is a schematic of that one restricted experiment; it is not evidence of a stochastic readout, nonzero detector noise or the band integration of a real spectrum.
```

A synthetic source has photon spectral density equal to that of 1 Jy at 700 nm,
held constant over a 1 nm rectangular interval. This is a reference spectrum
chosen to make the arithmetic exact, not an integration of a flat-$F_\nu$
spectrum, whose photon density varies as $1/\lambda$ across the interval (see
{ref}`radiometry-jy-photons`). A collecting area of 1 m$^2$ gathers it for 1 s,
and the photons fall equally on four pixels of an ideal detector with constant
quantum efficiency $q$ and no dark current, clock-induced charge or read noise.
For independent Poisson photoelectrons the summed variance is the expected
electron count (see {ref}`radiometry-detector-counts`), so

$$
\sum_p \operatorname{Var}(E_p) = q\,\Phi_{\lambda,\mathrm{nm}}\,A\,\Delta\lambda\,t,
$$

where $\Phi_{\lambda,\mathrm{nm}}$ is the photon spectral density, $A$ the area,
$\Delta\lambda$ the interval and $t$ the exposure. With the numbers above the
right-hand side is $q$ times the photon density in photon s$^{-1}$ m$^{-2}$
nm$^{-1}$, read as electron$^2$.

```{code-cell} python
from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from optixstuff import IdealDetector

AREA_M2, BANDWIDTH_NM, EXPOSURE_S, N_PIXELS = 1.0, 1.0, 1.0, 4
RTOL, ATOL = 1e-12, 1e-10
```

## The independent anchor

The Planck constant is exact in the SI, $h = 6.62607015\times10^{-34}$ J s
({ref}`BIPM 2019, section 2.2 <source-bipm2019>`), and 1 Jy is
$10^{-26}$ W m$^{-2}$ Hz$^{-1}$. Dividing the energy flux per unit frequency by
the photon energy and converting the wavelength measure gives
$\Phi_{\lambda,\mathrm{nm}} = 10^{-26}F_{\rm Jy}/(h\lambda_{\rm nm})$, derived in
{ref}`radiometry-jy-photons`. Decimal arithmetic with 40 significant digits
evaluates it without the library.

```{code-cell} python
getcontext().prec = 40
anchor = Decimal("1e-26") / (Decimal("6.62607015e-34") * Decimal(700))
library = float(jy_to_photons_per_nm_per_m2(1.0, 700.0))
print(f"independent anchor : {anchor:.13f} photon s-1 m-2 nm-1")
print(f"hwoutils           : {library:.13f} photon s-1 m-2 nm-1")
print(f"relative difference: {abs(library - float(anchor)) / float(anchor):.2e}")
```

## Through the detector

The detector's deterministic `noise_variance` is evaluated for three quantum
efficiencies. The expected value comes from the anchor, not from either library.

```{code-cell} python
def summed_variance(qe, photon_density):
    detector = IdealDetector(
        pixel_scale_arcsec=1.0,
        shape=(2, 2),
        quantum_efficiency=qe,
        dark_current_rate_e_per_s=0.0,
        read_noise_e=0.0,
        clock_induced_charge_rate_e_per_frame=0.0,
        frame_time_s=1.0,
    )
    rate = jnp.full((2, 2), photon_density * AREA_M2 * BANDWIDTH_NM / N_PIXELS)
    return float(np.asarray(detector.noise_variance(rate, EXPOSURE_S)).sum())


print(f"{'QE':>4} {'expected (e2)':>20} {'optixstuff (e2)':>20} {'|difference|':>13}")
for qe in (0.0, 0.5, 1.0):
    expected = float(anchor) * AREA_M2 * BANDWIDTH_NM * EXPOSURE_S * qe
    measured = summed_variance(qe, library)
    within = np.isclose(measured, expected, rtol=RTOL, atol=ATOL)
    print(f"{qe:4.1f} {expected:20.10f} {measured:20.10f} {abs(measured - expected):13.2e}"
          f"  {'within' if within else 'OUTSIDE'} tolerance")
```

The tolerance is a relative $10^{-12}$ with an absolute floor of $10^{-10}$ in
the units of each comparison: photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ for the density
and electron$^2$ for the variance. The calculation is a handful of float64
multiplications, whose rounding error is near $10^{-16}$ relative, so the
tolerance is a conservative floating-point floor. It is not an HWO accuracy
requirement and says nothing about the uncertainty of a real measurement.

## Negative controls

A check is only useful if it fails for the errors it is meant to catch. Two
deliberate mistakes are applied outside the libraries: a second application of
the quantum efficiency, and a factor of 1000 in the spectral measure, the size of
a per-micrometer density read as per-nanometer.

```{code-cell} python
qe = 0.5
expected = float(anchor) * qe
controls = {
    "QE applied twice": qe * summed_variance(qe, library),
    "per-um density read as per-nm": summed_variance(qe, 1000.0 * library),
}
for name, value in controls.items():
    relative = abs(value - expected) / expected
    print(f"{name:32s} relative error {relative:9.3e} "
          f"({'rejected' if relative > RTOL else 'NOT rejected'})")
```

Both controls miss the anchor by many orders of magnitude more than the
tolerance, and the repository's tests assert that the comparison rejects them.

## The four-pixel count budget (analytic only)

The radiometry chapter's {ref}`worked count example <radiometry-count-example>`
adds background, dark current, clock-induced charge and read noise. Its numbers
are recalculated below from the stated primitives. No library executes this
experiment here: the reference case above covers only the zero-noise detector,
and the {ref}`limitations page <limitations-radiometry>`
records that the detector's stochastic readout omits configured read noise and
clock-induced charge (`radiometry-idealdetector-accepts-noise-stochastic-path`,
reproduced at optixstuff 3.0.0), so a library result for this budget would not
be evidence of the detector experiment.

```{code-cell} python
pixels, qe, live, frames = 4, 0.5, 100.0, 10
rows = {
    "planet photoelectrons": 20 * qe * live,
    "background photoelectrons": 80 * qe * live,
    "dark mean and variance": pixels * 0.01 * live,
    "CIC mean and variance": pixels * 0.02 * frames,
}
read_variance = pixels * 2.0**2 * frames
mean = sum(rows.values())
variance = mean + read_variance
for name, value in rows.items():
    print(f"{name:28s} {value:10.1f}")
print(f"{'read variance (e2)':28s} {read_variance:10.1f}")
print(f"raw mean {mean:.1f} e, variance {variance:.1f} e2, "
      f"SNR after subtracting known means {rows['planet photoelectrons'] / np.sqrt(variance):.4f}")
```

## Evidence and its limits

The case's status for this build, the exact test identifiers it requires and the
dependency versions it ran with are on the [evidence page](../evidence/index.md).
A passing case shows that these two library functions reproduce this one
experiment at the recorded versions. It does not adopt the proposed profile,
verify the general photon-to-electron boundary, or validate anything against
measured data. The case is not attached to a node of the suite's validation
hierarchy: no existing node lists summed source shot variance as its response
quantity, and a code-verification case raises no validation level.
