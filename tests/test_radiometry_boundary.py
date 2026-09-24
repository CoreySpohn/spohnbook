"""Reference case jy-to-ideal-detector-shot-variance.

A synthetic source of 1 Jy at 700 nm, taken as a constant photon spectral
density over a 1 nm rectangular interval, is collected by 1 m2 for 1 s and
spread over four equally illuminated pixels of an ideal detector with constant
quantum efficiency and no dark current, clock-induced charge or read noise. The
summed deterministic source shot variance must then equal QE times the photon
count. The expected photon density comes from the exact SI value of the Planck
constant and the definition of the jansky (1e-26 W m-2 Hz-1, not an SI unit),
calculated here with decimal arithmetic, never from the
library under test.

This case verifies the Jy-to-photon conversion feeding the deterministic shot
variance. It does not verify stochastic readout, band integration of a real
spectrum, coronagraph throughput, nonzero detector noise, dQE or an exposure time
calculator.

Tolerance basis: rtol 1e-12 with atol 1e-10 in the units of each assertion
(photon s-1 m-2 nm-1 for the density, electron2 for the variance). The case is a
handful of float64 multiplications, whose rounding error is near 1e-16 relative,
so the tolerance is a conservative floating-point floor, not an HWO accuracy
requirement. The negative controls show it still rejects a doubled QE and a
factor-1000 spectral-measure error by many orders of magnitude.
"""

from decimal import Decimal, getcontext

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from optixstuff import IdealDetector

CASE = "jy-to-ideal-detector-shot-variance"
SRQ = "summed source shot variance"
RTOL, ATOL = 1e-12, 1e-10

# 1 Jy at 700 nm in photon s-1 m-2 nm-1, from h = 6.62607015e-34 J s (exact, SI
# 2019) and 1 Jy = 1e-26 W m-2 Hz-1: F_nu / (h lambda) with lambda in m, times
# 1e-9 m per nm, which reduces to 1e-26 / (h * 700).
EXPECTED_PHOTON_DENSITY = 21559.8597091736
AREA_M2, BANDWIDTH_NM, EXPOSURE_S, N_PIXELS = 1.0, 1.0, 1.0, 4


def _si_anchor():
    getcontext().prec = 40
    return Decimal("1e-26") / (Decimal("6.62607015e-34") * Decimal(700))


def _detector(qe):
    return IdealDetector(
        pixel_scale_arcsec=1.0,
        shape=(2, 2),
        quantum_efficiency=qe,
        dark_current_rate_e_per_s=0.0,
        read_noise_e=0.0,
        clock_induced_charge_rate_e_per_frame=0.0,
        frame_time_s=1.0,
    )


def _summed_variance(detector, photon_density):
    rate = jnp.full((2, 2), photon_density * AREA_M2 * BANDWIDTH_NM / N_PIXELS)
    return float(np.asarray(detector.noise_variance(rate, EXPOSURE_S)).sum())


def _assert_boundary(photon_density, variance, qe):
    np.testing.assert_allclose(
        photon_density, EXPECTED_PHOTON_DENSITY, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        variance,
        EXPECTED_PHOTON_DENSITY * AREA_M2 * BANDWIDTH_NM * EXPOSURE_S * qe,
        rtol=RTOL,
        atol=ATOL,
    )


@vv.case(CASE, "code-verification", srq=SRQ)
def test_expected_density_matches_si_definitions():
    np.testing.assert_allclose(
        EXPECTED_PHOTON_DENSITY, float(_si_anchor()), rtol=RTOL, atol=ATOL
    )


@vv.case(CASE, "code-verification", srq=SRQ)
@vv.seam("hwoutils", "optixstuff", case=CASE)
@pytest.mark.parametrize("qe", [0.0, 0.5, 1.0])
def test_jy_to_detector_shot_variance(qe):
    photon_density = float(jy_to_photons_per_nm_per_m2(1.0, 700.0))
    variance = _summed_variance(_detector(qe), photon_density)
    _assert_boundary(photon_density, variance, qe)


@vv.case(CASE, "code-verification", srq=SRQ)
def test_negative_control_second_qe_application_is_rejected():
    qe = 0.5
    photon_density = float(jy_to_photons_per_nm_per_m2(1.0, 700.0))
    variance = qe * _summed_variance(_detector(qe), photon_density)
    with pytest.raises(AssertionError):
        _assert_boundary(photon_density, variance, qe)


@vv.case(CASE, "code-verification", srq=SRQ)
def test_negative_control_spectral_measure_error_is_rejected():
    qe = 0.5
    photon_density = 1000.0 * float(jy_to_photons_per_nm_per_m2(1.0, 700.0))
    variance = _summed_variance(_detector(qe), photon_density)
    with pytest.raises(AssertionError):
        _assert_boundary(photon_density, variance, qe)


def test_four_pixel_count_budget_arithmetic():
    """Recompute the radiometry chapter's worked count example from primitives.

    Analytic only: no library executes this experiment here, so the test carries
    no evidence marker and never enters the ledger. Four pixels, 20 and
    80 photon/s planet and background, QE 0.5, 100 live seconds in ten frames,
    dark 0.01 e/s/pixel, CIC 0.02 e/frame/pixel, read noise 2 e/read.
    """
    pixels, qe, live, frames = 4, 0.5, 100.0, 10
    planet = 20 * qe * live
    background = 80 * qe * live
    dark = pixels * 0.01 * live
    cic = pixels * 0.02 * frames
    read_variance = pixels * 2.0**2 * frames
    mean = planet + background + dark + cic
    variance = planet + background + dark + cic + read_variance
    assert (planet, background) == (1000.0, 4000.0)
    np.testing.assert_allclose([dark, cic, read_variance], [4.0, 0.8, 160.0])
    np.testing.assert_allclose(mean, 5004.8, rtol=1e-12)
    np.testing.assert_allclose(variance, 5164.8, rtol=1e-12)
    np.testing.assert_allclose(planet / np.sqrt(variance), 13.9147, atol=5e-5)
