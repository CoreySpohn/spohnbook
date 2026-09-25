"""Independent analytic anchors for the dust-model conventions chapter.

Every expected value below is derived by hand from geometry or algebra and
typed into the test. None is computed by a production scene or zodi
renderer. The tolerances rtol = atol = 1e-12 are float64 fixture tolerances
(basis: floating-point), not production accuracy requirements.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from hwoutils.constants import AU2m, arcsec2rad, pc2m

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "dust_reference_cases.py"
_SPEC = importlib.util.spec_from_file_location("dust_reference_cases", _TOOLS)
drc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drc)

RTOL = 1e-12
ATOL = 1e-12

# Constant-emissivity sphere: radius 2 AU at the origin, emissivity
# j = 3 photon s^-1 m^-2 nm^-1 sr^-1 AU^-1.
RADIUS_AU = 2.0
EMISSIVITY = 3.0

# (observer AU, unit direction, positive-ray path AU, radiance)
SPHERE_CASES = [
    ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, 3.0),  # inside, toward the near wall
    ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 3.0, 9.0),  # inside, across the center
    ((5.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 4.0, 12.0),  # outside, full chord
    ((5.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0, 0.0),  # outside, looking away
    ((2.0, -3.0, 0.0), (0.0, 1.0, 0.0), 0.0, 0.0),  # tangent grazing ray
]


@pytest.mark.parametrize(("observer", "direction", "path", "radiance"), SPHERE_CASES)
def test_sphere_positive_half_ray(observer, direction, path, radiance):
    length = drc.sphere_path_length(observer, direction, RADIUS_AU)
    np.testing.assert_allclose(length, path, rtol=RTOL, atol=ATOL)
    value = drc.sphere_ray_radiance(observer, direction, RADIUS_AU, EMISSIVITY)
    np.testing.assert_allclose(value, radiance, rtol=RTOL, atol=ATOL)


def test_sphere_path_is_never_negative():
    rng = np.random.default_rng(20260924)
    observers = rng.uniform(-6.0, 6.0, size=(500, 3))
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    lengths = [
        drc.sphere_path_length(o, n, RADIUS_AU)
        for o, n in zip(observers, directions, strict=True)
    ]
    assert len(lengths) == 500
    assert min(lengths) >= 0.0
    assert max(lengths) <= 2.0 * RADIUS_AU + ATOL


def test_full_line_mutation_fails_the_inside_anchor():
    """A kernel that integrates the whole line (both signs of s) is wrong.

    Basis: mutation. Integrating the full chord instead of the positive
    half-ray gives 4 AU for both inside cases, which the anchors reject.
    """
    observer, direction, path, _ = SPHERE_CASES[0]
    full_chord = drc.sphere_chord_length(observer, direction, RADIUS_AU)
    assert full_chord == pytest.approx(4.0, rel=RTOL)
    assert not np.isclose(full_chord, path, rtol=RTOL, atol=ATOL)


def test_non_unit_direction_is_rejected():
    with pytest.raises(ValueError, match="unit"):
        drc.sphere_path_length((5.0, 0.0, 0.0), (-2.0, 0.0, 0.0), RADIUS_AU)


def test_cumulative_ray_integral_reaches_the_total():
    observer, direction = (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)
    s = np.linspace(0.0, 5.0, 11)
    cumulative = drc.cumulative_sphere_radiance(
        observer, direction, RADIUS_AU, EMISSIVITY, s
    )
    # Inside the sphere for s in [0, 3] AU, so j * s there, then flat at 9.
    expected = EMISSIVITY * np.minimum(s, 3.0)
    np.testing.assert_allclose(cumulative, expected, rtol=RTOL, atol=ATOL)


def test_scattering_angle_signs():
    """Forward scattering: grain between star and observer; backward behind."""
    # Observer at +5 AU looking back toward the star along -x.
    direction = (-1.0, 0.0, 0.0)
    forward = drc.scattering_angle((1.0, 0.0, 0.0), direction)
    backward = drc.scattering_angle((-1.0, 0.0, 0.0), direction)
    side = drc.scattering_angle((0.0, 1.0, 0.0), direction)
    np.testing.assert_allclose(forward, 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(backward, math.pi, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(side, math.pi / 2, rtol=RTOL, atol=ATOL)
    # The planetary illumination angle at the same point is the supplement.
    alpha = drc.illumination_angle((1.0, 0.0, 0.0), direction)
    np.testing.assert_allclose(alpha, math.pi, rtol=RTOL, atol=ATOL)


# Uniform patch: a 2 arcsec square field of uniform photon radiance.
FIELD_ARCSEC = 2.0
RADIANCE_SR = 7.0  # photon s^-1 m^-2 nm^-1 sr^-1
# Typed once from its definition, then checked against hwoutils.
ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def test_arcsec_constant_parity():
    assert arcsec2rad == pytest.approx(ARCSEC_TO_RAD, rel=1e-15)


@pytest.mark.parametrize("n_pix", [24, 48, 96])
def test_uniform_patch_conserves_integrated_flux(n_pix):
    pixels = drc.uniform_patch_pixel_flux(RADIANCE_SR, FIELD_ARCSEC, n_pix)
    assert pixels.shape == (n_pix, n_pix)
    expected_total = RADIANCE_SR * (FIELD_ARCSEC * ARCSEC_TO_RAD) ** 2
    expected_pixel = expected_total / n_pix**2
    np.testing.assert_allclose(pixels.sum(), expected_total, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(pixels, expected_pixel, rtol=RTOL, atol=0.0)


def test_pixel_flux_quarters_while_total_holds():
    coarse = drc.uniform_patch_pixel_flux(RADIANCE_SR, FIELD_ARCSEC, 24)
    fine = drc.uniform_patch_pixel_flux(RADIANCE_SR, FIELD_ARCSEC, 48)
    np.testing.assert_allclose(fine[0, 0] / coarse[0, 0], 0.25, rtol=RTOL)
    np.testing.assert_allclose(fine.sum(), coarse.sum(), rtol=RTOL)


def test_missing_solid_angle_mutation_fails_conservation():
    """Summing radiance samples without d(Omega) grows as N^2.

    Basis: mutation. The sampled-sum ladder is the negative control the
    conservation test must reject.
    """
    sums = [drc.sampled_radiance_sum(RADIANCE_SR, n) for n in (24, 48, 96)]
    np.testing.assert_allclose(np.diff(np.log2(sums)), [2.0, 2.0], rtol=RTOL)
    assert not np.isclose(sums[0], sums[2], rtol=1e-3)


# Distant face-on disk: uniform radiance out to a physical radius.
DISK_RADIUS_AU = 3.0
DISTANCE_PC = 10.0
STAR_PHOTON_LUMINOSITY = 4.0e44  # photon s^-1 nm^-1


def test_distance_doubling_invariants():
    near = drc.distant_disk_observables(
        RADIANCE_SR, DISK_RADIUS_AU, DISTANCE_PC, STAR_PHOTON_LUMINOSITY
    )
    far = drc.distant_disk_observables(
        RADIANCE_SR, DISK_RADIUS_AU, 2.0 * DISTANCE_PC, STAR_PHOTON_LUMINOSITY
    )
    # Radiance along a physical sightline does not depend on distance.
    np.testing.assert_allclose(far["radiance"], near["radiance"], rtol=RTOL)
    # The disk subtends a quarter of the solid angle and delivers a quarter
    # of the flux; the star dims by the same factor, so contrast holds.
    np.testing.assert_allclose(
        far["solid_angle_sr"], near["solid_angle_sr"] / 4, rtol=RTOL
    )
    np.testing.assert_allclose(far["disk_flux"], near["disk_flux"] / 4, rtol=RTOL)
    np.testing.assert_allclose(far["star_flux"], near["star_flux"] / 4, rtol=RTOL)
    np.testing.assert_allclose(far["contrast"], near["contrast"], rtol=RTOL)


def test_distant_disk_absolute_values():
    obs = drc.distant_disk_observables(
        RADIANCE_SR, DISK_RADIUS_AU, DISTANCE_PC, STAR_PHOTON_LUMINOSITY
    )
    au_m = 1.495978707e11  # IAU 2012 exact definition
    assert au_m == pytest.approx(AU2m, rel=1e-15)
    # 1 pc = 648000 / pi AU by the IAU 2015 definition.
    assert pc2m == pytest.approx(648000.0 / math.pi * au_m, rel=1e-12)
    distance_m = DISTANCE_PC * pc2m
    theta = DISK_RADIUS_AU * au_m / distance_m  # small-angle radius, rad
    solid_angle = math.pi * theta**2
    star_flux = STAR_PHOTON_LUMINOSITY / (4.0 * math.pi * distance_m**2)
    np.testing.assert_allclose(obs["solid_angle_sr"], solid_angle, rtol=1e-12)
    np.testing.assert_allclose(obs["disk_flux"], RADIANCE_SR * solid_angle, rtol=RTOL)
    np.testing.assert_allclose(
        obs["contrast"], RADIANCE_SR * solid_angle / star_flux, rtol=RTOL
    )


def test_pixel_contrast_needs_physical_area():
    """Host contrast of one pixel scales with D^2 d(Omega), not d(Omega) alone.

    At fixed angular pixel size the physical area grows as D^2 while the
    star dims as D^-2 and the radiance is unchanged, so the contrast of a
    fixed angular pixel grows as D^2 (it covers more of the cloud).
    """
    d_omega = (0.01 * ARCSEC_TO_RAD) ** 2
    near = drc.pixel_host_contrast(RADIANCE_SR, d_omega, 10.0, STAR_PHOTON_LUMINOSITY)
    far = drc.pixel_host_contrast(RADIANCE_SR, d_omega, 20.0, STAR_PHOTON_LUMINOSITY)
    np.testing.assert_allclose(far / near, 4.0, rtol=RTOL)
    # Same physical area at both distances: contrast holds.
    far_matched = drc.pixel_host_contrast(
        RADIANCE_SR, d_omega / 4.0, 20.0, STAR_PHOTON_LUMINOSITY
    )
    np.testing.assert_allclose(far_matched, near, rtol=RTOL)


def test_amplitude_product_is_degenerate():
    morphology = np.array([[0.0, 1.0], [2.0, 0.5]])
    a = drc.product_amplitude_image(2.0, 0.15, morphology)
    b = drc.product_amplitude_image(1.0, 0.30, morphology)
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=0.0)


def test_amplitude_gradient_nonzero_but_fisher_singular():
    """Differentiable is not identifiable: each partial is nonzero, yet the
    Fisher matrix for (nzodis, albedo) has rank one."""
    morphology = np.array([[0.0, 1.0], [2.0, 0.5]])
    jac = drc.product_amplitude_jacobian(2.0, 0.15, morphology)
    assert np.all(np.abs(jac).sum(axis=0) > 0.0)
    fisher = jac.T @ jac
    assert np.linalg.matrix_rank(fisher) == 1
    # Doubling the amplitude is a real change the image must see.
    doubled = drc.product_amplitude_image(4.0, 0.15, morphology)
    base = drc.product_amplitude_image(2.0, 0.15, morphology)
    assert not np.allclose(doubled, base)


@pytest.mark.parametrize("g", [-0.5, 0.0, 0.3, 0.9])
def test_henyey_greenstein_normalized(g):
    """Integral of p over 4 pi sr is one (closed form: 2 / (1 - g^2) times
    (1 - g^2) / (4 pi) times 2 pi). Gauss-Legendre in mu; basis:
    floating-point plus quadrature error, checked at 1e-10."""
    mu, w = np.polynomial.legendre.leggauss(4000)
    total = 2.0 * math.pi * np.sum(w * drc.henyey_greenstein(mu, g))
    np.testing.assert_allclose(total, 1.0, rtol=1e-10)


def test_henyey_greenstein_mixture_needs_normalized_weights():
    mu, w = np.polynomial.legendre.leggauss(4000)
    mixed = drc.henyey_greenstein_mixture(mu, (0.7, 0.3), (0.8, -0.2))
    np.testing.assert_allclose(2.0 * math.pi * np.sum(w * mixed), 1.0, rtol=1e-10)
    with pytest.raises(ValueError, match="sum to one"):
        drc.henyey_greenstein_mixture(mu, (0.7, 0.7), (0.8, -0.2))


# Thin uniform slab (face-on radius R, thickness h) seen by a distant observer.
SLAB_RADIUS_AU = 4.0
SLAB_THICKNESS_AU = 0.1
SLAB_EMISSIVITY = 5.0  # photon s^-1 m^-2 nm^-1 sr^-1 AU^-1


@pytest.mark.parametrize("inclination_deg", [60.0, 120.0])
def test_thin_slab_path_is_positive_on_both_sides(inclination_deg):
    """|cos i| = 1/2 at 60 and 120 deg, so the path is 2h inside the ellipse."""
    inc = math.radians(inclination_deg)
    half_minor = SLAB_RADIUS_AU * 0.5
    inside = drc.thin_slab_radiance(
        0.0, 0.9 * half_minor, SLAB_RADIUS_AU, SLAB_THICKNESS_AU, SLAB_EMISSIVITY, inc
    )
    outside = drc.thin_slab_radiance(
        0.0, 1.1 * half_minor, SLAB_RADIUS_AU, SLAB_THICKNESS_AU, SLAB_EMISSIVITY, inc
    )
    np.testing.assert_allclose(inside, 2.0 * SLAB_THICKNESS_AU * SLAB_EMISSIVITY)
    assert outside == 0.0


def test_signed_slab_weight_goes_negative_past_ninety():
    """The mutant keeps the sign of cos i: a physically impossible negative
    radiance that a floored log display would hide."""
    inc = math.radians(120.0)
    signed = drc.thin_slab_radiance(
        0.0, 0.0, SLAB_RADIUS_AU, SLAB_THICKNESS_AU, SLAB_EMISSIVITY, inc, signed=True
    )
    np.testing.assert_allclose(signed, -2.0 * SLAB_THICKNESS_AU * SLAB_EMISSIVITY)


# Power-law energy radiance B(lambda) = B0 (lambda / lambda0)^k.
PLANCK_H = 6.62607015e-34  # J s, exact SI
LIGHT_C = 299792458.0  # m s^-1, exact SI


def test_planck_and_light_constants_parity():
    from hwoutils.constants import c, h

    assert h == pytest.approx(PLANCK_H, rel=1e-15)
    assert c == pytest.approx(LIGHT_C, rel=1e-15)


@pytest.mark.parametrize("k", [0.0, -1.0, -4.0, 1.5])
def test_photon_band_integral_matches_quadrature(k):
    """Convert to photons inside the integral; Gauss-Legendre is the
    independent check of the closed form (basis: floating-point)."""
    b0, lam0, lo, hi = 2.0e-8, 550.0, 500.0, 600.0
    lam, w = np.polynomial.legendre.leggauss(64)
    lam = 0.5 * (hi - lo) * lam + 0.5 * (hi + lo)
    density = b0 * (lam / lam0) ** k * lam * 1e-9 / (PLANCK_H * LIGHT_C)
    quad = 0.5 * (hi - lo) * np.sum(w * density)
    exact = drc.power_law_photon_band_integral(b0, lam0, k, lo, hi)
    np.testing.assert_allclose(exact, quad, rtol=1e-12)


def test_center_conversion_is_exact_only_for_flat_energy():
    b0, lam0, lo, hi = 2.0e-8, 550.0, 500.0, 600.0
    flat = drc.center_conversion_band_integral(b0, lam0, 0.0, lo, hi)
    np.testing.assert_allclose(
        flat, drc.power_law_photon_band_integral(b0, lam0, 0.0, lo, hi), rtol=1e-12
    )
    steep_exact = drc.power_law_photon_band_integral(b0, lam0, -4.0, lo, hi)
    steep_center = drc.center_conversion_band_integral(b0, lam0, -4.0, lo, hi)
    assert abs(steep_center / steep_exact - 1.0) > 1e-3


# Circular Gaussian clump of photon radiance, integrated exactly per pixel.
CLUMP_PEAK_SR = 7.0
CLUMP_SIGMA_ARCSEC = 0.25


def _clump_total():
    """Closed form: peak * 2 pi sigma^2 * erf(F / (2 sqrt2 sigma))^2, in sr."""
    sigma = CLUMP_SIGMA_ARCSEC * ARCSEC_TO_RAD
    half = 0.5 * FIELD_ARCSEC / CLUMP_SIGMA_ARCSEC
    return CLUMP_PEAK_SR * 2.0 * math.pi * sigma**2 * math.erf(half / math.sqrt(2)) ** 2


@pytest.mark.parametrize("n_pix", [24, 48, 96])
def test_gaussian_clump_pixel_integrals_conserve_flux(n_pix):
    pixels = drc.gaussian_patch_pixel_flux(
        CLUMP_PEAK_SR, CLUMP_SIGMA_ARCSEC, FIELD_ARCSEC, n_pix
    )
    assert pixels.shape == (n_pix, n_pix)
    assert np.all(pixels > 0.0)
    np.testing.assert_allclose(pixels.sum(), _clump_total(), rtol=1e-12, atol=0.0)


def test_gaussian_clump_center_pixel_approaches_radiance_times_area():
    """As pixels shrink, center-pixel flux / pixel solid angle -> peak radiance."""
    ratios = []
    for n_pix in (24, 48, 96):
        pixels = drc.gaussian_patch_pixel_flux(
            CLUMP_PEAK_SR, CLUMP_SIGMA_ARCSEC, FIELD_ARCSEC, n_pix
        )
        d_omega = (FIELD_ARCSEC / n_pix * ARCSEC_TO_RAD) ** 2
        # Even grids: the four central pixels share the peak.
        c = n_pix // 2
        ratios.append(pixels[c, c] / d_omega)
    errors = np.abs(np.array(ratios) / CLUMP_PEAK_SR - 1.0)
    assert errors[0] > errors[1] > errors[2]
    # Smooth integrand, pixel-average error is second order in pixel size.
    observed_order = np.log2(errors[:-1] / errors[1:])
    np.testing.assert_allclose(observed_order, 2.0, atol=0.05)


def test_raster_distance_rescale_keeps_contrast_per_pixel():
    """A distant-observer contrast raster moves from D to D' exactly by
    scaling the angular pixel by D / D': the physical pixel area D^2 dOmega,
    and so each pixel's contrast, is unchanged. Basis: exact algebra."""
    pixscale_mas = 2.0
    new_scale = drc.raster_pixscale_at_distance(pixscale_mas, 10.0, 25.0)
    np.testing.assert_allclose(new_scale, 0.8, rtol=RTOL)
    d_old = (pixscale_mas * 1e-3 * ARCSEC_TO_RAD) ** 2
    d_new = (new_scale * 1e-3 * ARCSEC_TO_RAD) ** 2
    old = drc.pixel_host_contrast(RADIANCE_SR, d_old, 10.0, STAR_PHOTON_LUMINOSITY)
    new = drc.pixel_host_contrast(RADIANCE_SR, d_new, 25.0, STAR_PHOTON_LUMINOSITY)
    np.testing.assert_allclose(new, old, rtol=RTOL)
