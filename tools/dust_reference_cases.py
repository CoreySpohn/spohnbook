"""Independent analytic fixtures for the dust-model conventions chapter.

These helpers are mathematical teaching fixtures, not a dust model. The
constant-emissivity sphere checks ray bounds and signs; the uniform patch
checks the radiance-to-pixel measure; the distant disk checks the distance
identities; the product-amplitude image shows a parameter degeneracy; the
Henyey-Greenstein functions check phase-function normalization.

Conventions (see the dust models chapter):

- A ray is x(s) = x_observer + s * n_hat with s >= 0 and n_hat a unit vector
  pointing from the observer into the scene.
- The star sits at the origin. Light incident on a grain at x propagates
  along x / |x|; light scattered toward the observer propagates along
  -n_hat. The scattering (deflection) angle Theta is the angle between
  these two propagation directions, so Theta = 0 is forward scattering.
- The planetary illumination angle at the same point is pi - Theta.

Only NumPy and hwoutils constants are used, so the fixtures never import a
scene, zodi or instrument library.
"""

import math

import numpy as np
from hwoutils.constants import AU2m, arcsec2rad, pc2m

_UNIT_TOL = 1e-12


def _unit(direction):
    """Return a direction as a float array, rejecting non-unit vectors."""
    n = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(n))
    if abs(norm - 1.0) > _UNIT_TOL:
        msg = f"direction must be a unit vector; got norm {norm!r}"
        raise ValueError(msg)
    return n


def _sphere_roots(observer, direction, radius):
    """Ray parameters where x(s) crosses a sphere centered on the origin.

    Returns:
        The pair (s_minus, s_plus), or None when the ray misses or grazes.
    """
    o = np.asarray(observer, dtype=float)
    n = _unit(direction)
    b = float(o @ n)
    c = float(o @ o) - radius**2
    disc = b * b - c
    if disc <= 0.0:
        return None
    root = math.sqrt(disc)
    return -b - root, -b + root


def sphere_chord_length(observer, direction, radius):
    """Full line chord through the sphere, ignoring the sign of s.

    This is the deliberately wrong (both-directions) integral that the
    positive half-ray anchors must reject.

    Args:
        observer: Observer position, AU, shape (3,).
        direction: Unit viewing direction from the observer, shape (3,).
        radius: Sphere radius, AU.

    Returns:
        Chord length in AU.
    """
    roots = _sphere_roots(observer, direction, radius)
    if roots is None:
        return 0.0
    return roots[1] - roots[0]


def sphere_path_length(observer, direction, radius):
    """Length of the positive half-ray s >= 0 inside the sphere.

    Args:
        observer: Observer position, AU, shape (3,).
        direction: Unit viewing direction from the observer, shape (3,).
        radius: Sphere radius, AU.

    Returns:
        Path length in AU; zero when the ray misses, grazes, or points away.
    """
    roots = _sphere_roots(observer, direction, radius)
    if roots is None:
        return 0.0
    s_enter = max(roots[0], 0.0)
    s_exit = max(roots[1], 0.0)
    return s_exit - s_enter


def sphere_ray_radiance(observer, direction, radius, emissivity):
    """Line-of-sight radiance of a constant-emissivity sphere.

    Args:
        observer: Observer position, AU, shape (3,).
        direction: Unit viewing direction from the observer, shape (3,).
        radius: Sphere radius, AU.
        emissivity: Photon emissivity per unit path, in
            photon s^-1 m^-2 nm^-1 sr^-1 AU^-1.

    Returns:
        Photon radiance, photon s^-1 m^-2 nm^-1 sr^-1.
    """
    return emissivity * sphere_path_length(observer, direction, radius)


def cumulative_sphere_radiance(observer, direction, radius, emissivity, s):
    """Radiance accumulated from the observer out to each distance s.

    Args:
        observer: Observer position, AU, shape (3,).
        direction: Unit viewing direction from the observer, shape (3,).
        radius: Sphere radius, AU.
        emissivity: Photon emissivity per unit path (see
            `sphere_ray_radiance`).
        s: Nonnegative distances along the ray, AU.

    Returns:
        Cumulative photon radiance at each s.
    """
    s = np.asarray(s, dtype=float)
    roots = _sphere_roots(observer, direction, radius)
    if roots is None:
        return np.zeros_like(s)
    s_enter = max(roots[0], 0.0)
    s_exit = max(roots[1], 0.0)
    inside = np.clip(s, s_enter, s_exit) - s_enter
    return emissivity * inside


def scattering_angle(position, direction):
    """Scattering angle Theta at a grain for a star at the origin.

    Args:
        position: Grain position relative to the star, shape (3,).
        direction: Unit viewing direction n_hat from the observer.

    Returns:
        Theta in radians, in [0, pi]; 0 is forward scattering.
    """
    x = np.asarray(position, dtype=float)
    k_in = x / np.linalg.norm(x)
    k_out = -_unit(direction)
    return float(np.arccos(np.clip(k_in @ k_out, -1.0, 1.0)))


def illumination_angle(position, direction):
    """Planetary illumination angle at the same point: pi minus Theta."""
    return math.pi - scattering_angle(position, direction)


def uniform_patch_pixel_flux(radiance_sr, field_arcsec, n_pix):
    """Pixel-integrated flux of a uniform square field.

    Uses the small-angle (flat tangent-plane) pixel solid angle
    (field / n_pix)^2 in steradians; for arcsecond fields the exact
    spherical correction is of order theta^2, about 1e-11 here.

    Args:
        radiance_sr: Uniform photon radiance, photon s^-1 m^-2 nm^-1 sr^-1.
        field_arcsec: Side of the square field, arcsec.
        n_pix: Pixels per side.

    Returns:
        (n_pix, n_pix) array of photon flux densities,
        photon s^-1 m^-2 nm^-1 per pixel.
    """
    pixel_rad = field_arcsec / n_pix * arcsec2rad
    return np.full((n_pix, n_pix), radiance_sr * pixel_rad**2)


def sampled_radiance_sum(radiance_sr, n_pix):
    """Sum of radiance samples with no solid-angle weight (the wrong measure).

    Args:
        radiance_sr: Uniform photon radiance, per sr.
        n_pix: Pixels per side.

    Returns:
        The unweighted sum, which grows as n_pix squared.
    """
    return float(np.full((n_pix, n_pix), radiance_sr).sum())


def distant_disk_observables(radiance_sr, disk_radius_au, distance_pc, luminosity):
    """Radiance, flux and host contrast of a face-on uniform disk.

    The disk has uniform photon radiance out to a physical radius and is
    viewed from a distant observer (small-angle limit). The star is a point
    source of isotropic photon luminosity.

    Args:
        radiance_sr: Photon radiance, photon s^-1 m^-2 nm^-1 sr^-1.
        disk_radius_au: Physical disk radius, AU.
        distance_pc: Observer distance, pc.
        luminosity: Stellar photon luminosity, photon s^-1 nm^-1.

    Returns:
        Dict with radiance, solid_angle_sr, disk_flux and star_flux (both
        photon s^-1 m^-2 nm^-1), and the dimensionless contrast.
    """
    distance_m = distance_pc * pc2m
    theta = disk_radius_au * AU2m / distance_m
    solid_angle = math.pi * theta**2
    disk_flux = radiance_sr * solid_angle
    star_flux = luminosity / (4.0 * math.pi * distance_m**2)
    return {
        "radiance": radiance_sr,
        "solid_angle_sr": solid_angle,
        "disk_flux": disk_flux,
        "star_flux": star_flux,
        "contrast": disk_flux / star_flux,
    }


def pixel_host_contrast(radiance_sr, pixel_solid_angle_sr, distance_pc, luminosity):
    """Host-relative contrast of one pixel of an extended source.

    Equals 4 pi I D^2 d(Omega) / L: the physical pixel area D^2 d(Omega)
    carries the distance dependence, not the angular area alone.

    Args:
        radiance_sr: Photon radiance, photon s^-1 m^-2 nm^-1 sr^-1.
        pixel_solid_angle_sr: Pixel solid angle, sr.
        distance_pc: Observer distance, pc.
        luminosity: Stellar photon luminosity, photon s^-1 nm^-1.

    Returns:
        Dimensionless pixel contrast.
    """
    distance_m = distance_pc * pc2m
    star_flux = luminosity / (4.0 * math.pi * distance_m**2)
    return radiance_sr * pixel_solid_angle_sr / star_flux


def product_amplitude_image(nzodis, albedo, morphology):
    """Image whose brightness depends only on the product nzodis * albedo.

    Args:
        nzodis: Dust abundance multiplier, dimensionless.
        albedo: Free spectral amplitude, dimensionless.
        morphology: Fixed spatial pattern, any shape.

    Returns:
        nzodis * albedo * morphology.
    """
    return nzodis * albedo * np.asarray(morphology, dtype=float)


def product_amplitude_jacobian(nzodis, albedo, morphology):
    """Analytic Jacobian of the flattened image with respect to (nzodis, albedo).

    Returns:
        (n_pixels, 2) array; its columns are parallel, so J^T J has rank one.
    """
    m = np.asarray(morphology, dtype=float).ravel()
    return np.stack([albedo * m, nzodis * m], axis=1)


def henyey_greenstein(cos_theta, g):
    """Henyey-Greenstein phase function normalized over 4 pi sr.

    Args:
        cos_theta: Cosine of the scattering angle Theta.
        g: Asymmetry parameter in (-1, 1); positive favors forward scattering.

    Returns:
        p(Theta) in sr^-1.
    """
    mu = np.asarray(cos_theta, dtype=float)
    return (1.0 - g**2) / (4.0 * math.pi * (1.0 + g**2 - 2.0 * g * mu) ** 1.5)


def henyey_greenstein_mixture(cos_theta, weights, gs):
    """Weighted sum of Henyey-Greenstein terms whose weights sum to one.

    Args:
        cos_theta: Cosine of the scattering angle.
        weights: Component weights; must sum to one.
        gs: Asymmetry parameter of each component.

    Returns:
        Mixture phase function in sr^-1.

    Raises:
        ValueError: If the weights do not sum to one.
    """
    weights = np.asarray(weights, dtype=float)
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        msg = f"mixture weights must sum to one; got {weights.sum()!r}"
        raise ValueError(msg)
    return sum(
        w * henyey_greenstein(cos_theta, g) for w, g in zip(weights, gs, strict=True)
    )
