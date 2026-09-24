#!/usr/bin/env python3
"""Reproduce the limitation records that carry a public reproduction.

Each function prints the installed versions it ran against and the observable
the limitations page quotes. The script only observes library behavior; it
repairs nothing. A later release may change the result, which is why the page
records the versions next to every reproduced finding.

Usage: reproduce_limitations.py [phase-angle | ideal-detector | color-roles]
"""

import sys
from importlib.metadata import version


def _versions(*names):
    print("versions: " + ", ".join(f"{n} {version(n)}" for n in names))


def phase_angle():
    """Contrast on the near and far sides of an edge-on circular orbit.

    With +z toward the observer, a quarter period after the ascending node the
    planet is between star and observer (new phase, Lambert phase 0) and three
    quarters after it is behind the star (full phase, Lambert phase 1).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from hwoutils.constants import G, Msun2kg
    from orbix.equations.orbit import mean_motion, period_n
    from orbix.kepler import get_grid_solver
    from orbix.orbit import KeplerianOrbit
    from skyscapes.physical_model import LambertianPhysicalModel
    from skyscapes.scene import Planet, Star

    _versions("orbix", "skyscapes", "hwoutils")
    t0 = 2461000.5
    star = Star(
        Ms_kg=Msun2kg,
        dist_pc=10.0,
        wavelengths_nm=jnp.linspace(500.0, 1000.0, 3),
        times_jd=t0 + jnp.linspace(-800.0, 800.0, 3),
        flux_density_jy=jnp.full((3, 3), 50.0),
        diameter_arcsec=0.001,
    )
    orbit = KeplerianOrbit(
        a_AU=1.0, e=0.0, W_rad=0.0, i_rad=jnp.pi / 2, w_rad=0.0, M0_rad=0.0, t0_d=t0
    )
    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        Mp_Mearth=jnp.array([1.0]),
        orbit=orbit,
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    period = float(period_n(mean_motion(orbit.a_AU, G * Msun2kg))[0])
    for label, fraction in (
        ("near side (new phase)", 0.25),
        ("far side (full phase)", 0.75),
    ):
        epoch = jnp.array([t0 + fraction * period])
        contrast = float(planet.contrast(solver, 700.0, epoch, star=star)[0, 0])
        print(f"{label:24s} contrast {contrast:.3e}")


def ideal_detector():
    """Deterministic variance against the sample variance of the stochastic readout."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from optixstuff import IdealDetector

    _versions("optixstuff")
    detector = IdealDetector(
        pixel_scale_arcsec=1.0,
        shape=(300, 300),
        quantum_efficiency=1.0,
        read_noise_e=2.0,
        clock_induced_charge_rate_e_per_frame=0.02,
        frame_time_s=1.0,
    )
    rate = jnp.zeros((300, 300))
    exposure = 10.0
    predicted = float(np.asarray(detector.noise_variance(rate, exposure)).mean())
    sample = detector.readout(rate, exposure, jax.random.PRNGKey(0))
    print(f"noise_variance per pixel      {predicted:.3f} e2")
    print(f"readout sample variance       {float(np.var(np.asarray(sample))):.3f} e2")


def color_roles():
    """Colormap each hwostyle role resolves to in the light and dark modes."""
    from hwostyle.styles import MODE_CMAPS, SHARED_CMAPS

    _versions("hwostyle")
    for mode in ("light", "dark"):
        resolved = dict(SHARED_CMAPS, **MODE_CMAPS.get(mode, {}))
        named = {k: v for k, v in resolved.items() if isinstance(v, str)}
        print(
            f"{mode}: {len(resolved)} roles; "
            + ", ".join(f"{k}={v}" for k, v in named.items())
        )


REPRODUCTIONS = {
    "phase-angle": phase_angle,
    "ideal-detector": ideal_detector,
    "color-roles": color_roles,
}


if __name__ == "__main__":
    for name in sys.argv[1:] or REPRODUCTIONS:
        print(f"== {name}")
        REPRODUCTIONS[name]()
