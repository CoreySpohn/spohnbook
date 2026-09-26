"""The physical observation, from a distant system to a measured product.

One composition answers the reader's first question: what physical system do
the libraries represent, and where along it does a given chapter sit? The top
row is physical light. A host star, a planet and an exozodiacal disk are drawn
side on with skyscapes.viz (observer to the right, +z toward the observer);
their light crosses a scale break to the observer, who sits inside the local
zodiacal cloud; a generic telescope aperture feeds a generic coronagraph rail
(eyepiece) that ends on a detector. The bottom row is data: a synthetic raw
frame, the records reduced from several visits (in the records chapter's
symbols), and an inferred orbit drawn as posterior draws (orbix.viz) given
the measured positions. Outlined chapter locators name the chapter that
defines each part.

``build_overview(layout, cast, highlight=None)`` draws the map. ``highlight``
names one chapter region (see ``HIGHLIGHTS``); every other region is dimmed
toward the background and the chapter's locator is drawn bold with a
"this chapter" tag, so the same drawing opens each chapter with its own part
picked out.

Labels carry no stroked halo; a label over an image, hatch or line sits on a
plain backing box instead, which keeps the vector exports small.

The relay stills follow one scene through five two-panel steps; the right
panel of each step returns, drawn by the same function, as the left panel of
the next.

No number is printed on any graphic. The synthetic frame's amplitudes are
schematic, not a brightness ratio; tests/test_explainer_d01.py pins that
contract and the sky orientation shared by the frame and the orbit panels.
"""

import functools
import math

import eyepiece as ep
import hwostyle
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import AU2m, G_si, Msun2kg, s2d
from hwoutils.conversions import au_to_arcsec
from matplotlib.collections import Collection
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.text import Text

from explainers import _common as ex

# The scene. One Earth-size planet on a nearly circular orbit inside one
# parametric exozodiacal disk around a solar-mass star at 10 pc, built with
# the same skyscapes types as the skyscapes scene-geometry page. The orbit is
# placed in the disk midplane: under skyscapes' sky rotation, inclination 60
# degrees and node angle equal to the disk position angle give the orbit the
# midplane's normal (tests/test_explainer_d01.py checks this), so in the side
# view the orbit lies along the disk.
#
# Sky profile. Every sky position and sky image here is in the geometry
# chapter's proposed observer profile: the right-handed basis (X, Y, Z) =
# (north, east, toward observer), r = R_Z(Omega) R_X(i) R_Z(omega_p) [r cos nu,
# r sin nu, 0], and exchanged offsets (xi, eta) = (east, north) = (Y, X) / d.
# orbix and skyscapes apply the same rotation but label their first rotated
# component RA (east) and their second Dec (north). The two profiles share
# the rotated vector and differ only in those labels, so the mapping is
# explicit: the chapter's (xi, eta) is (second, first) of the library output,
# and a library sky image indexed (second, first) is transposed. In the
# chapter profile the ascending node lies at position angle Omega east of
# north; tests/test_explainer_d01.py recomputes the positions from the
# chapter's formula and pins the node.
PROFILE = (
    "geometry chapter observer profile (proposed): (X, Y, Z) = (north, east, "
    "toward observer), r = R_Z(Omega) R_X(i) R_Z(omega_p) r_perifocal, offsets "
    "(xi, eta) = (east, north) = (Y, X) / d; library (RA, Dec) outputs mapped "
    "by exchanging their two sky components"
)
SCENE = {
    "dist_pc": 10.0,
    "a_AU": 3.0,
    "e": 0.05,
    "i_deg": 60.0,
    "W_deg": 30.0,
    "w_deg": 0.0,
    "M0_deg": 0.0,
    "disk_incl_deg": 60.0,
    "disk_pa_deg": 30.0,
    "disk_r_in_AU": 1.0,
    "disk_r_out_AU": 5.0,
    "wavelength_nm": 550.0,
}
# The epoch drawn in the system panel and the synthetic frame, and the four
# visit epochs whose positions feed the orbit panel. The drawn epoch is the
# third visit: the planet is on the far side of the sky plane (lit gibbous)
# at a projected separation well outside the frame's stellar leakage.
T_EPOCH_JD = 1186.0
T_VISITS_JD = (870.0, 1030.0, 1186.0, 1345.0)

# The synthetic raw frame: a small detector cutout displayed on the sky
# (east to the left, north up). Expected counts per pixel are a uniform
# local-zodi level, the skyscapes disk image block-summed to the frame
# pixels, a pixel-integrated Gaussian planet image and a stellar leakage core
# with a few speckles; the frame is one Poisson realization. The amplitudes
# are schematic and chosen so every contribution is visible; they are not a
# star-planet brightness ratio and none is printed.
FRAME = {
    "n_pix": 25,
    "pixel_arcsec": 0.04,
    "oversample": 5,
    "local_zodi": 10.0,
    "exozodi_peak": 40.0,
    "planet_total": 700.0,
    "psf_sigma_pix": 0.8,
    "leak_core": 2500.0,
    "display_max": 300.0,
    "leak_core_pix": 1.0,
    "speckle_amp": 70.0,
    "n_speckles": 6,
    "speckle_radius_pix": 2.6,
    "seed": 20260925,
}
# Posterior draws by importance resampling. The measured positions are the
# true positions at the four visits plus independent Gaussian noise of
# ``astrometry_sigma_arcsec`` per axis. The prior: eccentricity uniform on
# [0, ``prior_e_max``], argument of periapsis and mean anomaly at epoch
# uniform on the circle, and semimajor axis, inclination and node angle
# Gaussian about the simulated values with the ``prior_*`` widths. It is
# sampled ``n_prior`` times, each sample is weighted by the Gaussian
# likelihood of the four measured positions, and ``n`` draws are taken by
# systematic resampling (with replacement) in proportion to the weights.
DRAWS = {
    "n": 30,
    "n_prior": 6_000_000,
    "chunk": 500_000,
    "prior_a_frac": 0.15,
    "prior_i_deg": 15.0,
    "prior_W_deg": 15.0,
    "prior_e_max": 0.6,
    "astrometry_sigma_arcsec": 0.012,
    "seed": 7,
}

# Chapter regions: the parts of the map each highlight keeps at full value.
HIGHLIGHTS = {
    "geometry": ("system", "path"),
    "dust": ("disk", "local_zodi"),
    "radiometry": ("path", "local_zodi", "aperture", "frame", "readout"),
    "optics": ("aperture", "instrument", "frame"),
    "inference": ("frame", "records", "orbit", "data"),
    "tutorial": (
        "system",
        "disk",
        "path",
        "local_zodi",
        "aperture",
        "instrument",
        "readout",
        "frame",
    ),
}
DIM = 0.28

# Chapter locators: each chapter's short name, placed on the map next to the
# part it defines. The names shorten the chapter titles.
CHAPTERS = {
    "geometry": "Geometry",
    "dust": "Dust",
    "radiometry": "Radiometry",
    "optics": "Optics",
    "inference": "Records",
}
# Where each locator sits: ((x, y), horizontal alignment) in parent units.
_CHIPS = {
    "geometry": (((0.3, 42.6), "left"),),
    "dust": (((7.0, 22.4), "left"), ((25.6, 21.2), "right")),
    "radiometry": (((55.6, 20.9), "right"),),
    "optics": (((54.0, 26.6), "center"),),
    "inference": (((34.6, 18.6), "center"),),
}

# Parent coordinates: 72 by 44 units, 0.1 inch per unit on the page.
_W, _H = 72.0, 44.0
_SYSTEM_BOX = (0.3, 23.8, 18.2, 14.4)  # x0, y0, width, height
_FRAME_BOX = (55.8, 3.0, 15.5, 15.5)
_ORBIT_BOX = (0.8, 3.0, 15.5, 15.5)
_RECORDS_BOX = (27.0, 4.4, 15.2, 12.0)
_INSTRUMENT_ORIGIN = (25.0, 22.5)
_INSTRUMENT_SIZE = (34.0, 17.0)
_APERTURE_OFFSET = (6.5, 8.8)
_RAY_DY = {"star": 2.4, "planet": 0.0, "exozodi": -2.4}


def _spread(cast):
    """Ray spacing at the aperture: wider on a slide, whose heads are larger."""
    return 1.3 if cast.layout.is_slide else 1.0


# Scene and synthetic data


@functools.cache
def scene():
    """The skyscapes system drawn in every panel."""
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.orbit import KeplerianOrbit
    from skyscapes import System
    from skyscapes.disk import GraterDisk
    from skyscapes.physical_model import LambertianPhysicalModel
    from skyscapes.scene import FlatStar, Planet

    star = FlatStar(Ms_kg=Msun2kg, dist_pc=SCENE["dist_pc"], flux_phot_per_nm_m2=1e9)
    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        Mp_Mearth=jnp.array([1.0]),
        orbit=_orbit(KeplerianOrbit, _elements()),
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    n_fine = FRAME["n_pix"] * FRAME["oversample"]
    disk = GraterDisk(
        sma_AU=jnp.array(3.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-3.0),
        ksi0_AU=jnp.array(0.1),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(SCENE["disk_r_in_AU"]),
        rmax_AU=jnp.array(SCENE["disk_r_out_AU"]),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.4, 0.4]),
        Ag_grid=jnp.array([0.3, 0.3]),
        nx=n_fine,
        ny=n_fine,
        pixel_scale_arcsec=FRAME["pixel_arcsec"] / FRAME["oversample"],
        dist_pc=SCENE["dist_pc"],
        n_slices_los=41,
    )
    return System(
        star=star,
        planets=(planet,),
        trig_solver=get_grid_solver(level="scalar", E=False, trig=True, jit=True),
        disk=disk,
        midplane_inc_deg=SCENE["disk_incl_deg"],
        midplane_pa_deg=SCENE["disk_pa_deg"],
    )


def _elements():
    """The true orbital elements as length-one arrays."""
    return {
        "a_AU": np.array([SCENE["a_AU"]]),
        "e": np.array([SCENE["e"]]),
        "W_rad": np.array([math.radians(SCENE["W_deg"])]),
        "i_rad": np.array([math.radians(SCENE["i_deg"])]),
        "w_rad": np.array([math.radians(SCENE["w_deg"])]),
        "M0_rad": np.array([math.radians(SCENE["M0_deg"])]),
    }


def _orbit(cls, elements):
    n = len(elements["a_AU"])
    return cls(
        a_AU=jnp.asarray(elements["a_AU"]),
        e=jnp.asarray(elements["e"]),
        W_rad=jnp.asarray(elements["W_rad"]),
        i_rad=jnp.asarray(elements["i_rad"]),
        w_rad=jnp.asarray(elements["w_rad"]),
        M0_rad=jnp.asarray(elements["M0_rad"]),
        t0_d=jnp.zeros(n),
    )


def period_d():
    """Orbital period of the true orbit [d], from its mean motion."""
    from orbix.equations.orbit import mean_motion, period_n

    n = mean_motion(jnp.asarray(SCENE["a_AU"] * AU2m), jnp.asarray(G_si * Msun2kg))
    return float(period_n(n)) * s2d


def sky_positions_arcsec(t_jd):
    """True planet offsets (xi, eta) = (east, north) [arcsec], shape ``(T, 2)``.

    In the chapter profile: the library's second rotated component is east
    and its first is north (see ``PROFILE``).
    """
    system = scene()
    planet = system.planets[0]
    r_au = np.asarray(
        planet.propagate(
            system.trig_solver,
            jnp.atleast_1d(jnp.asarray(t_jd, float)),
            star=system.star,
        )[0]
    )[0]
    return np.stack(
        [
            au_to_arcsec(r_au[1], SCENE["dist_pc"]),
            au_to_arcsec(r_au[0], SCENE["dist_pc"]),
        ],
        axis=-1,
    )


def planet_xyz_au(t_jd):
    """True planet position (X, Y, Z) = (north, east, toward observer) [AU].

    The library's rotated vector is the chapter's (X, Y, Z) component for
    component; only its sky labels differ. Shape ``(3,)``.
    """
    system = scene()
    planet = system.planets[0]
    r_au = planet.propagate(
        system.trig_solver, jnp.atleast_1d(jnp.asarray(t_jd, float)), star=system.star
    )[0]
    return np.asarray(r_au)[0, :, 0]


def _pixel_gaussian(n_pix, center_rc, sigma):
    """Unit-sum Gaussian integrated over each pixel, centered at (row, col)."""
    edges = np.arange(n_pix + 1) - 0.5
    root2 = math.sqrt(2.0) * sigma

    def one_d(c):
        cdf = np.array([0.5 * (1.0 + math.erf((x - c) / root2)) for x in edges])
        return np.diff(cdf)

    return np.outer(one_d(center_rc[0]), one_d(center_rc[1]))


def sky_to_display_pixel(xy_arcsec):
    """Display (row, col) of a sky offset in the frame: east left, north up.

    Row increases with Dec (origin at the bottom); column increases toward
    the west, so an eastward (positive RA) offset lands left of center.
    """
    center = 0.5 * (FRAME["n_pix"] - 1)
    scale = FRAME["pixel_arcsec"]
    return center + xy_arcsec[1] / scale, center - xy_arcsec[0] / scale


@functools.cache
def disk_image():
    """Model disk surface brightness on the sky, chapter profile.

    Indexed ``[north, east]``, both increasing with the index (row 0 is the
    southernmost). skyscapes indexes its image by (second, first) rotated
    component, which is (east, north) in the chapter profile, so its array
    is transposed here (see ``PROFILE``).
    """
    system = scene()
    fine = np.asarray(
        system.disk.surface_brightness(
            jnp.asarray(SCENE["wavelength_nm"]),
            jnp.asarray(0.0),
            jnp.asarray(SCENE["disk_incl_deg"]),
            jnp.asarray(SCENE["disk_pa_deg"]),
        ),
        dtype=float,
    )
    return fine.T


@functools.cache
def frame_components():
    """Expected counts per pixel of each contribution, display orientation.

    Returns:
        A dict of ``(n_pix, n_pix)`` arrays: ``local_zodi``, ``exozodi``,
        ``planet`` and ``leakage``.
    """
    n = FRAME["n_pix"]
    over = FRAME["oversample"]
    # The frame is drawn east to the left, so flip the columns once, here.
    binned = disk_image().reshape(n, over, n, over).sum(axis=(1, 3))[:, ::-1]
    exozodi = FRAME["exozodi_peak"] * binned / binned.max()
    local = np.full((n, n), FRAME["local_zodi"])
    planet_rc = sky_to_display_pixel(sky_positions_arcsec(T_EPOCH_JD)[0])
    planet = FRAME["planet_total"] * _pixel_gaussian(
        n, planet_rc, FRAME["psf_sigma_pix"]
    )
    center = 0.5 * (n - 1)
    leakage = FRAME["leak_core"] * _pixel_gaussian(
        n, (center, center), FRAME["leak_core_pix"]
    )
    rng = np.random.default_rng(FRAME["seed"])
    for _ in range(FRAME["n_speckles"]):
        radius = FRAME["speckle_radius_pix"] * math.sqrt(rng.uniform(0.3, 1.0))
        angle = rng.uniform(0.0, 2.0 * math.pi)
        rc = (center + radius * math.sin(angle), center + radius * math.cos(angle))
        leakage = leakage + FRAME["speckle_amp"] * rng.uniform(
            0.5, 1.0
        ) * _pixel_gaussian(n, rc, 0.8) * (2.0 * math.pi * 0.64)
    return {
        "local_zodi": local,
        "exozodi": exozodi,
        "planet": planet,
        "leakage": leakage,
    }


@functools.cache
def raw_frame():
    """One Poisson realization of the summed expected counts."""
    expected = sum(frame_components().values())
    rng = np.random.default_rng(FRAME["seed"] + 1)
    return rng.poisson(expected).astype(float)


def _sky_tracks_arcsec(elements, t_jd):
    """Offsets (xi, eta) [arcsec] of a batch of orbits, shape ``(K, T, 2)``."""
    from orbix.orbit import KeplerianOrbit
    from skyscapes.physical_model import LambertianPhysicalModel
    from skyscapes.scene import Planet

    k = len(elements["a_AU"])
    system = scene()
    batch = Planet(
        Rp_Rearth=jnp.ones(k),
        Mp_Mearth=jnp.ones(k),
        orbit=_orbit(KeplerianOrbit, elements),
        physical_model=LambertianPhysicalModel(Ag=jnp.full(k, 0.3)),
    )
    r_au = np.asarray(
        batch.propagate(system.trig_solver, jnp.asarray(t_jd, float), star=system.star)[
            0
        ]
    )
    return np.stack(
        [
            au_to_arcsec(r_au[:, 1, :], SCENE["dist_pc"]),
            au_to_arcsec(r_au[:, 0, :], SCENE["dist_pc"]),
        ],
        axis=-1,
    )


def _prior_chunk(rng, n):
    """``n`` samples of the prior on the elements (see ``DRAWS``)."""
    base = _elements()
    two_pi = 2.0 * math.pi
    return {
        "a_AU": base["a_AU"] * (1.0 + DRAWS["prior_a_frac"] * rng.standard_normal(n)),
        "e": rng.uniform(0.0, DRAWS["prior_e_max"], n),
        "W_rad": base["W_rad"]
        + math.radians(DRAWS["prior_W_deg"]) * rng.standard_normal(n),
        "i_rad": base["i_rad"]
        + math.radians(DRAWS["prior_i_deg"]) * rng.standard_normal(n),
        "w_rad": rng.uniform(0.0, two_pi, n),
        "M0_rad": rng.uniform(0.0, two_pi, n),
    }


@functools.cache
def posterior_draws():
    """Posterior draws of the elements given the four measured positions.

    Importance resampling: prior samples weighted by the Gaussian likelihood
    of the measured positions, then systematic resampling with replacement
    (see ``DRAWS``). Deterministic for the seed.

    Returns:
        ``(elements, measured, sigma, ess)``: a dict of ``(K,)`` element
        arrays, the measured positions ``(P, 2)`` [arcsec], the per-axis
        uncertainty [arcsec] and the effective sample size of the weights.
    """
    rng = np.random.default_rng(DRAWS["seed"])
    sigma = DRAWS["astrometry_sigma_arcsec"]
    truth = sky_positions_arcsec(np.array(T_VISITS_JD))
    measured = truth + sigma * rng.standard_normal(truth.shape)
    chunks, chi2 = [], []
    for start in range(0, DRAWS["n_prior"], DRAWS["chunk"]):
        size = min(DRAWS["chunk"], DRAWS["n_prior"] - start)
        prior = _prior_chunk(rng, size)
        predicted = _sky_tracks_arcsec(prior, np.array(T_VISITS_JD))
        chi2.append(np.sum(((predicted - measured[None]) / sigma) ** 2, axis=(1, 2)))
        chunks.append(prior)
    chi2 = np.concatenate(chi2)
    weights = np.exp(-0.5 * (chi2 - chi2.min()))
    weights /= weights.sum()
    ess = 1.0 / float(np.sum(weights**2))
    k = DRAWS["n"]
    positions = (rng.uniform() + np.arange(k)) / k
    pick = np.minimum(np.searchsorted(np.cumsum(weights), positions), len(weights) - 1)
    elements = {
        key: np.concatenate([chunk[key] for chunk in chunks])[pick] for key in chunks[0]
    }
    return elements, measured, sigma, ess


@functools.cache
def orbit_draws():
    """Posterior orbit tracks and the measured positions they are drawn from.

    Returns:
        ``(tracks, measured, sigma)``: tracks ``(K, T, 2)`` [arcsec] over one
        period, measured ``(P, 2)`` [arcsec], and the per-axis uncertainty.
    """
    elements, measured, sigma, _ = posterior_draws()
    t = np.linspace(0.0, 1.02 * period_d(), 200)
    return _sky_tracks_arcsec(elements, t), measured, sigma


# Small drawing helpers


def _inset(ax, box):
    """An inset axes at ``box`` = (x0, y0, width, height) in parent data."""
    inset = ax.inset_axes(box, transform=ax.transData, zorder=5)
    inset.patch.set_visible(False)
    return inset


def _fit_limits(ax, box, *, pad=0.0):
    """Widen an equal-scale inset's limits so its aspect matches ``box``.

    Keeps the data centered and the scale equal in both directions, and
    switches the inset to automatic aspect, so points in it map to the
    parent with one fixed affine transform.
    """
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    flip_x = ax.get_xlim()[0] > ax.get_xlim()[1]
    w, h = (x1 - x0) * (1 + pad), (y1 - y0) * (1 + pad)
    target = box[2] / box[3]
    if w / h < target:
        w = h * target
    else:
        h = w / target
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    xl = (cx + 0.5 * w, cx - 0.5 * w) if flip_x else (cx - 0.5 * w, cx + 0.5 * w)
    ax.set_aspect("auto")
    ax.set_xlim(*xl)
    ax.set_ylim(cy - 0.5 * h, cy + 0.5 * h)


def _backing(cast):
    """A plain backing box in the background color, for a label over marks.

    Labels here carry no stroked halo (a halo turns every glyph into an
    outline path and bloats the vector exports); a label that sits over an
    image, a hatch or a line gets this box instead.
    """
    return {
        "boxstyle": "round,pad=0.15,rounding_size=0.25",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _label(ax, xy, text, color, cast, *, size=None, box=False, **kw):
    """A direct label in an entity's color, optionally on a backing box."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    if box:
        kw["bbox"] = _backing(cast)
    return ax.text(
        *xy,
        text,
        color=color,
        fontsize=size or cast.layout.small_pt,
        zorder=9,
        **kw,
    )


def _title(ax, xy, text, cast, **kw):
    """A panel title in the text color, plain weight."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "bottom")
    return ax.text(
        *xy, text, color=cast.text, fontsize=cast.layout.font_pt, zorder=9, **kw
    )


def _plain_text(fig):
    """Remove the stroked halos the shared helpers put on their labels."""
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
    return fig


# Panels shared by the map and the relay


def panel_system(ax, cast, *, box=None, rays="arrow"):
    """The system side on: star, planet, orbit, exozodiacal disk, rays out.

    Drawn with ``skyscapes.viz.plot_system(view="side")``: the line of sight
    runs left to right with the observer to the right (+z), and the vertical
    axis is the sky offset along the disk's projected minor axis. Axes are
    hidden, the view is cropped to the disk, and the library's own labels
    are replaced by the shared cast's direct labels.

    Args:
        ax: Axes to draw into.
        cast: The active cast.
        box: The parent box of an inset, to match its aspect; None keeps
            equal aspect.
        rays: ``"arrow"`` draws the outgoing rays with heads at the right
            edge; ``"edge"`` stops them at the edge without heads, for a map
            that continues them.

    Returns:
        ``(result, ends)``: the skyscapes ``PlotResult`` and the right-edge
        point of each outgoing ray, keyed ``star``, ``planet``, ``exozodi``.
    """
    from skyscapes import viz

    system = scene()
    small = cast.layout.small_pt
    track = jnp.linspace(0.0, period_d(), 200)
    result = viz.plot_system(
        system,
        T_EPOCH_JD,
        view="side",
        ax=ax,
        track_t_jd=track,
        planet_labels=["planet"],
        labels=True,
    )
    ax.axis("off")
    gids = {}
    for group in result.artists.values():
        for artist in group if isinstance(group, list) else [group]:
            gids[artist.get_gid()] = artist
    # The library's labels and observer arrow give way to the cast's labels.
    for gid, artist in gids.items():
        if gid.startswith("label/") or gid == "observer":
            artist.set_visible(False)
    gids["star"].set_sizes([(1.3 * cast.layout.marker_pt) ** 2])
    gids["star"].set_zorder(6)
    gids["planet/planet"].set_sizes([(0.85 * cast.layout.marker_pt) ** 2])
    gids["planet/planet"].set_edgecolor(cast.background)
    gids["planet/planet"].set_zorder(6)
    for name in ("disk/near", "disk/far"):
        gids[name].set_linewidth(1.2 * cast.layout.lw)
    if "track/planet" in gids:
        gids["track/planet"].set_visible(False)
    # The dust layer: a hatched band around each midplane half, so the disk
    # reads as a region of dust rather than a line.
    thick = 0.42
    for name in ("disk/near", "disk/far"):
        (h0, h1), (v0, v1) = gids[name].get_data()
        d = np.array([h1 - h0, v1 - v0])
        n = np.array([-d[1], d[0]]) / np.hypot(*d) * thick
        corners = [
            (h0 + n[0], v0 + n[1]),
            (h1 + n[0], v1 + n[1]),
            (h1 - n[0], v1 - n[1]),
            (h0 - n[0], v0 - n[1]),
        ]
        ex.region(ax, "exozodi", Polygon(corners, closed=True), cast).set_gid(
            f"disk/band/{name[5:]}"
        )
    (h_far0, h_far1), (v_far0, v_far1) = gids["disk/far"].get_data()
    (_, h_near1), (_, v_near1) = gids["disk/near"].get_data()
    # Crop to the disk: the far end sets the left edge, the near end the top.
    half_h = 1.08 * max(abs(h_far1), abs(h_near1))
    half_v = 1.35 * max(abs(v_far1), abs(v_near1))
    ax.set_xlim(-half_h, half_h)
    ax.set_ylim(-half_v, half_v)
    ax.set_aspect("equal")
    plane = gids.get("sky_plane")
    if plane is not None:
        plane.set_data([0.0, 0.0], [-half_v, half_v])

    def text(xy, label, color, offset, ha, va, gid, style=None):
        artist = ax.annotate(
            label,
            xy,
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=color,
            fontsize=small,
            fontstyle=style,
            linespacing=0.95,
            zorder=8,
        )
        artist.set_gid(gid)
        return artist

    planet_h, planet_v = (float(v) for v in gids["planet/planet"].get_offsets()[0])
    text(
        (0.0, 0.0),
        "star",
        cast["star"].color,
        (0, 0.7 * cast.layout.marker_pt),
        "center",
        "bottom",
        "note/star",
    )
    text(
        (planet_h, planet_v),
        "planet",
        cast["planet"].color,
        (0.6 * cast.layout.marker_pt, -0.35 * cast.layout.marker_pt),
        "left",
        "top",
        "note/planet",
    ).set_bbox(_backing(cast))
    text(
        (h_far1, v_far1),
        "exozodiacal\ndust",
        cast["exozodi"].color,
        (-0.4 * cast.layout.marker_pt, -1.5 * cast.layout.marker_pt),
        "center",
        "top",
        "disk/label",
    )
    exo_label = ax.findobj(lambda a: a.get_gid() == "disk/label")[0]
    exo_label.set_bbox(_backing(cast))
    exo_label.set_zorder(12)
    text(
        (0.0, half_v),
        "sky plane",
        cast["scenery"].color,
        (-3, -2),
        "right",
        "top",
        "note/sky-plane",
        "italic",
    )
    if box is not None:
        _fit_limits(ax, box)
    right = max(ax.get_xlim())
    ends = {}
    if rays:
        # Starlight illuminating the planet, then the three contributions
        # leaving toward the observer (+z, to the right).
        # Starlight reaching the planet: a short ray, so its head is drawn
        # at a fixed small size rather than the layout's ray head.
        ax.annotate(
            "",
            (planet_h + 0.3, planet_v + 0.17),
            (-0.3, -0.17),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.5,head_width=0.25",
                "mutation_scale": 1.2 * cast.layout.marker_pt,
                "color": cast["star"].color,
                "lw": cast["ray"].lw,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=5,
        ).set_gid("ray/star-planet")
        grain = (0.85 * h_far1 + 0.15 * h_far0, 0.85 * v_far1 + 0.15 * v_far0)
        # Ray labels sit below the shaft and clear of the arrowhead, whose
        # length grows with the square of the layout's marker size.
        mk = cast.layout.marker_pt
        back = 0.13 * mk * mk + 3.0 if rays == "arrow" else 2.0
        starts = {
            "star": (0.5, 0.0),
            "planet": (planet_h + 0.35, planet_v),
            "exozodi": (grain[0] + 0.3, grain[1]),
        }
        for key, (h, v) in starts.items():
            gid = "disk/ray" if key == "exozodi" else f"ray/{key}"
            if rays == "edge":
                (line,) = ax.plot(
                    [h, right],
                    [v, v],
                    color=cast[key].color,
                    lw=cast["ray"].lw,
                    zorder=4,
                )
                line.set_gid(gid)
            else:
                ex.arrow(ax, (h, v), (right, v), "ray", cast, source=key)[0].set_gid(
                    gid
                )
            ends[key] = (right, v)
        text(
            ends["star"],
            "starlight",
            cast["star"].color,
            (-back, -2),
            "right",
            "top",
            "note/starlight",
        )
        text(
            ends["planet"],
            "reflected",
            cast["planet"].color,
            (-back, -2),
            "right",
            "top",
            "note/reflected",
        )
        text(
            ends["exozodi"],
            "scattered",
            cast["exozodi"].color,
            (-back, -2),
            "right",
            "top",
            "disk/scattered",
        )
    return result, ends


def panel_sky(ax, cast):
    """The model sky before any optics: disk brightness, star, planet."""
    from skyscapes import viz

    # The chapter-profile image, passed as an array: skyscapes draws column
    # index increasing to the left, which is east here.
    result = viz.plot_disk_image(
        disk_image(),
        pixel_scale_arcsec=FRAME["pixel_arcsec"] / FRAME["oversample"],
        outline=False,
        colorbar=False,
        ax=ax,
    )
    result.artists["image"].set_cmap(ex.image_cmap("intensity"))
    xy = sky_positions_arcsec(T_EPOCH_JD)[0]
    ax.plot(
        [xy[0]],
        [xy[1]],
        zorder=6,
        mec=cast.background,
        mew=0.8,
        **cast["planet"].marker_kw(cast.layout.marker_pt),
    )
    _label(
        ax,
        (xy[0], xy[1] - 0.06),
        "planet",
        cast["planet"].color,
        cast,
        va="top",
        box=True,
    )
    for artist in result.artists.get("scatter", []):
        if artist.get_gid() == "star":
            artist.set_sizes([(1.4 * cast.layout.marker_pt) ** 2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    _compass(ax, cast)
    return result


def _compass(ax, cast, *, corner="upper left"):
    """East and north arrows in one corner of a sky panel.

    Every sky panel here is displayed east to the left and north up, so the
    east arrow points left and the north arrow up on the page.
    """
    fx = 0.2 if corner.endswith("left") else 0.9
    fy = 0.70 if corner.startswith("upper") else 0.10
    length = 0.13
    color = cast["annotation"].color
    for end, text, ha, va, off in (
        ((fx - length, fy), "E", "right", "center", (-2, 0)),
        ((fx, fy + length), "N", "center", "bottom", (0, 2)),
    ):
        ax.annotate(
            "",
            end,
            (fx, fy),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 0.8 * cast.layout.lw,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=8,
        )
        ax.annotate(
            text,
            end,
            xycoords="axes fraction",
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=color,
            fontsize=cast.layout.small_pt,
            zorder=8,
            bbox=_backing(cast),
        )


def panel_frame(ax, cast, *, labels=True):
    """The synthetic raw frame: readouts colormap, raw pixels, log norm.

    The norm tops out at ``FRAME["display_max"]``, so the stellar leakage
    core saturates and the fainter dust stays visible; the amplitudes are
    schematic either way.
    """
    frame = raw_frame()
    n = FRAME["n_pix"]
    floor = 0.6 * FRAME["local_zodi"]
    image = ax.imshow(
        np.clip(frame, floor, None),
        cmap=ex.image_cmap("readouts"),
        norm=LogNorm(vmin=floor, vmax=FRAME["display_max"], clip=True),
        interpolation="nearest",
        origin="lower",
        extent=(-0.5, n - 0.5, -0.5, n - 0.5),
        zorder=1,
    )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(cast["detector"].color)
        spine.set_linewidth(1.2 * cast.layout.lw)
    if not labels:
        return image
    c = 0.5 * (n - 1)
    row, col = sky_to_display_pixel(sky_positions_arcsec(T_EPOCH_JD)[0])
    _label(
        ax, (col, row - 1.6), "planet", cast["planet"].color, cast, va="top", box=True
    )
    _label(
        ax,
        (c - 2.4, c),
        "stellar\nleakage",
        cast["star"].color,
        cast,
        ha="right",
        linespacing=0.95,
        box=True,
    )
    ez = frame_components()["exozodi"]
    # Label the dust on its brightest pixel well away from the star.
    rr, cc = np.indices(ez.shape)
    keep = (np.hypot(rr - c, cc - c) > 6.0) & (np.hypot(rr - row, cc - col) > 8.0)
    r_ez, c_ez = np.unravel_index(np.argmax(np.where(keep, ez, -1.0)), ez.shape)
    _label(
        ax,
        (c_ez, r_ez + 1.2),
        "exozodi",
        cast["exozodi"].color,
        cast,
        va="bottom",
        box=True,
    )
    _label(
        ax,
        (n - 0.9, n - 0.9),
        "local zodi (uniform floor)",
        cast["local_zodi"].color,
        cast,
        ha="right",
        va="top",
        box=True,
    )
    return image


def panel_orbit(ax, cast, *, labels=True, compass="lower left"):
    """Posterior tracks (orbix.viz) given the measured positions.

    Offsets are in the chapter profile, drawn east to the left and north up.
    The measured position of the visit shown in the raw frame is ringed.
    """
    from orbix.viz import plot_sky_track

    tracks, measured, sigma = orbit_draws()
    result = plot_sky_track(tracks, ax=ax, style=hwostyle.roles.model)
    # The renderer's neutral central dot becomes the shared star glyph.
    result.parts["star"].remove()
    ex.mark(ax, "star", (0.0, 0.0), cast, scale=0.8)
    for k in range(len(tracks)):
        line = result.parts[f"path/{k}"]
        line.set_linewidth(0.45 * cast.layout.lw)
        line.set_alpha(0.35)
    ax.errorbar(
        measured[:, 0],
        measured[:, 1],
        xerr=sigma,
        yerr=sigma,
        fmt="o",
        ms=0.6 * cast.layout.marker_pt,
        color=cast["planet"].color,
        mec=cast.background,
        mew=0.6,
        elinewidth=0.9 * cast.layout.lw,
        capsize=0,
        zorder=7,
    )
    current = measured[T_VISITS_JD.index(T_EPOCH_JD)]
    ax.plot(
        [current[0]],
        [current[1]],
        marker="o",
        ms=1.9 * cast.layout.marker_pt,
        mfc="none",
        mec=cast.text,
        mew=0.9 * cast.layout.lw,
        ls="none",
        zorder=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # A square window centered on the draws, widened toward the measured
    # (west, right-hand) side so the labels there fit.
    xs, ys = tracks[..., 0], tracks[..., 1]
    half = 0.66 * max(np.ptp(xs), np.ptp(ys))
    cx = 0.5 * (xs.min() + xs.max()) - 0.12 * half
    cy = 0.5 * (ys.min() + ys.max()) + 0.06 * half
    ax.set_xlim(cx + half, cx - half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color(cast["scenery"].color)
        spine.set_linewidth(0.6 * cast.layout.lw)
    if compass is not None:
        _compass(ax, cast, corner=compass)
    if labels:
        small = cast.layout.small_pt
        ring = 1.0 * cast.layout.marker_pt + 2.0
        # Outside the panel the lowest position has a clear run to its label;
        # inside, the highest has clear space beside it.
        pick = np.argmin if labels == "outside" else np.argmax
        other = measured[int(pick(measured[:, 1]))]
        for xy, text, color in (
            (current, "current visit", cast.text),
            (other, "measured", cast["planet"].color),
        ):
            if labels == "outside":
                # Labels in the free space right of the panel, with leaders.
                frac_y = (xy[1] - (cy - half)) / (2.0 * half)
                ax.annotate(
                    text,
                    xy,
                    xytext=(1.06, frac_y),
                    textcoords="axes fraction",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=small,
                    zorder=9,
                    annotation_clip=False,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": color,
                        "lw": 0.6 * cast.layout.lw,
                        "shrinkA": 1.0,
                        "shrinkB": ring,
                    },
                )
            else:
                ax.annotate(
                    text,
                    xy,
                    xytext=(ring, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=small,
                    zorder=9,
                    bbox=_backing(cast),
                )
        ax.text(
            0.5 * (xs.min() + xs.max()),
            ys.max() + 0.04 * half,
            "posterior draws",
            color=hwostyle.roles.model,
            fontsize=small,
            ha="center",
            va="bottom",
            zorder=9,
            bbox=_backing(cast),
        )
    return tracks, measured


def _aperture_xy(origin):
    return (origin[0] + _APERTURE_OFFSET[0], origin[1] + _APERTURE_OFFSET[1])


def draw_local_zodi(ax, origin, cast):
    """The local zodiacal cloud around the observer, with its light."""
    ox, oy = origin
    ax_, ay_ = _aperture_xy(origin)
    cloud = ex.region(ax, "local_zodi", Ellipse((ox + 7.8, ay_), 14.2, 16.4), cast)
    if cast.mode == "dark":
        # Full-strength green dots over a large cloud overpower the dark
        # page; the hatch keeps its hue at a lower value.
        cloud.set_hatchcolor(_blend(cast["local_zodi"].color, 0.45, cast.background))
    if cast.layout.is_slide:
        # The slide cloud is larger on the page than the hatch spacing.
        cloud.set_hatch(".")
    _label(
        ax, (ox + 7.8, oy - 1.3), "local zodiacal dust", cast["local_zodi"].color, cast
    )
    # The local dust is a foreground along the same line of sight, so its
    # light arrives nearly parallel to the target's rays.
    for sign in (1.0, -1.0):
        y = ay_ + sign * 3.55 * _spread(cast)
        ex.arrow(
            ax,
            (ox + 1.4, y + sign * 0.35),
            (ax_ - 0.3, y),
            "ray",
            cast,
            source="local_zodi",
        )


def draw_aperture(ax, origin, cast):
    """The generic telescope aperture, edge on, facing the incoming light."""
    xy = _aperture_xy(origin)
    ex.aperture(ax, xy, 7.4 * _spread(cast), cast)
    _label(
        ax,
        (xy[0], xy[1] + 5.3),
        "telescope\naperture",
        cast["aperture"].color,
        cast,
        va="bottom",
        linespacing=0.95,
        box=True,
    )
    return xy


def draw_rail(ax, origin, cast):
    """The generic coronagraph (eyepiece rail) ending on the detector.

    Returns:
        The detector glyph's position in the axes' data coordinates.
    """
    ox, oy = origin
    aperture = _aperture_xy(origin)
    rail_box = (ox + 10.4, oy + 1.8, 23.4, 14.6)
    rail = _inset(ax, rail_box)
    planes = [
        ("pupil", "pupil"),
        ("focal\nmask", "fpm"),
        ("Lyot\nstop", "lyot"),
        ("detector", "detector"),
    ]
    positions = (0.12, 0.38, 0.64, 0.92)
    result = ep.rail(planes, ax=rail, positions=positions)
    # The beam is shaded from the pupil on; before it (and from the aperture
    # to the rail) only its edges are drawn.
    clip = Rectangle(
        (positions[0], -1e3), 1e3, 2e3, transform=rail.transData, visible=False
    )
    rail.add_patch(clip)
    result.artists["fill"].set_clip_path(clip)
    for text in result.artists["text"]:
        text.set_fontsize(cast.layout.small_pt)
        text.set_linespacing(0.95)
        text.set_bbox(_backing(cast))
    # Beam edges from the aperture into the rail's entrance.
    y0 = rail_box[1] + 0.52 * rail_box[3]
    half = 0.20 * rail_box[3]
    for sign in (-1.0, 1.0):
        ax.plot(
            [aperture[0] + 0.2, rail_box[0] + 0.02 * rail_box[2]],
            [aperture[1] + sign * 3.7 * _spread(cast), y0 + sign * half],
            color=cast["scenery"].color,
            lw=0.7 * cast.layout.lw,
            zorder=3,
        )
    _label(
        ax,
        (rail_box[0] + 0.47 * rail_box[2], oy + 1.0),
        "generic coronagraph,\nnot a flight design",
        cast["annotation"].color,
        cast,
        fontstyle="italic",
        linespacing=0.95,
    )
    return (rail_box[0] + 0.92 * rail_box[2], y0)


# Dimming for the highlighted variants


def _blend(color, f, bg):
    r, g, b, a = to_rgba(color)
    br, bgc, bb, _ = to_rgba(bg)
    return (br + f * (r - br), bgc + f * (g - bgc), bb + f * (b - bb), a)


def dim(artist, f, bg):
    """Fade an artist (and everything inside an axes) toward the background."""
    if hasattr(artist, "child_axes") and hasattr(artist, "get_children"):
        for child in artist.get_children():
            if child is not getattr(artist, "patch", None):
                dim(child, f, bg)
        return
    if isinstance(artist, Text):
        artist.set_color(_blend(artist.get_color(), f, bg))
        box = artist.get_bbox_patch()
        if box is not None:
            box.set_edgecolor(_blend(box.get_edgecolor(), f, bg))
        arrow_patch = getattr(artist, "arrow_patch", None)
        if arrow_patch is not None:
            dim(arrow_patch, f, bg)
    elif isinstance(artist, Line2D):
        artist.set_color(_blend(artist.get_color(), f, bg))
        for getter, setter in (
            ("get_markerfacecolor", "set_markerfacecolor"),
            ("get_markeredgecolor", "set_markeredgecolor"),
        ):
            value = getattr(artist, getter)()
            if value not in ("none", "None"):
                getattr(artist, setter)(_blend(value, f, bg))
    elif isinstance(artist, Patch):
        face = artist.get_facecolor()
        if face[3] > 0:
            artist.set_facecolor(_blend(face, f, bg))
        artist.set_edgecolor(_blend(artist.get_edgecolor(), f, bg))
        if artist.get_hatch():
            artist.set_hatchcolor(_blend(artist.get_hatchcolor(), f, bg))
    elif isinstance(artist, Collection):
        faces = artist.get_facecolors()
        if len(faces):
            artist.set_facecolors([_blend(c, f, bg) for c in faces])
        edges = artist.get_edgecolors()
        if len(edges):
            artist.set_edgecolors([_blend(c, f, bg) for c in edges])
    elif isinstance(artist, AxesImage):
        artist.set_alpha(f)


# The map


def build_overview(layout, cast, highlight=None, tag_text=None):
    """The full observation map, optionally with one chapter's region lit.

    Args:
        layout: ``ex.DOC`` or ``ex.SLIDE``.
        cast: The active cast.
        highlight: None for the plain map, or a key of ``HIGHLIGHTS``.
        tag_text: The tag written on the highlighted chapter's locator; None
            writes "this tutorial" for the tutorial region and "this
            chapter" otherwise.

    Returns:
        The figure.

    Raises:
        ValueError: If ``highlight`` is not a key of ``HIGHLIGHTS``.
    """
    if highlight is not None and highlight not in HIGHLIGHTS:
        msg = f"unknown highlight {highlight!r}; known: {sorted(HIGHLIGHTS)}"
        raise ValueError(msg)
    if tag_text is None:
        tag_text = "this tutorial" if highlight == "tutorial" else "this chapter"
    fig, ax = ex.figure(layout, doc_height_in=4.6)
    ax.set_xlim(0, _W)
    ax.set_ylim(0, _H)
    ax.set_aspect("equal")
    ax.axis("off")
    tagger = _Tagger(ax)

    with tagger("system"):
        sys_ax = _inset(ax, _SYSTEM_BOX)
        _, ends = panel_system(sys_ax, cast, box=_SYSTEM_BOX, rays="edge")
        top = _SYSTEM_BOX[1] + _SYSTEM_BOX[3]
        _title(
            ax,
            (_SYSTEM_BOX[0] + 0.5 * _SYSTEM_BOX[2], top + 1.6),
            "target system, side view",
            cast,
        )
        _label(
            ax,
            (_SYSTEM_BOX[0] + 0.5 * _SYSTEM_BOX[2], top + 0.2),
            "observer to the right",
            cast["annotation"].color,
            cast,
            va="bottom",
            fontstyle="italic",
        )

    aperture = _aperture_xy(_INSTRUMENT_ORIGIN)
    for key in ("star", "planet", "exozodi"):
        with tagger("path", "disk") if key == "exozodi" else tagger("path"):
            start = _inset_to_parent(sys_ax, _SYSTEM_BOX, ends[key])
            _path_ray(
                ax,
                key,
                start,
                (aperture[0] - 0.3, aperture[1] + _RAY_DY[key] * _spread(cast)),
                cast,
            )
    with tagger("path"):
        _label(
            ax,
            (22.6, 24.4),
            "interstellar\ndistance,\nnot to scale",
            cast["annotation"].color,
            cast,
            fontstyle="italic",
            linespacing=0.95,
        )

    with tagger("local_zodi"):
        draw_local_zodi(ax, _INSTRUMENT_ORIGIN, cast)
    with tagger("aperture"):
        draw_aperture(ax, _INSTRUMENT_ORIGIN, cast)
    with tagger("instrument"):
        detector = draw_rail(ax, _INSTRUMENT_ORIGIN, cast)

    fx, fy, fw, fh = _FRAME_BOX
    with tagger("frame"):
        frame_ax = _inset(ax, _FRAME_BOX)
        panel_frame(frame_ax, cast)
        _title(ax, (fx + 0.5 * fw, 0.4), "raw frame, synthetic", cast)

    rx, ry, rw, rh = _RECORDS_BOX
    ox, _, ow, _ = _ORBIT_BOX
    with tagger("data", "readout"):
        ex.arrow(
            ax,
            (detector[0], _INSTRUMENT_ORIGIN[1] + 2.4),
            (detector[0], fy + fh + 0.4),
            "data",
            cast,
        )
        _label(
            ax,
            (detector[0] + (1.8 if layout.is_slide else 1.0), 20.9),
            "read out",
            cast["data"].color,
            cast,
            ha="left",
        )
    with tagger("data"):
        mid = ry + 0.5 * rh
        ex.arrow(ax, (fx - 0.6, mid), (rx + rw + 0.8, mid), "data", cast)
        _label(
            ax,
            (0.5 * (fx + rx + rw), mid + 1.6),
            "reduce",
            cast["data"].color,
            cast,
            va="bottom",
        )
        ex.arrow(ax, (rx - 0.8, mid), (ox + ow + 0.6, mid), "data", cast)
        _label(
            ax,
            (0.5 * (rx + ox + ow), mid + 1.6),
            "infer with\nan assumed\nmodel",
            cast["data"].color,
            cast,
            va="bottom",
            linespacing=0.95,
        )

    with tagger("records"):
        _records_card(ax, (rx, ry), (rw, rh), cast)
        _title(ax, (rx + 0.5 * rw, 0.4), "records, one per visit", cast)

    with tagger("orbit"):
        orbit_ax = _inset(ax, _ORBIT_BOX)
        panel_orbit(orbit_ax, cast, labels="outside")
        _title(ax, (ox + 0.5 * ow, 0.4), "inferred orbit, sky", cast)

    for key, spots in _CHIPS.items():
        lit = key == highlight
        with tagger(f"chip:{key}"):
            for xy, ha in spots:
                _chip(ax, xy, CHAPTERS[key], cast, ha=ha, strong=lit)

    ex.badge(
        ax, cast, "schematic of the simulation scope, not to scale", loc="upper right"
    )
    _key(ax, cast, highlight, tag_text)
    if highlight is not None:
        _apply_highlight(cast, tagger, highlight, sys_ax)
    return _plain_text(fig)


def _inset_to_parent(inset, box, xy):
    """Map a point from an automatic-aspect inset to parent data units."""
    x0, x1 = inset.get_xlim()
    y0, y1 = inset.get_ylim()
    return (
        box[0] + (xy[0] - x0) / (x1 - x0) * box[2],
        box[1] + (xy[1] - y0) / (y1 - y0) * box[3],
    )


def _path_ray(ax, key, start, end, cast, *, break_x=21.4):
    """One contribution's ray across the interstellar gap, with a break."""
    ex.arrow(ax, start, end, "ray", cast, source=key)
    frac = (break_x - start[0]) / (end[0] - start[0])
    y = start[1] + frac * (end[1] - start[1])
    along = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    ex.scale_break(ax, (break_x, y), cast, along_deg=along)


def _records_card(ax, corner, size, cast):
    """The acquired records: what each visit reports, as symbols."""
    x0, y0 = corner
    w, h = size
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=0.8",
            facecolor=to_rgba(cast["data"].color, 0.08),
            edgecolor=cast["data"].color,
            lw=0.8 * cast.layout.lw,
            zorder=3,
        )
    )
    # The chapter's symbols: D is the reporting event, (xi, eta) the east and
    # north offsets, C their covariance; k counts the visits.
    rows = [
        (r"epoch $t_k$", cast.text),
        (r"event $D_k$", cast.text),
        (r"offsets $(\xi_k,\eta_k)$", cast["planet"].color),
        (r"covariance $C_k$", cast.text),
        ("calibration revision", cast.text),
    ]
    step = h / len(rows)
    for j, (text, color) in enumerate(rows):
        _label(ax, (x0 + 1.0, y0 + h - (j + 0.5) * step), text, color, cast, ha="left")


def _chip(ax, xy, text, cast, *, ha="center", strong=False):
    """A chapter locator: the chapter's short name in an outlined box.

    ``strong`` marks the locator of the chapter a highlighted map opens:
    bold text and a heavier outline in the text color.
    """
    return ax.text(
        *xy,
        text,
        color=cast.text,
        fontsize=cast.layout.small_pt,
        fontweight="bold" if strong else "normal",
        ha=ha,
        va="center",
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.4",
            "facecolor": cast.background,
            "edgecolor": cast.text if strong else cast["scenery"].color,
            "lw": (1.4 if strong else 0.7) * cast.layout.lw,
        },
    )


def _key(ax, cast, highlight=None, tag_text=None):
    """The key: light versus transformed data, and the chapter locators.

    A highlighted map adds a row of its own: a bold locator sample keyed as
    ``tag_text`` for a chapter, or, for the tutorial, a bold line saying the
    parts at full strength are the tutorial's.
    """
    x = 27.0
    step = 2.4 if cast.layout.is_slide else 1.45
    length = 4.6 if cast.layout.is_slide else 3.2
    top = 43.0 if cast.layout.is_slide else 43.3
    for j, (kind, text) in enumerate(
        (("ray", "light, propagating"), ("data", "data, transformed"))
    ):
        y = top - j * step
        ex.arrow(ax, (x, y), (x + length, y), kind, cast)
        _label(
            ax, (x + length + 0.6, y), text, cast["annotation"].color, cast, ha="left"
        )
    # The locator key sits under the status badge.
    rows = [(53.4, 40.3, "chapter that defines it", False)]
    if highlight == "tutorial":
        _label(
            ax,
            (53.4, 38.7),
            f"full strength: {tag_text}",
            cast.text,
            cast,
            ha="left",
            fontweight="bold",
        )
    elif highlight is not None:
        rows.append((60.4, 38.7, tag_text, True))
    for x, y, text, strong in rows:
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 0.6),
                2.4,
                1.2,
                boxstyle="round,pad=0.1,rounding_size=0.4",
                facecolor=cast.background,
                edgecolor=cast.text if strong else cast["scenery"].color,
                lw=(1.4 if strong else 0.7) * cast.layout.lw,
                zorder=10,
            )
        )
        _label(
            ax,
            (x + 3.1, y),
            text,
            cast.text if strong else cast["annotation"].color,
            cast,
            ha="left",
            fontweight="bold" if strong else "normal",
        )


class _Tagger:
    """Record which parent-axes artists belong to which map parts.

    ``with tagger("part", ...)`` tags every artist added to the parent axes
    (inset axes included) inside the block.
    """

    def __init__(self, ax):
        self.ax = ax
        self.tags = []
        self._parts = None
        self._before = None

    def __call__(self, *parts):
        self._parts = parts
        return self

    def __enter__(self):
        self._before = {id(c) for c in self.ax.get_children()}
        return self

    def __exit__(self, *exc):
        new = [c for c in self.ax.get_children() if id(c) not in self._before]
        self.tags.append((set(self._parts), new))
        return False


def _apply_highlight(cast, tagger, highlight, sys_ax):
    """Dim every part of the map outside the highlighted chapter's region."""
    keep = set(HIGHLIGHTS[highlight]) | {f"chip:{highlight}"}
    if highlight == "tutorial":
        keep |= {f"chip:{key}" for key in CHAPTERS if key != "inference"}
    bg = cast.background
    for parts, artists in tagger.tags:
        if parts & keep:
            continue
        for artist in artists:
            if artist is sys_ax and "disk" in keep:
                # Keep the dust inside the system panel; dim the rest.
                for child in sys_ax.get_children():
                    gid = child.get_gid() or ""
                    if not (gid.startswith("disk/") or gid == "ray/exozodi"):
                        dim(child, DIM, bg)
                continue
            dim(artist, DIM, bg)


# The relay: one scene followed into an instrument, an image and a record


def _instrument_axes(ax, cast, *, incoming=True):
    """The instrument group in its own axes, at the map's scale."""
    origin = (0.0, 0.0)
    ax.set_xlim(-2.0, _INSTRUMENT_SIZE[0])
    ax.set_ylim(-3.8, _INSTRUMENT_SIZE[1] + 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_local_zodi(ax, origin, cast)
    aperture = draw_aperture(ax, origin, cast)
    draw_rail(ax, origin, cast)
    if incoming:
        for key in ("star", "planet", "exozodi"):
            y = aperture[1] + _RAY_DY[key] * _spread(cast)
            ex.arrow(ax, (-1.8, y), (aperture[0] - 0.3, y), "ray", cast, source=key)


def _carried(ax, cast, *, top=True):
    """Tag a panel that returns from the previous step."""
    y, va = (0.985, "top") if top else (0.015, "bottom")
    ax.text(
        0.015,
        y,
        "(from the previous step)",
        transform=ax.transAxes,
        ha="left",
        va=va,
        color=cast["annotation"].color,
        fontstyle="italic",
        fontsize=cast.layout.small_pt,
        zorder=9,
        bbox=_backing(cast),
    )


def _relay_figure(layout, cast, title):
    fig, axes = ex.figure(layout, doc_height_in=3.5, ncols=2)
    fig.suptitle(title, color=cast.text)
    return fig, axes


def _panel_title(ax, text, cast):
    ax.set_title(text, color=cast.text, fontsize=cast.layout.font_pt)


def _sky_badge(ax, cast, text):
    ex.badge(ax, cast, text, loc="lower right")


def build_relay_1(layout, cast):
    """Step one: the scene side on, and the same scene on the sky."""
    fig, (left, right) = _relay_figure(
        layout, cast, "One scene: a star, a planet and dust, seen from far away"
    )
    panel_system(left, cast, rays="arrow")
    _panel_title(left, "side view, observer to the right", cast)
    ex.badge(left, cast, "schematic, not to scale", loc="lower right")
    panel_sky(right, cast)
    _panel_title(right, "the same scene on the sky", cast)
    _sky_badge(right, cast, "model dust brightness, noiseless")
    return _plain_text(fig)


def build_relay_2(layout, cast):
    """Step two: the sky scene, then the telescope and coronagraph."""
    fig, (left, right) = _relay_figure(
        layout, cast, "Its light and the local zodiacal light enter a telescope"
    )
    panel_sky(left, cast)
    _panel_title(left, "the same scene on the sky", cast)
    _sky_badge(left, cast, "model dust brightness, noiseless")
    _carried(left, cast)
    _instrument_axes(right, cast)
    _panel_title(right, "the observer, inside the local dust", cast)
    ex.badge(right, cast, "schematic, not to scale", loc="upper right")
    return _plain_text(fig)


def _records_axes(ax, cast):
    """The record card alone, in its own axes, at the map's proportions."""
    w, h = _RECORDS_BOX[2], _RECORDS_BOX[3]
    ax.set_xlim(-1.5, w + 1.5)
    ax.set_ylim(-1.5, h + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    _records_card(ax, (0.0, 0.0), (w, h), cast)


def build_relay_3(layout, cast):
    """Step three: the instrument, then the raw frame it records."""
    fig, (left, right) = _relay_figure(
        layout, cast, "Every contribution lands in the same detector pixels"
    )
    _instrument_axes(left, cast)
    _panel_title(left, "the observer, inside the local dust", cast)
    ex.badge(left, cast, "schematic, not to scale", loc="upper right")
    _carried(left, cast, top=False)
    panel_frame(right, cast)
    _panel_title(right, "raw frame, one visit", cast)
    _sky_badge(right, cast, "synthetic, amplitudes schematic")
    return _plain_text(fig)


def build_relay_4(layout, cast):
    """Step four: the raw frame, then the record it reduces to."""
    fig, (left, right) = _relay_figure(layout, cast, "Each visit reduces to one record")
    panel_frame(left, cast)
    _panel_title(left, "raw frame, one visit", cast)
    _sky_badge(left, cast, "synthetic, amplitudes schematic")
    _carried(left, cast)
    _records_axes(right, cast)
    _panel_title(right, r"the record of visit $k$", cast)
    return _plain_text(fig)


def build_relay_5(layout, cast):
    """Step five: the record, then the orbit the records of four visits give."""
    fig, (left, right) = _relay_figure(
        layout, cast, "Positions from several visits constrain the orbit"
    )
    _records_axes(left, cast)
    _panel_title(left, r"the record of visit $k$", cast)
    _carried(left, cast)
    panel_orbit(right, cast, labels=True)
    _panel_title(right, "inferred orbit on the sky", cast)
    _sky_badge(right, cast, "simulated positions, posterior draws")
    return _plain_text(fig)


# Registry

_SCOPE = (
    "An original schematic of the physical system this book's libraries simulate, "
    "not an identity, a convention or an instrument design."
)
_PRIOR_TEXT = (
    "The prior takes eccentricity uniform on [0, 0.6], the argument of periapsis and the mean "
    "anomaly at epoch uniform on the circle, and the semimajor axis, inclination and node angle "
    "Gaussian about the simulated values (standard deviations of 15 percent, 15 degrees and 15 "
    "degrees); six million prior samples are weighted by the Gaussian likelihood of the measured "
    "positions and thirty draws are taken by systematic resampling, so the draws gather along the "
    "measured arc and fan out on the unobserved one. "
)
_PROFILE_TEXT = (
    "Sky offsets follow the proposed observer profile of {ref}`geometry-observer-basis` "
    "(north, east, toward the observer, with the node angle measured from north toward east); "
    "the library outputs, which label the same rotated components right ascension and "
    "declination, are mapped into it by exchanging those two components. "
)
_FRAME_TEXT = (
    "The raw frame carries no compass: how detector rows and columns map to east and north "
    "depends on the roll and on the pending image-coordinate decision ({ref}`optics-roll-rotation`), "
    "so its pixels are placed as the sky would appear at zero roll under one possible mapping. "
)
_MAP_CAPTION = (
    "The physical observation, from a distant system to the products an analysis reports. "
    "Top row, physical light (solid arrows with filled heads): a host star, and a planet on an orbit "
    "in the midplane of an exozodiacal dust disk, seen side on, with the observer to the right along "
    "the positive z direction of the proposed observer-toward-positive-Z profile "
    "({ref}`geometry-illumination`); the planet is on the far side, so the observer sees mostly its lit "
    "hemisphere. "
    "Reflected planet light, scattered dust light and starlight cross an interstellar distance "
    "that is not drawn to scale and enter a generic telescope aperture and coronagraph that end on a "
    "detector ({ref}`radiometry-response-ownership`), together with local zodiacal light from the dust "
    "cloud around the observer, a foreground along the same line of sight (the same scattering process "
    "seen from inside the cloud, {doc}`/conventions/dust-models`). Bottom row, data (hollow block "
    "arrows): a synthetic raw frame in which planet light, stellar leakage, exozodiacal light and the "
    "uniform local zodiacal floor share the same pixels; one record per visit $k$, with its epoch "
    "$t_k$, reporting event $D_k$, east and north offsets $(\\xi_k,\\eta_k)$, their covariance $C_k$ "
    "and its calibration revision ({ref}`records-reporting-law`, {ref}`records-covariance`, "
    "{ref}`records-identities`); and, in the inferred-orbit sky panel, posterior orbit draws given the "
    "positions measured on four visits, drawn with crosshairs of plus and minus one standard "
    "deviation, the visit shown in the raw frame ringed. "
    + _PRIOR_TEXT
    + _PROFILE_TEXT
    + "The inferred-orbit panel, the one sky panel on this map, is shown east to the left and north "
    "up, as its compass marks. "
    + _FRAME_TEXT
    + "The frame amplitudes are schematic, not a "
    "brightness ratio. Each outlined name locates the chapter that defines that part: Geometry and "
    "time, Dust models, Radiometry, Optical fields, and Measurements, probability, and records. "
    + _SCOPE
)
_MAP_ALT = (
    "A two-row schematic. Top row, left to right: a side view titled target system, side view, "
    "observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, "
    "planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled "
    "starlight, reflected and scattered leave the system toward the right, cross paired slash marks "
    "labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture "
    "inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly "
    "parallel to the others into the aperture; the beam continues through a rail labeled pupil, "
    "focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow "
    "arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, "
    "with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor "
    "labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model "
    "lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi "
    "sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with "
    "an east and north compass, the star at the center, four cyan measured positions, one ringed "
    "and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the "
    "measured arc and fanning out on the opposite side. Small outlined tags name the chapter for "
    "each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics "
    "at the coronagraph and Records above the record card. A key at the top separates light arrows "
    "from data arrows and explains the tags, and a badge reads schematic of the simulation scope, "
    "not to scale."
)

_HIGHLIGHT_TEXT = {
    "geometry": "the target system and the direction to the observer",
    "dust": "the exozodiacal disk and the local zodiacal cloud",
    "radiometry": "the light arriving at the aperture and the counts in the detector pixels",
    "optics": "the aperture, the coronagraph planes and the detector image",
    "inference": "the raw frame, the records and the inferred orbit",
    "tutorial": "the scene, the instrument and the raw frame",
}


def _highlight_spec(key):
    region = _HIGHLIGHT_TEXT[key]
    tag = "this tutorial" if key == "tutorial" else "this chapter"
    locator = (
        "a bold key line reading full strength: this tutorial"
        if key == "tutorial"
        else f"its {CHAPTERS[key]} tag drawn bold and keyed as this chapter"
    )
    return ex.FigureSpec(
        slug=f"d01-overview-{key}",
        build=functools.partial(build_overview, highlight=key),
        caption=(
            f"Where {tag} sits in the physical observation: {region}, at full strength, with the "
            "rest of the map dimmed ({ref}`full map <fig-explainer-d01-overview>`). "
            + _SCOPE
        ),
        alt=(
            f"The observation map with {region} at full strength, {locator}, and every other "
            "part faded toward the background. " + _MAP_ALT
        ),
        status="schematic of the simulation scope, not to scale",
        params={"highlight": key, "parts": list(HIGHLIGHTS[key])},
    )


_PARAMS = {
    "profile": PROFILE,
    "scene": SCENE,
    "t_epoch_jd": T_EPOCH_JD,
    "t_visits_jd": T_VISITS_JD,
    "frame": FRAME,
    "draws": DRAWS,
}
_STEP = "Step {} of five: "

FIGURES = [
    ex.FigureSpec(
        slug="d01-overview",
        build=build_overview,
        caption=_MAP_CAPTION,
        alt=_MAP_ALT,
        status="schematic of the simulation scope, not to scale",
        params=_PARAMS,
    ),
    *(_highlight_spec(key) for key in HIGHLIGHTS),
    ex.FigureSpec(
        slug="d01-relay-1",
        build=build_relay_1,
        caption=(
            _STEP.format("one")
            + "one scene. Left, the star, planet and exozodiacal disk side on, "
            "the observer to the right ({ref}`geometry-illumination`); right, the same disk's model "
            "surface brightness on the sky, east to the left and north up as the compass marks, with "
            "the star and planet positions marked. "
            + _PROFILE_TEXT
            + "Noiseless model. "
            + _SCOPE
        ),
        alt=(
            "Two panels. Left: a side view of a star, a planet on its orbit and a hatched, tilted dust "
            "disk, with rays labeled starlight, reflected and scattered leaving to the right toward the "
            "observer. Right: a log-scaled sky image of the inclined dust ring with the star at the "
            "center, the planet marked, and east and north arrows."
        ),
        status="schematic; model brightness, noiseless",
        params=_PARAMS,
    ),
    ex.FigureSpec(
        slug="d01-relay-2",
        build=build_relay_2,
        caption=(
            _STEP.format("two")
            + "the sky scene from step one, then its light entering a generic "
            "telescope aperture together with the local zodiacal light of the cloud around the "
            "observer, a foreground along the same line of sight, and a generic coronagraph ending on "
            "a detector ({ref}`radiometry-response-ownership`). Not a flight design. "
            + _SCOPE
        ),
        alt=(
            "Two panels. Left: the sky image of the dust ring from the previous step, with east and "
            "north arrows. Right: yellow, cyan and purple rays, and two green rays from a dotted cloud "
            "nearly parallel to them, enter an edge-on aperture, then a rail of pupil, focal mask, "
            "Lyot stop and detector."
        ),
        status="schematic, not to scale",
        params=_PARAMS,
    ),
    ex.FigureSpec(
        slug="d01-relay-3",
        build=build_relay_3,
        caption=(
            _STEP.format("three")
            + "the instrument from step two, then a synthetic raw frame of one "
            "visit in which planet light, stellar leakage, exozodiacal light and the uniform local "
            "zodiacal floor share the detector pixels ({ref}`radiometry-detector-counts`). One Poisson "
            "realization of schematic amplitudes. " + _FRAME_TEXT + _SCOPE
        ),
        alt=(
            "Two panels. Left: the instrument from the previous step. Right: a pixelated raw frame in a "
            "dark-to-bright colormap with a bright center labeled stellar leakage, a blob labeled "
            "planet, a faint arc labeled exozodi and a floor labeled local zodi, uniform floor."
        ),
        status="synthetic frame, amplitudes schematic",
        params=_PARAMS,
    ),
    ex.FigureSpec(
        slug="d01-relay-4",
        build=build_relay_4,
        caption=(
            _STEP.format("four")
            + "the raw frame from step three, then the record that visit $k$ "
            "reduces to: its epoch $t_k$, reporting event $D_k$, east and north offsets "
            "$(\\xi_k,\\eta_k)$, their covariance $C_k$ and its calibration revision "
            "({ref}`records-reporting-law`, {ref}`records-covariance`, {ref}`records-identities`). "
            "The card lists what a record carries, not values. " + _SCOPE
        ),
        alt=(
            "Two panels. Left: the raw frame from the previous step. Right: a record card listing "
            "epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and "
            "calibration revision."
        ),
        status="schematic",
        params=_PARAMS,
    ),
    ex.FigureSpec(
        slug="d01-relay-5",
        build=build_relay_5,
        caption=(
            _STEP.format("five")
            + "the record card from step four, then the planet positions "
            "measured on four visits, with crosshairs of plus and minus one standard deviation, and "
            "posterior orbit draws given them ({ref}`records-reporting-law`); the ringed position is "
            "the visit of the raw frame in steps three and four. The sky panel is shown east to the "
            "left and north up, as its compass marks. "
            + _PRIOR_TEXT
            + _PROFILE_TEXT
            + _SCOPE
        ),
        alt=(
            "Two panels. Left: the record card from the previous step. Right: a sky panel with east "
            "and north arrows, the star at the center, four cyan measured positions, one ringed and "
            "labeled current visit, and a bundle of thin pink posterior orbit ellipses, tight where "
            "they pass through the measured positions and fanning out on the opposite side."
        ),
        status="simulated positions, posterior draws",
        params=_PARAMS,
    ),
]
ANIMATIONS = []
