"""Star, planet, observer, and the orbital reference planes.

Every construction here is computed from the geometry chapter's own
formulas, not from a library's projection: the proposed observer profile
puts the star at the origin with the right-handed basis
``(X, Y, Z) = (north, east, toward the observer)`` and places the orbit with
``r = R_Z(Omega) R_X(i) R_Z(omega_p) [r cos(nu), r sin(nu), 0]``. The current
orbix projection labels its first rotated component RA and its second Dec,
so for the same elements its sky plot is this profile mirrored about the
north-east diagonal; it is not drawn here under this name.

Views are orthographic cameras. A view is either one angle ``g`` (degrees),
a camera turned about the north axis only, 0 for the side view from the east
(sky plane edge-on, observer to the right, north up) and 90 for the
observer's own view (north up, east to the left), or a pair ``(g, el)`` that
also raises the camera ``el`` degrees toward north. Every view uses one page
scale per unit of distance, so a change of view never rescales the orbit.

Figures:
    d02-observer-construction: side-view star, planet, observer construction.
    d02-orbit-elements: an inclined eccentric orbit and its three angles,
        with the observer's view beside it.
    d02-phase-epochs: behind, on, and in front of the sky plane, in side
        view and in the observer's view (the still of d02-orbit-clock).

Animations:
    d02-viewpoint-sweep (ground "viewpoint"): the camera turns about north
        from the side view to the observer's view; the orbit and the planet
        are frozen.
    d02-orbit-clock (ground "narration"): both cameras fixed; orbital time
        runs one period at equal time steps, and the projected offset and the
        illumination change.

Artists that carry a convention (angle arcs, their arrowheads, the motion
arrows, the angular-momentum vector and the construction's rays) have a
``gid`` so the tests can check the drawn geometry, not only the helpers.
"""

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from explainers import ManimSpec
from explainers import _common as ex

# Scientific inputs: one set for every figure and animation.
ORBIT = {"a": 1.0, "e": 0.35, "i_deg": 55.0, "Omega_deg": 130.0, "omega_deg": 70.0}
# Side-view construction: planet distance and observer-axis angle.
SIDE = {"r": 1.0, "beta_deg": 55.0}
# The oblique camera of the orbit-elements figure: turned 15 deg from the
# east axis toward the observer, then raised 15 deg toward north. Chosen so
# h stands 30 deg off the node line and the i arc spans 50 deg on the page.
OBLIQUE_VIEW = (15.0, 15.0)
# Planet epoch frozen during the viewpoint sweep, as an argument of latitude.
SWEEP_U_DEG = 140.0
# Behind, on, and in front of the sky plane, as arguments of latitude.
EPOCH_U_DEG = (270.0, 0.0, 90.0)
EPOCH_NAMES = (
    "behind the sky plane",
    "on the sky plane (ascending node)",
    "in front of the sky plane",
)
# Equal-time ticks per period drawn along the projected track.
N_TICKS = 24
# Motion arrows: argument of latitude at the head, and the span behind it.
MOTION_U_DEG = (35.0, 215.0)
MOTION_SPAN_DEG = 7.0

NORTH = np.array([1.0, 0.0, 0.0])
EAST = np.array([0.0, 1.0, 0.0])
TOWARD_OBSERVER = np.array([0.0, 0.0, 1.0])

PROFILE = "proposed observer profile (pending)"
STATUS_ANGLES = f"{PROFILE}; schematic, not to scale"
STATUS_ORBIT = f"{PROFILE}; orbit to scale, observer distance not drawn"
STATUS_DISK = f"{PROFILE}; orbit to scale, planet disk enlarged"


# Geometry (pure numpy; every anchor below is checked in the tests)


def rot_x(angle):
    """Active right-handed rotation about +X by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(angle):
    """Active right-handed rotation about +Z by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orbit_rotation(orbit=ORBIT):
    """The chapter's ``R_Z(Omega) R_X(i) R_Z(omega_p)``."""
    return (
        rot_z(np.deg2rad(orbit["Omega_deg"]))
        @ rot_x(np.deg2rad(orbit["i_deg"]))
        @ rot_z(np.deg2rad(orbit["omega_deg"]))
    )


def radius(nu, orbit=ORBIT):
    """Conic radius at true anomaly ``nu`` (radians)."""
    a, e = orbit["a"], orbit["e"]
    return a * (1.0 - e**2) / (1.0 + e * np.cos(nu))


def position(nu, orbit=ORBIT):
    """Star-centered position(s) at true anomaly ``nu``, shape ``(..., 3)``."""
    nu = np.asarray(nu, dtype=float)
    r = radius(nu, orbit)
    perifocal = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)], -1)
    return perifocal @ orbit_rotation(orbit).T


def nu_from_u(u_deg, orbit=ORBIT):
    """True anomaly (radians) for an argument of latitude ``u = nu + omega_p``."""
    return np.deg2rad(np.asarray(u_deg, dtype=float) - orbit["omega_deg"])


def angular_momentum_direction(orbit=ORBIT):
    """Unit orbit normal along the angular momentum."""
    return orbit_rotation(orbit) @ TOWARD_OBSERVER


def node_direction(orbit=ORBIT):
    """Unit vector from the star toward the ascending node, in the sky plane."""
    om = np.deg2rad(orbit["Omega_deg"])
    return np.array([np.cos(om), np.sin(om), 0.0])


def ascending_node(orbit=ORBIT):
    """Position of the ascending node (``nu = -omega_p``)."""
    return position(nu_from_u(0.0, orbit), orbit)


def descending_node(orbit=ORBIT):
    """Position of the descending node (``nu = pi - omega_p``)."""
    return position(nu_from_u(180.0, orbit), orbit)


def periapsis(orbit=ORBIT):
    """Position of periapsis (``nu = 0``)."""
    return position(0.0, orbit)


def solve_kepler(mean_anomaly, e):
    """Eccentric anomaly for mean anomaly(ies), by Newton iteration."""
    m = np.asarray(mean_anomaly, dtype=float)
    ecc = m + e * np.sin(m)
    for _ in range(50):
        ecc = ecc - (ecc - e * np.sin(ecc) - m) / (1.0 - e * np.cos(ecc))
    return ecc


def nu_from_time_fraction(frac, orbit=ORBIT):
    """True anomaly at time ``(t - t_p) / P = frac``."""
    e = orbit["e"]
    ecc = solve_kepler(2.0 * np.pi * np.asarray(frac, dtype=float), e)
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(0.5 * ecc), np.sqrt(1.0 - e) * np.cos(0.5 * ecc)
    )


def time_fraction_from_nu(nu, orbit=ORBIT):
    """``(t - t_p) / P`` in [0, 1) at true anomaly ``nu``."""
    e = orbit["e"]
    ecc = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(0.5 * nu), np.sqrt(1.0 + e) * np.cos(0.5 * nu)
    )
    return ((ecc - e * np.sin(ecc)) / (2.0 * np.pi)) % 1.0


def illumination_angle(r_vec, o_hat=TOWARD_OBSERVER):
    """Distant-observer illumination angle, ``cos(alpha) = -r . o / |r|``."""
    r_vec = np.asarray(r_vec, dtype=float)
    cos_a = -(r_vec @ o_hat) / np.linalg.norm(r_vec, axis=-1)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def lambert_phase(alpha):
    """Lambert-sphere phase function ``(sin a + (pi - a) cos a) / pi``."""
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi


def side_state(side=SIDE):
    """Planet in the side-view plane: ``(Z, rho, alpha)`` from ``beta_axis``."""
    beta = np.deg2rad(side["beta_deg"])
    z = side["r"] * np.cos(beta)
    rho = side["r"] * np.sin(beta)
    alpha = np.pi - np.arctan2(rho, z)
    return z, rho, alpha


def camera(view):
    """Screen basis ``(right, up, back)`` for a view ``g`` or ``(g, el)``.

    ``back`` points from the scene toward the camera: along east at
    ``g = 0`` and along the observer direction at ``g = 90``, then tilted
    ``el`` degrees toward north. ``right x up = back``.
    """
    g, el = (float(view), 0.0) if np.ndim(view) == 0 else map(float, view)
    g, el = np.deg2rad(g), np.deg2rad(el)
    back = np.cos(el) * (np.cos(g) * EAST + np.sin(g) * TOWARD_OBSERVER)
    back = back + np.sin(el) * NORTH
    right = np.cross(NORTH, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return right, up, back


def project(points, view):
    """Orthographic screen coordinates of 3D point(s), shape ``(..., 2)``."""
    right, up, _ = camera(view)
    points = np.asarray(points, dtype=float)
    return np.stack([points @ right, points @ up], -1)


def arc_points(start_vec, axis, angle, radius_, n=49):
    """Points on a circle arc from ``start_vec`` rotated about ``axis``."""
    k = axis / np.linalg.norm(axis)
    v = start_vec / np.linalg.norm(start_vec)
    t = np.linspace(0.0, angle, n)[:, None]
    rotated = (
        v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * (k @ v) * (1.0 - np.cos(t))
    )
    return radius_ * rotated


def lit_polygon(center, toward_star, alpha, radius_, n=64):
    """Outline of a Lambert sphere's lit part as seen at illumination ``alpha``.

    The bright limb is the half of the disk facing the star's projected
    direction ``toward_star``; the terminator is a half ellipse whose
    semi-axis along that direction is ``radius * cos(alpha)``.
    """
    u = np.asarray(toward_star, dtype=float)
    u = u / np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    t = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
    limb = np.cos(t)[:, None] * u + np.sin(t)[:, None] * w
    term = -np.cos(alpha) * np.cos(t)[::-1, None] * u + np.sin(t)[::-1, None] * w
    return np.asarray(center) + radius_ * np.vstack([limb, term])


def epoch_state(u_deg, orbit=ORBIT):
    """Position, illumination angle and phase function at argument of latitude."""
    r_vec = position(nu_from_u(u_deg, orbit), orbit)
    alpha = illumination_angle(r_vec)
    return r_vec, alpha, lambert_phase(alpha)


# Inks: the light mode's scenery and note grays are one step darker so small
# labels keep their contrast at the documentation column width.


def _is_light(cast):
    return cast.mode != "dark"


def _scen_ink(cast):
    return cast.neutral(0.62) if _is_light(cast) else cast["scenery"].color


def _note_ink(cast):
    return cast.neutral(0.72) if _is_light(cast) else cast["annotation"].color


def _unlit(cast):
    """Fill of a planet's unlit side: dark gray in both modes."""
    return cast.neutral(0.6) if _is_light(cast) else cast.neutral(0.22)


# Drawing helpers local to this diagram


def _vector(ax, start, end, color, lw, *, ms=12.0, zorder=4, gid=None):
    """A geometric vector: thin shaft, open V head (not a light ray)."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="->,head_length=0.5,head_width=0.25",
        mutation_scale=ms,
        color=color,
        lw=lw,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        gid=gid,
    )
    ax.add_patch(patch)
    return patch


def _label(ax, xy, text, cast, *, color=None, fontsize=None, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ex.halo(
        ax.text(
            *xy,
            text,
            color=color or _note_ink(cast),
            fontsize=fontsize or cast.layout.small_pt,
            zorder=9,
            **kw,
        ),
        cast,
    )


def _note(ax, xy, text, cast, **kw):
    kw.setdefault("color", _note_ink(cast))
    return ex.note(ax, xy, text, cast, transform=ax.transAxes, **kw)


def _headline(fig, layout, text):
    """A one-line question heading the talk slide (not the documentation still)."""
    if layout.is_slide:
        fig.suptitle(text, fontsize=layout.title_pt)


def _place_radial(text, xy, dist, *, toward=None):
    """Put ``text`` beside ``xy``, pushed ``dist`` away from the star."""
    xy = np.asarray(xy, dtype=float)
    d = xy if toward is None else np.asarray(toward, dtype=float)
    norm = np.linalg.norm(d)
    u = d / norm if norm > 1e-9 else np.array([0.0, -1.0])
    text.set_position(xy + dist * u)
    text.set_ha("left" if u[0] > 0.35 else ("right" if u[0] < -0.35 else "center"))
    text.set_va("bottom" if u[1] > 0.35 else ("top" if u[1] < -0.35 else "center"))


def _arc3d(ax, points2d, cast, *, label, label_xy, gid):
    """A projected angle arc; its arrowhead sits at the measured end."""
    color = cast.neutral(0.85)
    ax.plot(*points2d.T, color=color, lw=cast.layout.lw, zorder=6, gid=gid)
    ax.add_patch(
        FancyArrowPatch(
            points2d[-4],
            points2d[-1],
            arrowstyle="-|>",
            mutation_scale=cast.layout.marker_pt * 1.4,
            color=color,
            lw=cast.layout.lw,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
            gid=f"{gid}-head",
        )
    )
    _label(ax, label_xy, label, cast, color=color, fontsize=cast.layout.font_pt)


def _split_orbit(view, orbit=ORBIT, n=721):
    """Projected orbit split into in-front (Z >= 0) and behind (Z < 0) parts."""
    nu = np.linspace(0.0, 2.0 * np.pi, n)
    pts = position(nu, orbit)
    xy = project(pts, view)
    parts = []
    for keep in (pts[:, 2] >= 0.0, pts[:, 2] < 0.0):
        # Widen each part by one sample so it meets the sky plane.
        grown = keep | np.roll(keep, 1) | np.roll(keep, -1)
        parts.append(np.where(grown[:, None], xy, np.nan))
    return parts[0], parts[1]


def _draw_orbit(ax, view, cast):
    front, behind = _split_orbit(view)
    color = cast["planet"].color
    lw = cast.layout.lw
    (f,) = ax.plot(*front.T, color=color, lw=1.3 * lw, zorder=3)
    (b,) = ax.plot(*behind.T, color=color, lw=0.9 * lw, ls=(0, (3, 2.5)), zorder=2)
    return f, b


def motion_ends(u_deg, view):
    """Tail and head of a motion arrow: the head is later in the orbit."""
    nu = nu_from_u(np.array([u_deg - MOTION_SPAN_DEG, u_deg]))
    return project(position(nu), view)


def _motion_arrows(ax, view, cast, prefix):
    patches = []
    for k, u in enumerate(MOTION_U_DEG):
        a, b = motion_ends(u, view)
        patch = FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=cast.layout.marker_pt * 1.9,
            color=cast["planet"].color,
            lw=0.01,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
            gid=f"{prefix}-motion-{k}",
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def _point(ax, xy, marker, cast, *, size=None, filled=True):
    color = cast["scenery"].color
    (line,) = ax.plot(
        [xy[0]],
        [xy[1]],
        marker=marker,
        ls="none",
        ms=size or 0.9 * cast.layout.marker_pt,
        mfc=color if filled else cast.background,
        mec=color,
        mew=0.8 * cast.layout.lw,
        zorder=7,
    )
    return line


def _move(line, xy):
    line.set_data([xy[0]], [xy[1]])


def _finish(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")


# The orbit in space

AXIS_LEN = 1.4  # basis-arrow length, in units of the semi-major axis
SKY_SIZE = 1.25  # radius of the drawn sky-plane disk
H_LEN = 1.3  # drawn length of the angular-momentum direction
SPACE_LIMITS = ((-1.8, 1.9), (-1.55, 1.9))
# Left and right panels of the orbit-elements figure: equal vertical spans and
# horizontal spans in the ratio of the panel widths, so both share one page
# scale whether the width or the height binds.
ELEMENTS_WIDTHS = (1.6, 1.0)
ELEMENTS_LIMITS = ((-1.3, 1.75), (-1.4, 1.9))
ELEMENTS_SKY_LIMITS = ((-1.0, 0.906), (-1.45, 1.85))
# Arc radii: Omega outermost, i in the middle, omega_p innermost.
ARC_R = {"Omega": 0.66, "i": 0.46, "omega": 0.26}
SKY_LIMITS = ((-1.45, 1.25), (-1.3, 1.35))
# Epoch and clock panels share one x span, so every panel has one page scale.
PAIR_X = (-1.45, 1.45)
SIDE_Y = (-1.55, 1.3)
SPACE_MARKS = (
    ("asc", "ascending node", ascending_node, 0.16),
    ("peri", "periapsis", periapsis, 0.14),
)
BASIS_TEXT = (
    (NORTH, r"north $\hat{\mathbf{X}}$"),
    (EAST, r"east $\hat{\mathbf{Y}}$"),
    (TOWARD_OBSERVER, "toward\nobserver " + r"$\hat{\mathbf{Z}}$"),
)


def _sky_disk(view):
    t = np.linspace(0.0, 2.0 * np.pi, 181)
    ring = SKY_SIZE * np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], -1)
    return project(ring, view)


def _space_panel(ax, view, cast, *, u_planet_deg=None):
    """The orbit in space from one camera; returns every projected artist."""
    art = {"disk": ex.region(ax, "reference_plane", Polygon(_sky_disk(view)), cast)}
    scen, ink = cast["scenery"].color, _scen_ink(cast)
    art["plane_label"] = _label(
        ax, (0, 0), "sky plane", cast, color=ink, fontstyle="italic"
    )
    art["basis"] = [
        (
            vec,
            _vector(
                ax,
                (0, 0),
                (0, 0),
                scen,
                0.9 * cast.layout.lw,
                ms=cast.layout.marker_pt * 1.4,
            ),
            _label(ax, (0, 0), text, cast, color=ink),
        )
        for vec, text in BASIS_TEXT
    ]
    art["endon"] = _label(ax, (0.0, 0.0), "", cast, color=ink, ha="left", va="bottom")
    art["endon"].set_transform(ax.transAxes)
    art["endon"].set_position((0.0, 0.07))
    art["orbit"] = _draw_orbit(ax, view, cast)
    art["motion"] = _motion_arrows(ax, view, cast, "d02")
    (art["nodes_line"],) = ax.plot([], [], color=scen, lw=0.8 * cast.layout.lw)
    art["asc"] = _point(ax, (0, 0), "^", cast)
    art["desc"] = _point(ax, (0, 0), "v", cast, filled=False)
    art["peri"] = _point(ax, (0, 0), "D", cast, size=0.75 * cast.layout.marker_pt)
    art["marks"] = [
        (fn, dist, _label(ax, (0, 0), text, cast)) for _, text, fn, dist in SPACE_MARKS
    ]
    ex.mark(ax, "star", (0, 0), cast, scale=0.9)
    if u_planet_deg is not None:
        (art["planet"],) = ax.plot(
            [], [], zorder=8, **cast["planet"].marker_kw(cast.layout.marker_pt)
        )
        art["planet_label"] = _label(
            ax, (0, 0), "planet", cast, color=cast["planet"].color
        )
    _set_space_view(art, view, u_planet_deg)
    return art


def _set_space_view(art, view, u_planet_deg=None):
    """Project every artist of ``_space_panel`` for one camera."""
    art["disk"].set_xy(_sky_disk(view))
    _place_radial(
        art["plane_label"], project(np.array([-0.7, -0.7, 0.0]) * SKY_SIZE, view), 0.05
    )
    endon = ""
    for (vec, arrow, lab), (_, text) in zip(art["basis"], BASIS_TEXT, strict=True):
        tip = project(AXIS_LEN * vec, view)
        arrow.set_positions((0, 0), tip)
        if np.linalg.norm(tip) < 0.3 * AXIS_LEN:
            # Seen end-on: the axis points at the camera, through the star.
            lab.set_visible(False)
            endon = (
                text.replace("\n", " ") + " points at the camera\n(end-on, at the star)"
            )
        else:
            lab.set_visible(True)
            _place_radial(lab, tip, 0.1)
    art["endon"].set_text(endon)
    front, behind = _split_orbit(view)
    art["orbit"][0].set_data(*front.T)
    art["orbit"][1].set_data(*behind.T)
    for patch, u in zip(art["motion"], MOTION_U_DEG, strict=True):
        patch.set_positions(*motion_ends(u, view))
    an, dn = ascending_node(), descending_node()
    art["nodes_line"].set_data(*project(np.array([dn, an]), view).T)
    _move(art["asc"], project(an, view))
    _move(art["desc"], project(dn, view))
    _move(art["peri"], project(periapsis(), view))
    for fn, dist, lab in art["marks"]:
        _place_radial(lab, project(fn(), view), dist)
    if u_planet_deg is not None:
        xy = project(position(nu_from_u(u_planet_deg)), view)
        _move(art["planet"], xy)
        _place_radial(art["planet_label"], xy, 0.14)


def _angle_arcs(ax, view, cast):
    """The three element angles and the vectors they are measured between."""
    h = angular_momentum_direction()
    n_hat = node_direction()
    i = np.deg2rad(ORBIT["i_deg"])
    om = np.deg2rad(ORBIT["Omega_deg"])
    w = np.deg2rad(ORBIT["omega_deg"])
    # Inclination: from +Z to h, about the node direction.
    axis_i = np.cross(TOWARD_OBSERVER, h)
    _arc3d(
        ax,
        project(arc_points(TOWARD_OBSERVER, axis_i, i, ARC_R["i"]), view),
        cast,
        label=r"$i$",
        label_xy=project(
            arc_points(TOWARD_OBSERVER, axis_i, i, ARC_R["i"] + 0.13)[24], view
        ),
        gid="d02-arc-i",
    )
    # Node longitude: in the sky plane, from north toward east.
    _arc3d(
        ax,
        project(arc_points(NORTH, TOWARD_OBSERVER, om, ARC_R["Omega"]), view),
        cast,
        label=r"$\Omega$",
        label_xy=project(
            arc_points(NORTH, TOWARD_OBSERVER, om, ARC_R["Omega"] + 0.16)[24], view
        ),
        gid="d02-arc-Omega",
    )
    # Argument of periapsis: in the orbit plane, from the node along the motion.
    _arc3d(
        ax,
        project(arc_points(n_hat, h, w, ARC_R["omega"]), view),
        cast,
        label=r"$\omega_p$",
        label_xy=project(arc_points(n_hat, h, w, ARC_R["omega"] + 0.13)[24], view),
        gid="d02-arc-omega",
    )
    ax.plot(
        *project(np.array([np.zeros(3), periapsis()]), view).T,
        color=cast["scenery"].color,
        lw=0.7 * cast.layout.lw,
        zorder=2,
    )
    # The angular-momentum direction carries the answer ink and a heavier line.
    tip = project(H_LEN * h, view)
    _vector(
        ax,
        (0, 0),
        tip,
        cast.text,
        1.5 * cast.layout.lw,
        ms=cast.layout.marker_pt * 1.8,
        zorder=5,
        gid="d02-h",
    )
    return tip


def _camera_locator(ax, cast, *, view=None):
    """Top view from north: star, sky plane edge-on, camera direction."""
    rp = cast["reference_plane"]
    scen, ink = cast["scenery"].color, _scen_ink(cast)
    ms = cast.layout.marker_pt * 1.3
    ax.plot([0, 0], [-1.0, 0.55], color=rp.color, ls=rp.ls, lw=cast.layout.lw)
    _vector(ax, (0, 0), (1.0, 0), scen, 0.9 * cast.layout.lw, ms=ms)
    _label(ax, (1.0, 0.1), "toward\nobserver", cast, color=ink, va="bottom")
    _vector(ax, (0, 0), (0, -0.9), scen, 0.9 * cast.layout.lw, ms=ms)
    _label(ax, (-0.08, -0.95), "east", cast, color=ink, ha="right", va="top")
    ex.mark(ax, "star", (0, 0), cast, scale=0.7)
    t = np.linspace(-0.5 * np.pi, 0.0, 60)
    ax.plot(
        0.75 * np.cos(t),
        0.75 * np.sin(t),
        color=cast.neutral(0.35),
        lw=0.6 * cast.layout.lw,
        ls=":",
    )
    (sight,) = ax.plot([], [], color=cast.text, lw=0.7 * cast.layout.lw, ls="--")
    (cam,) = ax.plot(
        [],
        [],
        marker="s",
        ls="none",
        ms=cast.layout.marker_pt,
        color=cast.text,
        zorder=6,
    )
    lab = _label(ax, (0, 0), "camera", cast, color=cast.text, ha="left")
    _finish(ax, (-0.55, 1.5), (-1.15, 0.75))
    art = {"cam": cam, "sight": sight, "label": lab}
    if view is not None:
        _set_locator(art, view)
    return art


def _set_locator(art, view):
    # Seen from north with the observer to the right, east points down.
    g = np.deg2rad(view if np.ndim(view) == 0 else view[0])
    xy = 0.75 * np.array([np.sin(g), -np.cos(g)])
    _move(art["cam"], xy)
    art["sight"].set_data([0.0, xy[0]], [0.0, xy[1]])
    art["label"].set_position(xy + np.array([0.1, -0.12]))


def _sky_omega_arc(ax, cast):
    """Omega as the observer measures it: from north, counterclockwise on the
    image (toward east), to the ascending node."""
    om = np.deg2rad(ORBIT["Omega_deg"])
    scen = cast["scenery"].color
    lw = 0.7 * cast.layout.lw
    ax.plot([0, 0], [0, 0.7], color=scen, lw=lw, ls="--", zorder=2)
    ax.plot(
        *project(np.array([np.zeros(3), ascending_node()]), 90.0).T, color=scen, lw=lw
    )
    _arc3d(
        ax,
        project(arc_points(NORTH, TOWARD_OBSERVER, om, 0.42), 90.0),
        cast,
        label=r"$\Omega$",
        label_xy=project(arc_points(NORTH, TOWARD_OBSERVER, om, 0.6)[24], 90.0),
        gid="d02-sky-arc-Omega",
    )


def _sky_panel(ax, cast, *, ticks=True, compass=(-0.95, -1.05), marks=True):
    """The observer's view: north up, east left, star at the center."""
    _draw_orbit(ax, 90.0, cast)
    if marks:
        _point(ax, project(ascending_node(), 90.0), "^", cast)
        _point(ax, project(descending_node(), 90.0), "v", cast, filled=False)
        _point(
            ax, project(periapsis(), 90.0), "D", cast, size=0.75 * cast.layout.marker_pt
        )
    _motion_arrows(ax, 90.0, cast, "d02-sky")
    if ticks:
        frac = np.arange(N_TICKS) / N_TICKS
        pts = project(position(nu_from_time_fraction(frac)), 90.0)
        ax.plot(
            *pts.T,
            ls="none",
            marker="o",
            ms=0.3 * cast.layout.marker_pt,
            color=cast["planet"].color,
            zorder=3,
        )
    if compass is not None:
        color = _note_ink(cast)
        origin = np.array(compass, dtype=float)
        for vec, text in ((NORTH, r"N, $\eta$"), (EAST, r"E, $\xi$")):
            d = project(vec, 90.0)
            _vector(ax, origin, origin + 0.36 * d, color, 0.8 * cast.layout.lw)
            lab = _label(ax, (0, 0), text, cast, color=color)
            _place_radial(lab, origin + 0.36 * d, 0.05, toward=d)
    ex.mark(ax, "star", (0, 0), cast, scale=0.8)


# Figure: side-view construction

PLANET_R = 0.06  # drawn planet radius in the construction
RAY_GAP = 0.035  # gap between a ray tip and the planet disk


def build_construction(layout, cast):
    fig, ax = ex.figure(layout, doc_height_in=3.4)
    _headline(
        fig, layout, "Where are the illumination and observer-axis angles measured?"
    )
    z, rho, alpha = side_state()
    b_deg = 180.0 - np.rad2deg(alpha)
    lw = layout.lw
    scen, ink, dim = cast["scenery"].color, _scen_ink(cast), _note_ink(cast)
    x_obs = 2.55
    ax.plot([-0.55, x_obs - 0.2], [0, 0], color=scen, lw=0.8 * lw, zorder=1)
    ex.scale_break(ax, (2.0, 0.0), cast)
    rp = cast["reference_plane"]
    ax.plot([0, 0], [-0.55, 1.18], color=rp.color, ls=rp.ls, lw=lw, zorder=1)
    _label(
        ax,
        (-0.05, 1.18),
        "sky plane\n(edge-on)",
        cast,
        color=ink,
        ha="right",
        va="top",
        fontstyle="italic",
    )
    ex.aperture(ax, (x_obs, 0.0), 0.34, cast)
    _label(ax, (x_obs, 0.26), "observer", cast, color=ink, va="bottom")
    _vector(ax, (0, 0), (0.42, 0), cast.text, 1.3 * lw, ms=layout.marker_pt * 1.6)
    _label(ax, (0.3, -0.1), r"$\hat{\mathbf{o}}$", cast, color=cast.text, va="top")
    # Projection rectangle: Z along the line of sight, rho in the sky plane.
    ax.plot([z, z], [0, rho - PLANET_R], color=scen, lw=0.7 * lw, ls=":", zorder=2)
    ax.plot([0, z - PLANET_R], [rho, rho], color=scen, lw=0.7 * lw, ls=":", zorder=2)
    for start, end in (((0.0, -0.3), (z, -0.3)), ((-0.17, 0.0), (-0.17, rho))):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="|-|,widthA=0.25,widthB=0.25",
                mutation_scale=layout.marker_pt,
                color=dim,
                lw=0.8 * lw,
                shrinkA=0,
                shrinkB=0,
            )
        )
    _label(ax, (0.5 * z, -0.37), r"$Z=\mathbf{r}\cdot\hat{\mathbf{o}}$", cast, va="top")
    _label(
        ax, (z + 0.08, -0.37), "line-of-sight\ncoordinate", cast, ha="left", va="top"
    )
    _label(
        ax, (-0.24, 0.5 * rho), r"$\rho$" + "\nprojected\nseparation", cast, ha="right"
    )
    # Starlight along r stops a gap short of the disk; reflected light leaves
    # the disk edge toward the observer.
    u_r = np.array([z, rho]) / np.hypot(z, rho)
    r_tip = np.array([z, rho]) - (PLANET_R + RAY_GAP) * u_r
    art = ex.arrow(ax, (0.0, 0.0), r_tip, "ray", cast, source="star")
    art[0].set_gid("d02-ray-r")
    _label(
        ax,
        (0.5 * z - 0.1, 0.5 * rho + 0.1),
        r"$\mathbf{r}$",
        cast,
        color=cast["star"].color,
        fontsize=layout.font_pt,
    )
    art = ex.arrow(
        ax, (z + PLANET_R + RAY_GAP, rho), (1.85, rho), "ray", cast, source="planet"
    )
    art[0].set_gid("d02-ray-observer")
    _label(
        ax,
        (1.25, rho + 0.07),
        r"toward observer, $\mathbf{R}_{\rm obs}-\mathbf{r}$",
        cast,
        color=cast["planet"].color,
        va="bottom",
    )
    _label(
        ax,
        (1.3, rho - 0.08),
        r"distant observer: parallel to $\hat{\mathbf{o}}$",
        cast,
        va="top",
    )
    ax.add_patch(
        Circle(
            (z, rho),
            PLANET_R,
            facecolor=_unlit(cast),
            edgecolor=cast["planet"].color,
            lw=0.8 * lw,
            zorder=7,
        )
    )
    ax.add_patch(
        Polygon(
            lit_polygon((z, rho), (-z, -rho), 0.5 * np.pi, PLANET_R),
            facecolor=cast["planet"].color,
            edgecolor="none",
            zorder=8,
        )
    )
    _label(
        ax,
        (z - 0.1, rho + 0.09),
        "planet (lit\nside faces star)",
        cast,
        color=cast["planet"].color,
        ha="right",
        va="bottom",
    )
    ex.mark(ax, "star", (0, 0), cast)
    _label(
        ax, (-0.06, -0.08), "star", cast, color=cast["star"].color, ha="right", va="top"
    )
    for gid, artists in (
        (
            "d02-arc-beta",
            ex.angle_arc(
                ax,
                (0, 0),
                0.0,
                b_deg,
                cast,
                radius=0.3,
                label=r"$\beta_{\rm axis}$",
                label_radius=1.6,
            ),
        ),
        (
            "d02-arc-alpha",
            ex.angle_arc(
                ax,
                (z, rho),
                0.0,
                b_deg - 180.0,
                cast,
                radius=0.22,
                label=r"$\alpha$",
                label_radius=1.6,
            ),
        ),
    ):
        artists[0].set_gid(gid)
        artists[1].set_gid(f"{gid}-head")
    _label(
        ax,
        (1.4, 0.3),
        r"$\alpha=\pi-\beta_{\rm axis}$" + "\n" + r"$\cos\alpha=-Z/r$",
        cast,
        color=cast.neutral(0.85),
        fontsize=layout.font_pt,
    )
    ex.note(
        ax,
        (-0.62, -0.66),
        "side view in the plane of star, planet and observer; "
        "the vertical is an unnamed sky-plane direction",
        cast,
        ha="left",
        va="bottom",
        color=_note_ink(cast),
    )
    ex.badge(ax, cast, STATUS_ANGLES, loc="upper right")
    _finish(ax, (-0.65, 2.75), (-0.68, 1.22))
    return fig


# Figure: orbit elements


def build_elements(layout, cast):
    fig, (ax, sky) = ex.figure(
        layout, doc_height_in=4.7, ncols=2, width_ratios=list(ELEMENTS_WIDTHS)
    )
    _headline(fig, layout, "Where is each orbital angle measured?")
    view = OBLIQUE_VIEW
    art = _space_panel(ax, view, cast)
    # The sky-plane label sits just outside the plane's southern rim.
    _place_radial(
        art["plane_label"],
        project(np.array([-SKY_SIZE, 0.0, 0.0]), view),
        0.06,
        toward=(0.0, -1.0),
    )
    for vec, _, lab in art["basis"]:
        if vec is TOWARD_OBSERVER:
            _place_radial(lab, project(AXIS_LEN * vec, view), 0.08, toward=(0.2, -1.0))
    h_tip = _angle_arcs(ax, view, cast)
    h_lab = _label(
        ax,
        (0, 0),
        r"$\hat{\mathbf{h}}$ angular momentum",
        cast,
        color=cast.text,
    )
    _place_radial(h_lab, h_tip, 0.08, toward=(1.0, -0.1))
    desc = _label(ax, (0, 0), "descending node", cast)
    _place_radial(desc, project(descending_node(), view), 0.1, toward=(1.0, 0.45))
    mo = _label(ax, (0, 0), "motion", cast, color=cast["planet"].color)
    _place_radial(mo, project(position(nu_from_u(MOTION_U_DEG[1])), view), 0.12)
    _note(
        ax,
        (1.0, 1.0),
        "oblique view: camera turned\n15 deg from east toward the\n"
        "observer, raised 15 deg north",
        cast,
        ha="right",
        va="top",
    )
    ex.badge(ax, cast, f"{PROFILE}\norbit to scale", loc="upper left")
    _finish(ax, *ELEMENTS_LIMITS)

    _sky_panel(sky, cast, ticks=False, compass=(0.85, -1.2))
    _sky_omega_arc(sky, cast)
    for fn, text in ((ascending_node, "ascending node"), (periapsis, "periapsis")):
        lab = _label(sky, (0, 0), text, cast)
        _place_radial(lab, project(fn(), 90.0), 0.12, toward=(0.0, -1.0))
    _note(
        sky,
        (0.5, 1.0),
        "observer's view from " + r"$+\hat{\mathbf{Z}}$" + "\n"
        r"offsets $(\xi,\eta)$ = (east, north)" + "\n"
        "both panels: solid in front of the\nsky plane ($Z>0$), dashed behind;\n"
        "observer distance not drawn",
        cast,
        ha="center",
        va="top",
    )
    _finish(sky, *ELEMENTS_SKY_LIMITS)
    return fig


# Planet glyph for the observer's view and the side view

GLYPH_R = 0.18


def _sky_dynamic(ax, cast):
    ink = _scen_ink(cast)
    (rho_line,) = ax.plot(
        [], [], color=cast["scenery"].color, lw=0.8 * cast.layout.lw, zorder=4
    )
    rho_label = _label(
        ax, (0, 0), r"$\rho$", cast, color=ink, fontsize=cast.layout.font_pt
    )
    dark = Circle(
        (0, 0),
        GLYPH_R,
        facecolor=_unlit(cast),
        edgecolor=cast["planet"].color,
        lw=0.8 * cast.layout.lw,
        zorder=8,
    )
    ax.add_patch(dark)
    lit = Polygon(
        np.zeros((4, 2)), facecolor=cast["planet"].color, edgecolor="none", zorder=9
    )
    ax.add_patch(lit)
    return {"rho": rho_line, "rho_label": rho_label, "dark": dark, "lit": lit}


def _set_sky(art, r_vec, alpha):
    xy = project(r_vec, 90.0)
    art["rho"].set_data([0.0, xy[0]], [0.0, xy[1]])
    normal = np.array([-xy[1], xy[0]]) / np.linalg.norm(xy)
    art["rho_label"].set_position(0.5 * xy + 0.13 * normal)
    art["rho_label"].set_alpha(1.0 if np.linalg.norm(xy) > 0.5 else 0.0)
    art["dark"].set_center(xy)
    art["lit"].set_xy(lit_polygon(xy, -xy, alpha, GLYPH_R))


def _side_panel(ax, cast, *, plane_label=True):
    """Side view from the east: sky plane edge-on, observer to the right.

    Under the sky plane sit the observer arrow and a front/behind readout,
    so both read as part of this panel.
    """
    rp = cast["reference_plane"]
    ax.plot([0, 0], [-1.05, 1.2], color=rp.color, ls=rp.ls, lw=cast.layout.lw, zorder=1)
    _draw_orbit(ax, 0.0, cast)
    _motion_arrows(ax, 0.0, cast, "d02-side")
    scen, ink = cast["scenery"].color, _scen_ink(cast)
    _vector(
        ax,
        (0.0, -1.18),
        (0.7, -1.18),
        scen,
        0.9 * cast.layout.lw,
        ms=cast.layout.marker_pt * 1.3,
    )
    _label(ax, (0.76, -1.18), "to observer", cast, color=ink, ha="left")
    where = _label(ax, (0.0, -1.33), "", cast, color=ink, va="top", fontstyle="italic")
    plane = _label(
        ax,
        (0.06, 1.2),
        "sky plane",
        cast,
        color=ink,
        ha="left",
        va="top",
        fontstyle="italic",
    )
    plane.set_visible(plane_label)
    ex.mark(ax, "star", (0, 0), cast, scale=0.8)
    (planet,) = ax.plot(
        [], [], zorder=8, **cast["planet"].marker_kw(cast.layout.marker_pt)
    )
    (drop,) = ax.plot([], [], color=scen, lw=0.9 * cast.layout.lw, ls=":", zorder=4)
    z_label = _label(ax, (0, 0), r"$Z$", cast, color=ink, fontsize=cast.layout.font_pt)
    return {"planet": planet, "drop": drop, "z_label": z_label, "where": where}


def _set_side(art, r_vec):
    x, y = project(r_vec, 0.0)
    _move(art["planet"], (x, y))
    art["drop"].set_data([0.0, x], [y, y])
    # The label sits on its own dotted segment; the halo keeps it legible.
    art["z_label"].set_position((0.5 * x, y))
    art["z_label"].set_alpha(1.0 if abs(x) > 0.3 else 0.0)
    if abs(r_vec[2]) < 1e-9:
        art["where"].set_text("on the sky plane: Z = 0")
    else:
        art["where"].set_text("in front: Z > 0" if r_vec[2] > 0 else "behind: Z < 0")


def _readout(alpha, *, name_phi=False, sep="\n"):
    phi = r"$\Phi$ (Lambert phase)" if name_phi else r"$\Phi$"
    return (
        rf"$\alpha$ = {np.rad2deg(alpha):.0f} deg"
        + sep
        + f"{phi} = {lambert_phase(alpha):.2f}"
    )


TICK_NOTE = f"dots: {N_TICKS} equal time steps per period"
DISK_NOTE = "planet disk enlarged;\ncolored = lit, gray = unlit"


# Figure: behind / node / in front (the still of the clock animation)


def build_epochs(layout, cast):
    fig, axes = ex.figure(layout, doc_height_in=5.4, nrows=2, ncols=3)
    _headline(fig, layout, "Why is the planet half lit where it crosses the sky plane?")
    for col, (u, name) in enumerate(zip(EPOCH_U_DEG, EPOCH_NAMES, strict=True)):
        r_vec, alpha, _ = epoch_state(u)
        side, sky = axes[0, col], axes[1, col]
        art = _side_panel(side, cast, plane_label=(col == 0))
        _set_side(art, r_vec)
        side.set_title(name, fontsize=layout.font_pt)
        _finish(side, PAIR_X, (SIDE_Y[0], 1.75))
        _sky_panel(sky, cast, compass=(-0.95, -1.05) if col == 0 else None, marks=False)
        _set_sky(_sky_dynamic(sky, cast), r_vec, alpha)
        sky.set_title(
            _readout(alpha, name_phi=(col == 0)),
            fontsize=layout.font_pt,
            color=cast.text,
        )
        _finish(sky, PAIR_X, (-1.3, 1.35))
    _note(axes[0, 0], (0.0, 1.0), "side view\nfrom the east", cast, ha="left", va="top")
    _note(axes[1, 0], (0.0, 1.0), "observer's\nview", cast, ha="left", va="top")
    _note(axes[1, 2], (1.0, 0.0), DISK_NOTE, cast, ha="right", va="bottom")
    _note(
        axes[1, 1],
        (1.0, 0.0),
        TICK_NOTE.replace("dots: ", "dots:\n"),
        cast,
        ha="right",
        va="bottom",
    )
    ex.badge(axes[0, 2], cast, f"{PROFILE}\norbit to scale", loc="upper right")
    return fig


# Animation: the camera turns, the orbit is frozen


def sweep_frames(layout):
    """Camera angles from the side view to the observer's view; epoch fixed."""
    n = layout.n_frames(96)
    return [
        {"view_deg": float(v), "u_deg": SWEEP_U_DEG} for v in np.linspace(0.0, 90.0, n)
    ]


def build_sweep(layout, cast):
    fig, (ax, loc) = ex.figure(
        layout, doc_height_in=4.0, ncols=2, width_ratios=[2.3, 1.0]
    )
    _headline(fig, layout, "What does the observer see of a three-dimensional orbit?")
    art = _space_panel(ax, 0.0, cast, u_planet_deg=SWEEP_U_DEG)
    _finish(ax, *SPACE_LIMITS)
    ex.badge(ax, cast, STATUS_ORBIT, loc="upper left")
    readout = loc.text(
        0.0,
        0.0,
        "",
        transform=loc.transAxes,
        ha="left",
        va="bottom",
        fontsize=layout.font_pt,
        color=cast.text,
    )
    locator = _camera_locator(loc, cast)
    loc.set_ylim(-1.75, 0.95)
    _note(
        loc, (0.0, 1.0), "camera direction,\nseen from north", cast, ha="left", va="top"
    )
    _note(
        ax,
        (0.0, 0.0),
        "orbit and planet frozen; only the camera turns",
        cast,
        ha="left",
        va="bottom",
    )

    def draw(fig_, frame):
        del fig_
        v = frame["view_deg"]
        _set_space_view(art, v, frame["u_deg"])
        _set_locator(locator, v)
        if v >= 89.999:
            readout.set_text("observer's view:\nnorth up, east left")
        elif v <= 0.001:
            readout.set_text("side view\nfrom the east")
        else:
            readout.set_text(f"camera turned\n{v:.0f} deg about north")

    frames = sweep_frames(layout)
    draw(fig, frames[0])
    return ex.AnimationScene(fig=fig, draw=draw, frames=frames)


# Animation: the cameras are fixed, orbital time runs


def clock_frames(layout):
    """Equal time steps over one period, starting behind the sky plane.

    The talk version takes 200 steps (20 s at 10 fps), so the planet moves
    at most about 4 deg of true anomaly per frame, even at periapsis.
    """
    n = layout.n_frames(200)
    f0 = time_fraction_from_nu(nu_from_u(EPOCH_U_DEG[0]))
    return [{"frac": float((f0 + k / n) % 1.0)} for k in range(n)]


def build_clock(layout, cast):
    fig, (side, sky) = ex.figure(layout, doc_height_in=3.9, ncols=2)
    _headline(
        fig, layout, "How do sky position and illumination change over one orbit?"
    )
    limits = (PAIR_X, (SIDE_Y[0], 1.75))
    side_art = _side_panel(side, cast)
    _finish(side, *limits)
    _sky_panel(sky, cast, marks=False)
    sky_art = _sky_dynamic(sky, cast)
    _finish(sky, *limits)
    _note(side, (0.0, 0.0), "side view\nfrom the east", cast, ha="left", va="bottom")
    _note(
        sky,
        (0.0, 1.0),
        "observer's view from " + r"$+\hat{\mathbf{Z}}$",
        cast,
        ha="left",
        va="top",
    )
    ex.badge(side, cast, f"{PROFILE}\norbit to scale", loc="upper right")
    _note(sky, (1.0, 0.0), DISK_NOTE + "\n" + TICK_NOTE, cast, ha="right", va="bottom")
    readout = sky.text(
        1.0,
        1.0,
        "",
        transform=sky.transAxes,
        ha="right",
        va="top",
        fontsize=layout.font_pt,
        color=cast.text,
    )

    def draw(fig_, frame):
        del fig_
        r_vec = position(nu_from_time_fraction(frame["frac"]))
        alpha = illumination_angle(r_vec)
        _set_side(side_art, r_vec)
        _set_sky(sky_art, r_vec, alpha)
        readout.set_text(
            rf"$t-t_p$ = {frame['frac']:.2f} P" + "\n" + _readout(alpha, name_phi=True)
        )

    frames = clock_frames(layout)
    draw(fig, frames[0])
    return ex.AnimationScene(fig=fig, draw=draw, frames=frames)


# Three rotations: the orbit placed one factor at a time
#
# Read left to right, the chapter's R_Z(Omega) R_X(i) R_Z(omega_p) is three
# turns, each about an axis that the earlier turns placed: about +Z by Omega,
# about the node line by i, and about the orbit normal by omega_p. The orbit
# after step k is the chapter's formula with the later angles still zero, so
# every state drawn below is that formula; the documentation strip and the
# talk clip both draw these states from ``rotation_geometry``.

ROTATION_STEPS = ("omega", "inclination", "periapsis")
# Fractions of (Omega, i, omega_p) applied before step 1 and after each step.
STEP_FRACTIONS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
)
FACE_ON = (90.0, 0.0)
# The turn in the sky plane is seen in the observer's view; the two turns out
# of it are seen in the oblique view of the orbit-elements figure.
STEP_VIEWS = (FACE_ON, OBLIQUE_VIEW, OBLIQUE_VIEW)
ROTATION_LIMITS = ((-1.5, 1.62), (-1.55, 1.62))
NORTH_REF_LEN = 0.9  # the dashed north reference Omega is measured from
SKY_OMEGA_R = 0.42  # Omega arc radius in the observer's view, inside periapsis
ROTATION_NAMES = (r"$R_Z(\Omega)$", r"$R_X(i)$", r"$R_Z(\omega_p)$")
OBLIQUE_NOTE = (
    f"oblique view: camera turned {OBLIQUE_VIEW[0]:.0f} deg from east\n"
    f"toward the observer, raised {OBLIQUE_VIEW[1]:.0f} deg north"
)


def staged_rotation(fractions, orbit=ORBIT):
    """``R_Z(f0 Omega) R_X(f1 i) R_Z(f2 omega_p)``: the construction part way."""
    f_om, f_i, f_w = fractions
    return (
        rot_z(f_om * np.deg2rad(orbit["Omega_deg"]))
        @ rot_x(f_i * np.deg2rad(orbit["i_deg"]))
        @ rot_z(f_w * np.deg2rad(orbit["omega_deg"]))
    )


def staged_position(nu, fractions, orbit=ORBIT):
    """Position(s) at true anomaly ``nu`` after the given fractions of the turns."""
    nu = np.asarray(nu, dtype=float)
    r = radius(nu, orbit)
    perifocal = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)], -1)
    return perifocal @ staged_rotation(fractions, orbit).T


def staged_axes(fractions, orbit=ORBIT):
    """Node-line direction and orbit normal after the given fractions.

    The first turn places the node line and the tilt, about that line, leaves
    it fixed. The tilt places the normal and the last turn, about the normal,
    leaves it fixed.
    """
    f_om, f_i, _ = fractions
    turn = rot_z(f_om * np.deg2rad(orbit["Omega_deg"]))
    node = turn @ NORTH
    normal = turn @ rot_x(f_i * np.deg2rad(orbit["i_deg"])) @ TOWARD_OBSERVER
    return node, normal


def rotation_geometry(fractions, view, orbit=ORBIT, n=361, omega_r=None):
    """Every projected shape of the construction for one state and one camera.

    Args:
        fractions: Applied fractions of ``(Omega, i, omega_p)``, each in [0, 1].
        view: A camera, as for ``camera``.
        orbit: Orbital elements.
        n: Samples around the orbit.
        omega_r: Radius of the Omega arc; None uses the orbit-elements
            figure's oblique-view radius. The observer's view uses
            ``SKY_OMEGA_R``, inside periapsis, as that figure's sky panel does.

    Returns:
        A dict of screen-coordinate arrays: the orbit split into the part on
        or in front of the sky plane and the part behind it (NaN gaps), the
        sky disk, the node line, the north reference, the three angle arcs as
        far as they are applied and a point beside the middle of each, the
        basis tips, the orbit normal tip, periapsis and both nodes, and the
        motion arrows as (tail, head) pairs.
    """
    f_om, f_i, f_w = fractions
    node, normal = staged_axes(fractions, orbit)
    nu = np.linspace(0.0, 2.0 * np.pi, n)
    pts = staged_position(nu, fractions, orbit)
    xy = project(pts, view)
    parts = []
    for keep in (pts[:, 2] >= -1e-9, pts[:, 2] < -1e-9):
        # Widen each part by one sample so it meets the sky plane.
        grown = keep | np.roll(keep, 1) | np.roll(keep, -1)
        parts.append(np.where(grown[:, None], xy, np.nan))
    om = f_om * np.deg2rad(orbit["Omega_deg"])
    inc = f_i * np.deg2rad(orbit["i_deg"])
    w = f_w * np.deg2rad(orbit["omega_deg"])

    def arc(start, axis, angle, rad, label_gap):
        return (
            project(arc_points(start, axis, angle, rad), view),
            project(arc_points(start, axis, angle, rad + label_gap)[24], view),
        )

    omega_r = ARC_R["Omega"] if omega_r is None else omega_r
    arc_om, label_om = arc(NORTH, TOWARD_OBSERVER, om, omega_r, 0.17)
    arc_i, label_i = arc(TOWARD_OBSERVER, node, inc, ARC_R["i"], 0.14)
    arc_w, label_w = arc(node, normal, w, ARC_R["omega"], 0.15)
    motion = [
        project(
            staged_position(
                nu_from_u(np.array([u - MOTION_SPAN_DEG, u]), orbit), fractions, orbit
            ),
            view,
        )
        for u in MOTION_U_DEG
    ]
    return {
        "front": parts[0],
        "behind": parts[1],
        "sky": _sky_disk(view),
        "node_line": project(np.array([-SKY_SIZE * node, SKY_SIZE * node]), view),
        "north_ref": project(np.array([np.zeros(3), NORTH_REF_LEN * NORTH]), view),
        "arc_Omega": arc_om,
        "arc_i": arc_i,
        "arc_omega": arc_w,
        "label_Omega": label_om,
        "label_i": label_i,
        "label_omega": label_w,
        "north": project(AXIS_LEN * NORTH, view),
        "east": project(AXIS_LEN * EAST, view),
        "toward": project(AXIS_LEN * TOWARD_OBSERVER, view),
        "h": project(H_LEN * normal, view),
        "peri": project(staged_position(0.0, fractions, orbit), view),
        "asc": project(staged_position(-w, fractions, orbit), view),
        "desc": project(staged_position(np.pi - w, fractions, orbit), view),
        "motion": motion,
    }


def angle_readout(step, orbit=ORBIT):
    """The printed value of the angle a step applies, e.g. ``Omega = 130 deg``."""
    key, symbol = (
        ("Omega_deg", r"\Omega"),
        ("i_deg", "i"),
        ("omega_deg", r"\omega_p"),
    )[step]
    return rf"${symbol}={orbit[key]:.0f}\degree$"


STEP_ACTIONS = (
    r"turn about $+\hat{\mathbf{Z}}$, north toward east",
    "tilt about the node line",
    r"turn about $\hat{\mathbf{h}}$, from the node",
)
# The clip's step list, as (text, whether it is mathtext).
CLIP_ACTIONS = (
    (STEP_ACTIONS[0], True),
    (
        "tilt about the node line; the ascending node\n"
        "(where the planet moves toward\nthe observer)",
        False,
    ),
    (STEP_ACTIONS[2], True),
)


def _backing(cast):
    """A plain backing box for a label over marks (no stroked halo)."""
    return {
        "boxstyle": "round,pad=0.12,rounding_size=0.2",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _plain(ax, xy, text, cast, *, color=None, box=True, **kw):
    """A label with no stroked halo; over marks it sits on a backing box."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", cast.layout.small_pt)
    kw.setdefault("zorder", 9)
    text_artist = ax.text(*xy, text, color=color or _note_ink(cast), **kw)
    if box:
        text_artist.set_bbox(_backing(cast))
    return text_artist


def _arc_plain(ax, pts, cast, *, gid):
    """An angle arc with its arrowhead at the measured end."""
    color = cast.neutral(0.85)
    ax.plot(*pts.T, color=color, lw=cast.layout.lw, zorder=6, gid=gid)
    ax.add_patch(
        FancyArrowPatch(
            pts[-4],
            pts[-1],
            arrowstyle="-|>",
            mutation_scale=cast.layout.marker_pt * 1.4,
            color=color,
            lw=cast.layout.lw,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
            gid=f"{gid}-head",
        )
    )


def _rotation_panel(ax, step, cast):
    """One step: the orbit before the turn (dotted) and after it."""
    layout = cast.layout
    view = STEP_VIEWS[step]
    omega_r = SKY_OMEGA_R if step == 0 else None
    geo = rotation_geometry(STEP_FRACTIONS[step + 1], view, omega_r=omega_r)
    ghost = rotation_geometry(STEP_FRACTIONS[step], view)
    scen, ink = cast["scenery"].color, _scen_ink(cast)
    lw, ms = layout.lw, 1.4 * layout.marker_pt
    tag = f"d02-rot{step + 1}"
    ex.region(ax, "reference_plane", Polygon(geo["sky"]), cast)
    for key in ("front", "behind"):
        ax.plot(
            *ghost[key].T,
            color=cast.neutral(0.5),
            lw=0.8 * lw,
            ls=":",
            zorder=2,
            gid=f"{tag}-before-{key}",
        )
    color = cast["planet"].color
    ax.plot(*geo["front"].T, color=color, lw=1.3 * lw, zorder=3, gid=f"{tag}-front")
    ax.plot(
        *geo["behind"].T,
        color=color,
        lw=0.9 * lw,
        ls=(0, (3, 2.5)),
        zorder=2,
        gid=f"{tag}-behind",
    )
    ax.plot(*geo["node_line"].T, color=scen, lw=0.8 * lw, zorder=2, gid=f"{tag}-nodes")
    for k, (tail, head) in enumerate(geo["motion"]):
        ax.add_patch(
            FancyArrowPatch(
                tail,
                head,
                arrowstyle="-|>",
                mutation_scale=layout.marker_pt * 1.9,
                color=color,
                lw=0.01,
                shrinkA=0,
                shrinkB=0,
                zorder=5,
                gid=f"{tag}-motion-{k}",
            )
        )
    ex.mark(ax, "star", (0, 0), cast, scale=0.8)
    arc_ink = cast.neutral(0.85)
    if step == 0:
        ax.plot(*geo["north_ref"].T, color=scen, lw=0.7 * lw, ls="--", zorder=2)
        _arc_plain(ax, geo["arc_Omega"], cast, gid=f"{tag}-arc-Omega")
        _plain(
            ax,
            geo["label_Omega"],
            r"$\Omega$",
            cast,
            color=arc_ink,
            fontsize=layout.font_pt,
        )
        _point(ax, geo["peri"], "D", cast, size=0.75 * layout.marker_pt)
        _plain(
            ax,
            geo["peri"] + np.array([0.1, -0.06]),
            "periapsis",
            cast,
            ha="left",
            va="top",
        )
        end = geo["node_line"][1]
        _plain(
            ax,
            end + np.array([-0.02, -0.04]),
            "node line",
            cast,
            color=ink,
            ha="right",
            va="top",
            box=False,
        )
        # A compass on this sky panel: north up, east left.
        origin = np.array([1.25, -1.2])
        for vec, text in ((NORTH, "N"), (EAST, "E")):
            d = project(vec, view)
            _vector(ax, origin, origin + 0.34 * d, ink, 0.8 * lw, ms=ms)
            _plain(ax, origin + 0.46 * d, text, cast, color=ink, box=False)
    else:
        for key, text in (("north", "N"), ("toward", r"$\hat{\mathbf{Z}}$")):
            tip = geo[key]
            _vector(ax, (0, 0), tip, scen, 0.9 * lw, ms=ms)
            _plain(
                ax,
                tip + 0.1 * tip / np.linalg.norm(tip),
                text,
                cast,
                color=ink,
                box=False,
            )
        _vector(
            ax,
            (0, 0),
            geo["h"],
            cast.text,
            1.5 * lw,
            ms=1.8 * layout.marker_pt,
            zorder=5,
            gid=f"{tag}-h",
        )
        _plain(
            ax,
            geo["h"] + np.array([0.05, 0.03]),
            r"$\hat{\mathbf{h}}$",
            cast,
            color=cast.text,
            ha="left",
            va="bottom",
            fontsize=layout.font_pt,
            box=False,
        )
        _point(ax, geo["asc"], "^", cast)
        _point(ax, geo["desc"], "v", cast, filled=False)
        _plain(
            ax,
            geo["desc"]
            + (np.array([0.1, 0.02]) if step == 1 else np.array([0.06, 0.1])),
            "descending\nnode" if step == 1 else "descending node",
            cast,
            ha="left",
            va="bottom",
        )
        if step == 1:
            # Periapsis still lies at the ascending node until the last turn.
            peri = _point(ax, geo["peri"], "D", cast, size=0.55 * layout.marker_pt)
            peri.set_markeredgecolor(cast.background)
            peri.set_zorder(8)
            _arc_plain(ax, geo["arc_i"], cast, gid=f"{tag}-arc-i")
            _plain(
                ax, geo["label_i"], r"$i$", cast, color=arc_ink, fontsize=layout.font_pt
            )
            _plain(
                ax,
                geo["asc"] + np.array([0.0, -0.16]),
                "ascending node (the planet\nmoves toward the observer),\nstill at periapsis",
                cast,
                va="top",
            )
        else:
            ax.plot(
                *np.array([[0.0, 0.0], geo["peri"]]).T,
                color=scen,
                lw=0.7 * lw,
                zorder=2,
            )
            _arc_plain(ax, geo["arc_omega"], cast, gid=f"{tag}-arc-omega")
            _plain(
                ax,
                geo["label_omega"],
                r"$\omega_p$",
                cast,
                color=arc_ink,
                fontsize=layout.font_pt,
            )
            _point(ax, geo["peri"], "D", cast, size=0.75 * layout.marker_pt)
            _plain(
                ax,
                geo["peri"] + np.array([0.1, -0.06]),
                "periapsis",
                cast,
                ha="left",
                va="top",
            )
            _plain(
                ax,
                geo["asc"] + np.array([0.0, -0.13]),
                "ascending node",
                cast,
                va="top",
            )
    ax.set_title(
        f"{step + 1}. {ROTATION_NAMES[step]}, {angle_readout(step)}\n{STEP_ACTIONS[step]}",
        fontsize=layout.font_pt,
        color=cast.text,
        linespacing=1.35,
    )
    _finish(ax, *ROTATION_LIMITS)
    return geo


def build_three_rotations(layout, cast):
    fig, axes = ex.figure(layout, doc_height_in=3.6, ncols=3)
    lines = [
        r"$\mathbf{r}=R_Z(\Omega)\,R_X(i)\,R_Z(\omega_p)\,"
        r"[r\cos\nu,\ r\sin\nu,\ 0]^{\mathsf{T}}$",
        "one factor per panel, left to right; dotted: the orbit before the turn",
    ]
    if layout.is_slide:
        lines.insert(0, "How do three rotations place the orbit?")
    fig.suptitle(
        "\n".join(lines), x=0.01, ha="left", fontsize=layout.font_pt, color=cast.text
    )
    fig.text(
        0.99,
        0.985,
        f"{PROFILE}\norbit to scale",
        ha="right",
        va="top",
        fontsize=layout.small_pt,
        fontstyle="italic",
        color=cast["annotation"].color,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": cast.background,
            "edgecolor": cast["scenery"].color,
            "lw": 0.6 * layout.lw,
            "ls": "--",
        },
        gid="d02-rot-badge",
    )
    for step, ax in enumerate(axes):
        _rotation_panel(ax, step, cast)
    views = ("observer's view", "oblique view", "oblique view")
    for ax, text in zip(axes, views, strict=True):
        _plain(
            ax,
            (0.0, 0.0),
            text,
            cast,
            box=False,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontstyle="italic",
        )
    return fig


# Talk clip: the three rotations played as transformations (Manim)
#
# The clip draws the states of the strip, from ``rotation_geometry``, with the
# chapter's own cameras and projection; Manim only moves them. Every frame is
# set from one state, interpolated between the beats below: the applied
# fractions, the camera, the Omega arc radius, the weight of each group of
# marks ("show"), the factor now being applied ("hl") and the factors already
# applied ("lit").

# Seconds of each beat: a number is scaled by the venue's pace; a turn and
# the final hold have their own lengths, so the documentation clip keeps its
# rotations slow enough to read while its other beats are brisk.
CLIP_PACE = {"doc": 0.6, "slide": 1.0}
CLIP_TURN_S = {"doc": 2.6, "slide": 3.2}
CLIP_FINAL_S = {"doc": 3.2, "slide": 3.5}
CLIP_SECTIONS = ROTATION_STEPS
CLIP_BEATS = (
    ("omega", 0.8, {}),
    (
        "omega",
        0.6,
        {
            "show": {
                "node_line": 1,
                "node_label": 1,
                "north_ref": 1,
                "ghost": 1,
                "step0": 1,
            },
            "hl": (1, 0, 0),
        },
    ),
    ("omega", "turn", {"f": (1, 0, 0), "show": {"arc_Omega": 1}}),
    (
        "omega",
        0.5,
        {
            "show": {"val0": 1, "ghost": 0, "north_ref": 0},
            "hl": (0, 0, 0),
            "lit": (1, 0, 0),
        },
    ),
    ("omega", 0.8, {}),
    (
        "inclination",
        1.8,
        {
            "view": OBLIQUE_VIEW,
            "omega_r": ARC_R["Omega"],
            "ghost_f": (1, 0, 0),
            "show": {"toward": 1, "view_face": 0, "node_label": 0},
        },
    ),
    (
        "inclination",
        0.6,
        {
            "show": {"ghost": 1, "step1": 1, "h": 1, "view_oblique": 1},
            "hl": (0, 1, 0),
        },
    ),
    ("inclination", "turn", {"f": (1, 1, 0), "show": {"arc_i": 1}}),
    (
        "inclination",
        0.5,
        {
            "show": {"val1": 1, "nodes": 1, "ghost": 0},
            "hl": (0, 0, 0),
            "lit": (1, 1, 0),
        },
    ),
    ("inclination", 0.8, {}),
    (
        "periapsis",
        0.6,
        {
            "ghost_f": (1, 1, 0),
            "show": {"ghost": 1, "step2": 1},
            "hl": (0, 0, 1),
        },
    ),
    (
        "periapsis",
        "turn",
        {"f": (1, 1, 1), "show": {"arc_omega": 1, "peri_line": 1}},
    ),
    (
        "periapsis",
        0.6,
        {
            "show": {"val2": 1, "ghost": 0, "desc_label": 1},
            "hl": (0, 0, 0),
            "lit": (1, 1, 1),
        },
    ),
    ("periapsis", "final", {}),
)
CLIP_GROUPS = (
    "sky",
    "orbit",
    "basis",
    "motion",
    "peri",
    "view_face",
    "node_label",
    "node_line",
    "north_ref",
    "ghost",
    "toward",
    "h",
    "arc_Omega",
    "arc_i",
    "arc_omega",
    "nodes",
    "desc_label",
    "peri_line",
    "view_oblique",
    "step0",
    "step1",
    "step2",
    "val0",
    "val1",
    "val2",
)
CLIP_VISIBLE_AT_START = ("sky", "orbit", "basis", "motion", "peri", "view_face")
CLIP_START = {
    "f": (0.0, 0.0, 0.0),
    "ghost_f": (0.0, 0.0, 0.0),
    "view": FACE_ON,
    "omega_r": SKY_OMEGA_R,
    "hl": (0.0, 0.0, 0.0),
    "lit": (0.0, 0.0, 0.0),
    "show": {g: float(g in CLIP_VISIBLE_AT_START) for g in CLIP_GROUPS},
}


def clip_states(start=CLIP_START, beats=CLIP_BEATS):
    """The state before and after every beat, starting from ``start``.

    Returns:
        A list of ``(section, seconds, before, after)``.
    """
    out, state = [], {**start, "show": dict(start["show"])}
    for section, seconds, change in beats:
        after = {**state, "show": {**state["show"], **change.get("show", {})}}
        for key in ("f", "ghost_f", "view", "hl", "lit"):
            if key in change:
                after[key] = tuple(float(v) for v in change[key])
        if "omega_r" in change:
            after["omega_r"] = float(change["omega_r"])
        out.append((section, seconds, state, after))
        state = after
    return out


def mix_state(a, b, t):
    """The state a fraction ``t`` of the way from ``a`` to ``b``.

    Every numeric field is interpolated. ``ghost_f`` takes its new value at
    once; the beats change it only while the ghost is hidden.
    """

    def lerp(x, y):
        return tuple((1.0 - t) * np.asarray(x, float) + t * np.asarray(y, float))

    return {
        "f": lerp(a["f"], b["f"]),
        "ghost_f": b["ghost_f"],
        "view": lerp(a["view"], b["view"]),
        "omega_r": (1.0 - t) * a["omega_r"] + t * b["omega_r"],
        "hl": lerp(a["hl"], b["hl"]),
        "lit": lerp(a["lit"], b["lit"]),
        "show": {k: (1.0 - t) * a["show"][k] + t * b["show"][k] for k in a["show"]},
    }


def beat_seconds(seconds, venue_name):
    """Length of one beat in a venue: a turn, the final hold, or paced."""
    if seconds == "turn":
        return CLIP_TURN_S[venue_name]
    if seconds == "final":
        return CLIP_FINAL_S[venue_name]
    return CLIP_PACE[venue_name] * seconds


def clip_seconds(venue_name):
    """Running time of the clip in one venue, before frame rounding."""
    return sum(beat_seconds(seconds, venue_name) for _, seconds, _ in CLIP_BEATS)


def three_rotations_scene():
    """The Manim scene class of the three-rotations clip (imports Manim)."""
    import manim

    from explainers import _manim as em

    class ThreeRotations(em.ExplainerScene):
        """R_Z(Omega), R_X(i) and R_Z(omega_p) applied to the orbit in turn."""

        def construct(self):
            s = self.style
            venue, cast, layout = s.venue, s.cast, s.venue.layout
            slide = layout.is_slide
            origin = np.array([-3.15, -0.45]) if slide else np.array([-3.65, -0.35])
            scale = 2.05 if slide else 2.2
            lw, mk = layout.lw, layout.marker_pt
            small, body = layout.small_pt, layout.font_pt
            scen, ink, note_ink = (
                cast["scenery"].color,
                _scen_ink(cast),
                _note_ink(cast),
            )
            arc_ink, planet = cast.neutral(0.85), cast["planet"].color
            items = []  # (mobject, group, stroke opacity, fill opacity)
            boxed = []  # (label, backing box)

            def at(xy):
                return origin + scale * np.asarray(xy, dtype=float)

            def keep(mob, group, stroke=None, fill=None):
                stroke = mob.get_stroke_opacity() if stroke is None else stroke
                fill = mob.get_fill_opacity() if fill is None else fill
                items.append((mob, group, stroke, fill))
                return mob

            def poly(group, **kw):
                return keep(em.Poly(s, **kw), group)

            def marker(group, shape, size_pt, *, filled=True):
                r = venue.units(0.5 * size_pt)
                offsets = {
                    "D": np.array([[1.2, 0], [0, 1.2], [-1.2, 0], [0, -1.2]]),
                    "^": np.array([[0, 1.2], [-1, -0.7], [1, -0.7]]),
                    "v": np.array([[0, -1.2], [-1, 0.7], [1, 0.7]]),
                }[shape]
                mob = em.Poly(
                    s,
                    color=scen,
                    width_pt=0.8 * lw,
                    fill=scen if filled else cast.background,
                    fill_opacity=1.0,
                )
                mob.offsets = r * offsets
                return keep(mob, group)

            def label(group, string, pt, *, color, italic=False, math=True, box=False):
                if math:
                    mob = em.MathLabel(s, string, pt, color=color, italic=italic)
                else:
                    mob = em.text(s, string, pt, color=color, italic=italic)
                if box:
                    back = em.Poly(
                        s,
                        color=cast.background,
                        width_pt=lw,
                        fill=cast.background,
                        fill_opacity=0.85,
                    )
                    boxed.append((mob, keep(back, group, stroke=0.0, fill=0.85)))
                return keep(mob, group, stroke=0.0, fill=1.0)

            # The orbit in space, back to front.
            rp = cast["reference_plane"]
            # A dashed outline cannot carry a fill, so the shaded disk and its
            # dash-dot rim are two polylines.
            sky_fill = poly(
                "sky",
                color=rp.color,
                width_pt=rp.lw,
                fill=rp.color,
                fill_opacity=rp.fill_alpha,
            )
            keep(sky_fill.set_stroke(opacity=0.0), "sky", stroke=0.0)
            sky = poly(
                "sky",
                color=rp.color,
                width_pt=rp.lw,
                dash_pt=tuple(v * rp.lw for v in (6.4, 1.6, 1.0, 1.6)),
            )
            ghost = [
                poly(
                    "ghost",
                    color=cast.neutral(0.5),
                    width_pt=0.8 * lw,
                    dash_pt=(0.8 * lw, 1.3 * lw),
                )
                for _ in range(2)
            ]
            behind = poly(
                "orbit", color=planet, width_pt=0.9 * lw, dash_pt=(2.7 * lw, 2.25 * lw)
            )
            node_line = poly("node_line", color=scen, width_pt=0.8 * lw)
            north_ref = poly(
                "north_ref", color=scen, width_pt=0.7 * lw, dash_pt=(2.6 * lw, 1.1 * lw)
            )
            peri_line = poly("peri_line", color=scen, width_pt=0.7 * lw)
            basis = {}
            for key, group in (
                ("north", "basis"),
                ("east", "basis"),
                ("toward", "toward"),
            ):
                vec = em.Vector(s, color=scen, width_pt=0.9 * lw, head_pt=0.7 * mk)
                keep(vec.shaft, group)
                keep(vec.head, group)
                basis[key] = vec
            front = poly("orbit", color=planet, width_pt=1.3 * lw)
            h_vec = em.Vector(s, color=cast.text, width_pt=1.5 * lw, head_pt=0.9 * mk)
            keep(h_vec.shaft, "h")
            keep(h_vec.head, "h")
            arcs = {}
            for key in ("Omega", "i", "omega"):
                arc = poly(f"arc_{key}", color=arc_ink, width_pt=lw)
                head = em.Head(s, color=arc_ink, size_pt=0.63 * mk, width_pt=0.6 * lw)
                arcs[key] = (arc, keep(head, f"arc_{key}"))
            motion = []
            for _ in MOTION_U_DEG:
                head = em.Head(s, color=planet, size_pt=0.76 * mk, width_pt=0.6 * lw)
                head.head_wid = venue.units(0.76 * mk)
                motion.append(keep(head, "motion"))
            peri = marker("peri", "D", 0.75 * mk)
            asc = marker("nodes", "^", 0.9 * mk)
            desc = marker("nodes", "v", 0.9 * mk, filled=False)
            star = manim.Star(
                n=5,
                outer_radius=venue.units(0.8 * mk),
                density=2,
                color=s.entity("star"),
                fill_opacity=1.0,
                stroke_width=0.0,
            ).move_to([*origin, 0.0])
            keep(star, "sky", stroke=0.0, fill=1.0)

            # Labels on the geometry.
            axis_pt = 1.15 * small
            basis_labels = {
                "north": label(
                    "basis", r"north $\hat{\mathbf{X}}$", axis_pt, color=ink, box=True
                ),
                "east": label(
                    "basis", r"east $\hat{\mathbf{Y}}$", axis_pt, color=ink, box=True
                ),
                "toward": label(
                    "toward",
                    r"toward observer $\hat{\mathbf{Z}}$",
                    axis_pt,
                    color=ink,
                    box=True,
                ),
            }
            arc_labels = {
                key: label(f"arc_{key}", tex, body, color=arc_ink, box=True)
                for key, tex in (
                    ("Omega", r"$\Omega$"),
                    ("i", r"$i$"),
                    ("omega", r"$\omega_p$"),
                )
            }
            h_label = label("h", r"$\hat{\mathbf{h}}$", body, color=cast.text, box=True)
            peri_label = label("peri", "periapsis", small, color=note_ink, math=False)
            node_label = label(
                "node_label", "node line", small, color=ink, math=False, box=True
            )
            asc_label = label(
                "nodes", "ascending node", small, color=note_ink, math=False, box=True
            )
            desc_label = label(
                "desc_label",
                "descending node",
                small,
                color=note_ink,
                math=False,
                box=True,
            )
            sky_label = label(
                "sky", "sky plane", small, color=ink, italic=True, math=False
            )

            # The equation, the steps and the notes, in the right-hand column.
            x0 = 1.0 if slide else 0.9
            top = 2.3 if slide else 2.95
            eq_pt = 1.2 * body
            dim, bright = s.neutral(0.45), s.text_color
            row = em.math_row(s, [r"$\mathbf{r}=$", *ROTATION_NAMES], eq_pt, gap_em=0.3)
            row.shift(np.array([x0, top, 0.0]))
            vec_row = em.math_row(
                s, [r"$\times\,[r\cos\nu,\ r\sin\nu,\ 0]^{\mathsf{T}}$"], eq_pt
            )
            vec_row.shift(
                np.array([row[1].get_left()[0], top - venue.units(1.9 * eq_pt), 0.0])
            )
            keep(row[0], "sky", stroke=0.0, fill=1.0)
            keep(vec_row[0], "sky", stroke=0.0, fill=1.0)
            factors = list(row[1:])
            unders = []
            for k, piece in enumerate(factors):
                keep(piece, "sky", stroke=0.0, fill=1.0)
                bar = em.Poly(s, color=cast.text, width_pt=1.2 * lw)
                y = piece.get_bottom()[1] - venue.units(0.3 * eq_pt)
                bar.redraw(
                    np.array([[piece.get_left()[0], y], [piece.get_right()[0], y]])
                )
                unders.append(keep(bar, f"under{k}"))
            step_y = top - venue.units(3.7 * eq_pt)
            step_pt, action_pt = 1.15 * body, 1.15 * small
            steps = []
            for k in range(3):
                head = label(
                    f"step{k}",
                    f"{k + 1}. {ROTATION_NAMES[k]}",
                    step_pt,
                    color=cast.text,
                )
                em.place(head, (x0, step_y), "left", "top")
                value = label(f"val{k}", angle_readout(k), step_pt, color=cast.text)
                em.place(
                    value,
                    (
                        head.get_right()[0] + venue.units(0.9 * step_pt),
                        head.get_center()[1],
                    ),
                    "left",
                    "center",
                )
                text, math = CLIP_ACTIONS[k]
                action = label(f"step{k}", text, action_pt, color=note_ink, math=math)
                em.place(
                    action,
                    (
                        x0 + venue.units(1.2 * step_pt),
                        head.get_bottom()[1] - venue.units(0.45 * step_pt),
                    ),
                    "left",
                    "top",
                )
                steps += [head, value, action]
                step_y = action.get_bottom()[1] - venue.units(0.75 * step_pt)
            annotation = cast["annotation"].color
            legend = label(
                "sky",
                "dotted: the orbit before the turn\ndashed: behind the sky plane",
                small,
                color=annotation,
                italic=True,
                math=False,
            )
            em.place(legend, (x0, -3.45), "left", "bottom")
            views = [
                label(
                    "view_face",
                    r"observer's view from $+\hat{\mathbf{Z}}$: north up, east left",
                    small,
                    color=annotation,
                    italic=True,
                ),
                label(
                    "view_oblique",
                    OBLIQUE_NOTE,
                    small,
                    color=annotation,
                    italic=True,
                    math=False,
                ),
            ]
            for mob in views:
                em.place(
                    mob,
                    (x0, legend.get_top()[1] + venue.units(0.6 * small)),
                    "left",
                    "bottom",
                )
            badge = em.badge(s, f"{PROFILE}; orbit to scale", corner=manim.UL)
            extras = [badge]
            if slide:
                headline = em.text(
                    s, "How do three rotations place the orbit?", layout.title_pt
                )
                em.place(
                    headline, (venue.frame_width / 2 - 0.25, 3.45), "right", "center"
                )
                extras.append(headline)

            def place_radial(mob, xy, dist, toward=None):
                d = (
                    np.asarray(xy, float)
                    if toward is None
                    else np.asarray(toward, float)
                )
                n = np.hypot(*d)
                u = d / n if n > 1e-9 else np.array([0.0, -1.0])
                ha = "left" if u[0] > 0.35 else ("right" if u[0] < -0.35 else "center")
                va = "bottom" if u[1] > 0.35 else ("top" if u[1] < -0.35 else "center")
                em.place(mob, at(np.asarray(xy) + dist * u), ha, va)

            pad = venue.units(0.15 * small)

            def apply(state):
                geo = rotation_geometry(
                    state["f"], state["view"], omega_r=state["omega_r"]
                )
                old = rotation_geometry(state["ghost_f"], state["view"])
                sky.redraw(at(geo["sky"]), closed=True)
                sky_fill.redraw(at(geo["sky"]), closed=True)
                ghost[0].redraw(at(old["front"]))
                ghost[1].redraw(at(old["behind"]))
                behind.redraw(at(geo["behind"]))
                front.redraw(at(geo["front"]))
                node_line.redraw(at(geo["node_line"]))
                north_ref.redraw(at(geo["north_ref"]))
                peri_line.redraw(at(np.array([[0.0, 0.0], geo["peri"]])))
                for key, vec in basis.items():
                    vec.redraw(at((0.0, 0.0)), at(geo[key]))
                toward = (0.1, -1.0) if state["view"][0] < 80.0 else None
                place_radial(basis_labels["north"], geo["north"], 0.07)
                place_radial(basis_labels["east"], geo["east"], 0.07)
                place_radial(basis_labels["toward"], geo["toward"], 0.1, toward=toward)
                h_vec.redraw(at((0.0, 0.0)), at(geo["h"]))
                place_radial(h_label, geo["h"], 0.06, toward=(1.0, 0.3))
                for key, (arc, head) in arcs.items():
                    pts = geo[f"arc_{key}"]
                    if np.hypot(*(pts[-1] - pts[0])) < 1e-3:
                        arc.clear_points()
                        head.clear_points()
                    else:
                        arc.redraw(at(pts))
                        head.redraw(at(pts[-1]), pts[-1] - pts[-4])
                    em.place(arc_labels[key], at(geo[f"label_{key}"]))
                for head, (tail, tip) in zip(motion, geo["motion"], strict=True):
                    head.redraw(at(tip), tip - tail)
                for mob, key in ((peri, "peri"), (asc, "asc"), (desc, "desc")):
                    mob.redraw(at(geo[key]) + mob.offsets, closed=True)
                # Radially out in the observer's view; to the right in the
                # oblique view, where periapsis starts on the ascending node.
                g = np.clip(
                    (state["view"][0] - OBLIQUE_VIEW[0]) / (90.0 - OBLIQUE_VIEW[0]),
                    0.0,
                    1.0,
                )
                out = geo["peri"] / np.hypot(*geo["peri"])
                # Off the node line and the north axis: outward, turned a little
                # counterclockwise.
                out = out + 0.7 * np.array([-out[1], out[0]])
                place_radial(
                    peri_label,
                    geo["peri"],
                    0.1,
                    toward=g * out + (1.0 - g) * np.array([1.0, 0.0]),
                )
                place_radial(asc_label, geo["asc"], 0.1, toward=(0.0, -1.0))
                place_radial(desc_label, geo["desc"], 0.1, toward=(1.0, 0.45))
                far = geo["node_line"][0]
                place_radial(node_label, 0.8 * far, 0.08, toward=(-far[1], far[0]))
                # The sky-plane label sits outside the rim, south-west in the
                # observer's view and south in the oblique view, clear of the orbit.
                g = np.clip(
                    (state["view"][0] - OBLIQUE_VIEW[0]) / (90.0 - OBLIQUE_VIEW[0]),
                    0,
                    1,
                )
                theta = np.deg2rad(180.0 + 45.0 * g)
                rim = SKY_SIZE * (np.cos(theta) * NORTH + np.sin(theta) * EAST)
                place_radial(sky_label, project(rim, state["view"]), 0.04)
                for mob, back in boxed:
                    (x0_, y0_), (x1_, y1_) = (
                        mob.get_corner(manim.DL)[:2],
                        mob.get_corner(manim.UR)[:2],
                    )
                    back.redraw(
                        np.array(
                            [
                                [x0_ - pad, y0_ - pad],
                                [x1_ + pad, y0_ - pad],
                                [x1_ + pad, y1_ + pad],
                                [x0_ - pad, y1_ + pad],
                            ]
                        ),
                        closed=True,
                    )
                for k, piece in enumerate(factors):
                    w = min(1.0, state["hl"][k] + state["lit"][k])
                    piece.set_fill(manim.interpolate_color(dim, bright, w))
                weights = {
                    **state["show"],
                    **{f"under{k}": state["hl"][k] for k in range(3)},
                }
                for mob, group, stroke, fill in items:
                    weight = weights.get(group, 1.0)
                    mob.set_stroke(opacity=stroke * weight)
                    mob.set_fill(opacity=fill * weight, family=True)

            layers = [sky_fill, sky, *ghost, behind, node_line, north_ref, peri_line]
            layers += [m for v in basis.values() for m in (v.shaft, v.head)]
            layers += [front, h_vec.shaft, h_vec.head]
            layers += [m for pair in arcs.values() for m in pair]
            layers += [*motion, peri, asc, desc, star]
            layers += [back for _, back in boxed]
            layers += [*basis_labels.values(), *arc_labels.values(), h_label]
            layers += [peri_label, node_label, asc_label, desc_label, sky_label]
            layers += [row[0], *factors, vec_row[0], *unders, *steps, legend, *views]
            rig = manim.VGroup(*layers)
            self.add(rig, *extras)
            beats = clip_states()
            apply(beats[0][2])
            section = None
            for name, seconds, before, after in beats:
                if name != section:
                    self.section(name)
                    section = name
                run = beat_seconds(seconds, venue.name)
                if before == after:
                    self.wait(run)
                    continue
                self.play(
                    manim.UpdateFromAlphaFunc(
                        rig, lambda _m, t, a=before, b=after: apply(mix_state(a, b, t))
                    ),
                    run_time=run,
                    rate_func=manim.smooth,
                )

    return ThreeRotations


# Registry

_PARAMS = {"orbit": ORBIT, "side": SIDE, "oblique_view": OBLIQUE_VIEW}

CONSTRUCTION_CAPTION = (
    "Where the illumination angle and the observer-axis angle are measured "
    "({ref}`geometry-illumination`). Side view in the plane that contains the "
    "star, the planet and the line of sight, with the star at the origin and "
    r"$+Z$ toward the observer as in the proposed observer profile. The vertical "
    "is an unnamed sky-plane direction, so the drawing does not choose a "
    r"north/east ordering. Starlight travels along $\mathbf r$; "
    r"$Z=\mathbf r\cdot\hat{\mathbf o}$ is the line-of-sight coordinate and "
    r"$\rho$ the projected separation in the sky plane. The observer-axis angle "
    r"$\beta_{\rm axis}=\operatorname{atan2}(\rho,Z)$ is measured at the star from "
    r"$\hat{\mathbf o}$. The illumination angle $\alpha$ is measured at the planet, "
    r"between the direction back to the star and the direction "
    r"$\mathbf R_{\rm obs}-\mathbf r$ to the observer "
    "({ref}`Savransky et al. 2019, Sec. 2.1, eq. 3 <source-savransky2019>`). "
    "For a point-source star and a distant observer that direction is parallel "
    r"to $\hat{\mathbf o}$, and $\alpha=\pi-\beta_{\rm axis}$ is an identity under "
    "those assumptions. The hemisphere facing the star is lit. Schematic, not to "
    "scale; the break marks the omitted observer distance."
)
CONSTRUCTION_ALT = (
    "Side-view diagram. A star sits at the origin on a horizontal line of sight "
    "that runs right to a distant observer, drawn as an aperture beyond a scale "
    "break; a vertical dash-dot line through the star is the sky plane seen "
    "edge-on. A yellow ray labeled r runs up and to the right from the star and "
    "stops just short of a planet whose star-facing half is colored and whose "
    "other half is gray. A cyan ray leaves the planet toward the observer, "
    "parallel to the unit vector o-hat. Dotted lines drop from the planet to the "
    "line of sight and to the sky plane; dimension bars mark Z along the line "
    "of sight and rho along the sky plane. An arc at the star marks beta-axis "
    "from o-hat to r, and an arc at the planet marks alpha from the observer "
    "direction back to the star, with the relation alpha equals pi minus "
    "beta-axis."
)

ELEMENTS_CAPTION = (
    "An inclined eccentric orbit in the proposed observer profile "
    "({ref}`geometry-observer-basis`), a candidate that is pending at the "
    "conventions stage. Left, an oblique view (camera turned 15 degrees from "
    "east toward the observer and raised 15 degrees toward north): the "
    r"right-handed basis $(\hat{\mathbf X},\hat{\mathbf Y},\hat{\mathbf Z})$ = "
    "(north, east, toward observer) with the sky plane through the star. The "
    "ascending node is the sky-plane crossing where the planet moves toward the "
    r"observer, $\dot Z>0$. $\Omega$ is measured in the sky plane from north "
    r"toward east to the node; $i$ is the angle from $+\hat{\mathbf Z}$ to the "
    r"angular momentum direction $\hat{\mathbf h}$; $\omega_p$ is measured in "
    "the orbit plane from the node toward periapsis in the direction of motion, "
    r"so that $\mathbf r=R_Z(\Omega)R_X(i)R_Z(\omega_p)"
    r"[r\cos\nu,\ r\sin\nu,\ 0]^{\mathsf T}$. Right, the same orbit as the "
    r"observer sees it from $+\hat{\mathbf Z}$, north up and east to the left, "
    r"where $\Omega$ runs counterclockwise on the image from north to the "
    r"ascending node. Exchanged offsets are ordered $(\xi,\eta)$ = (east, north), "
    "which is not the dynamical axis order. The observer direction, rotation "
    "order and node agree with {ref}`Savransky et al. (2019), Sec. 2.1 and "
    r"Fig. 1 <source-savransky2019>`; placing east along $\hat{\mathbf Y}$ is "
    "this chapter's reading, which one sentence of that paper contradicts, and "
    "it awaits the three-axis basis transform fixture "
    "({ref}`geometry-savransky-profile`). The drawing is computed from the "
    "chapter's formulas, not from orbix. The current orbix projection applies "
    "the same rotation but labels its first rotated component RA (east) and its "
    "second Dec (north), so for the same elements the tutorial's sky plot is "
    "this panel mirrored about the north-east diagonal: the node appears at "
    r"position angle $90^\circ-\Omega$ and the apparent motion is reversed. A "
    "reflection is not a rotation, so relabeling cannot migrate fitted elements "
    "between the two profiles (the migration rule in "
    r"{ref}`geometry-savransky-profile`). Orbit to scale with $e=0.35$, "
    r"$i=55^\circ$, $\Omega=130^\circ$, $\omega_p=70^\circ$; both panels share "
    "one page scale; the observer distance is not drawn."
)
ELEMENTS_ALT = (
    "Two panels. Left: an oblique view of a star at the center of a shaded "
    "elliptical disk labeled sky plane, with basis arrows labeled north X-hat "
    "(up), east Y-hat and toward observer Z-hat. A cyan eccentric orbit crosses "
    "the sky plane along a line of nodes; the part in front of the plane is "
    "solid and the part behind is dashed. A filled triangle marks the ascending "
    "node, an open triangle the descending node, a diamond periapsis, and cyan "
    "arrowheads the direction of motion. A heavier arrow labeled h-hat angular "
    "momentum rises from the star. Three arcs with arrowheads mark i from Z-hat "
    "to h-hat, Omega in the sky plane from north to the ascending node, and "
    "omega-p in the orbit plane from the node to periapsis; the three arcs have "
    "different radii, Omega outermost and omega-p innermost. Right: the same "
    "orbit seen by the "
    "observer, north up and east left, with the nodes and periapsis marked, an "
    "arc Omega from north counterclockwise to the ascending node, and a compass "
    "labeled N, eta and E, xi."
)

EPOCHS_CAPTION = (
    "Illumination follows the line-of-sight coordinate, not the projected "
    "separation ({ref}`geometry-illumination`), in the proposed observer "
    "profile, which is pending ({ref}`geometry-observer-basis`). The orbit of "
    r"the previous figure at three arguments of latitude $u=\nu+\omega_p$: "
    r"$270^\circ$ (behind the sky plane), $0^\circ$ (the ascending node, on the "
    r"sky plane) and $90^\circ$ (in front). Top: side view from the east, with "
    "the sky plane edge-on and the observer to the right; the dotted segment is "
    r"the line-of-sight coordinate $Z$. Bottom: the observer's view, north up "
    r"and east to the left, with the projected separation $\rho$ and the planet "
    "disk enlarged and lit as a Lambert sphere at the illumination angle "
    "(colored lit, gray unlit); dots mark 24 equal time steps per period. "
    r"Because $\cos\alpha=-Z/r$ and $Z=r\sin i\sin u$, the three states have "
    r"$\alpha=90^\circ-i$, $90^\circ$ and $90^\circ+i$, and a planet is at "
    r"quadrature exactly where it crosses the sky plane. $\Phi(\alpha)$ is the "
    "Lambert phase function ({ref}`Perryman 2018, Sec. 6.15.1, eq. 6.95 "
    "<source-perryman2018>`). Orbit to scale; planet disk enlarged."
)
EPOCHS_ALT = (
    "Six panels in two rows and three columns, titled behind the sky plane, on "
    "the sky plane (ascending node), and in front of the sky plane. The top row "
    "shows the orbit from the east with the sky plane as a vertical dash-dot "
    "line and, under it, an arrow to the observer and the sign of Z; the planet "
    "sits left of the plane, on it, and right of it, with a dotted segment "
    "marking Z. The bottom row shows the observer's view of the same orbit with "
    "equal-time dots, a line from the star to an enlarged planet disk, and the "
    "lit part of the disk in color with the unlit part gray: nearly full at "
    "alpha 35 degrees with Phi 0.84, half lit at alpha 90 degrees with Phi "
    "0.32, and a thin crescent at alpha 145 degrees with Phi 0.02."
)

SWEEP_CAPTION = (
    "The orbit of the orbit-elements figure and one planet position, seen by a "
    "camera that turns about the north axis only, from the side view from the "
    "east (sky plane edge-on, observer to the right) to the observer's own view "
    r"from $+\hat{\mathbf Z}$ (north up, east to the left). The orbit, the planet "
    "and the page scale are fixed; only the viewpoint changes, so the observer's "
    "view is a projection of one three-dimensional state. The inset shows the "
    "camera direction seen from north. Unlike the oblique still, this camera is "
    "not raised toward north, so at the start the east axis points at the "
    "camera. Proposed observer profile ({ref}`geometry-observer-basis`), "
    "pending; orbit to scale."
)
SWEEP_ALT = (
    "Animation. A camera turns 90 degrees about the north axis. At the start "
    "the sky plane is edge-on, the east axis points at the camera and the "
    "observer direction points right; at the end the sky plane is face-on, the "
    "observer direction points at the camera, north is up and east is left. "
    "The cyan orbit, its nodes, periapsis and the planet do not move in space. "
    "A small inset seen from north shows a square camera marker moving along a "
    "quarter circle from the east direction to the observer direction."
)

CLOCK_CAPTION = (
    "One orbital period at equal time steps, with both cameras fixed: the side "
    "view from the east (left) and the observer's view (right). As the planet "
    "moves, its line-of-sight coordinate, its projected offset and its "
    r"illumination angle change together. The planet is at quadrature, "
    r"$\alpha=90^\circ$, where it crosses the sky plane; the readouts give "
    r"$\alpha$ and the Lambert phase function $\Phi$ "
    "({ref}`geometry-illumination`; {ref}`Perryman 2018, Sec. 6.15.1, eq. 6.95 "
    r"<source-perryman2018>`). Time is counted from periapsis passage $t_p$ in "
    "units of the period. Proposed observer profile, pending; orbit to scale, "
    "planet disk enlarged."
)
CLOCK_ALT = (
    "Animation of one orbital period with fixed cameras. Left: side view from "
    "the east with the sky plane as a vertical dash-dot line; the planet moves "
    "around the tilted orbit, a dotted segment shows its distance Z from the "
    "sky plane, and a line under the sky plane says whether it is in front or "
    "behind. Right: the observer's view; an enlarged planet disk moves along "
    "the projected orbit past equal-time dots, its lit part changing from "
    "nearly full behind the sky plane to half lit at the node to a thin "
    "crescent in front. Readouts give the time since periapsis, alpha and Phi."
)

ROTATIONS_CAPTION = (
    "The orbit of the orbit-elements figure built one factor at a time, in the "
    "proposed observer profile ({ref}`geometry-observer-basis`), which is "
    "pending. Read left to right, "
    r"$\mathbf r=R_Z(\Omega)R_X(i)R_Z(\omega_p)[r\cos\nu,\ r\sin\nu,\ 0]^{\mathsf T}$"
    " is three active right-handed rotations, each about an axis that the "
    "rotations before it placed. Each panel applies one factor to the orbit "
    "left by the panel before it, which is drawn dotted. Left, in the "
    r"observer's view from $+\hat{\mathbf Z}$ (north up, east to the left): "
    r"$R_Z(\Omega)$ turns the ellipse about $+\hat{\mathbf Z}$, carrying the "
    r"line that becomes the node line from north toward east by $\Omega$. "
    "Middle, in the oblique view of the next figure: "
    r"$R_X(i)$ tilts the orbit by $i$ about the node line, so the orbit normal "
    r"$\hat{\mathbf h}$ leaves $+\hat{\mathbf Z}$ by $i$, and the crossing on "
    "the node line where the planet moves toward the observer, "
    r"$\dot Z>0$, is the ascending node. Until the last factor, periapsis lies "
    r"on the node line. Right: $R_Z(\omega_p)$ turns the orbit about "
    r"$\hat{\mathbf h}$ within its own plane and carries periapsis away from "
    r"the ascending node by $\omega_p$ in the direction of motion. Every panel "
    "is the chapter's formula with the later angles set to zero, so the strip "
    "is an identity of the rotation product. The order of the factors "
    "matters: the same three angles applied in another order place a "
    "different orbit. Read right to left, the same product is three turns "
    "about the fixed axes instead, and it gives the same orbit only because "
    "it keeps this order of factors. The "
    "rotation order and the node agree with {ref}`Savransky et al. (2019), "
    "Sec. 2.1 and Fig. 1 <source-savransky2019>`; placing east along "
    r"$\hat{\mathbf Y}$ awaits the fixture in "
    "{ref}`geometry-savransky-profile`. Orbit to scale with $e=0.35$, "
    r"$i=55^\circ$, $\Omega=130^\circ$, $\omega_p=70^\circ$; the three "
    "panels share one page scale; the observer distance is not drawn."
)
ROTATIONS_ALT = (
    "Three panels, each with a star at the center of a shaded sky-plane disk, "
    "a cyan eccentric orbit, and, dotted, the orbit before that panel's "
    "rotation. Left, titled R_Z(Omega), Omega = 130 degrees, in the observer's "
    "view with north up and east left: the orbit lies in the sky plane, a line "
    "through the star and periapsis is labeled node line, and an arc from a "
    "dashed north line turns counterclockwise to it; a compass marks N and E. "
    "Middle, titled R_X(i), i = 55 degrees, in an oblique view with arrows N "
    "and Z-hat: the orbit is tilted about the node line, with its part behind "
    "the sky plane dashed, a heavier arrow h-hat leaves Z-hat by an arc "
    "labeled i, a filled triangle marks the ascending node, labeled as where "
    "the planet moves toward the observer, with the periapsis diamond still on "
    "it, and an open triangle marks the descending node. Right, titled R_Z(omega-p), omega-p = 70 "
    "degrees, in the same oblique view: the orbit has turned within its tilted "
    "plane, an arc labeled omega-p runs from the node line to periapsis, "
    "marked by a diamond, and both nodes are labeled."
)
CLIP_CAPTION = (
    "The construction of the three-rotation figure played as motion, in the "
    "proposed observer profile ({ref}`geometry-observer-basis`), which is "
    r"pending. The orbit turns by $\Omega$ about $+\hat{\mathbf Z}$ in the "
    "observer's view, the camera turns to the oblique view, the orbit tilts by "
    r"$i$ about the node line, and it turns by $\omega_p$ about "
    r"$\hat{\mathbf h}$; the factor being applied is underlined in "
    r"$\mathbf r=R_Z(\Omega)R_X(i)R_Z(\omega_p)[r\cos\nu,\ r\sin\nu,\ 0]^{\mathsf T}$."
    " The clip ends on the geometry of the orbit-elements figure. Every frame "
    "is the chapter's formula with the angles partly applied, drawn with the "
    "projection of the stills. Orbit to scale; the observer distance is not "
    "drawn."
)
CLIP_ALT = (
    "Animation. A cyan eccentric orbit around a star turns about the line of "
    "sight in the observer's view, north up and east left, while an arc from "
    "north grows to 130 degrees. The view then turns to an oblique camera, the "
    "orbit tilts by 55 degrees about the node line while an arrow h-hat leaves "
    "the Z-hat axis, and the orbit finally turns by 70 degrees within its "
    "tilted plane, carrying periapsis away from the ascending node. Beside it, "
    "the equation r = R_Z(Omega) R_X(i) R_Z(omega-p) times the perifocal "
    "position underlines each factor as it is applied, and the three angle "
    "values appear one by one."
)

FIGURES = [
    ex.FigureSpec(
        slug="d02-observer-construction",
        build=build_construction,
        caption=CONSTRUCTION_CAPTION,
        alt=CONSTRUCTION_ALT,
        status=STATUS_ANGLES,
        params=_PARAMS,
    ),
    ex.FigureSpec(
        slug="d02-orbit-elements",
        build=build_elements,
        caption=ELEMENTS_CAPTION,
        alt=ELEMENTS_ALT,
        status=STATUS_ORBIT,
        params=_PARAMS,
    ),
    ex.FigureSpec(
        slug="d02-phase-epochs",
        build=build_epochs,
        caption=EPOCHS_CAPTION,
        alt=EPOCHS_ALT,
        status=STATUS_DISK,
        params={**_PARAMS, "epoch_u_deg": EPOCH_U_DEG, "n_ticks": N_TICKS},
    ),
    ex.FigureSpec(
        slug="d02-three-rotations",
        build=build_three_rotations,
        caption=ROTATIONS_CAPTION,
        alt=ROTATIONS_ALT,
        status=STATUS_ORBIT,
        params={**_PARAMS, "step_fractions": STEP_FRACTIONS},
    ),
]

ANIMATIONS = [
    ex.AnimationSpec(
        slug="d02-viewpoint-sweep",
        build=build_sweep,
        ground="viewpoint",
        caption=SWEEP_CAPTION,
        alt=SWEEP_ALT,
        status=STATUS_ORBIT,
        params={**_PARAMS, "sweep_u_deg": SWEEP_U_DEG},
        preview_frames=(0, 10, 29),
    ),
    ex.AnimationSpec(
        slug="d02-orbit-clock",
        build=build_clock,
        ground="narration",
        caption=CLOCK_CAPTION,
        alt=CLOCK_ALT,
        status=STATUS_DISK,
        params={**_PARAMS, "n_ticks": N_TICKS},
        preview_frames=(0, 9, 14, 20),
    ),
]

MANIM = [
    ManimSpec(
        slug="d02-three-rotations-clip",
        scene=three_rotations_scene,
        still="d02-three-rotations",
        ground="narration",
        caption=CLIP_CAPTION,
        alt=CLIP_ALT,
        status=STATUS_ORBIT,
        params={
            **_PARAMS,
            "pace": CLIP_PACE,
            "turn_s": CLIP_TURN_S,
            "final_s": CLIP_FINAL_S,
        },
        doc_max_s=16.0,
    ),
]
