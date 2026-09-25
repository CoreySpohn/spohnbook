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
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

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
