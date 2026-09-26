"""Barycenter, reflex motion, and the sign of the stellar radial velocity.

Every quantity drawn here is computed from the geometry chapter's own reflex
relations in the proposed observer profile, ``(X, Y, Z) = (north, east,
toward the observer)``, not from a library:

    r = r_p - r_star,   r_star = -Mp / (M* + Mp) r,   r_p = M* / (M* + Mp) r,
    v_r,star = -dZ_star/dt = Mp / (M* + Mp) dZ/dt.

The orbit is the chapter's worked signed example: a circular relative orbit
with ``i = 90 deg`` and ``Omega = omega_p = M_0 = 0``, so the orbit lies in
the north / toward-observer plane and the side view from the east shows it
face-on. The mass ratio is exaggerated on the page so the star's reflex
orbit is visible; the Earth/Sun amplitude printed beside the curve is
computed separately from hwoutils constants.

Geometry helpers (the chapter's rotation, orthographic cameras and the
lit-half outline of a planet disk) are imported from the orbit-geometry
explainer, so both diagrams place a body with one formula.

Figures:
    d11-barycenter-rv: three side views at the worked example's states
        (quadrature at t0, the planet at +Z, the planet at -Z) above the
        stellar radial-velocity curve with the three states marked (the
        still of d11-reflex-clock).

Animations:
    d11-reflex-clock (ground "rate"): the side view and the RV curve with
        orbital time running at equal steps over one period; the concept is
        a rate, the line-of-sight component of the star's velocity.

Artists that carry a convention (bodies, velocity vectors, the relative
vector, the RV curve and its state markers) have a ``gid`` so the tests can
check the drawn geometry, not only the helpers.
"""

import matplotlib.pyplot as plt
import numpy as np
from hwoutils import constants as const
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.text import Text

from explainers import _common as ex
from explainers.d02_orbit_geometry import lit_polygon, orbit_rotation, position, project

# Scientific inputs.
# The chapter's worked signed example (geometry-rv-worked-example).
ORBIT = {"a": 1.0, "e": 0.0, "i_deg": 90.0, "Omega_deg": 0.0, "omega_deg": 0.0}
# Drawn planet-to-star mass ratio, exaggerated so the reflex orbit shows.
MASS_RATIO = 0.5
# The worked example's three states, as (t - t0) / P (t0 = t_p here, M_0 = 0).
STATE_FRACS = (0.0, 0.25, 0.75)
STATE_TIMES = (r"$t=t_0$", r"$t=t_0+P/4$", r"$t=t_0+3P/4$")
STATE_NAMES = (
    "quadrature",
    r"planet nearest the observer ($+Z$)",
    r"planet farthest ($-Z$)",
)
STATE_SIDES = ("(half lit)", "(dark side to us)", "(lit side to us)")
# The extra RV state marked on the curve: the star approaches at t0 + P/2.
HALF_FRAC = 0.5
# Velocity arrows: drawn length per unit of the relative speed n a.
VEL_SCALE = 0.8

NORTH = np.array([1.0, 0.0, 0.0])
TOWARD_OBSERVER = np.array([0.0, 0.0, 1.0])
SIDE_VIEW = 0.0  # the orbit-geometry camera: side view from the east

PROFILE = "proposed observer profile and RV sign (pending)"
STATUS = f"{PROFILE}; mass ratio exaggerated, planet disk enlarged"


# Physics (pure numpy; every anchor is checked in the tests)


def mass_fractions(q=MASS_RATIO):
    """``(Mp / (M* + Mp), M* / (M* + Mp))`` for ``q = Mp / M*``."""
    return q / (1.0 + q), 1.0 / (1.0 + q)


def relative_state(frac, orbit=ORBIT):
    """Relative position and velocity of the planet about the star.

    Args:
        frac: ``(t - t0) / P``; the example has ``M_0 = 0``, so this is also
            the time since periapsis in periods.
        orbit: A circular relative orbit.

    Returns:
        ``(r, v)``: ``r`` in units of ``a`` and ``v`` in units of ``n a``,
        each of shape ``(..., 3)`` in ``(north, east, toward observer)``.
    """
    if orbit["e"] != 0.0:
        msg = "relative_state is written for the circular worked example"
        raise ValueError(msg)
    nu = 2.0 * np.pi * np.asarray(frac, dtype=float)
    r = position(nu, orbit) / orbit["a"]
    tangent = np.stack([-np.sin(nu), np.cos(nu), np.zeros_like(nu)], -1)
    v = tangent @ orbit_rotation(orbit).T
    return r, v


def barycentric_states(frac, q=MASS_RATIO, orbit=ORBIT):
    """Barycentric star and planet states from the relative state.

    Returns:
        A dict with ``r_star``, ``v_star``, ``r_p``, ``v_p`` (barycenter at
        the origin) and the relative ``r``, ``v``.
    """
    r, v = relative_state(frac, orbit)
    f_p, f_star = mass_fractions(q)
    return {
        "r": r,
        "v": v,
        "r_star": -f_p * r,
        "v_star": -f_p * v,
        "r_p": f_star * r,
        "v_p": f_star * v,
    }


def stellar_rv_over_k(frac, q=MASS_RATIO, orbit=ORBIT):
    """Stellar RV ``-dZ_star/dt`` in units of the chapter's ``K_star``.

    For this orbit ``K_star = Mp / (M* + Mp) n a sin(i)``.
    """
    s = barycentric_states(frac, q, orbit)
    f_p, _ = mass_fractions(q)
    return -s["v_star"][..., 2] / (f_p * np.sin(np.deg2rad(orbit["i_deg"])))


def k_star_si(m_star_kg, m_p_kg, a_m, i_rad=0.5 * np.pi, e=0.0):
    """The chapter's ``K_star = Mp/(M*+Mp) n a sin(i) / sqrt(1 - e^2)``, m/s."""
    mu = const.G_si * (m_star_kg + m_p_kg)
    n = np.sqrt(mu / a_m**3)
    return m_p_kg / (m_star_kg + m_p_kg) * n * a_m * np.sin(i_rad) / np.sqrt(1 - e**2)


EARTH_SUN_K = float(k_star_si(const.Msun2kg, const.Mearth2kg, const.AU2m))
EARTH_SUN_Q = float(const.Mearth2kg / const.Msun2kg)
K_TEXT = f"{EARTH_SUN_K:+.5f} m/s"
K_CM_TEXT = f"about {EARTH_SUN_K * 100:.0f} cm/s"
Q_TEXT = rf"${EARTH_SUN_Q * 1e6:.1f}\times10^{{-6}}$"


# Inks and text


def _is_light(cast):
    return cast.mode != "dark"


def _scen_ink(cast):
    return cast.neutral(0.62) if _is_light(cast) else cast["scenery"].color


def _note_ink(cast):
    return cast.neutral(0.72) if _is_light(cast) else cast["annotation"].color


def _unlit(cast):
    return cast.neutral(0.6) if _is_light(cast) else cast.neutral(0.22)


def _vec_ink(cast):
    """The relative vector: a strong neutral, not a palette hue."""
    return cast.neutral(0.85)


def _backing(cast):
    """A plain box in the background color for a label over marks."""
    return {
        "boxstyle": "round,pad=0.15,rounding_size=0.25",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _label(ax, xy, text, cast, *, color=None, fontsize=None, boxed=True, **kw):
    """A label in data coordinates, on a plain backing box, never a halo."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    t = ax.text(
        *xy,
        text,
        color=color or _note_ink(cast),
        fontsize=fontsize or cast.layout.small_pt,
        zorder=9,
        **kw,
    )
    if boxed:
        t.set_bbox(_backing(cast))
    return t


def _note(ax, xy, text, cast, **kw):
    """An italic note in axes coordinates."""
    kw.setdefault("color", _note_ink(cast))
    kw.setdefault("fontsize", cast.layout.small_pt)
    return ax.text(*xy, text, transform=ax.transAxes, fontstyle="italic", **kw)


def _plain_text(fig):
    """Strip any stroked halo a shared helper may have added."""
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
    return fig


def _headline(fig, layout, text):
    """A one-line question heading the talk slide only."""
    if layout.is_slide:
        fig.suptitle(text, fontsize=layout.title_pt)


def _vector(ax, cast, color, lw, *, gid, ms=None, zorder=6):
    """A geometric vector: thin shaft, open V head (not a light ray)."""
    patch = FancyArrowPatch(
        (0, 0),
        (1, 1),
        arrowstyle="->,head_length=0.5,head_width=0.28",
        mutation_scale=ms or cast.layout.marker_pt * 1.5,
        color=color,
        lw=lw,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        gid=gid,
    )
    ax.add_patch(patch)
    return patch


def _barycenter(ax, cast, *, size=None):
    """Barycenter glyph: a small open circle with a cross (hand-rolled)."""
    ms = size or 0.95 * cast.layout.marker_pt
    color = cast.text
    ax.plot(
        [0],
        [0],
        marker="o",
        ms=ms,
        mfc=cast.background,
        mec=color,
        mew=0.8 * cast.layout.lw,
        ls="none",
        zorder=7,
        gid="d11-barycenter",
    )
    ax.plot(
        [0],
        [0],
        marker="+",
        ms=ms,
        mec=color,
        mew=0.8 * cast.layout.lw,
        ls="none",
        zorder=7,
    )


def side_xy(vec):
    """Page coordinates of a 3D vector in the side view from the east."""
    return project(vec, SIDE_VIEW)


# Side view: barycenter, reflex orbits, bodies and velocities

PLANET_R = 0.085  # drawn planet-disk radius, in units of a (enlarged)
SIDE_X = (-1.1, 1.25)
SIDE_Y = (-1.32, 1.02)
OBS_Y = -1.2  # height of the "to observer" arrow
SKY_TOP, SKY_BOTTOM = 1.0, -0.84


def _side_panel(ax, cast, *, labels=True, sky_label=True):
    """Static scenery and moving artists of one side view from the east.

    The camera is the orbit-geometry explainer's side view: north up and the
    observer to the right along +Z. With ``i = 90 deg`` and ``Omega = 0`` the
    orbit plane is the page.
    """
    layout = cast.layout
    lw = layout.lw
    f_p, f_star = mass_fractions()
    rp = cast["reference_plane"]
    ax.plot(
        [0, 0], [SKY_BOTTOM, SKY_TOP], color=rp.color, ls=rp.ls, lw=0.9 * lw, zorder=1
    )
    t = np.linspace(0.0, 2.0 * np.pi, 361)
    ring = np.stack([np.cos(t), np.zeros_like(t), np.sin(t)], -1)
    ax.plot(
        *side_xy(f_star * ring).T,
        color=cast["planet"].color,
        lw=0.8 * lw,
        zorder=2,
        gid="d11-planet-orbit",
    )
    ax.plot(
        *side_xy(f_p * ring).T,
        color=cast["star"].color,
        lw=0.8 * lw,
        ls=(0, (3, 2)),
        zorder=2,
        gid="d11-star-orbit",
    )
    ink = _scen_ink(cast)
    obs = FancyArrowPatch(
        (-0.2, OBS_Y),
        (0.3, OBS_Y),
        arrowstyle="->,head_length=0.5,head_width=0.28",
        mutation_scale=layout.marker_pt * 1.4,
        color=cast["scenery"].color,
        lw=0.9 * lw,
        shrinkA=0,
        shrinkB=0,
        gid="d11-to-observer",
    )
    ax.add_patch(obs)
    _label(
        ax, (0.36, OBS_Y), r"to observer, $+Z$", cast, color=ink, ha="left", boxed=False
    )
    if sky_label:
        _label(
            ax,
            (0.05, SKY_TOP),
            "sky plane",
            cast,
            color=ink,
            ha="left",
            va="top",
            fontstyle="italic",
            boxed=False,
        )
    _barycenter(ax, cast)
    art = {"ax": ax}
    art["rel"] = _vector(ax, cast, _vec_ink(cast), 0.9 * lw, gid="d11-relative")
    art["v_star"] = _vector(ax, cast, cast["star"].color, 1.2 * lw, gid="d11-v-star")
    art["v_p"] = _vector(ax, cast, cast["planet"].color, 1.2 * lw, gid="d11-v-planet")
    # The line-of-sight part of the star's velocity, -v_r: page x is Z.
    (art["v_los"],) = ax.plot(
        [],
        [],
        color=cast["star"].color,
        lw=2.2 * lw,
        ls=(0, (0.8, 1.2)),
        marker="|",
        markevery=[1],
        ms=0.9 * layout.marker_pt,
        zorder=7,
        gid="d11-v-star-los",
    )
    (art["star"],) = ax.plot(
        [],
        [],
        zorder=8,
        gid="d11-star",
        **cast["star"].marker_kw(1.2 * layout.marker_pt),
    )
    art["dark"] = Circle(
        (0, 0),
        PLANET_R,
        facecolor=_unlit(cast),
        edgecolor=cast["planet"].color,
        lw=0.8 * lw,
        zorder=8,
        gid="d11-planet",
    )
    ax.add_patch(art["dark"])
    art["lit"] = Polygon(
        np.zeros((4, 2)), facecolor=cast["planet"].color, edgecolor="none", zorder=9
    )
    ax.add_patch(art["lit"])
    fs = layout.font_pt
    art["r_label"] = _label(
        ax, (0, 0), r"$\mathbf{r}$", cast, color=_vec_ink(cast), fontsize=fs
    )
    art["vs_label"] = _label(
        ax,
        (0, 0),
        r"$\mathbf{v}_\bigstar$",
        cast,
        color=cast["star"].color,
        fontsize=fs,
    )
    art["vp_label"] = _label(
        ax, (0, 0), r"$\mathbf{v}_p$", cast, color=cast["planet"].color, fontsize=fs
    )
    art["name_labels"] = []
    if labels:
        art["name_labels"] = [
            _label(ax, (0, 0), "star", cast, color=cast.text),
            _label(ax, (0, 0), "planet", cast, color=cast.text),
            _label(ax, (0, 0), "barycenter", cast, color=cast.text),
        ]
    return art


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 1.0])


def _set_side(art, frac):
    """Place every moving artist of a side view at ``(t - t0) / P = frac``."""
    s = barycentric_states(frac)
    xs, xp = side_xy(s["r_star"]), side_xy(s["r_p"])
    vs, vp = VEL_SCALE * side_xy(s["v_star"]), VEL_SCALE * side_xy(s["v_p"])
    art["star"].set_data([xs[0]], [xs[1]])
    art["dark"].set_center(xp)
    art["lit"].set_xy(lit_polygon(xp, xs - xp, 0.5 * np.pi, PLANET_R))
    # The relative vector runs from the star to the planet's near limb.
    u = _unit(xp - xs)
    art["rel"].set_positions(xs, xp - 1.15 * PLANET_R * u)
    art["v_star"].set_positions(xs, xs + vs)
    art["v_los"].set_data([xs[0], xs[0] + vs[0]], [xs[1], xs[1]])
    art["v_p"].set_positions(xp, xp + vp)
    # r is labeled on the side of the vector the planet's velocity leaves.
    normal = -_unit(vp)
    art["r_label"].set_position(xs + 0.7 * (xp - xs) + 0.1 * normal)
    art["vs_label"].set_position(xs + vs + 0.12 * _unit(vs))
    art["vp_label"].set_position(xp + vp + 0.12 * _unit(vp))
    if art["name_labels"]:
        # Names sit radially outside each body, on the side the velocity leaves.
        star_l, planet_l, bary_l = art["name_labels"]
        star_l.set_position(xs + 0.2 * _unit(xs))
        planet_l.set_position(xp - 0.12 * _unit(vp))
        planet_l.set_ha(
            "right" if vp[0] > 0.1 else ("left" if vp[0] < -0.1 else "center")
        )
        planet_l.set_va(
            "bottom" if vp[1] < -0.1 else ("top" if vp[1] > 0.1 else "center")
        )
        # The barycenter label sits on the side of r the velocities leave.
        bary_l.set_position(0.1 * _unit(vp))
        bary_l.set_ha(
            "left" if vp[0] > 0.1 else ("right" if vp[0] < -0.1 else "center")
        )
        bary_l.set_va(
            "bottom" if vp[1] > 0.1 else ("top" if vp[1] < -0.1 else "center")
        )


def _finish(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")


# RV curve

RV_Y = (-2.05, 1.65)


def _rv_panel(ax, cast):
    """Stellar RV in units of K over one period, positive = recession."""
    layout = cast.layout
    f = np.linspace(0.0, 1.0, 241)
    ax.axhline(0.0, color=cast["scenery"].color, lw=0.7 * layout.lw, zorder=1)
    ax.plot(
        f,
        stellar_rv_over_k(f),
        color=cast["star"].color,
        lw=1.5 * layout.lw,
        zorder=3,
        gid="d11-rv-curve",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(*RV_Y)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "1/4", "1/2", "3/4", "1"])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_yticklabels([r"$-K_\bigstar$", "0", r"$+K_\bigstar$"])
    ax.set_xlabel(r"$(t-t_0)\,/\,P$")
    ax.set_ylabel(
        r"$v_{r,\bigstar}=-\dot Z_\bigstar$",
        labelpad=2.0 * layout.font_pt if layout.is_slide else 4.0,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _label(
        ax,
        (0.5, 0.5),
        "positive = recession:\nstar moving away",
        cast,
        color=cast.text,
        gid="d11-rv-recession",
    )
    for k, fr in enumerate((*STATE_FRACS, HALF_FRAC)):
        y = float(stellar_rv_over_k(fr))
        ax.plot(
            [fr],
            [y],
            marker="o",
            ms=layout.marker_pt,
            mfc=cast.background,
            mec=cast.text,
            mew=layout.lw,
            ls="none",
            zorder=5,
            gid=f"d11-rv-state-{k}",
        )
    _label(
        ax,
        (0.03, 1.1),
        r"quadrature: $+K_\bigstar$" + "\n" + f"Earth and Sun: {K_TEXT} ({K_CM_TEXT})",
        cast,
        color=cast.text,
        ha="left",
        va="bottom",
        gid="d11-rv-k-label",
    )
    _label(
        ax,
        (0.5, -1.16),
        r"$t_0+P/2$: $-K_\bigstar$, negative = approach:"
        + "\nstar moving toward the observer",
        cast,
        color=cast.text,
        va="top",
        gid="d11-rv-approach",
    )
    _label(
        ax,
        (0.24, -0.12),
        r"planet nearest ($+Z$)",
        cast,
        color=cast.text,
        ha="right",
        va="top",
    )
    _label(
        ax,
        (0.76, -0.12),
        r"planet farthest ($-Z$)",
        cast,
        color=cast.text,
        ha="left",
        va="top",
    )


def _status_row(fig, grid_cell, cast):
    """A thin row that carries the status badge under the panels."""
    ax = fig.add_subplot(grid_cell)
    ax.axis("off")
    text = (
        f"{PROFILE}; planet disk enlarged\n"
        rf"mass ratio exaggerated: drawn $M_p/M_\bigstar$ = {MASS_RATIO:g} "
        f"(Earth and Sun: {Q_TEXT})"
    )
    badge = ex.badge(ax, cast, text, loc="upper right")
    badge.set_gid("d11-status")
    return ax


# Figure: the worked example's three states above the RV curve


def build_states(layout, cast):
    fig = plt.figure(figsize=layout.size(5.3), layout="constrained")
    ratios = [2.6, 1.5, 0.18] if layout.is_slide else [2.75, 1.35, 0.2]
    grid = fig.add_gridspec(3, 3, height_ratios=ratios)
    _headline(
        fig,
        layout,
        "Which way does the star move when the planet moves toward the observer?",
    )
    for col, (fr, when, name, lit) in enumerate(
        zip(STATE_FRACS, STATE_TIMES, STATE_NAMES, STATE_SIDES, strict=True)
    ):
        ax = fig.add_subplot(grid[0, col])
        art = _side_panel(ax, cast, labels=(col == 0))
        _set_side(art, fr)
        for a in (
            art["rel"],
            art["v_star"],
            art["v_p"],
            art["v_los"],
            art["star"],
            art["dark"],
        ):
            a.set_gid(f"{a.get_gid()}-{col}")
        rv = float(stellar_rv_over_k(fr))
        rv_text = r"$v_{r,\bigstar}=+K_\bigstar$" if rv > 0.5 else r"$v_{r,\bigstar}=0$"
        ax.set_title(f"{when}: {rv_text}\n{name}\n{lit}", fontsize=layout.font_pt)
        _finish(ax, SIDE_X, SIDE_Y)
        if col == 0:
            _note(
                ax, (0.0, 0.1), "side view\nfrom the east", cast, ha="left", va="bottom"
            )
    rv_ax = fig.add_subplot(grid[1, :])
    _rv_panel(rv_ax, cast)
    _status_row(fig, grid[2, :], cast)
    return _plain_text(fig)


# Animation: orbital time runs, the camera is fixed


def clock_frames(layout):
    """Equal time steps over one period, starting at t0 (quadrature).

    The count is a multiple of 4, so the worked example's three states are
    frames: 200 steps in the talk version, 28 in the documentation player.
    """
    n = layout.n_frames(200)
    n -= n % 4
    return [{"frac": k / n} for k in range(n)]


def build_clock(layout, cast):
    fig = plt.figure(figsize=layout.size(4.3), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.12], width_ratios=[1.0, 1.25])
    _headline(fig, layout, "The star's line-of-sight velocity over one orbit")
    side = fig.add_subplot(grid[0, 0])
    rv = fig.add_subplot(grid[0, 1])
    # The moving velocity labels sweep every corner of this panel, so its key
    # sits in the title rather than on the scene.
    art = _side_panel(side, cast, sky_label=False)
    _finish(side, SIDE_X, SIDE_Y)
    side.set_title(
        "side view from the east\ndash-dot: sky plane\n"
        + r"dotted: line-of-sight part of $\mathbf{v}_\bigstar$",
        fontsize=layout.small_pt,
        color=_note_ink(cast),
        fontstyle="italic",
    )
    _rv_panel(rv, cast)
    _status_row(fig, grid[1, :], cast)
    (cursor,) = rv.plot(
        [0, 0], list(RV_Y), color=_scen_ink(cast), lw=0.7 * layout.lw, ls=":", zorder=2
    )
    (dot,) = rv.plot(
        [],
        [],
        marker="o",
        ms=layout.marker_pt,
        mfc=cast["star"].color,
        mec=cast.text,
        mew=0.8 * layout.lw,
        ls="none",
        zorder=6,
        gid="d11-rv-now",
    )

    def draw(fig_, frame):
        del fig_
        fr = frame["frac"]
        _set_side(art, fr)
        y = float(stellar_rv_over_k(fr))
        dot.set_data([fr], [y])
        cursor.set_xdata([fr, fr])
        rv.set_title(
            rf"$(t-t_0)/P$ = {fr:.2f},   $v_{{r,\bigstar}}/K_\bigstar$ = {round(y, 2) + 0.0:+.2f}",
            fontsize=layout.font_pt,
        )

    frames = clock_frames(layout)
    draw(fig, frames[0])
    return ex.AnimationScene(fig=_plain_text(fig), draw=draw, frames=frames)


# Registry

_PARAMS = {
    "orbit": ORBIT,
    "drawn_mass_ratio": MASS_RATIO,
    "state_fracs": STATE_FRACS,
    "velocity_scale": VEL_SCALE,
    "earth_sun_k_m_s": round(EARTH_SUN_K, 8),
}

STATES_CAPTION = (
    "The star and the planet both orbit their common barycenter, and the sign "
    "of the stellar radial velocity follows from the star's reflex motion "
    "({ref}`geometry-origin-mass-phase`, {ref}`geometry-radial-velocity`). "
    "Top: side views from the east, north up and the observer to the right "
    r"along $+Z$, of the chapter's worked signed example "
    r"({ref}`geometry-rv-worked-example`): a circular relative orbit with "
    r"$i=90^\circ$ and $\Omega=\omega_p=M_0=0$, so the orbit lies in the "
    "north and line-of-sight plane and is seen face-on. The barycenter is the "
    r"crossed circle. The relative vector $\mathbf r=\mathbf r_p-\mathbf r_\star$ "
    "runs from the star to the planet through the barycenter, with "
    r"$\mathbf r_\star=-\frac{M_p}{M_\star+M_p}\mathbf r$, so the star (dashed "
    "orbit) and the planet (solid orbit) are always on opposite sides of it. "
    "Velocity arrows share one scale. "
    r"At $t_0$ the planet is north of the star and moves toward the observer, "
    r"$\dot Z>0$; the star moves away, so its recession velocity "
    r"$v_{r,\star}=-\dot Z_\star$ is positive, and the planet is at quadrature. "
    "A quarter period later the planet is nearest the observer, at $+Z$, with "
    "its unlit side toward the observer, and three quarters later it is "
    "farthest, at $-Z$, with its lit side toward the observer; both times the "
    "star moves across the line of sight and its radial velocity is zero, so "
    "brightness alone does not fix the sign. Bottom: $v_{r,\star}$ over one "
    r"period in units of "
    r"$K_\star=\frac{M_p}{M_\star+M_p}\frac{na\sin i}{\sqrt{1-e^2}}$, "
    r"$v_{r,\star}/K_\star=\cos(\nu+\omega_p)+e\cos\omega_p$, with the three "
    r"states and $t_0+P/2$, where the star approaches at $-K_\star$, marked. "
    "The bracket uses the planet's argument of periastron. The literature "
    "writes this curve with the star's argument of periastron "
    "({ref}`Lovis and Fischer 2010, Sec. 2.1, eqs. 10-12 <source-lovis2010>`). "
    "If the same node convention and orbital phase are retained, the stellar "
    r"periapsis argument is $\omega_\star=\omega_p+\pi$, and rewriting the "
    r"bracket with $\omega_\star$ introduces a minus sign. A familiar RV "
    r"formula using an unlabeled $\omega$ is therefore insufficient evidence "
    "of compatibility: its body, node, observer direction and velocity sign "
    r"must travel together, and the literature form gives $+K_\star$ at $t_0$ "
    "only under its own conventions. "
    r"For the Earth and Sun, $v_{r,\star}(t_0)=+K_\star=+0.08946$ m/s "
    "(about 9 cm/s). The recorded limitations include a report of an "
    r"implementation with the opposite sign at $t_0$ "
    "({ref}`limitations-geometry`). Positive recession "
    "and the observer profile are proposed conventions, pending at the "
    "conventions stage. Two-body reflex motion with the systemic velocity "
    r"removed. The drawn mass ratio is exaggerated to $M_p/M_\star=0.5$ (the "
    r"Earth and Sun have $3.0\times10^{-6}$) so the star's orbit is visible; "
    "the planet disk is enlarged and lit on its star-facing half. Computed from "
    "the chapter's reflex relations, not from a library."
)
STATES_ALT = (
    "Four panels. The top row has three side views, titled t equals t sub 0, "
    "v r equal to plus K, quadrature, half lit; t equals t sub 0 plus a "
    "quarter period, v r equal to zero, planet nearest the observer, plus Z, "
    "dark side to us; and t equals t sub 0 plus three quarters of a period, v r "
    "equal to zero, planet farthest, minus Z, lit side to us. Each "
    "shows a crossed circle labeled barycenter on a vertical dash-dot sky "
    "plane, a small dashed yellow circle that is the star's orbit, a larger "
    "cyan circle that is the planet's orbit, an arrow at the bottom pointing "
    "right labeled to observer, plus Z, the yellow star and the half-lit cyan "
    "planet on opposite sides of the barycenter, a gray vector r from the star "
    "to the planet, and velocity arrows v star in yellow and v p in cyan. In "
    "the first panel the planet is at the top moving right, toward the "
    "observer, and the star is below the barycenter moving left, away from "
    "the observer. In the second the planet is right of the barycenter, its "
    "lit half facing away from the observer, moving down, and the star moves "
    "up. In the third the planet is left of the barycenter, its lit half "
    "facing the observer, moving up, and the star moves down. The bottom "
    "panel plots v r star equals minus Z-dot star against time over one "
    "period as a yellow cosine from plus K through minus K back to plus K, "
    "with open circles at plus K at time zero, labeled quadrature, zero at a "
    "quarter period, labeled planet nearest, plus Z, minus K at half a "
    "period, labeled negative equals approach, star moving toward the "
    "observer, and zero at three quarters, labeled planet farthest, minus Z. "
    "Text marks positive as recession, the star moving away, and gives the "
    "Earth and Sun value as plus 0.08946 meters per second, about 9 "
    "centimeters per second. A badge says the observer profile and RV sign "
    "are proposed and pending, and that the mass ratio is exaggerated to 0.5 "
    "against 3.0 times ten to the minus six for the Earth and Sun."
)

CLOCK_CAPTION = (
    "One orbital period at equal time steps, with the camera fixed: the side "
    "view from the east of the worked signed example (left) and the stellar "
    "radial velocity (right) "
    "({ref}`geometry-radial-velocity`, {ref}`geometry-rv-worked-example`). The "
    "star and the planet circle their barycenter on opposite sides. The dotted "
    "segment at the star is the line-of-sight part of its velocity, "
    r"$\dot Z_\star$ along $+Z$; the dot on the curve is its negative, the "
    r"recession velocity $v_{r,\star}=-\dot Z_\star$, in units of $K_\star$. "
    "It is largest where the star moves straight away from the observer, most "
    "negative where it moves straight toward the observer, and zero where it "
    "moves across the line of sight. Time starts at $t_0$, at quadrature. "
    "Positive recession and the observer profile are proposed conventions, "
    r"pending. The drawn mass ratio is exaggerated to $M_p/M_\star=0.5$; the "
    "planet disk is enlarged. Computed from the chapter's reflex relations, "
    "not from a library."
)
CLOCK_ALT = (
    "Animation of one orbital period with a fixed camera. Left: a side view "
    "from the east with the observer to the right; a yellow star and a "
    "half-lit cyan planet circle a crossed-circle barycenter on opposite "
    "sides, joined by a gray vector r, with yellow and cyan velocity arrows "
    "and a dotted yellow horizontal segment at the star showing the "
    "line-of-sight part of its velocity. "
    "Right: a yellow cosine curve of the stellar radial velocity against time "
    "over one period, positive labeled recession, with the marked states and "
    "a dot and a dotted cursor moving along it. Readouts give the time in "
    "periods and v r over K; the dot starts at plus one, when the star moves "
    "away from the observer and its dotted segment points left, passes zero "
    "when the planet is nearest the observer and the segment vanishes, "
    "reaches minus one when the segment points right, toward the observer, "
    "and passes zero again when the planet is farthest."
)

FIGURES = [
    ex.FigureSpec(
        slug="d11-barycenter-rv",
        build=build_states,
        caption=STATES_CAPTION,
        alt=STATES_ALT,
        status=STATUS,
        params=_PARAMS,
    ),
]

ANIMATIONS = [
    ex.AnimationSpec(
        slug="d11-reflex-clock",
        build=build_clock,
        ground="rate",
        caption=CLOCK_CAPTION,
        alt=CLOCK_ALT,
        status=STATUS,
        params=_PARAMS,
        preview_frames=(0, 3, 7, 11, 14, 21),
    ),
]
