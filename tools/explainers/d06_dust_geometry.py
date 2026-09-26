"""Local zodiacal light and an exozodiacal disk: one process, two observer geometries.

The opening figure draws two separate systems. Top: the telescope sits inside
the Solar-system dust cloud and looks out along a half-ray; the grain on that
ray receives sunlight and scatters some of it back to the telescope, and the
grain is drawn large beside the view, in its own scattering plane. Bottom: the
telescope is far outside a host star's circumstellar disk, beyond a scale
break; the disk is drawn side on (inclination, near and far halves, the
sightline's chord through the dust layer, a grain inset) and as the projected
image. Both rows name the illuminating star, the grain, the incident and
outgoing rays, and the integration direction along the sightline.

The geometry views come from ``skyscapes.viz`` (``plot_local_zodi_geometry``,
``plot_disk_geometry`` and ``plot_disk_image``); the projected image is the
``surface_brightness`` of a declared synthetic ``GraterDisk``. The library's
angle conventions (``+z`` toward the observer, near half toward ``+z``,
scattering angle 0 forward) agree with the dust chapter. The image axes are
named by the disk (line of nodes, projected minor axis), not by east and
north, because the observer basis is pending.

The animation sweeps the viewing inclination of the same synthetic disk with
the distance and the color scale fixed, beside a disk-frame glyph of the
turning view direction and the disk's phase function with its near and far
minor-axis grains.

Hand-rolled shapes (no primitive exists): the schematic cloud envelope, the
integration arrows and the outgoing local-zodi ray laid along the library
sightlines, the telescope glyph that replaces the library's observer mark in
both views, the scale break on the disk sightline, the leader-labeled dust
path, the disk-frame view glyph, the phase-function panel with its fixed
key, and the edge and half labels. Library
artists are restyled through their gids (neutral sightlines, hatched layer,
smaller inclination arc, legible inset text).
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
from hwoutils.conversions import au_to_arcsec
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.text import Text

from explainers import _common as ex

STATUS = "schematic, not to scale"
STATUS_IMAGE = "simulated example disk"

# Local zodiacal light: the two Leinert inputs of the look direction, the
# observer's heliocentric distance, and where the marked grain sits.
ZODI = {
    "ecliptic_lat_deg": 30.0,
    "solar_lon_deg": 135.0,
    "observer_distance_AU": 1.0,
    "grain_distance_AU": 1.0,
    "ray_length_AU": 1.6,
}
# Schematic cloud envelope: an oblate ellipsoid about the Sun, drawn only to
# show that the telescope is inside the dust. It is not a density model.
CLOUD_AU = {"radius": 2.4, "half_height": 0.8}

# Exozodiacal disk: a declared synthetic GRaTeR disk (Augereau et al. 1999,
# section 3.1) with a Henyey-Greenstein phase function.
DISK = {
    "sma_AU": 3.0,
    "alpha_in": 5.0,
    "alpha_out": -3.0,
    "ksi0_AU": 0.1,
    "gamma": 2.0,
    "beta": 1.0,
    "rmin_AU": 1.5,
    "rmax_AU": 5.0,
    "g_HG": 0.4,
    "albedo": 0.3,
    "n_pix": 121,
    "pixel_scale_arcsec": 0.0085,
    "dist_pc": 10.0,
    "n_slices_los": 41,
    "wavelength_nm": 550.0,
}
INCL_DEG = 60.0
PA_DEG = 0.0
GRAIN_RADIUS_AU = 3.0
LAYER_THICKNESS_AU = 0.6
DYNAMIC_RANGE = 3.0e2
# Inclination sweep of the animation: face-on to 80 degrees, short of the
# edge-on limit and of the pending sense of inclinations above 90 degrees.
SWEEP_DEG = (0.0, 80.0)
SWEEP_FRAMES = 49


# Independent geometry (the chapter's definitions, no plotting library)


def zodi_look(ecliptic_lat_deg, solar_lon_deg):
    """Unit look direction n-hat for the two Leinert angles.

    Ecliptic frame centered on the Sun with the observer on +x, so the Sun
    lies along -x from the observer; +z is ecliptic north. The longitude
    difference is measured from the Sun direction toward +y.
    """
    b = np.radians(ecliptic_lat_deg)
    d = np.radians(solar_lon_deg)
    return np.array([-np.cos(b) * np.cos(d), np.cos(b) * np.sin(d), np.sin(b)])


def scattering_angle_deg(star_to_grain, n_hat):
    """Scattering angle: cos Theta = k_in . (-n_hat), in degrees."""
    k_in = np.asarray(star_to_grain, float) / np.linalg.norm(star_to_grain)
    return float(np.degrees(np.arccos(np.clip(k_in @ (-np.asarray(n_hat)), -1, 1))))


def zodi_numbers(p=ZODI):
    """Angles printed on the local zodiacal panels."""
    n = zodi_look(p["ecliptic_lat_deg"], p["solar_lon_deg"])
    obs = np.array([p["observer_distance_AU"], 0.0, 0.0])
    grain = obs + p["grain_distance_AU"] * n
    theta = scattering_angle_deg(grain, n)
    elong = float(np.degrees(np.arccos(n @ np.array([-1.0, 0.0, 0.0]))))
    return {"theta": theta, "alpha": 180.0 - theta, "elongation": elong}


def disk_minor_axis_theta(incl_deg, near=True):
    """Scattering angle of a midplane grain on the projected minor axis."""
    return 90.0 - incl_deg if near else 90.0 + incl_deg


# The synthetic disk


@functools.cache
def _renderer():
    import jax
    import jax.numpy as jnp
    from skyscapes.disk import GraterDisk

    d = DISK
    disk = GraterDisk(
        sma_AU=jnp.array(d["sma_AU"]),
        alpha_in=jnp.array(d["alpha_in"]),
        alpha_out=jnp.array(d["alpha_out"]),
        ksi0_AU=jnp.array(d["ksi0_AU"]),
        gamma=jnp.array(d["gamma"]),
        beta=jnp.array(d["beta"]),
        rmin_AU=jnp.array(d["rmin_AU"]),
        rmax_AU=jnp.array(d["rmax_AU"]),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([d["g_HG"], d["g_HG"]]),
        Ag_grid=jnp.array([d["albedo"], d["albedo"]]),
        nx=d["n_pix"],
        ny=d["n_pix"],
        pixel_scale_arcsec=d["pixel_scale_arcsec"],
        dist_pc=d["dist_pc"],
        n_slices_los=d["n_slices_los"],
    )
    render = jax.jit(
        lambda incl: disk.surface_brightness(
            jnp.asarray(d["wavelength_nm"]),
            jnp.asarray(0.0),
            incl,
            jnp.asarray(PA_DEG),
        )
    )
    return disk, render


@functools.cache
def disk_images(incls):
    """Render the synthetic disk at each inclination, normalized together.

    Every image is divided by the brightest pixel of the whole set, so the
    frames of the sweep share one scale (render all frames first).

    Args:
        incls: Tuple of inclinations in degrees.

    Returns:
        A tuple of ``(ny, nx)`` arrays with a common peak of 1.
    """
    _, render = _renderer()
    raw = [np.asarray(render(float(i)), dtype=float) for i in incls]
    peak = max(float(np.nanmax(r)) for r in raw)
    return tuple(r / peak for r in raw)


def sweep_incls(layout):
    """Inclinations of the animation frames."""
    return tuple(
        float(v) for v in np.linspace(*SWEEP_DEG, layout.n_frames(SWEEP_FRAMES))
    )


def still_image():
    """The still's image, normalized to its own peak."""
    return disk_images((INCL_DEG,))[0]


def ring_point_arcsec(incl_deg, x_arcsec=-0.2):
    """A point of the projected middle ring at ``x_arcsec`` on the line of nodes.

    The ring of radius ``(r_in + r_out) / 2`` projects to an ellipse with
    semi-major axis ``a = r / D`` (arcsec, with r in AU and D in pc) and
    semi-minor axis ``a cos i``; the near half is at positive offsets.
    """
    a = float(au_to_arcsec(0.5 * (DISK["rmin_AU"] + DISK["rmax_AU"]), DISK["dist_pc"]))
    y = a * np.cos(np.radians(incl_deg)) * np.sqrt(1.0 - (x_arcsec / a) ** 2)
    return float(x_arcsec), float(y)


def hg_phase(theta_deg, g):
    """Henyey-Greenstein phase function, normalized to one over 4 pi sr."""
    mu = np.cos(np.radians(theta_deg))
    return (1.0 - g**2) / (4.0 * np.pi * (1.0 + g**2 - 2.0 * g * mu) ** 1.5)


def drawn_solar_lon_deg():
    """Signed lambda - lambda_sun of the drawn top-view sightline, in degrees.

    Seen from the ecliptic north pole with the observer on +x, ecliptic
    longitude increases counterclockwise, the Sun lies at 180 degrees and the
    library draws the look direction toward +y, at 180 - |dlon|.
    """
    n = zodi_look(ZODI["ecliptic_lat_deg"], ZODI["solar_lon_deg"])
    lam = np.degrees(np.arctan2(n[1], n[0]))
    return float((lam - 180.0 + 180.0) % 360.0 - 180.0)


# Styling helpers


def _viz():
    from skyscapes import viz

    return viz


def _legible(ax, layout, *, inset_pt=None):
    """Fit the library view's fixed sizes to the layout.

    Text in the panel is lifted to the layout's small size and inset text to
    ``inset_pt``. On a slide, marks, lines and arrow heads drawn at the
    library's fixed point sizes are scaled by the layout's line-width ratio.
    """
    floor = layout.small_pt if layout.is_slide else 7.5
    inset_pt = floor if inset_pt is None else inset_pt
    for a in [ax, *ax.child_axes]:
        lift = floor if a is ax else inset_pt
        for t in a.get_children():
            if isinstance(t, Text) and (t.get_fontsize() < lift or a is not ax):
                t.set_fontsize(lift)
    if not layout.is_slide:
        return
    k = layout.lw / ex.DOC.lw
    for a in [ax, *[c for c in ax.child_axes if c.get_gid() == "inset"]]:
        for coll in a.collections:
            if isinstance(coll, PathCollection):
                coll.set_sizes(np.asarray(coll.get_sizes()) * k**2)
        for line in a.lines:
            if line.get_gid() not in _OWN:
                line.set_linewidth(line.get_linewidth() * k)
        for t in a.texts:
            if t.get_gid() in ("label/sun", "label/observer") and hasattr(t, "xyann"):
                t.xyann = tuple(k * v for v in t.xyann)
            patch = getattr(t, "arrow_patch", None)
            if patch is not None:
                patch.set_linewidth(patch.get_linewidth() * k)
                patch.set_mutation_scale(patch.get_mutation_scale() * k)


# Lines this module draws at layout sizes; the slide scaling leaves them.
_OWN = {"d06/telescope", "d06/break"}


def _by_gid(ax, gid):
    for a in [ax, *ax.child_axes]:
        for art in a.get_children():
            if art.get_gid() == gid:
                return art
    return None


def _restyle_inset(ax, cast, layout):
    """Give the grain inset's illumination-angle arc a visible weight."""
    arc = _by_gid(ax, "inset/illumination_angle")
    if arc is not None:
        arc.set_linewidth(ex.DOC.lw)
        arc.set_color(cast.neutral(0.7))
    for gid in ("inset/illumination_angle/label", "inset/illumination_angle/value"):
        t = _by_gid(ax, gid)
        if t is not None:
            t.set_color(cast.neutral(0.7))


def _move_integration(artists, start, end):
    """Move an integration arrow drawn by ``ex.arrow``."""
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    artists[0].set_positions(end - 0.02 * (end - start), end)
    artists[1].set_positions(start, end)


def _soften(patch, cast):
    """Keep a scenery cloud's hatch but at half weight, so rays read over it."""
    color = cast["local_zodi"].color
    patch.set_edgecolor(to_rgba(color, 0.6))
    dark = cast.mode == "dark"
    patch.set_facecolor(to_rgba(color, 0.06 if dark else 0.08))
    patch.set_hatchcolor(to_rgba(color, 0.3 if dark else 0.4))


def _neutral_sightline(ax, cast, layout):
    line = _by_gid(ax, "sightline")
    if line is not None and isinstance(line, Line2D):
        line.set_color(cast.neutral(0.55))
        line.set_linewidth(0.7 * ex.DOC.lw)
    lab = _by_gid(ax, "label/sightline")
    if lab is not None:
        lab.set_color(cast.neutral(0.55))


def _panel_title(ax, text, layout):
    ax.set_title(text, loc="left", fontsize=layout.font_pt)


def _backing(cast, pad=0.15):
    """A plain backing box in the background color, for a label over marks.

    Labels here carry no stroked halo (at documentation size a halo turns
    every glyph into a blob); a label over an image, hatching or a line sits
    on this box instead.
    """
    return {
        "boxstyle": f"round,pad={pad},rounding_size=0.25",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _plain_text(fig, cast):
    """Replace the stroked halos the shared and library helpers add.

    Every label that carried a stroke gets the plain backing box instead,
    unless it already sits on a box of its own.
    """
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
            if text.get_bbox_patch() is None:
                # Grain-inset labels sit tight against their rays: a slim box.
                inset = text.axes is not None and text.axes.get_gid() == "inset"
                text.set_bbox(_backing(cast, pad=0.04 if inset else 0.15))
    return fig


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# Local zodiacal light


def zodi_elongation_deg(p=ZODI):
    """Solar elongation of the look direction: cos e = cos(beta) cos(dlon)."""
    b = np.radians(p["ecliptic_lat_deg"])
    d = np.radians(p["solar_lon_deg"])
    return float(np.degrees(np.arccos(np.cos(b) * np.cos(d))))


def zodi_top(ax, cast, layout, *, inset_bounds, inset_pt=None):
    """Local zodiacal light, looking down on the ecliptic from its north pole."""
    p = ZODI
    viz = _viz()
    viz.plot_local_zodi_geometry(
        p["ecliptic_lat_deg"],
        p["solar_lon_deg"],
        view="top",
        observer_distance_AU=p["observer_distance_AU"],
        grain_distance_AU=p["grain_distance_AU"],
        ray_length_AU=p["ray_length_AU"],
        inset=True,
        inset_bounds=inset_bounds,
        ax=ax,
    )
    cloud = ex.region(ax, "local_zodi", Circle((0.0, 0.0), CLOUD_AU["radius"]), cast)
    cloud.set_zorder(0)
    _soften(cloud, cast)
    ex.note(
        ax,
        (0.98, 0.02),
        "zodiacal dust cloud\n(schematic extent,\nno density model)",
        cast,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=cast["local_zodi"].color,
    )
    _neutral_sightline(ax, cast, layout)
    n = zodi_look(p["ecliptic_lat_deg"], p["solar_lon_deg"])
    proj = n[:2]
    along = _unit(proj)
    right = np.array([along[1], -along[0]])  # lower-right of the sightline
    obs = np.array([p["observer_distance_AU"], 0.0])
    grain = obs + p["grain_distance_AU"] * proj
    # Label the observer's orbit where it leaves the lower right of the Sun,
    # clear of the rays (upper right) and of the inset corners.
    r_obs = p["observer_distance_AU"]
    orbit = ex.note(
        ax,
        (r_obs * np.cos(np.radians(-60.0)), r_obs * np.sin(np.radians(-60.0))),
        f"Earth orbit, {r_obs:g} AU",
        cast,
        ha="left",
        va="top",
        color=cast["scenery"].color,
    )
    orbit.set_gid("d06/orbit_label")
    # Integration along the half-ray s >= 0, on the lower-right side.
    label = ex.arrow(
        ax,
        obs + 0.18 * right,
        obs + 0.18 * right + 1.35 * proj,
        "integration",
        cast,
        label=r"integrate, $s\geq0$",
        label_frac=0.25,
        label_offset_pt=(9, -5),
    )[-1]
    label.set_ha("left")
    label.set_va("top")
    # Outgoing light: grain to telescope, along the sightline itself.
    scattered = _by_gid(ax, "scattered")
    if scattered is not None:
        scattered.set_visible(False)
    out = ex.arrow(
        ax,
        grain - 0.06 * along,
        obs + 0.16 * along,
        "ray",
        cast,
        source="local_zodi",
    )[0]
    if layout.is_slide:
        # Keep the head from covering the telescope and the arc.
        out.set_mutation_scale(0.6 * out.get_mutation_scale())
    # The same telescope glyph as the disk panel, facing the sightline.
    observer = _by_gid(ax, "observer")
    if observer is not None:
        observer.set_visible(False)
    scope = ex.aperture(
        ax,
        tuple(obs),
        0.2,
        cast,
        angle_deg=float(np.degrees(np.arctan2(along[1], along[0]))) + 90.0,
    )
    scope.set_gid("d06/telescope")
    # The projected longitude difference: the library's arc, with its label
    # moved outside the incident ray so the two do not cross.
    arc = _by_gid(ax, "look_angle/label")
    if arc is not None:
        arc.set_text(
            r"$|\lambda-\lambda_\odot|$"
            + "\n"
            + rf"= {p['solar_lon_deg']:.0f}$\degree$"
        )
        mid = np.radians(0.5 * (180.0 + np.degrees(np.arctan2(along[1], along[0]))))
        r_label = 0.85 if layout.is_slide else 0.72
        arc.set_position(tuple(obs + r_label * np.array([np.cos(mid), np.sin(mid)])))
        arc.set_linespacing(1.0)
    lbl = _by_gid(ax, "label/observer")
    if lbl is not None:
        lbl.set_text("telescope")
        lbl.xyann = (0, -9)
        lbl.set_ha("center")
        lbl.set_va("top")
    ro = _by_gid(ax, "label/angles")
    if ro is not None:
        ro.set_visible(False)
    ax.set_xlabel(r"$x$, Sun to telescope [AU]")
    ax.set_ylabel(r"$y$ [AU]")
    _restyle_inset(ax, cast, layout)
    _legible(ax, layout, inset_pt=inset_pt)
    return ax


def zodi_notes(ax, cast, xy, *, ha="left", sign=True):
    """What the top view cannot show: latitude, elongation, the angle's sign."""
    p = ZODI
    text = (
        rf"sightline {p['ecliptic_lat_deg']:.0f}$\degree$ above the ecliptic "
        r"($\beta$), drawn projected;" + "\n"
        rf"its solar elongation is {zodi_elongation_deg():.0f}$\degree$"
    )
    if sign:
        text += (
            "\n"
            rf"drawn at $\lambda-\lambda_\odot$ = {drawn_solar_lon_deg():.0f}$\degree$;"
            + "\nthe table is symmetric in its sign"
        )
    return ex.note(ax, xy, text, cast, transform=ax.transAxes, ha=ha, va="top")


# Exozodiacal disk


def disk_side(
    ax,
    cast,
    layout,
    incl_deg=INCL_DEG,
    *,
    inset_bounds,
    ylim=None,
    inset_pt=None,
    incl_label="upper left",
):
    """The exozodiacal disk side on, with the telescope beyond a scale break.

    Every label that moves with the inclination is anchored where no other
    artist can reach it at any inclination from 0 to 80 degrees: the
    telescope label under the telescope, the scattering-angle label under
    the sightline just past the grain, and the dust-path label at the top,
    left of the sky plane, with a leader to the chord. ``incl_label`` puts
    the inclination label upper left of the star (room while the sightline
    is high) or lower right (every inclination). The sky-plane label stands
    upright along the bottom of its line, on the side away from the far
    half, and the near and far labels sit just beyond their outer ends.
    """
    viz = _viz()
    disk, _ = _renderer()
    r_out = DISK["rmax_AU"]
    res = viz.plot_disk_geometry(
        disk,
        incl_deg=incl_deg,
        pa_deg=PA_DEG,
        grain_radius_AU=GRAIN_RADIUS_AU,
        thickness_AU=LAYER_THICKNESS_AU,
        inset=True,
        inset_bounds=inset_bounds,
        ax=ax,
    )
    dark = cast.mode == "dark"
    purple = cast["exozodi"].color
    for gid in ("disk/layer_near", "disk/layer_far"):
        layer = _by_gid(ax, gid)
        if layer is not None:
            layer.set_alpha(None)
            layer.set_hatch(cast["exozodi"].hatch)
            layer.set_facecolor(to_rgba(purple, 0.3 if dark else 0.14))
            layer.set_edgecolor(to_rgba(purple, 0.0))
            layer.set_hatchcolor(to_rgba(purple, 0.75 if dark else 0.45))
            layer.set_linewidth(0.0)
    for gid in ("disk/near", "disk/far"):
        mid = _by_gid(ax, gid)
        if mid is not None:
            mid.set_linewidth(1.1 * ex.DOC.lw)
    inc = _by_gid(ax, "incident")
    if inc is not None:
        inc.set_zorder(7)
    # The library's "to observer" arrow is replaced by a telescope on the
    # sightline beyond a scale break.
    for gid in ("observer", "label/observer"):
        art = _by_gid(ax, gid)
        if art is not None:
            art.set_visible(False)
    _neutral_sightline(ax, cast, layout)
    ax.set_xlim(-1.2 * r_out, 1.6 * r_out)
    if ylim is not None:
        ax.set_ylim(*ylim)
    y_lo, y_hi = ax.get_ylim()
    sky = _by_gid(ax, "label/sky_plane")
    if sky is not None:
        # Upright along the bottom of its line, right of it: the far half
        # lies at z < 0 below 90 degrees and on the line only face-on.
        sky.set_rotation(90.0)
        sky.set_position((0.07 * r_out, y_lo + 0.02 * (y_hi - y_lo)))
        sky.set_ha("left")
        sky.set_va("bottom")
    # The sightline's chord through the dust layer, labeled from the top so
    # the label never meets the chord, the rays or the disk.
    path = _by_gid(ax, "sightline/path")
    path_label = ax.annotate(
        "path through\nthe dust layer",
        (0.0, 0.0),
        xytext=(-0.04 * r_out, y_hi - 0.03 * (y_hi - y_lo)),
        textcoords="data",
        ha="right",
        va="top",
        color=purple,
        fontsize=layout.small_pt,
        arrowprops={
            "arrowstyle": "-",
            "color": purple,
            "lw": 0.6 * ex.DOC.lw,
            "shrinkA": 2,
            "shrinkB": 3,
            # From the label's lower right corner, beside the sky plane, so
            # the leader stays right of the integration label.
            "relpos": (1.0, 0.0),
        },
    )
    path_label.set_gid("d06/path_label")
    ex.halo(path_label, cast)

    z_break, z_tel = 1.3 * r_out, 1.5 * r_out
    bg = ax.fill([0.0], [0.0], color=cast.background, lw=0, zorder=6)[0]
    (slash1,) = ax.plot([], [], color=cast["scenery"].color, lw=layout.lw, zorder=7)
    (slash2,) = ax.plot([], [], color=cast["scenery"].color, lw=layout.lw, zorder=7)
    slash1.set_gid("d06/break")
    slash2.set_gid("d06/break")
    scope = ex.aperture(ax, (z_tel, 0.0), 0.9, cast)
    scope.set_gid("d06/telescope")
    scope_label = ex.note(
        ax,
        (0.0, 0.0),
        f"telescope,\n{DISK['dist_pc']:.0f} pc away",
        cast,
        ha="right",
        va="top",
        color=cast["aperture"].color,
        fontstyle="normal",
    )
    scope_label.set_gid("d06/telescope_label")
    sight_label = ex.note(ax, (0.0, 0.0), "sightline", cast, ha="left", va="top")
    off = 0.35  # AU, on the side of the sightline away from the star
    arrows = ex.arrow(
        ax,
        (z_break - 0.1, 0.0),
        (-1.1 * r_out, 0.0),
        "integration",
        cast,
        label=r"integrate, $s\geq0$",
        label_offset_pt=(0, 2),
    )
    # The shaft starts at the break, not at s = 0 (the telescope is beyond
    # it), so it carries no start bar.
    arrows[1].set_arrowstyle("-")
    arrows[-1].set_ha("left")
    arrows[-1].set_va("bottom")

    def place(incl):
        res.update(incl)
        grain = _by_gid(ax, "grain").get_offsets()[0]
        z_grain, u = float(grain[0]), float(grain[1])
        line = _by_gid(ax, "sightline")
        if isinstance(line, Line2D):
            line.set_data([-1.2 * r_out, z_tel], [u, u])
        h, w = 0.45, 0.14
        bg.set_xy(
            [
                (z_break - w - 0.2, u - h),
                (z_break + w - 0.2, u - h),
                (z_break + w + 0.2, u + h),
                (z_break - w + 0.2, u + h),
            ]
        )
        for sl, dz in ((slash1, -w), (slash2, w)):
            sl.set_data([z_break + dz - 0.2, z_break + dz + 0.2], [u - h, u + h])
        scope.set_data([z_tel, z_tel], [u - 0.45, u + 0.45])
        scope_label.set_position((1.6 * r_out - 0.1, u - 0.5))
        sight_label.set_position((-1.18 * r_out, u - 0.1))
        _move_integration(arrows, (z_break - 0.3, u + off), (-1.1 * r_out, u + off))
        arrows[-1].xy = (-1.18 * r_out, u + off)
        # Inclination arc: smaller than the library's.
        arc = _by_gid(ax, "inclination")
        if isinstance(arc, Line2D):
            x, y = arc.get_data()
            arc.set_data(0.45 * np.asarray(x), 0.45 * np.asarray(y))
        # Near and far labels just beyond their outer ends: the near label
        # up and to the right (clear of the dust-path label when the disk is
        # nearly face-on, and above the integration arrow when it is nearly
        # edge-on), the far label below its end.
        ir = np.radians(incl)
        d = np.array([np.sin(ir), np.cos(ir)])
        far_dx = 0.3 * (np.sin(ir) - np.cos(ir))
        for gid, pos in (
            ("label/near_side", r_out * d + np.array([0.8, 0.6])),
            ("label/far_side", -r_out * d + np.array([far_dx, -0.6])),
        ):
            t = _by_gid(ax, gid)
            if t is not None:
                t.set_position(tuple(pos))
        lab = _by_gid(ax, "inclination/label")
        if lab is not None:
            if incl_label == "upper left":
                lab.set_position((-0.15, float(np.clip(u - 0.3, 0.15, 0.45))))
                lab.set_ha("right")
                lab.set_va("center")
            else:
                lab.set_position((0.3, -0.3))
                lab.set_ha("left")
                lab.set_va("top")
        # Scattering angle: under the sightline where the arc meets it, so
        # the label never sits on the grain or the dust path.
        theta = _by_gid(ax, "scattering_angle/label")
        if theta is not None:
            theta.set_position((z_grain + 0.85, u - 0.25))
            theta.set_ha("center")
            theta.set_va("top")
        if path is not None:
            zs = np.asarray(path.get_xdata(), float)[:2]
            if np.all(np.isfinite(zs)):
                path_label.xy = (float(zs.mean()), u)
            path_label.set_visible(bool(np.all(np.isfinite(zs))))

    _restyle_inset(ax, cast, layout)
    ax.set_xlabel(r"$z$, toward the telescope [AU]")
    ax.set_ylabel("along the projected\nminor axis [AU]")
    _legible(ax, layout, inset_pt=inset_pt)
    place(incl_deg)
    return place


def disk_sky(ax, cast, layout, image, incl_deg=INCL_DEG, *, cbar_label, edges=False):
    """The projected image of the synthetic disk.

    The axes are the line of nodes and the projected minor axis, drawn
    right-handed with +z toward the telescope, as in the side view; no
    east or north is implied.
    """
    viz = _viz()
    res = viz.plot_disk_image(
        image,
        incl_deg=incl_deg,
        pa_deg=PA_DEG,
        pixel_scale_arcsec=DISK["pixel_scale_arcsec"],
        dist_pc=DISK["dist_pc"],
        radii_AU=(DISK["rmin_AU"], DISK["rmax_AU"]),
        grain_radius_AU=GRAIN_RADIUS_AU,
        dynamic_range=DYNAMIC_RANGE,
        vmax=1.0,
        colorbar=True,
        ax=ax,
    )
    res.artists["image"].set_cmap(ex.image_cmap("intensity"))
    res.artists["cbar"].set_label(cbar_label)
    half = 0.5 * DISK["n_pix"] * DISK["pixel_scale_arcsec"]
    ax.set_xlim(-half, half)
    ax.set_xlabel("along the line of nodes [arcsec]")
    ax.set_ylabel("along the projected\nminor axis [arcsec]")
    # Near and far half labels in the left corners, each with a leader to
    # its half of the ring on the minor-axis side of the image.
    halves = []
    for name, y_text, va in (("near half", 0.97, "top"), ("far half", 0.03, "bottom")):
        ann = ax.annotate(
            name,
            (0.0, 0.0),
            xytext=(0.03, y_text),
            textcoords="axes fraction",
            ha="left",
            va=va,
            color=cast.text,
            fontsize=layout.small_pt,
            arrowprops={
                "arrowstyle": "-",
                "color": cast.text,
                "lw": 0.6 * ex.DOC.lw,
                "shrinkA": 1,
                "shrinkB": 0,
            },
        )
        ann.set_gid(f"d06/{name.replace(' ', '_')}")
        halves.append(ex.halo(ann, cast))
    near, far = halves

    def place_halves(incl):
        """Point each leader at its half of the ring's middle radius."""
        x, y = ring_point_arcsec(incl)
        near.xy = (x, y)
        far.xy = (x, -y)

    place_halves(incl_deg)
    if edges:
        c = np.cos(np.radians(incl_deg))
        scale = float(au_to_arcsec(1.0, DISK["dist_pc"]))  # arcsec per AU
        # Leader lines to points on each projected edge (ellipse parameter t).
        for r, t_deg, xt, yt, va in (
            (DISK["rmin_AU"], -40.0, 0.27, -0.3, "top"),
            (DISK["rmax_AU"], 40.0, 0.3, 0.4, "bottom"),
        ):
            t = np.radians(t_deg)
            ann = ax.annotate(
                f"{r:g} AU edge",
                (r * scale * np.cos(t), r * c * scale * np.sin(t)),
                xytext=(xt, yt),
                textcoords="data",
                ha="center",
                va=va,
                color=cast.text,
                fontsize=layout.small_pt,
                arrowprops={"arrowstyle": "-", "color": cast.text, "lw": 0.6},
            )
            ex.halo(ann, cast)
    _legible(ax, layout)
    return res, (near, far), place_halves


# Figures


def _divider(fig, xy0, xy1, cast, layout):
    fig.add_artist(
        Line2D(
            [xy0[0], xy1[0]],
            [xy0[1], xy1[1]],
            transform=fig.transFigure,
            color=cast["scenery"].color,
            ls=":",
            lw=layout.lw,
        )
    )


INPUTS_ZODI = (
    r"Leinert table inputs: $\beta$, $|\lambda-\lambda_\odot|$, observer distance, "
    "wavelength"
)
INPUTS_DISK = (
    r"model inputs: density $n(r,z)$, phase $p(\Theta)$, orientation ($i$, "
    r"position angle), stellar luminosity $L_\star$, distance $D$"
)
TITLE_ZODI = "Local zodiacal light: inside the Sun's dust cloud"
TITLE_DISK = "Exozodiacal light: far outside the host's disk"
TOP_TITLE = r"top view from the ecliptic north pole"


def build_two_systems(layout, cast):
    """The opening figure: two systems, one scattering process.

    The documentation still has two rows: the local zodiacal geometry from
    the ecliptic pole with its grain drawn large beside it, then the disk
    side on (with its grain inset) and its projected image. The talk still
    keeps the two geometry panels in two columns.
    """
    if layout.is_slide:
        fig = plt.figure(figsize=layout.size(), layout="constrained")
        left, right = fig.subfigures(1, 2, width_ratios=[0.9, 1.1], wspace=0.03)
        ax_top = left.subplots()
        ax_side = right.subplots()
        left.suptitle(TITLE_ZODI.replace(": ", ":\n"), fontsize=layout.title_pt)
        right.suptitle(TITLE_DISK.replace(": ", ":\n"), fontsize=layout.title_pt)
        zodi_top(
            ax_top,
            cast,
            layout,
            inset_bounds=(0.0, 0.0, 0.36, 0.36),
        )
        _panel_title(ax_top, "(a) " + TOP_TITLE, layout)
        zodi_notes(ax_top, cast, (0.02, 0.9), sign=False)
        ex.badge(ax_top, cast, STATUS, loc="upper left")
        disk_side(
            ax_side,
            cast,
            layout,
            inset_bounds=(0.63, 0.02, 0.36, 0.36),
            ylim=(-4.2, 5.8),
            inset_pt=0.8 * layout.small_pt,
        )
        _panel_title(ax_side, "(b) side on: inclination $i$", layout)
        ex.badge(ax_side, cast, STATUS, loc="upper right")
        _divider(fig, (0.45, 0.04), (0.45, 0.97), cast, layout)
        left.supxlabel(
            INPUTS_ZODI, fontsize=layout.small_pt, color=cast["local_zodi"].color
        )
        right.supxlabel(
            INPUTS_DISK.replace("angle), ", "angle),\n"),
            fontsize=layout.small_pt,
            color=cast["exozodi"].color,
        )
        return _plain_text(fig, cast)

    fig = plt.figure(figsize=layout.size(7.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.12, 1.0], hspace=0.05)
    ax_top = top.subplots()
    top.suptitle(TITLE_ZODI, fontsize=layout.title_pt)
    zodi_top(
        ax_top,
        cast,
        layout,
        inset_bounds=(1.08, 0.42, 0.62, 0.58),
        inset_pt=layout.small_pt,
    )
    _panel_title(ax_top, "(a) " + TOP_TITLE, layout)
    ex.note(
        ax_top,
        (1.08, 1.02),
        "the grain, magnified in its scattering plane",
        cast,
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
    )
    zodi_notes(ax_top, cast, (1.08, 0.36))
    ex.badge(ax_top, cast, STATUS, loc="upper left")
    top.supxlabel(INPUTS_ZODI, fontsize=layout.small_pt, color=cast["local_zodi"].color)

    ax_side, ax_img = bottom.subplots(1, 2, width_ratios=[1.5, 1.0])
    bottom.suptitle(TITLE_DISK, fontsize=layout.title_pt)
    disk_side(
        ax_side,
        cast,
        layout,
        inset_bounds=(0.575, 0.015, 0.42, 0.37),
        ylim=(-4.2, 5.8),
        inset_pt=7.0,
    )
    _panel_title(ax_side, "(b) side on: inclination $i$", layout)
    ex.badge(ax_side, cast, STATUS, loc="upper right")
    disk_sky(
        ax_img,
        cast,
        layout,
        still_image(),
        cbar_label="surface brightness / image peak",
        edges=True,
    )
    _panel_title(ax_img, "(c) projected image", layout)
    ex.badge(ax_img, cast, STATUS_IMAGE, loc="lower right")
    bottom.supxlabel(INPUTS_DISK, fontsize=layout.small_pt, color=cast["exozodi"].color)
    _divider(fig, (0.03, 0.47), (0.97, 0.47), cast, layout)
    return _plain_text(fig, cast)


# Animation


def _phase_panel(ax, cast, layout):
    """The declared phase function with the two minor-axis grains on it.

    The two grains are named by a fixed key in the empty upper right, so no
    label follows a marker into the frame or the title.
    """
    theta = np.linspace(0.0, 180.0, 181)
    g = DISK["g_HG"]
    purple = cast["exozodi"].color
    ax.plot(theta, hg_phase(theta, g), color=purple, lw=layout.lw)
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 1.12 * hg_phase(0.0, g))
    ax.set_xticks([0, 90, 180])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel(r"scattering angle $\Theta$ [deg]", labelpad=1)
    ax.set_ylabel(r"$p(\Theta)$ [sr$^{-1}$]", labelpad=1)
    ax.set_title(
        f"Henyey-Greenstein, g = {g:g}",
        loc="left",
        fontsize=layout.small_pt,
    )
    size = cast.layout.marker_pt
    filled = {"color": purple, "ms": size}
    hollow = {"mfc": cast.background, "mec": purple, "ms": size}
    (near,) = ax.plot([], [], "o", zorder=5, **filled)
    (far,) = ax.plot([], [], "o", zorder=5, **hollow)
    for y, style, text in (
        (0.84, filled, r"near-half grain, $\Theta=90\degree-i$"),
        (0.64, hollow, r"far-half grain, $\Theta=90\degree+i$"),
    ):
        ax.plot([0.4], [y], "o", transform=ax.transAxes, clip_on=False, **style)
        ax.text(
            0.44,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=cast.text,
            fontsize=layout.small_pt,
        )

    def place(incl):
        tn, tf = disk_minor_axis_theta(incl), disk_minor_axis_theta(incl, near=False)
        near.set_data([tn], [hg_phase(tn, g)])
        far.set_data([tf], [hg_phase(tf, g)])

    return place


def _view_panel(ax, cast, layout):
    """Disk frame: the disk stays put and the direction to the telescope turns.

    The telescope label hangs off the arrow tip on the side away from the
    arc, and the normal is labeled on its left above the arrow head, so
    neither meets the arc or the arrow at any inclination.
    """
    ax.set_xlim(-1.25, 2.3)
    ax.set_ylim(-0.4, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ex.region(ax, "exozodi", Rectangle((-1.0, -0.06), 2.0, 0.12), cast)
    ex.mark(ax, "star", (0.0, 0.0), cast, scale=0.6)
    ax.plot(
        [0.0, 0.0],
        [0.0, 1.15],
        color=cast["scenery"].color,
        ls="--",
        lw=0.8 * layout.lw,
    )
    ex.note(ax, (-0.08, 1.05), "normal", cast, ha="right", va="center")
    ex.note(ax, (0.0, -0.12), "disk, fixed", cast, ha="center", va="top")
    arrow = ex.arrow(ax, (0.0, 0.0), (0.0, 1.0), "ray", cast, color=cast.text)[0]
    (arc,) = ax.plot([], [], color=cast.text, lw=layout.lw)
    label = ex.halo(
        ax.text(
            -1.2,
            0.1,
            "",
            color=cast.text,
            ha="left",
            va="bottom",
            fontsize=layout.small_pt,
        ),
        cast,
    )
    label.set_gid("d06/view_incl")
    if layout.is_slide:
        arrow.set_mutation_scale(0.6 * arrow.get_mutation_scale())
    tel = ex.halo(
        ax.text(
            0,
            0,
            "to telescope",
            color=cast.text,
            ha="left",
            va="center",
            fontsize=layout.small_pt,
        ),
        cast,
    )
    tel.set_gid("d06/view_telescope")

    def place(incl):
        i = np.radians(incl)
        tip = np.array([np.sin(i), np.cos(i)])
        arrow.set_positions((0.0, 0.0), tuple(0.85 * tip))
        t = np.linspace(0.0, i, 30)
        arc.set_data(0.45 * np.sin(t), 0.45 * np.cos(t))
        label.set_text(rf"$i$ = {incl:.0f}$\degree$")
        tel.set_position(tuple(0.85 * tip + np.array([0.08, 0.0])))

    return place


def build_sweep(layout, cast):
    """Inclination sweep of the synthetic disk, camera distance and scale fixed."""
    incls = sweep_incls(layout)
    images = disk_images(incls)
    height = 6.0 if not layout.is_slide else None
    fig = plt.figure(figsize=layout.size(height), layout="constrained")
    fig.suptitle(
        "Same disk, seen from different inclinations",
        fontsize=layout.title_pt,
        color=cast.text,
    )
    note = (
        "only the viewing direction turns; the disk, its distance and the "
        "color scale are fixed"
    )
    note_kw = {
        "fontsize": layout.small_pt,
        "fontstyle": cast["annotation"].fontstyle,
        "color": cast["annotation"].color,
    }
    band_ratio = 1.6 if layout.is_slide else 1.75
    band, main = fig.subfigures(2, 1, height_ratios=[band_ratio, 3.2])
    band.suptitle(note, **note_kw)
    ax_phase, ax_view = band.subplots(1, 2, width_ratios=[1.3, 1.0])
    place_phase = _phase_panel(ax_phase, cast, layout)
    place_view = _view_panel(ax_view, cast, layout)
    ax_side, ax_img = main.subplots(1, 2, width_ratios=[1.3, 1.0])
    place_side = disk_side(
        ax_side,
        cast,
        layout,
        incls[0],
        inset_bounds=(0.6, 0.015, 0.395, 0.31 if layout.is_slide else 0.34),
        inset_pt=7.5 if not layout.is_slide else 0.8 * layout.small_pt,
        # Headroom for the two-line dust-path label above the integration
        # label when the sightline is highest (face-on).
        ylim=(-6.0, 7.0),
        incl_label="lower right",
    )
    sky, (near, far), place_halves = disk_sky(
        ax_img,
        cast,
        layout,
        images[0],
        incls[0],
        cbar_label="surface brightness / brightest frame",
    )
    _panel_title(ax_side, "side on, in the sky frame", layout)
    _panel_title(ax_img, "projected image", layout)
    ex.badge(ax_side, cast, STATUS, loc="upper right")
    ex.badge(ax_img, cast, STATUS_IMAGE, loc="lower right")

    def draw(fig_, frame):
        del fig_
        incl = frame["incl"]
        place_side(incl)
        place_phase(incl)
        place_view(incl)
        sky.update(images[frame["index"]], incl_deg=incl)
        place_halves(incl)
        tilted = incl >= 5.0
        near.set_visible(tilted)
        far.set_visible(tilted)

    frames = [{"index": k, "incl": v} for k, v in enumerate(incls)]
    draw(fig, frames[0])
    _plain_text(fig, cast)
    return ex.AnimationScene(fig=fig, draw=draw, frames=frames)


# Captions

_PARAMS = {
    "zodi": ZODI,
    "cloud_envelope_AU": CLOUD_AU,
    "disk": DISK,
    "incl_deg": INCL_DEG,
    "pa_deg": PA_DEG,
    "grain_radius_AU": GRAIN_RADIUS_AU,
    "layer_thickness_AU": LAYER_THICKNESS_AU,
    "dynamic_range": DYNAMIC_RANGE,
}

TWO_SYSTEMS_CAPTION = (
    "Two separate systems, one scattering process (schematic, not to scale; the "
    "sections Rays, frames and angles and The single-scattering integral). Top, "
    "local zodiacal light: the telescope sits inside the Solar-system dust cloud, "
    "1 AU from the Sun, and integrates along the half-ray "
    r"$\boldsymbol x(s)=\boldsymbol x_{\rm obs}+s\hat{\boldsymbol n}$, $s\ge0$ "
    "(dashed integration arrow). (a) The view from the ecliptic north pole; the "
    "dotted circle is the Earth orbit at 1 AU. The sightline differs from the Sun "
    r"in ecliptic longitude by $|\lambda-\lambda_\odot|=135^\circ$ (the arc) and "
    r"rises $\beta=30^\circ$ above the ecliptic, so it is drawn projected; its "
    r"true angle from the Sun, the solar elongation $\epsilon$ with "
    r"$\cos\epsilon=\cos\beta\cos(\lambda-\lambda_\odot)$, is $128^\circ$. It is "
    r"drawn at $\lambda-\lambda_\odot=-135^\circ$, and the Leinert table is "
    "symmetric in that sign. The cloud outline is a schematic extent, not a "
    "density model. The grain 1 AU along the sightline receives sunlight "
    r"(yellow ray) and sends light to the telescope along $-\hat{\boldsymbol n}$ "
    "(green ray). Beside it the grain is drawn in its own scattering plane: "
    r"scattering angle $\Theta=154^\circ$, measured from the incident propagation "
    r"direction to $-\hat{\boldsymbol n}$ with $\Theta=0$ forward, and illumination "
    r"angle $\alpha=\pi-\Theta=26^\circ$. Bottom, exozodiacal light: the same "
    "telescope is about 10 pc from a host star, far outside its disk; the scale "
    r"break marks the omitted distance. (b) The disk side on at inclination $i=60^\circ$, "
    "in the plane of the sightline and the projected minor axis; the layer "
    "thickness is schematic. A midplane grain on the minor axis of the near half, "
    "the half toward the telescope in the chapter's proposed near-side profile, "
    r"scatters forward at $\Theta=90^\circ-i=30^\circ$ (inset); the sightline's "
    "path through the dust layer is highlighted and labeled. (c) The projected image of a "
    "synthetic GRaTeR disk (Augereau et al. 1999, section 3.1; edges 1.5 and 5 AU, "
    "Henyey-Greenstein asymmetry 0.4) at 550 nm, rendered by the skyscapes "
    "GraterDisk example implementation and divided by its peak: the "
    "forward-scattering near half is brighter. Its axes are the line of nodes and "
    "the projected minor axis, with the minor axis as in (b); the image is "
    "symmetric about the minor axis, and how the axes map to east and north "
    "depends on the pending observer basis. The circle marks the sightline of (b). "
    "The disk orientation is set independently of any planet orbit. The input "
    "lines name what each profile takes: the Leinert table (Leinert et al. 1998, "
    "section 8.1, equation 14) returns the brightness along one sightline from "
    "inside the cloud and supports no outside view; the parametric disk is "
    "evaluated for a distant observer from its density, phase function, "
    r"orientation (inclination and position angle), stellar luminosity $L_\star$ and distance $D$. Both pictures "
    "keep the chapter's scope: optically thin single scattering of a point star, "
    "with thermal emission, extinction and polarization excluded. The analytic "
    "ray-support and scattering-angle fixtures follow in "
    "{ref}`the dust geometry figure <fig-dust-geometry>`."
)
TWO_SYSTEMS_ALT = (
    "Two rows separated by a dotted line. Top row, local zodiacal light: a view "
    "from the ecliptic north pole with the Sun at the center of a hatched green "
    "cloud outline and a telescope on the dotted 1 AU Earth orbit; a thin "
    "sightline leaves the telescope at 135 degrees in longitude from the Sun "
    "with a dashed integration arrow beside it, and a grain 1 AU out on it "
    "receives a yellow ray from the Sun and sends a green ray back along the "
    "sightline to the telescope. Beside the view, the grain is magnified in its "
    "scattering plane with a scattering angle of 154 degrees and an "
    "illumination angle of 26 degrees, and notes say the sightline rises 30 "
    "degrees above the ecliptic and its solar elongation is 128 degrees. "
    "Bottom row, exozodiacal light: a disk seen side on, tilted 60 degrees from "
    "the sky plane, with near and far halves; a grain on the near half receives a "
    "yellow ray from the star and sends a purple ray along the sightline to a "
    "telescope 10 pc away beyond a scale break; a dashed integration arrow runs "
    "from the break through the disk, and the path through the dust layer is "
    "labeled; an inset gives a scattering angle of 30 "
    "degrees and an illumination angle of 150 degrees. Beside it, the projected "
    "image: an inclined ring whose upper, near half is brighter, with leaders "
    "to the near and far halves, a circle on the marked sightline and its 1.5 "
    "and 5 AU edges labeled."
)
SWEEP_CAPTION = (
    "The viewpoint changes, the disk does not (a simulated example with schematic "
    "geometry; the section Rays, frames and angles, inclination and near side, a "
    "proposed profile). The synthetic disk of the opening figure is seen at "
    "inclinations from 0 to 80 degrees with its density, position angle and "
    "distance fixed; the small disk-frame glyph shows the direction to the "
    "telescope turning while the disk stays put. Left: the sightline through a "
    "midplane grain 3 AU out on the near half's minor axis, with its path through "
    "the schematic dust layer highlighted and the integration running from the "
    "telescope beyond the scale break; the inset gives the grain's scattering "
    r"angle $\Theta=90^\circ-i$ and illumination angle $\alpha=\pi-\Theta$. Top: "
    "the disk's Henyey-Greenstein phase function (asymmetry 0.4) with the "
    r"near-side grain at $\Theta=90^\circ-i$ and its far-side mirror at "
    r"$\Theta=90^\circ+i$. Right: the skyscapes GraterDisk image at 550 nm, every "
    "frame on one logarithmic scale divided by the brightest frame; the circle is "
    "the same sightline. Two effects change the image as the disk tilts. The "
    r"path through a thin layer lengthens as $h/|\cos i|$, which brightens the "
    "whole ring. The near and far grains move to opposite ends of the phase "
    r"function: from $i=0$ to $80^\circ$ the near grain's $p(\Theta)$ rises from "
    r"0.054 to 0.29 sr$^{-1}$ and the far grain's falls from 0.054 to 0.025 "
    r"sr$^{-1}$, which brightens the near half and dims the far half, so the "
    "near half ends brighter than the far half. The sweep "
    "stops at 80 degrees, short of edge-on and of the pending sense of "
    r"$i>90^\circ$. Optically thin single scattering; thermal emission excluded."
)
SWEEP_ALT = (
    "Animation. Along the top: the title same disk, seen from different "
    "inclinations; a phase-function curve with a filled near-half dot and a "
    "hollow far-half dot moving apart as the inclination grows, named in a "
    "fixed key; and a small fixed disk with an arrow to the telescope swinging "
    "away from the disk normal. Below left: the disk side on, "
    "its midplane turning from vertical to nearly horizontal while a grain on its "
    "near half stays on a horizontal sightline to a telescope beyond a scale "
    "break, with the path through the dust layer highlighted and labeled and an "
    "inset giving "
    "the scattering angle as it falls from 90 degrees. Below right: the projected "
    "image on a fixed color scale, a ring that flattens into an ellipse and "
    "brightens, its near half more than its far half."
)

FIGURES = [
    ex.FigureSpec(
        slug="d06-dust-two-systems",
        build=build_two_systems,
        caption=TWO_SYSTEMS_CAPTION,
        alt=TWO_SYSTEMS_ALT,
        status=STATUS,
        params=_PARAMS,
    ),
]

ANIMATIONS = [
    ex.AnimationSpec(
        slug="d06-dust-inclination-sweep",
        build=build_sweep,
        ground="viewpoint",
        caption=SWEEP_CAPTION,
        alt=SWEEP_ALT,
        status=STATUS_IMAGE,
        params={**_PARAMS, "sweep_deg": SWEEP_DEG, "sweep_frames": SWEEP_FRAMES},
        hold_s=(1.0, 4.0),
        preview_frames=(0, 15, 29),
    ),
]
