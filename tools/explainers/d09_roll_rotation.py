"""Telescope roll is not active image rotation.

Teaches that a positive telescope roll turns the detector basis under a
fixed sky, so a fixed source lands in a new pixel and the sky appears to turn
clockwise in detector coordinates, while an active image rotation is an
operation on stored data that moves the content counterclockwise in fixed
pixels. The pixels drawn are the exact examples of the optics chapter
(section "East/north, telescope roll, and active image rotation"): on a
4-by-4 grid the source at (x, y) = (1.5 s, 0.5 s) occupies (r, c) = (2, 3),
moves to (0, 2) after +90 degrees of roll, and to (3, 1) after an active
+90-degree rotation. The tests recompute them by hand.

Motion ground for the animation: viewpoint. The detector turns under the
fixed sky while the roll angle runs from 0 to +90 degrees, and the stored
array beside it shows where the fixed source falls; the still carries the
two end states and the active rotation without it.
"""

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Affine2D

from explainers import _common as ex

# The chapter's worked example, in units of the pixel spacing s.
N_GRID = 4
SOURCE_XY = (1.5, 0.5)
ROLL_DEG = 90.0
ACTIVE_DEG = 90.0

# Sky-panel half width, in units of s; the grid spans -2 to +2.
SKY_LIM = 3.5
SKY_LIM_ANIM = 3.9  # the turning grid reaches a radius of about 3.5
EDGE_GAP = 0.28  # offset of the detector axis arrows from the grid edge
GHOST_GAP = 0.62  # offset of the faint roll-0 arrows, outside the live ones
WEDGE = 0.8  # side of the corner wedge that marks pixel (0, 0)
HEAD_MIN_DEG = 40.0  # shortest animated roll arc that carries an arrowhead


# Geometry (pure functions; the tests check them against hand arithmetic)


def optical_origin(n=N_GRID):
    """Geometric optical origin (c_x = c_y) of an n-by-n grid."""
    return (n - 1) / 2.0


def rotation(alpha_deg):
    """The chapter's R(alpha): counterclockwise by alpha in the (x, y) plane."""
    a = np.deg2rad(alpha_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def detector_xy(sky_xy, roll_deg):
    """Detector coordinates d = R(-theta) s of a fixed sky source."""
    return rotation(-roll_deg) @ np.asarray(sky_xy, dtype=float)


def actively_rotated_xy(xy, alpha_deg):
    """Where an active rotation by alpha sends the content at ``xy``."""
    return rotation(alpha_deg) @ np.asarray(xy, dtype=float)


def pixel_of(xy, n=N_GRID):
    """The (row, column) holding physical ``(x, y)``, in units of s."""
    c0 = optical_origin(n)
    col = int(np.floor(xy[0] + c0 + 0.5))
    row = int(np.floor(xy[1] + c0 + 0.5))
    return row, col


def cases():
    """The three states of the still: stored pixel and detector (x, y)."""
    before = np.asarray(SOURCE_XY, dtype=float)
    rolled = detector_xy(before, ROLL_DEG)
    active = actively_rotated_xy(before, ACTIVE_DEG)
    return {
        "zero": {"xy": before, "pixel": pixel_of(before)},
        "roll": {"xy": rolled, "pixel": pixel_of(rolled)},
        "active": {"xy": active, "pixel": pixel_of(active)},
    }


def roll_states(layout):
    """Roll angles shown by the animation, as self-contained frame states."""
    step = 5.0 if not layout.is_slide else 2.0
    n = round(ROLL_DEG / step) + 1
    thetas = np.linspace(0.0, ROLL_DEG, layout.n_frames(n))
    return [{"theta": float(t)} for t in thetas]


# Text without stroked halos


def _backing(cast):
    """A plain backing box in the background color, for a label over marks."""
    return {
        "boxstyle": "round,pad=0.15,rounding_size=0.2",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _label(ax, xy, text, cast, *, color=None, box=True, **kw):
    """A label in data coordinates, on the plain backing box."""
    kw.setdefault("fontsize", cast.layout.small_pt)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    t = ax.text(*xy, text, color=cast.text if color is None else color, **kw)
    if box:
        t.set_bbox(_backing(cast))
    return t


def _plain_text(fig, cast):
    """Replace any stroked halo a shared helper added with the backing box."""
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
            if text.get_bbox_patch() is None:
                text.set_bbox(_backing(cast))
    return fig


def _ink(cast):
    """Ink for construction lines: readable, below the entities."""
    return cast.neutral(0.7)


# Shared pieces


def _vector(ax, start, end, color, lw, cast, *, transform=None, zorder=4):
    """A plain direction arrow (a basis vector, not a ray or a data arrow)."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=cast.layout.marker_pt * 1.3,
        color=color,
        lw=lw,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    if transform is not None:
        patch.set_transform(transform)
    ax.add_patch(patch)
    return patch


def _sky_compass(ax, cast, origin):
    """Sky-chart axes: x_sky = E to the right, y_sky = N up (example chart).

    Drawn in the lower left corner of a sky panel, clear of the detector's
    own arrows, with each letter just beyond its arm.
    """
    color = cast.text
    length = 0.6
    ox, oy = origin
    _vector(ax, (ox, oy), (ox + length, oy), color, cast.layout.lw, cast)
    _vector(ax, (ox, oy), (ox, oy + length), color, cast.layout.lw, cast)
    _label(ax, (ox + length + 0.1, oy), "E", cast, color=color, ha="left", box=False)
    _label(ax, (ox, oy + length + 0.1), "N", cast, color=color, va="bottom", box=False)


def _source(ax, xy, cast, *, ghost=False, zorder=6):
    """The fixed off-axis source (a planet), or its earlier place as a ghost."""
    kw = cast["planet"].marker_kw(0.9 * cast.layout.marker_pt)
    if ghost:
        kw.update(markerfacecolor="none", alpha=0.75)
    (line,) = ax.plot([xy[0]], [xy[1]], zorder=zorder, **kw)
    return line


def _star(ax, cast):
    """The star on the optical axis, at the physical origin (0, 0)."""
    return ex.mark(ax, "star", (0.0, 0.0), cast, scale=0.55)


def _pixel_patch(pixel, cast, *, transform=None):
    """The highlighted pixel holding the source, in detector (x, y)."""
    c0 = optical_origin()
    r, c = pixel
    patch = Rectangle(
        (c - c0 - 0.5, r - c0 - 0.5),
        1.0,
        1.0,
        facecolor=to_rgba(cast["planet"].color, 0.3),
        edgecolor=cast["planet"].color,
        lw=cast.layout.lw,
        zorder=2.5,
        gid="d09-lit-pixel",
    )
    if transform is not None:
        patch.set_transform(transform)
    return patch


def _detector(ax, cast, rot, *, lit=None, faint=False):
    """The 4-by-4 detector drawn in detector (x, y), carried by ``rot``.

    ``rot`` is ``Affine2D`` (the roll) composed with the data transform, so
    the whole detector turns with one mutable transform. The corner pixel
    (0, 0) carries a small wedge, because a turned square grid looks like
    the unturned one; the column and row arrows along its edges give the
    detector basis.
    """
    ent = cast["detector"]
    alpha = 0.45 if faint else 1.0
    half = N_GRID / 2.0
    art = {"lines": []}
    for k in range(1, N_GRID):
        v = -half + k
        for xs, ys in (([v, v], [-half, half]), ([-half, half], [v, v])):
            (ln,) = ax.plot(
                xs,
                ys,
                color=cast.neutral(0.45),
                lw=0.5 * ent.lw,
                transform=rot,
                alpha=alpha,
                zorder=2,
            )
            art["lines"].append(ln)
    frame = Rectangle(
        (-half, -half),
        2 * half,
        2 * half,
        facecolor="none",
        edgecolor=ent.color,
        lw=1.5 * ent.lw,
        alpha=alpha,
        zorder=3,
    )
    frame.set_transform(rot)
    ax.add_patch(frame)
    wedge = Polygon(
        [(-half, -half), (-half + WEDGE, -half), (-half, -half + WEDGE)],
        closed=True,
        facecolor=ent.color,
        edgecolor="none",
        alpha=alpha,
        zorder=3,
    )
    wedge.set_transform(rot)
    ax.add_patch(wedge)
    y_edge = -half - EDGE_GAP
    art["col_arrow"] = _vector(
        ax,
        (-half, y_edge),
        (half + 0.25, y_edge),
        ent.color,
        cast.layout.lw,
        cast,
        transform=rot,
    )
    art["row_arrow"] = _vector(
        ax,
        (-half - EDGE_GAP, -half),
        (-half - EDGE_GAP, half + 0.25),
        ent.color,
        cast.layout.lw,
        cast,
        transform=rot,
    )
    for artist in (art["col_arrow"], art["row_arrow"]):
        artist.set_alpha(alpha)
    art["frame"], art["wedge"] = frame, wedge
    if lit is not None:
        art["lit"] = ax.add_patch(_pixel_patch(lit, cast, transform=rot))
    return art


def _axis_labels(ax, cast, theta_deg, *, faint=False):
    """Name the detector arrows c and r at their heads, upright text."""
    half = N_GRID / 2.0
    color = cast["detector"].color
    rot = rotation(theta_deg)
    tips = {
        "c": (half + 0.55, -half - EDGE_GAP),
        "r": (-half - EDGE_GAP, half + 0.55),
    }
    labels = {}
    for name, det in tips.items():
        xy = rot @ np.asarray(det)
        labels[name] = _label(
            ax, xy, name, cast, color=color, fontstyle="italic", box=False
        )
        if faint:
            labels[name].set_alpha(0.45)
    return labels


def _sky_axes(ax, lim):
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _sky_panel(ax, cast, theta_deg, lit, *, faint=False, lim=SKY_LIM, compass=None):
    """Physical view in the sky chart: fixed sky, detector rolled by theta."""
    _sky_axes(ax, lim)
    rot = Affine2D().rotate_deg(theta_deg)
    art = _detector(ax, cast, rot + ax.transData, lit=lit, faint=faint)
    art["affine"] = rot
    art["labels"] = _axis_labels(ax, cast, theta_deg, faint=faint)
    _star(ax, cast)
    art["source"] = _source(ax, SOURCE_XY, cast)
    corner = (-lim + 0.15, -lim + 0.2)
    _sky_compass(ax, cast, corner if compass is None else compass)
    return art


def _ghost_basis(ax, cast):
    """Faint dashed column and row arrows where they pointed at roll 0.

    They sit outside the turned detector's own arrows, so a quarter turn,
    which leaves a square outline unchanged, still reads as a turn.
    """
    half = N_GRID / 2.0
    off = half + GHOST_GAP
    color = to_rgba(cast["detector"].color, 0.5)
    for start, end in (((-half, -off), (half, -off)), ((-off, -half), (-off, half))):
        patch = _vector(ax, start, end, color, 0.8 * cast.layout.lw, cast)
        patch.set_linestyle((0, (3, 2)))
    # Only the roll-0 r arrow is named: the roll arc names where c pointed.
    _label(
        ax,
        (-off, half + 0.12),
        "r",
        cast,
        color=color,
        fontstyle="italic",
        box=False,
        va="bottom",
    )


def _roll_arc(ax, cast):
    """The roll angle, outside the frame: from E (the roll-0 c) to the new c.

    The vertex is where the new column arrow's line meets the roll-0 column
    arrow's line, at the lower right; dotted rays mark both directions.
    """
    half = N_GRID / 2.0
    vx, vy = half + EDGE_GAP, -(half + GHOST_GAP)
    ray = {"color": _ink(cast), "lw": 0.7 * cast.layout.lw, "ls": ":", "zorder": 3}
    ax.plot([half, vx + 0.62], [vy, vy], **ray)
    ax.plot([vx, vx], [vy, -half], **ray)
    ex.angle_arc(
        ax,
        (vx, vy),
        0.0,
        ROLL_DEG,
        cast,
        radius=0.46,
        label=r"$\theta$",
        color=cast.text,
        label_radius=1.75,
    )


def _name_marks(ax, cast):
    """Direct labels for the star and the source in a sky panel."""
    _label(ax, (0.0, -0.42), "star", cast, color=cast.text, va="top")
    _label(
        ax,
        (SOURCE_XY[0], SOURCE_XY[1] + 0.62),
        "source",
        cast,
        color=cast["planet"].color,
        va="bottom",
    )


def _pixel_text(pixel):
    return f"(r, c) = ({pixel[0]}, {pixel[1]})"


def _array_axes(ax, cast):
    """The stored array, displayed with origin lower: column c, row r."""
    lim = (-0.5, N_GRID - 0.5)
    ax.set(xlim=lim, ylim=lim, aspect="equal")
    ax.set_xticks(range(N_GRID))
    ax.set_yticks(range(N_GRID))
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("column c", labelpad=1)
    ax.set_ylabel("row r", labelpad=1)
    ex.pixel_grid(ax, (-0.5, -0.5), (N_GRID, N_GRID), 1.0, cast)
    for spine in ax.spines.values():
        spine.set_visible(False)
    c0 = optical_origin()
    ex.mark(ax, "star", (c0, c0), cast, scale=0.55)


def _index(xy):
    """Physical (x, y) in units of s to display coordinates (c, r)."""
    c0 = optical_origin()
    return (xy[0] + c0, xy[1] + c0)


def _lit_cell(ax, pixel, cast):
    r, c = pixel
    patch = Rectangle(
        (c - 0.5, r - 0.5),
        1.0,
        1.0,
        facecolor=to_rgba(cast["planet"].color, 0.3),
        edgecolor=cast["planet"].color,
        lw=cast.layout.lw,
        zorder=2.5,
    )
    return ax.add_patch(patch)


def _turn_arc(ax, start_xy, end_deg, cast, color):
    """Arc about the optical origin from ``start_xy`` through ``end_deg``."""
    c0 = optical_origin()
    radius = float(np.hypot(*start_xy))
    a0 = float(np.degrees(np.arctan2(start_xy[1], start_xy[0])))
    ts = np.radians(np.linspace(a0, a0 + end_deg, 60))
    xs, ys = c0 + radius * np.cos(ts), c0 + radius * np.sin(ts)
    ax.plot(xs[:-4], ys[:-4], color=color, lw=cast.layout.lw, zorder=4)
    _vector(ax, (xs[-5], ys[-5]), (xs[-3], ys[-3]), color, cast.layout.lw, cast)


def _start_label(ax, cast, *, below=False):
    """Name the open circle: where the source was stored before the turn.

    The label goes on the side the turn arc leaves free.
    """
    x, y = _index(SOURCE_XY)
    _label(
        ax,
        (x, y - 0.2 if below else y + 0.2),
        "start",
        cast,
        color=cast["planet"].color,
        va="top" if below else "bottom",
    )


def _array_panel(ax, cast, case, *, turn_deg=None):
    """Stored array with the source's pixel, and the turn from (2, 3)."""
    _array_axes(ax, cast)
    _lit_cell(ax, case["pixel"], cast)
    if turn_deg is not None:
        before = np.asarray(SOURCE_XY)
        _source(ax, _index(before), cast, ghost=True)
        _start_label(ax, cast, below=turn_deg > 0)
        _turn_arc(ax, before, turn_deg, cast, _ink(cast))
    _source(ax, _index(case["xy"]), cast)


# The still


HEADLINE = (
    "Roll turns the detector under a fixed sky; active rotation moves the content"
)


def _titles(layout):
    if layout.is_slide:
        return (
            "(a) roll 0",
            r"(b) telescope roll $\theta = +90\degree$",
            r"(c) active rotation $+90\degree$",
        )
    return (
        "(a) roll 0",
        r"(b) roll $\theta = +90\degree$",
        r"(c) active rotation $+90\degree$",
    )


def build_still(layout, cast):
    """Still: sky view (top) and stored array (bottom) for three cases."""
    slide = layout.is_slide
    fig, grid = ex.figure(
        layout,
        doc_height_in=5.4,
        nrows=3,
        ncols=3,
        height_ratios=[1.08, 1.0, 0.08],
    )
    # The last row is one strip for the status note, so the layout engine
    # keeps it clear of the axis labels and of the provenance stamp.
    for ax in grid[2]:
        ax.remove()
    foot = fig.add_subplot(grid[0, 0].get_gridspec()[2, :])
    foot.axis("off")
    axes = grid[:2]
    if slide:
        fig.suptitle(HEADLINE, fontsize=layout.title_pt, fontweight="bold")
    got = cases()
    titles = _titles(layout)
    top, bottom = axes
    # Top row: the physical picture in the sky chart.
    _sky_panel(top[0], cast, 0.0, got["zero"]["pixel"])
    _name_marks(top[0], cast)
    _sky_panel(top[1], cast, ROLL_DEG, got["roll"]["pixel"])
    _ghost_basis(top[1], cast)
    _roll_arc(top[1], cast)
    _sky_panel(top[2], cast, 0.0, None, faint=True)
    top[2].set_xlabel(
        "nothing physical turns",
        fontsize=layout.small_pt,
        fontstyle="italic",
        color=cast.text,
    )
    for ax, title in zip(top, titles, strict=True):
        ax.set_title(title, fontsize=layout.font_pt)
    # Row names at the left, in the reading order of the figure.
    row_kw = {"fontsize": layout.small_pt, "fontstyle": "italic", "color": cast.text}
    for ax, name, x in (
        (top[0], "on the sky chart", -0.1),
        (bottom[0], "stored array (origin lower)", -0.3),
    ):
        ax.text(
            x,
            0.5,
            name,
            transform=ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            **row_kw,
        )
    # Pixel captions under the sky panels.
    for ax, key in ((top[0], "zero"), (top[1], "roll")):
        ax.set_xlabel(
            f"source in {_pixel_text(got[key]['pixel'])}",
            fontsize=layout.small_pt,
            color=cast.text,
        )
    # Bottom row: the stored arrays.
    _array_panel(bottom[0], cast, got["zero"])
    _array_panel(bottom[1], cast, got["roll"], turn_deg=-ROLL_DEG)
    _array_panel(bottom[2], cast, got["active"], turn_deg=ACTIVE_DEG)
    notes = (
        "starting pixel",
        "sky turns clockwise",
        "content turns\ncounterclockwise",
    )
    for ax, key, note in zip(bottom, ("zero", "roll", "active"), notes, strict=True):
        ax.set_title(
            f"{_pixel_text(got[key]['pixel'])}\n{note}",
            fontsize=layout.small_pt,
            color=cast.text,
            linespacing=1.1,
        )
    # The status sits in the footer, in the badge style, clear of the titles.
    status = foot.text(
        0.5,
        0.5,
        "schematic; origin at the grid center; proposed pixel profile; "
        "image-coordinate profile pending",
        transform=foot.transAxes,
        ha="center",
        va="center",
        fontsize=layout.small_pt,
        fontstyle="italic",
        color=cast["annotation"].color,
    )
    status.set_bbox(
        {
            "boxstyle": "round,pad=0.3",
            "facecolor": cast.background,
            "edgecolor": cast["scenery"].color,
            "lw": 0.6 * cast.layout.lw,
            "ls": "--",
        }
    )
    return _plain_text(fig, cast)


# The animation


def build_roll_animation(layout, cast):
    """Animation: the detector turns under the fixed sky, theta 0 to +90."""
    slide = layout.is_slide
    fig, (sky, arr) = ex.figure(layout, doc_height_in=3.7, ncols=2)
    if slide:
        fig.suptitle(
            "Positive roll turns the detector counterclockwise under a fixed sky",
            fontsize=layout.title_pt,
            fontweight="bold",
        )
    start = pixel_of(SOURCE_XY)
    # Extra room between the slide headline and the panel titles, whose
    # mathtext the layout engine measures a little short.
    title_pad = 0.9 * layout.font_pt if slide else 0.4 * layout.font_pt
    # A wider sky panel, so the turning grid and its arrows never reach the
    # compass in the corner.
    lim = SKY_LIM_ANIM
    art = _sky_panel(sky, cast, 0.0, start, lim=lim)
    _name_marks(sky, cast)
    (arc_line,) = sky.plot([], [], color=cast.text, lw=layout.lw, zorder=4)
    arc_head = _vector(sky, (0.62, 0.0), (0.62, 0.001), cast.text, layout.lw, cast)
    theta_lab = _label(sky, (0.0, 0.0), r"$\theta$", cast, color=cast.text, box=False)
    # Titles start with full-length text, so the layout reserves their room.
    sky_title = sky.set_title(
        r"sky chart, roll $\theta = +90\degree$",
        fontsize=layout.font_pt,
        pad=title_pad,
    )
    sky.set_xlabel(
        "sky fixed, detector turns",
        fontsize=layout.small_pt,
        fontstyle="italic",
        color=cast.text,
    )

    _array_axes(arr, cast)
    arr_title = arr.set_title(
        f"stored array: source in {_pixel_text(start)}",
        fontsize=layout.font_pt,
        pad=title_pad,
    )
    lit = _lit_cell(arr, start, cast)
    _source(arr, _index(SOURCE_XY), cast, ghost=True)
    _start_label(arr, cast)
    (trail,) = arr.plot([], [], color=_ink(cast), lw=layout.lw, zorder=4)
    now = _source(arr, _index(SOURCE_XY), cast)
    ex.badge(sky, cast, "schematic", loc="upper right")
    _plain_text(fig, cast)
    half = N_GRID / 2.0
    lab_det = {
        "c": np.array([half + 0.55, -half - EDGE_GAP]),
        "r": np.array([-half - EDGE_GAP, half + 0.55]),
    }

    def draw(fig, frame):
        theta = frame["theta"]
        art["affine"].clear().rotate_deg(theta)
        rot = rotation(theta)
        for name, det in lab_det.items():
            art["labels"][name].set_position(tuple(rot @ det))
        d = detector_xy(SOURCE_XY, theta)
        pixel = pixel_of(d)
        c0 = optical_origin()
        art["lit"].set_xy((pixel[1] - c0 - 0.5, pixel[0] - c0 - 0.5))
        lit.set_xy((pixel[1] - 0.5, pixel[0] - 0.5))
        ts = np.radians(np.linspace(0.0, theta, 40))
        radius = 0.62
        if theta > 0.0:
            # The head is fixed in points, so on a short arc it would dwarf
            # the arc; below HEAD_MIN_DEG the arc is drawn without it.
            head = theta >= HEAD_MIN_DEG
            keep = ts[:-3] if head else ts
            arc_line.set_data(radius * np.cos(keep), radius * np.sin(keep))
            arc_head.set_positions(
                (radius * np.cos(ts[-4]), radius * np.sin(ts[-4])),
                (radius * np.cos(ts[-1]), radius * np.sin(ts[-1])),
            )
            arc_head.set_visible(head)
            mid = np.radians(0.5 * theta)
            theta_lab.set_position((0.34 * np.cos(mid), 0.34 * np.sin(mid)))
            theta_lab.set_visible(True)
        else:
            arc_line.set_data([], [])
            arc_head.set_visible(False)
            theta_lab.set_visible(False)
        phis = np.linspace(0.0, theta, 40)
        path = np.array([detector_xy(SOURCE_XY, p) for p in phis])
        trail.set_data(path[:, 0] + c0, path[:, 1] + c0)
        now.set_data([d[0] + c0], [d[1] + c0])
        sky_title.set_text(rf"sky chart, roll $\theta = {theta:+.0f}\degree$")
        arr_title.set_text(f"stored array: source in {_pixel_text(pixel)}")

    frames = roll_states(layout)
    # Settle the constrained layout before the recorder draws (and freezes)
    # frame 0: its first pass places the slide headline over the titles.
    draw(fig, frames[0])
    for _ in range(2):
        fig.canvas.draw()
    return ex.AnimationScene(fig=fig, draw=draw, frames=frames)


# Captions

_CLAUSE = (
    "Telescope roll and active image rotation under the example chart of "
    "{ref}`optics-roll-rotation`: the sky chart is "
    "$\\mathbf{s}=(x_{\\rm sky},y_{\\rm sky})=(E,N)$, a fixed sky source has "
    "detector coordinates $\\mathbf{d}=R(-\\theta)\\mathbf{s}$ for a positive "
    "telescope roll $\\theta$, and no calibrated zero-point orientation or "
    "reflection is applied. Stored pixels follow the proposed profile of "
    "{ref}`optics-pixel-directions`, $x=(c-c_x)s$ and $y=(r-c_y)s$, with the "
    "geometric origin $c_x=c_y=1.5$ of a 4-by-4 grid, where the star on the "
    "optical axis sits between four pixels. "
)
_PENDING = (
    "The rotation relations are identities given this example chart and "
    "profile. The image-coordinate profile itself is pending under the "
    "{ref}`image coordinates and PSFlet origin decision "
    "<decision-image-coordinates-and-psflet-origin>`, and the matrix taking "
    "$(E,N)$ into the sky chart is pending under the {ref}`observer basis and "
    "node decision <decision-observer-basis-and-node>`. The sky chart is drawn "
    "with its stored axes, east to the right; a conventional north-up, "
    "east-left display reverses only its horizontal display axis, which also "
    "reverses the apparent turning sense in the sky-chart view; the "
    "stored-array senses are unchanged. Telescope "
    "roll is neither an astronomical position angle nor a parallactic angle. "
    "Schematic, not to scale."
)
CAPTION_STILL = (
    _CLAUSE + "Top row: the physical picture in the sky chart, with the detector "
    "outline, its corner pixel (0, 0) marked by a wedge, and its column (c) and "
    "row (r) directions; in (b) faint dashed arrows keep the roll-0 directions, "
    "and the arc $\\theta$ outside the frame runs from the E direction, the "
    "roll-0 column direction, to the turned column direction. Bottom row: the same detector as a stored array, "
    "displayed with origin lower. (a) At zero roll the source at "
    "$(x,y)=(1.5s,0.5s)$ occupies $(r,c)=(2,3)$. (b) A roll of $\\theta=+90$ "
    "degrees turns the detector basis counterclockwise under the fixed sky; the "
    "source's detector coordinates become $(0.5s,-1.5s)$, pixel (0, 2), so in "
    "the stored array the sky turns clockwise. (c) An active +90-degree image "
    "rotation changes nothing physical: it moves the stored content "
    "counterclockwise in fixed pixels, to $(-0.5s,1.5s)$, pixel (3, 1). These "
    "pixels follow from geometry, independently of a rotation helper. The same "
    "senses hold in the example implementation of {doc}`the scene-to-detection "
    "example <../examples/scene-to-detection>` (section Detection): coronagraphoto "
    "places sources with the hwoutils matrix `ccw_rotation_matrix(-telescope_pa_deg)`, "
    "which turns the scene clockwise on the readout for a positive angle, and "
    "hwoutils `rotate_image` turns an image counterclockwise for a positive "
    "angle, so derotating a frame by its own roll angle is an active rotation by "
    "$+\\theta$. coronagraphoto's `telescope_pa_deg` plays the role of $\\theta$ "
    "here; despite its name it is a roll, not a position angle. " + _PENDING
)
ALT_STILL = (
    "Six panels in two rows and three columns, titled (a) roll 0, (b) roll theta "
    "= +90 degrees and (c) active rotation +90 degrees. Top row, labeled on the "
    "sky chart: each panel has sky axes with E to the right and N up in its "
    "lower left corner, a yellow star at the center, a cyan source up and to "
    "the right of the star, and a 4-by-4 detector grid centered on the star, "
    "with a large filled wedge in its pixel (0, 0) corner and arrows labeled c "
    "and r along two edges. In (a) the wedge is at the lower left, c points "
    "right, r points up, the star and source are labeled, and the source's "
    "pixel is shaded; below the panel it reads source in (r, c) = (2, 3). In "
    "(b) the grid is turned a quarter turn counterclockwise: the wedge is at "
    "the lower right, c points up along the right edge and r points left along "
    "the bottom edge, while faint dashed arrows outside them keep the roll-0 c "
    "direction, pointing right, and r direction, pointing up. Outside the lower "
    "right corner an arc labeled theta turns from the roll-0 c direction, which "
    "is the E direction, to the new c direction. The source has not moved; its "
    "shaded pixel is given below the panel as (r, c) = (0, 2). In (c) the grid "
    "is drawn faint in its unturned position, and the note below reads nothing "
    "physical turns. Bottom row, labeled stored array (origin lower): three "
    "4-by-4 arrays with column c from 0 to 3 to the right and row r from 0 to 3 "
    "upward, the star at the center. (a) shades pixel (2, 3), titled starting "
    "pixel. (b) shades pixel (0, 2), with an open circle labeled start at (2, 3) "
    "and a clockwise arc to the source, titled sky turns clockwise. (c) shades "
    "pixel (3, 1), with an open circle labeled start at (2, 3) and a "
    "counterclockwise arc to the source, titled content turns counterclockwise. "
    "A dashed status box at the bottom reads schematic; origin at the grid "
    "center; proposed pixel profile; image-coordinate profile pending."
)
CAPTION_ANIMATION = (
    "The detector turns under a fixed sky as the telescope roll $\\theta$ runs "
    "from 0 to +90 degrees; only $\\theta$ changes, and every scale is fixed. "
    "Left: the sky chart, with the star, the fixed source at "
    "$(x,y)=(1.5s,0.5s)$ and the detector grid turning counterclockwise, its "
    "corner pixel (0, 0) marked by a wedge. Right: the stored array, displayed "
    "with origin lower, where the same source moves clockwise along "
    "$\\mathbf{d}=R(-\\theta)\\mathbf{s}$ from pixel (2, 3) to pixel (0, 2); the "
    "shaded pixel is the one holding the source. coronagraphoto's "
    "`telescope_pa_deg` plays the role of $\\theta$; despite its name it is a "
    "roll, not a position angle. " + _CLAUSE + _PENDING
)
ALT_ANIMATION = (
    "Animation with two panels. Left, titled sky chart, roll theta = followed "
    "by the current angle from +0 to +90 degrees: sky axes with E to the right "
    "and N up in the lower left corner, a yellow star at the center, a cyan "
    "source up and to the right of it that never moves, and a 4-by-4 detector "
    "grid with a corner wedge and edge arrows labeled c and r that turns "
    "counterclockwise about the star, with a growing arc labeled theta from "
    "the E direction, which gains an arrowhead once the arc is long enough; "
    "the source's pixel in the turned grid is shaded, and the note below reads "
    "sky fixed, detector turns. Right, titled stored array: source in (r, c) = "
    "followed by the current pixel, from (2, 3) to (0, 2): a 4-by-4 grid "
    "displayed with origin lower, with column c and row r from 0 to 3 and the "
    "star at the center, where the cyan source moves clockwise from an open "
    "circle labeled start at pixel (2, 3), leaving a trail, and the shaded "
    "pixel follows it to pixel (0, 2)."
)

PARAMS = {
    "grid": N_GRID,
    "source_xy_in_s": list(SOURCE_XY),
    "roll_deg": ROLL_DEG,
    "active_rotation_deg": ACTIVE_DEG,
    "chart": "(x_sky, y_sky) = (E, N)",
}

FIGURES = [
    ex.FigureSpec(
        slug="d09-roll-rotation",
        build=build_still,
        caption=CAPTION_STILL,
        alt=ALT_STILL,
        status="schematic; proposed pixel profile, image-coordinate profile pending",
        params=PARAMS,
    ),
]
ANIMATIONS = [
    ex.AnimationSpec(
        slug="d09-roll-viewpoint",
        build=build_roll_animation,
        ground="viewpoint",
        caption=CAPTION_ANIMATION,
        alt=ALT_ANIMATION,
        status="schematic; proposed pixel profile, image-coordinate profile pending",
        params=PARAMS,
        fps=6,
    ),
]
