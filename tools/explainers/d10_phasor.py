"""Surface, OPD, phase, phasor: why a coherent intensity change is signed.

A relay strip of five panels, each carrying the object the previous one
introduced. A mirror surface moves by h; at normal reflection the path grows
by W = 2h; under the proposed coherent profile that OPD turns the field
arrow at the marked mirror point by +2 pi W / lambda; at one image pixel the
nominal field and the field the change adds sum as arrows; and the pixel's
brightness is the squared length of the sum, so the change can be negative.

The phase arrow is the output of a physicaloptix ``PhaseScreen`` applied to
a unit field, so the drawn rotation is the one the library implements. The
field sum and the intensities use the chapter's worked numbers, which are
not derived from the h of the first panels.

The still only; no motion ground is claimed.
"""

import functools
import math
from fractions import Fraction

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.text import Text

from explainers import _common as ex

# Scientific inputs. The surface step is a fraction of the wavelength so that
# the phase is exact; it is exaggerated for the drawing (a deformable mirror
# moves by nanometers, not tenths of a wave).
PARAMS = {
    "wavelength_nm": 550.0,
    "surface_h_waves": "1/16",
    # The chapter's worked example (optics-signed-intensity): real fields.
    "e0": 0.2,
    "delta_e": -0.1,
    # Pupil grid for the PhaseScreen evaluation; the marked point is the
    # center of the bump.
    "npup": 8,
}

STATUS = "schematic, not to scale; worked numbers"


def h_waves():
    """Surface displacement h in waves, as an exact fraction."""
    return Fraction(PARAMS["surface_h_waves"])


def opd_waves():
    """OPD at normal reflection, W = 2h, in waves."""
    return 2 * h_waves()


@functools.cache
def phase_arrow():
    """The field at the marked mirror point after the OPD, from physicaloptix.

    A unit field on a small pupil grid passes a ``PhaseScreen`` whose OPD is
    W at the central samples and zero elsewhere; the value at the center is
    the rotated arrow.

    Returns:
        The complex field value, a Python complex.
    """
    import jax
    import jax.numpy as jnp
    import physicaloptix as po

    n = PARAMS["npup"]
    lam = PARAMS["wavelength_nm"]
    opd_nm = float(opd_waves()) * lam
    bump = np.zeros((n, n))
    bump[n // 2 - 1 : n // 2 + 1, n // 2 - 1 : n // 2 + 1] = 1.0
    with jax.enable_x64(True):
        grid = po.Grid.pupil(n)
        basis = po.ModeBasis(B=jnp.asarray(bump[None]), coeffs=jnp.asarray([opd_nm]))
        screen = po.PhaseScreen(basis, grid, wavelength_nm=lam)
        field = po.Field(
            data=jnp.ones((n, n), dtype=complex), grid=grid, plane=po.PlaneKind.PUPIL
        )
        out = screen(field)
    return complex(np.asarray(out.data)[n // 2, n // 2])


def worked():
    """The chapter's worked numbers and what follows from them."""
    e0 = PARAMS["e0"]
    de = PARAMS["delta_e"]
    e = e0 + de
    i0 = abs(e0) ** 2
    i1 = abs(e) ** 2
    return {
        "e0": e0,
        "de": de,
        "e": e,
        "i0": i0,
        "i1": i1,
        "di": i1 - i0,
        "cross": 2.0 * (np.conj(e0) * de).real,
        "square": abs(de) ** 2,
    }


def fmt(value):
    """Two-decimal label with a typographic minus sign."""
    return f"${value:.2f}$"


def frac_label(fraction):
    """``lambda/16`` style label for a fraction of a wave."""
    if fraction.numerator == 1:
        return rf"\lambda/{fraction.denominator}"
    return rf"{fraction.numerator}\lambda/{fraction.denominator}"


def phase_label(fraction_of_turn):
    """``pi/4`` style label for a phase given as a fraction of 2 pi."""
    half_turns = 2 * fraction_of_turn
    if half_turns.numerator == 1:
        return rf"\pi/{half_turns.denominator}"
    return rf"{half_turns.numerator}\pi/{half_turns.denominator}"


# Labels: no stroked halos; a label over marks sits on a plain backing box.


def _backing(cast):
    return {
        "boxstyle": "round,pad=0.15,rounding_size=0.25",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _label(ax, xy, text, cast, *, color=None, box=True, **kw):
    kw.setdefault("fontsize", cast.layout.small_pt)
    kw.setdefault("linespacing", 1.0)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    label = ax.text(*xy, text, color=cast.text if color is None else color, **kw)
    if box:
        label.set_bbox(_backing(cast))
    label.set_zorder(9)
    return label


def _plain_text(fig, cast):
    """Replace any stroked halo a shared helper added with a backing box."""
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
            if text.get_bbox_patch() is None:
                text.set_bbox(_backing(cast))
    return fig


def phasor(ax, start, end, color, cast, *, lw_scale=1.4, zorder=5):
    """A field arrow on the complex plane: a solid shaft with a filled head.

    The head grows linearly with the layout's marker size, like the shaft.
    """
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_length=0.5,head_width=0.28",
        mutation_scale=1.6 * cast.layout.marker_pt,
        color=color,
        lw=lw_scale * cast.layout.lw,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def ray(ax, start, end, cast):
    """A starlight ray from the shared helper, with its head kept linear in size.

    ``ex.arrow`` scales both the head length and the mutation scale with the
    marker size, so its head grows with the square of the layout scale; on
    a slide that head would cover the mirror.
    """
    artists = ex.arrow(ax, start, end, "ray", cast, source="star")
    head = artists[0]
    head.set_mutation_scale(
        head.get_mutation_scale() * ex.DOC.marker_pt / cast.layout.marker_pt
    )
    return artists


def _complex_axes(ax, cast, xlim, ylim, *, show_labels=True):
    """Equal-aspect complex plane with thin Re and Im axes and no frame."""
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    axis_kw = {"color": cast.neutral(0.45), "lw": 0.7 * cast.layout.lw, "zorder": 1}
    ax.plot(xlim, (0, 0), **axis_kw)
    ax.plot((0, 0), ylim, **axis_kw)
    if show_labels:
        small = cast.layout.small_pt
        _label(
            ax,
            (xlim[1], 0),
            " Re",
            cast,
            box=False,
            ha="right",
            va="bottom",
            color=cast.neutral(0.6),
            fontsize=small,
        )
        _label(
            ax,
            (0, ylim[1]),
            " Im",
            cast,
            box=False,
            ha="left",
            va="top",
            color=cast.neutral(0.6),
            fontsize=small,
        )


# The side view of the mirror


FACE_BACK = 0.45  # mirror thickness in drawing units
H_DRAW = 0.3  # the drawn displacement, exaggerated
FLAT = 0.2  # half-width of the flat bottom of the bump
TAPER = 0.45  # half-width where the bump returns to the rest surface
HALF_Y = 0.85  # half-height of the mirror


def face_x(y):
    """Drawn face position: 0 at rest, H_DRAW over the flat bottom of the bump."""
    y = np.abs(np.asarray(y, dtype=float))
    ramp = np.clip((y - FLAT) / (TAPER - FLAT), 0.0, 1.0)
    return H_DRAW * 0.5 * (1.0 + np.cos(np.pi * ramp))


def draw_mirror(ax, cast, layout, *, name="mirror"):
    """The mirror in side view, its rest surface, and the marked point."""
    ax.set(xlim=(-1.05, 0.55), ylim=(-1.12, 1.0), aspect="equal", anchor="S")
    ax.axis("off")
    ys = np.linspace(-HALF_Y, HALF_Y, 241)
    xs = face_x(ys)
    outline = np.column_stack(
        [
            np.concatenate([xs, [FACE_BACK, FACE_BACK]]),
            np.concatenate([ys, [HALF_Y, -HALF_Y]]),
        ]
    )
    optics = cast["optics"]
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor=to_rgba(optics.color, 0.35),
            edgecolor=optics.color,
            lw=cast.layout.lw,
            zorder=3,
        )
    )
    rest = cast["reference_plane"]
    ax.plot((0.0, 0.0), (-0.6, 0.6), color=rest.color, ls=rest.ls, lw=rest.lw, zorder=4)
    _label(
        ax,
        (FACE_BACK, -HALF_Y - 0.05),
        name,
        cast,
        color=cast.neutral(0.7),
        box=False,
        ha="right",
        va="top",
        fontstyle="italic" if name != "mirror" else "normal",
    )
    _label(
        ax,
        (-0.03, 0.62),
        "at rest",
        cast,
        color=rest.color,
        box=False,
        ha="right",
        va="bottom",
    )
    ax.add_patch(Circle((H_DRAW, 0.0), 0.045, color=cast.text, zorder=6, lw=0))


def build_surface(ax, cast, layout):
    """Panel 1: the surface moves away from the light by h."""
    draw_mirror(ax, cast, layout)
    # Dimension of h between the rest surface and the moved surface.
    y = 0.1
    ax.add_patch(
        FancyArrowPatch(
            (0.0, y),
            (H_DRAW, y),
            arrowstyle="<|-|>,head_length=0.3,head_width=0.15",
            mutation_scale=cast.layout.marker_pt,
            color=cast.text,
            lw=0.8 * cast.layout.lw,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )
    _label(ax, (-0.06, y), rf"$h={frac_label(h_waves())}$", cast, box=False, ha="right")
    ray(ax, (-1.0, -0.4), (-0.3, -0.4), cast)
    _label(
        ax,
        (-0.65, -0.47),
        "starlight",
        cast,
        color=cast["star"].color,
        box=False,
        va="top",
    )


def build_path(ax, cast, layout):
    """Panel 2: the reflected path grows by h on the way in and h on the way out."""
    draw_mirror(ax, cast, layout, name="(the mirror from 1)")
    d = 0.1
    star = cast["star"].color
    band = to_rgba(cast.text, 0.3)
    for y in (d, -d):
        ax.plot(
            (0.0, H_DRAW),
            (y, y),
            color=band,
            lw=5.0 * cast.layout.lw,
            solid_capstyle="butt",
            zorder=4,
        )
    ray(ax, (-1.0, d), (H_DRAW, d), cast)
    ray(ax, (H_DRAW, -d), (-1.0, -d), cast)
    _label(
        ax, (-1.0, d + 0.06), "in", cast, color=star, box=False, ha="left", va="bottom"
    )
    _label(
        ax, (-1.0, -d - 0.06), "out", cast, color=star, box=False, ha="left", va="top"
    )
    # Each h sits just outside its shaded extra-path band, inside the dip.
    _label(ax, (0.12, d + 0.05), "$h$", cast, va="bottom")
    _label(ax, (0.12, -d - 0.05), "$h$", cast, va="top")
    _label(ax, (-0.5, -0.55), rf"$W={frac_label(opd_waves())}$", cast, box=False)


def build_phase(ax, cast, layout):
    """Panel 3: the OPD turns the field arrow at the marked point."""
    after = phase_arrow()
    _complex_axes(ax, cast, (-0.3, 1.3), (-0.95, 1.15))
    theta = np.radians(np.linspace(-18.0, 105.0, 124))
    scenery = cast["scenery"]
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=scenery.color,
        lw=scenery.lw,
        ls=":",
        zorder=2,
    )
    # Same color roles as the next panel: the nominal field in the star
    # color, the changed field in the text color.
    before_color = cast["star"].color
    after_color = cast.text
    phasor(ax, (0, 0), (1.0, 0.0), before_color, cast)
    phasor(ax, (0, 0), (after.real, after.imag), after_color, cast)
    angle = math.degrees(math.atan2(after.imag, after.real))
    ex.angle_arc(ax, (0.0, 0.0), 0.0, angle, cast, radius=0.4, color=cast.neutral(0.6))
    mid = math.radians(0.5 * angle)
    _label(
        ax,
        (0.5 * math.cos(mid) + 0.04, 0.5 * math.sin(mid)),
        rf"$\phi={phase_label(opd_waves())}$",
        cast,
        ha="left",
        box=False,
    )
    _label(
        ax,
        (1.0, -0.07),
        "before",
        cast,
        color=before_color,
        ha="right",
        va="top",
    )
    _label(
        ax,
        (after.real - 0.02, after.imag + 0.06),
        "after",
        cast,
        color=after_color,
        box=False,
        ha="right",
        va="bottom",
    )
    _label(
        ax,
        (0.7, -0.3),
        r"$E\,e^{+i2\pi W/\lambda}$" + "\n(this book's sign\nconvention)",
        cast,
        color=cast["annotation"].color,
        box=False,
        va="top",
    )


def build_sum(ax, cast, layout):
    """Panel 4: at one image pixel the fields add as arrows."""
    w = worked()
    lift = 0.055
    _complex_axes(ax, cast, (-0.02, 0.34), (-0.2, 0.165))
    # Other phases of the same added field: the tips land on this circle.
    theta = np.linspace(0, 2 * np.pi, 361)
    r = abs(w["de"])
    scenery = cast["scenery"]
    ax.plot(
        w["e0"] + r * np.cos(theta),
        r * np.sin(theta),
        color=scenery.color,
        lw=scenery.lw,
        ls=":",
        zorder=2,
    )
    star = cast["star"].color
    change = cast.neutral(0.55)
    phasor(ax, (0, 0), (w["e0"], 0.0), star, cast)
    phasor(ax, (w["e0"], lift), (w["e"], lift), change, cast)
    phasor(ax, (0, -lift), (w["e"], -lift), cast.text, cast)
    drop = {
        "color": cast.neutral(0.45),
        "lw": 0.6 * cast.layout.lw,
        "ls": ":",
        "zorder": 2,
    }
    ax.plot((w["e0"], w["e0"]), (0, lift), **drop)
    ax.plot((w["e"], w["e"]), (-lift, lift), **drop)
    # The true sum, on the real axis and on the circle.
    ax.add_patch(Circle((w["e"], 0.0), 0.006, color=cast.text, zorder=6, lw=0))
    _label(
        ax,
        (0.34, 0.165),
        "offset for visibility;\nall three fields are real",
        cast,
        color=cast["annotation"].color,
        box=False,
        ha="right",
        va="top",
        fontstyle="italic",
    )
    _label(
        ax,
        (0.16, -0.2),
        "worked numbers from the chapter,\nnot from panel 3",
        cast,
        color=cast["annotation"].color,
        box=False,
        va="bottom",
        fontstyle="italic",
    )
    _label(
        ax,
        (0.015, 0.01),
        rf"$E_0={w['e0']:.1f}$",
        cast,
        color=star,
        ha="left",
        va="bottom",
    )
    _label(
        ax,
        (0.5 * (w["e0"] + w["e"]), lift + 0.012),
        rf"$\delta E={w['de']:.1f}$",
        cast,
        color=change,
        va="bottom",
    )
    _label(
        ax,
        (0.015, -lift - 0.012),
        rf"$E={w['e']:.1f}$",
        cast,
        box=False,
        ha="left",
        va="top",
    )
    _label(
        ax,
        (w["e0"] + 0.02, -0.112),
        r"$\delta E$ at other phases",
        cast,
        color=cast.neutral(0.6),
        box=False,
        va="top",
    )


def build_intensity(ax, cast, layout):
    """Panel 5: brightness is the squared arrow length; the change is signed."""
    w = worked()
    residual = ex.image_cmap("residual")
    colors = [cast["star"].color, cast.text, residual(0.12)]
    values = [w["i0"], w["i1"], w["di"]]
    names = [r"$|E_0|^2$", r"$|E|^2$", r"$\Delta I$"]
    bars = ax.bar(range(3), values, width=0.62, color=colors, zorder=3)
    bars[2].set_hatch("//")
    bars[2].set_edgecolor(cast.background)
    ax.axhline(0.0, color=cast.neutral(0.6), lw=0.8 * cast.layout.lw, zorder=4)
    ax.set(xlim=(-0.55, 2.55), ylim=(-0.043, 0.05), anchor="S")
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    for x, v in zip(range(3), values, strict=True):
        above = v >= 0
        _label(
            ax,
            (x, v + (0.002 if above else -0.002)),
            fmt(v),
            cast,
            box=False,
            va="bottom" if above else "top",
        )
        _label(
            ax,
            (x, -0.003 if above else 0.003),
            names[x],
            cast,
            box=False,
            va="top" if above else "bottom",
        )
    _label(
        ax,
        (0.5, -0.027),
        "the pixel\ndims",
        cast,
        box=False,
        color=cast["annotation"].color,
        fontstyle="italic",
    )


TITLES = (
    "1  the surface\nmoves by $h$",
    "2  the path grows\nby $W=2h$",
    "3  the phase turns\nby $2\\pi W/\\lambda$",
    "4  at one pixel, the\nfields add as arrows",
    "5  brightness =\n(arrow length)$^2$",
)
# Panel widths as fractions of the figure width, left to right.
WIDTHS = (0.14, 0.14, 0.19, 0.31, 0.15)


def build_strip(layout, cast):
    """The five-panel relay strip, placed by hand so the titles align."""
    height = 2.95
    fig, ax0 = ex.figure(layout, doc_height_in=height)
    ax0.remove()
    fig.set_layout_engine("none")
    fh = fig.get_size_inches()[1]
    slide = layout.is_slide
    title_top = 1.0 - (1.2 if slide else 0.04) / fh
    title_h = (
        (2.4 if slide else 2.3) * layout.small_pt / 72.0 * (1.35 if slide else 1.0)
    )
    bottom = (1.4 if slide else 0.3) / fh
    bridge_h = 3.3 * layout.small_pt / 72.0
    top = title_top - (title_h + bridge_h + 0.08) / fh
    gap = (1.0 - sum(WIDTHS) - 0.02) / 4.0
    x = 0.01
    axes = []
    for width, title in zip(WIDTHS, TITLES, strict=True):
        ax = fig.add_axes([x, bottom, width, top - bottom])
        fig.text(
            x + 0.005,
            title_top,
            title,
            ha="left",
            va="top",
            fontsize=layout.font_pt if slide else layout.small_pt,
            linespacing=1.1,
            color=cast.text,
        )
        axes.append(ax)
        x += width + gap
    # The bridge from a pupil point to an image pixel: a thin separator in
    # the gap before panel 4 and one sentence under panel 4's title.
    x4 = axes[3].get_position().x0
    xs = x4 - 0.5 * gap
    fig.add_artist(
        Line2D(
            (xs, xs),
            (bottom, title_top),
            transform=fig.transFigure,
            color=cast.neutral(0.45),
            lw=0.6 * cast.layout.lw,
        )
    )
    fig.text(
        x4 + 0.005,
        title_top - (title_h + 0.03) / fh,
        "pupil point $\\rightarrow$ image pixel: the pixel's\n"
        "field is a sum of many turned arrows",
        ha="left",
        va="top",
        fontsize=layout.small_pt,
        fontstyle="italic",
        linespacing=1.1,
        color=cast["annotation"].color,
    )
    build_surface(axes[0], cast, layout)
    build_path(axes[1], cast, layout)
    build_phase(axes[2], cast, layout)
    build_sum(axes[3], cast, layout)
    build_intensity(axes[4], cast, layout)
    overlay = fig.add_axes([0, 0, 1, 1], zorder=-1)
    overlay.axis("off")
    ex.badge(overlay, cast, "schematic, not to scale", loc="lower right")
    if slide:
        fig.suptitle(
            "A mirror step becomes a phase; fields add as arrows, so light can cancel",
            fontsize=layout.title_pt,
            x=0.02,
            y=0.97,
            ha="left",
        )
        w = worked()
        fig.text(
            0.5,
            0.05,
            rf"$\Delta I = 2\,\mathrm{{Re}}(E_0^*\,\delta E) + |\delta E|^2"
            rf" = {w['cross']:.2f} + {w['square']:.2f} = {w['di']:.2f}$",
            ha="center",
            va="bottom",
            fontsize=layout.font_pt,
            color=cast.text,
        )
    return _plain_text(fig, cast)


STRIP_CAPTION = (
    "From a mirror surface to a signed intensity change, in five steps. "
    "(1) A deformable-mirror surface moves by h away from the incoming starlight, the path-increasing "
    "direction. "
    "(2) At normal reflection the light travels h farther on the way in and h farther on the way out, "
    "so the OPD is W = 2h ({ref}`Hecht 2017, Sec. 9.4.2, text before eq. 9.44 <source-hecht2017>`); "
    "the incoming and outgoing rays are drawn apart for clarity. The OPD is a path difference, not a "
    "surface height, and the conversion is made once, at the hardware ({ref}`optics-coherent-phase`). "
    "(3) At the marked mirror point the field is an arrow in the complex plane, and the OPD turns it by "
    "2 pi W/lambda, counterclockwise for a positive OPD under the proposed coherent profile, in which "
    "E becomes E exp(+i 2 pi W/lambda). The drawn step, h = lambda/16 and so W = lambda/8, turns the "
    "arrow by pi/4; the turned arrow is the output of the physicaloptix PhaseScreen applied to a unit "
    "field. "
    "(4) The field at one image pixel is a sum of the turned arrows from every point of the pupil, so "
    "the change adds a field delta E to the nominal field E_0 there. The numbers in this panel are "
    "the chapter's worked example, not a result of panel 3: E_0 = 0.2 and delta E = -0.1, both real, "
    "point in opposite directions, and the sum is E = 0.1, marked by the dot on the real axis. The "
    "arrows for delta E and E are drawn above and below the axis only for visibility. The dotted "
    "circle marks where the sum would end for other phases of the same delta E. "
    "(5) The brightness of the pixel is the squared length of its arrow: 0.04 before and 0.01 after. "
    "The change, Delta I = 2 Re(conj(E_0) delta E) + |delta E|^2 = -0.04 + 0.01 = -0.03, is negative "
    "while both intensities stay nonnegative ({ref}`optics-signed-intensity`); clipping it at zero "
    "before adding it to the nominal intensity would leave 0.04 and erase the cancellation. "
    "Panels 1, 2, 4 and 5 are identities for scalar, monochromatic, coherent light at normal "
    "incidence; the sense of rotation in panel 3 is the book's proposed pairing, not an adopted "
    "standard. Schematic, not to scale: the drawn step is exaggerated."
)

STRIP_ALT = (
    "A strip of five numbered panels. Panel 1, the surface moves by h: a gray mirror in side view "
    "whose face has a smooth, flat-bottomed dip, a dash-dot line labeled at rest across the dip, a "
    "double-headed arrow labeled h equals lambda over 16 between the rest line and the bottom of the "
    "dip, a dot on the dip, and a yellow arrow labeled starlight approaching from the left. Panel 2, "
    "the path grows by W equals 2h: the same mirror, tagged the mirror from 1, with a yellow ray "
    "labeled in going right into the dip and a ray labeled out coming back, each with a shaded extra "
    "segment between the rest line and the dip, labeled h beside it, and the label W equals lambda "
    "over 8. Panel 3, the phase turns by 2 pi W over lambda: a complex plane with a dotted arc of the "
    "unit circle, a yellow arrow labeled before along the real axis and a dark arrow labeled after at "
    "45 degrees, joined by a counterclockwise arc labeled phi equals pi over 4, and the note E times "
    "exp of plus i 2 pi W over lambda, this book's sign convention. A thin vertical line separates "
    "panel 3 from panel 4, under the note pupil point to image pixel: the pixel's field is a sum of "
    "many turned arrows. Panel 4, at one pixel the fields add as arrows: a yellow arrow labeled E0 "
    "equals 0.2 along the real axis, a gray arrow labeled delta E equals minus 0.1 running back from "
    "its tip, drawn just above the axis, and a dark arrow labeled E equals 0.1 from the origin, drawn "
    "just below the axis, with a dot on the real axis at 0.1 and a dotted circle of radius 0.1 around "
    "the tip of E0 labeled delta E at other phases. Notes read offset for visibility; all three "
    "fields are real, and worked numbers from the chapter, not from panel 3. Panel 5, brightness "
    "equals arrow length squared: a yellow bar of 0.04 labeled the magnitude of E0 squared, a dark "
    "bar of 0.01 labeled the magnitude of E squared, and a blue hatched bar below zero, minus 0.03, "
    "labeled Delta I, with the note the pixel dims. A badge reads schematic, not to scale."
)


FIGURES = [
    ex.FigureSpec(
        slug="d10-surface-to-phasor",
        build=build_strip,
        caption=STRIP_CAPTION,
        alt=STRIP_ALT,
        status=STATUS,
        params=PARAMS,
    ),
]
ANIMATIONS = []
