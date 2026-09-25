"""Shared vocabulary, layouts, drawing helpers and specs for explainer diagrams.

Everything a diagram module needs comes from here:

- ``DOC`` and ``SLIDE``, the two ``Layout`` objects a builder receives. A
  builder branches on ``layout.is_slide`` when a talk needs a different
  arrangement, and creates its figure with ``figure(layout, ...)``.
- ``Cast``, the one entity vocabulary every diagram shares. A builder never
  picks a color: it names an entity (``cast["star"]``) and uses the style
  that entity carries. Every entity differs from every other in at least one
  channel other than color (line style, marker, hatch, arrow style or font
  style), listed by ``Entity.signature``.
- ``image_cmap``, the colormap for an image quantity, by the role names of
  the figures-and-color chapter.
- Drawing helpers for shapes eyepiece does not provide: typed arrows, angle
  arcs, scale breaks, plane cards, a status badge, reference planes, lenslet
  and pixel grids, an aperture and a lens. Optical trains, images, orbits,
  provenance stamps and animation recording come from eyepiece directly.
- ``FigureSpec``, ``AnimationSpec`` and ``AnimationScene``, the registry
  records a diagram module declares.

Helpers take sizes in data units unless the argument name says points, and
draw into an axes the caller owns. Angle arcs, lenslet grids and plane cards
assume an axes with equal aspect.
"""

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import eyepiece as ep
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from hwostyle.colors import wavelength_to_hex
from matplotlib import patheffects, transforms
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle

# Layouts


@dataclass(frozen=True)
class Layout:
    """Page geometry and type sizes for one output venue.

    Attributes:
        name: ``"doc"`` (documentation column) or ``"slide"`` (16:9 talk).
        width_in: Figure width in inches. The exported figure has exactly
            this width; nothing is cropped or grown at save time.
        height_in: Default figure height in inches. A documentation builder
            may ask for another height; a slide is always this height.
        dpi: PNG export dpi for stills.
        font_pt: Body text size in points (labels, axis text).
        small_pt: Secondary text size in points (annotations, notes).
        title_pt: Panel title size in points.
        lw: Base line width in points.
        marker_pt: Base marker size in points.
        stamp_pt: Provenance stamp size in points.
        frame_budget: Largest frame count an animation may have in this
            layout, or None for no limit.
        anim_dpi: Rasterization dpi for animation frames.
    """

    name: str
    width_in: float
    height_in: float
    dpi: int
    font_pt: float
    small_pt: float
    title_pt: float
    lw: float
    marker_pt: float
    stamp_pt: float
    frame_budget: int | None
    anim_dpi: int

    @property
    def is_slide(self):
        """Whether this is the talk layout."""
        return self.name == "slide"

    def size(self, doc_height_in=None):
        """Figure size in inches, honoring a documentation height request.

        Args:
            doc_height_in: Height for the documentation layout. Ignored for
                a slide, whose height is fixed by the 16:9 frame.

        Returns:
            A ``(width, height)`` tuple in inches.
        """
        if self.is_slide or doc_height_in is None:
            return (self.width_in, self.height_in)
        return (self.width_in, float(doc_height_in))

    def n_frames(self, n):
        """Frame count for an animation of ``n`` frames in this layout.

        Args:
            n: Frame count the talk version uses.

        Returns:
            ``n``, capped at the layout's frame budget.
        """
        if self.frame_budget is None:
            return int(n)
        return min(int(n), self.frame_budget)

    def rc(self):
        """Matplotlib rcParams this layout applies while a figure is built.

        Returns:
            A dict for ``matplotlib.rc_context``. It fixes the type sizes and
            line weights and sets a fixed SVG hash salt, so vector exports do
            not change between identical builds.
        """
        return {
            "font.size": self.font_pt,
            "axes.titlesize": self.title_pt,
            "axes.labelsize": self.font_pt,
            "xtick.labelsize": self.small_pt,
            "ytick.labelsize": self.small_pt,
            "legend.fontsize": self.small_pt,
            "figure.titlesize": self.title_pt,
            "lines.linewidth": self.lw,
            "lines.markersize": self.marker_pt,
            "hatch.linewidth": 0.4 * self.lw,
            "svg.hashsalt": "spohnbook-explainers",
        }


# A documentation column is about 700 to 900 CSS pixels wide. At 7.2 inches
# a 10 pt label renders near the 16 px body text at that width, so labels stay
# readable without the reader zooming. The slide layout is 16 by 9 inches at
# 120 dpi, exactly 1920 by 1080 pixels, which is also eyepiece's talk preset.
DOC = Layout(
    name="doc",
    width_in=7.2,
    height_in=3.6,
    dpi=200,
    font_pt=10.0,
    small_pt=8.5,
    title_pt=11.0,
    lw=1.2,
    marker_pt=7.0,
    stamp_pt=6.0,
    frame_budget=30,
    anim_dpi=150,
)
SLIDE = Layout(
    name="slide",
    width_in=16.0,
    height_in=9.0,
    dpi=ep.PRESETS["talk"]["dpi"],
    font_pt=20.0,
    small_pt=16.0,
    title_pt=26.0,
    lw=2.4,
    marker_pt=14.0,
    stamp_pt=9.0,
    frame_budget=None,
    anim_dpi=ep.PRESETS["talk"]["dpi"],
)
LAYOUTS = {"doc": DOC, "slide": SLIDE}


def figure(layout, *, doc_height_in=None, nrows=1, ncols=1, **subplots_kw):
    """Create a constrained-layout figure sized for ``layout``.

    Args:
        layout: ``DOC`` or ``SLIDE``, as handed to the builder.
        doc_height_in: Height for the documentation layout, in inches. A
            slide always uses the full 16:9 frame.
        nrows: Subplot rows.
        ncols: Subplot columns.
        **subplots_kw: Passed to ``plt.subplots`` (``width_ratios``,
            ``gridspec_kw``, ``subplot_kw`` and so on).

    Returns:
        ``(fig, axes)`` exactly as ``plt.subplots`` returns them.
    """
    return plt.subplots(
        nrows,
        ncols,
        figsize=layout.size(doc_height_in),
        layout="constrained",
        **subplots_kw,
    )


# Entity vocabulary

ENTITY_KEYS = (
    "star",
    "planet",
    "exozodi",
    "local_zodi",
    "aperture",
    "optics",
    "lenslet",
    "detector",
    "ray",
    "integration",
    "data",
    "reference_plane",
    "scenery",
    "furniture",
    "annotation",
)
ARROW_KINDS = ("ray", "integration", "data")


@dataclass(frozen=True)
class Entity:
    """The drawing style of one recurring entity.

    Attributes:
        key: The cast key, for example ``"star"``.
        noun: A plain display name for labels.
        color: Resolved color for the active mode.
        rank: ``"furniture"``, ``"scenery"``, ``"data"`` or ``"answer"``.
        ls: Line style of the entity's outline or line.
        marker: Marker used when the entity is drawn as a point.
        hatch: Hatch pattern used when it is drawn as a region.
        arrowstyle: Arrow style used when it is drawn as an arrow.
        fontstyle: Font style used when it is drawn as text.
        lw: Line width in points.
        fill_alpha: Face opacity of a region.
    """

    key: str
    noun: str
    color: str
    rank: str
    ls: str = "-"
    marker: str | None = None
    hatch: str | None = None
    arrowstyle: str | None = None
    fontstyle: str | None = None
    lw: float = 1.0
    fill_alpha: float = 0.0

    def signature(self):
        """The channels other than color that identify this entity.

        Returns:
            ``(ls, marker, hatch, arrowstyle, fontstyle)``. No two entities of
            a cast share a signature, so a grayscale print still separates
            them.
        """
        return (self.ls, self.marker, self.hatch, self.arrowstyle, self.fontstyle)

    def line_kw(self):
        """Keyword arguments for ``ax.plot`` of this entity as a line."""
        return {"color": self.color, "ls": self.ls, "lw": self.lw}

    def marker_kw(self, size=None):
        """Keyword arguments for ``ax.plot`` of this entity as a point.

        Args:
            size: Marker size in points; None keeps the rcParams default.
        """
        kw = {"color": self.color, "marker": self.marker, "ls": "none"}
        if size is not None:
            kw["ms"] = size
        return kw

    def patch_kw(self):
        """Keyword arguments for a Matplotlib patch of this entity as a region."""
        kw = {
            "facecolor": to_rgba(self.color, self.fill_alpha),
            "edgecolor": self.color,
            "lw": self.lw,
            "ls": self.ls if self.ls != "none" else "-",
        }
        if self.hatch:
            kw["hatch"] = self.hatch
            kw["hatchcolor"] = self.color
        return kw


def neutral(level):
    """Blend from the background (0) toward the text color (1).

    Args:
        level: Blend fraction in [0, 1].

    Returns:
        An RGB tuple for the active rcParams.
    """
    bg = np.array(to_rgb(plt.rcParams["axes.facecolor"]))
    fg = np.array(to_rgb(plt.rcParams["text.color"]))
    return tuple(float(v) for v in bg + level * (fg - bg))


class Cast(Mapping):
    """The entity vocabulary for one mode and one layout.

    Build it with ``make_cast`` inside the active style; the export machinery
    does this and hands it to every builder. Index it by entity key.

    Attributes:
        mode: The hwostyle mode the colors were resolved in.
        layout: The layout whose line widths were applied.
        text: The text color, reserved for the answer rank.
        background: The axes background color.
    """

    def __init__(self, entities, *, mode, layout):
        """Wrap resolved entities; use ``make_cast`` rather than this."""
        self._entities = dict(entities)
        self.mode = mode
        self.layout = layout
        self.text = plt.rcParams["text.color"]
        self.background = plt.rcParams["axes.facecolor"]

    def __getitem__(self, key):
        """Entity for ``key``."""
        return self._entities[key]

    def __iter__(self):
        """Iterate over entity keys in declaration order."""
        return iter(self._entities)

    def __len__(self):
        """Number of entities."""
        return len(self._entities)

    def neutral(self, level):
        """Neutral gray between the background (0) and the text color (1)."""
        return neutral(level)


def make_cast(layout):
    """Resolve the shared entity vocabulary in the active hwostyle mode.

    Colors follow the figures-and-color chapter: star yellow, planet cyan,
    exozodiacal dust purple, local zodiacal light green, and neutral grays
    for hardware, scenery, furniture and notes. Hardware entities take no
    hue; they are told apart by shape, marker and hatch.

    Args:
        layout: The layout whose base line width scales every entity.

    Returns:
        A ``Cast``.
    """
    roles = hwostyle.roles
    light = hwostyle.current_mode() != "dark"
    lw = layout.lw
    hardware = neutral(0.7)
    entities = [
        Entity("star", "star", roles.star, "data", ls="none", marker="*", lw=lw),
        Entity("planet", "planet", roles.planet, "data", ls="none", marker="o", lw=lw),
        Entity(
            "exozodi",
            "exozodiacal dust",
            roles.disk,
            "data",
            hatch="///",
            lw=0.8 * lw,
            fill_alpha=0.12 if light else 0.22,
        ),
        Entity(
            "local_zodi",
            "local zodiacal dust",
            hwostyle.palette.green,
            "data",
            hatch="..",
            lw=0.8 * lw,
            fill_alpha=0.10 if light else 0.18,
        ),
        Entity("aperture", "aperture", hardware, "data", marker="|", lw=3.0 * lw),
        Entity(
            "optics",
            "instrument optics",
            hardware,
            "data",
            marker="d",
            lw=lw,
            fill_alpha=0.15,
        ),
        Entity("lenslet", "lenslet", hardware, "data", marker="H", lw=0.8 * lw),
        Entity(
            "detector",
            "detector",
            hardware,
            "data",
            marker="s",
            hatch="xx",
            lw=lw,
            fill_alpha=0.10,
        ),
        Entity("ray", "light ray", roles.star, "data", arrowstyle="-|>", lw=lw),
        Entity(
            "integration",
            "integration direction",
            neutral(0.8),
            "scenery",
            ls="--",
            arrowstyle="]->",
            lw=lw,
        ),
        Entity(
            "data",
            "data flow",
            neutral(0.6),
            "scenery",
            arrowstyle="simple",
            lw=0.8 * lw,
            fill_alpha=0.18,
        ),
        Entity(
            "reference_plane",
            "reference plane",
            neutral(0.45),
            "scenery",
            ls="-.",
            lw=0.8 * lw,
            fill_alpha=0.10,
        ),
        Entity("scenery", "scenery", neutral(0.45), "scenery", lw=0.7 * lw),
        Entity(
            "furniture",
            "furniture",
            neutral(0.2),
            "furniture",
            ls="none",
            lw=0.5 * lw,
            fill_alpha=1.0,
        ),
        Entity(
            "annotation",
            "annotation",
            neutral(0.6),
            "scenery",
            ls="none",
            fontstyle="italic",
            lw=lw,
        ),
    ]
    return Cast(
        {e.key: e for e in entities}, mode=hwostyle.current_mode(), layout=layout
    )


def wavelength_color(wavelength_nm):
    """The fixed wavelength-to-color mapping for a spectral channel.

    Args:
        wavelength_nm: Wavelength in nanometers.

    Returns:
        A hex color string.
    """
    return wavelength_to_hex(float(wavelength_nm))


IMAGE_ROLES = (
    "pupil",
    "opd",
    "phase",
    "intensity",
    "readouts",
    "residual",
    "statistic",
    "probability",
    "mask",
)

# The chapter's proposed map per image quantity. hwostyle does not yet ship
# the pupil and statistic roles and resolves opd to the residual map, so the
# book's figures read the proposal from this one table, not from hwostyle.
_NAMED_MAPS = {
    "opd": "BrBG",
    "phase": "twilight",
    "intensity": "viridis",
    "readouts": "magma",
    "residual": "RdBu_r",
    "statistic": "PuOr",
}


def image_cmap(role):
    """The colormap for an image quantity, by its role name.

    Args:
        role: One of ``IMAGE_ROLES``.

    Returns:
        A Matplotlib colormap for the active mode.

    Raises:
        ValueError: If ``role`` is not a known image role.
    """
    if role in _NAMED_MAPS:
        return plt.colormaps[_NAMED_MAPS[role]]
    bg = plt.rcParams["axes.facecolor"]
    cyan = hwostyle.palette.cyan
    if role == "pupil":
        return LinearSegmentedColormap.from_list("explainer_pupil", [bg, cyan], N=256)
    if role == "probability":
        fg = plt.rcParams["text.color"]
        return LinearSegmentedColormap.from_list(
            "explainer_probability", [bg, cyan, fg], N=256
        )
    if role == "mask":
        return ListedColormap([neutral(0.2), neutral(0.45)], name="explainer_mask")
    msg = f"unknown image role {role!r}; known: {IMAGE_ROLES}"
    raise ValueError(msg)


# Style appliers for entities that are ordinary Matplotlib artists


def mark(ax, key, xy, cast, *, scale=1.0, label=None, label_offset_pt=None):
    """Draw a point entity (star, planet, or any entity with a marker).

    Args:
        ax: Axes to draw into.
        key: Cast key of an entity that has a marker.
        xy: Position in data coordinates.
        cast: The active ``Cast``.
        scale: Marker size relative to the layout's base size. The star
            glyph is drawn 1.8 times larger than other markers at equal
            scale, so a star and a planet read as different sizes.
        label: Optional text placed near the mark, in the mark's color.
        label_offset_pt: Label offset from the mark, in points. None puts
            the label just below the mark, clear of it at any size.

    Returns:
        The ``Line2D`` of the mark.
    """
    entity = cast[key]
    size = cast.layout.marker_pt * scale * (1.8 if key == "star" else 1.0)
    (line,) = ax.plot([xy[0]], [xy[1]], zorder=5, **entity.marker_kw(size))
    if label_offset_pt is None:
        label_offset_pt = (0, -0.6 * size - 2)
    if label is not None:
        text_color = entity.color if entity.rank == "data" else cast["annotation"].color
        halo(
            ax.annotate(
                label,
                xy,
                xytext=label_offset_pt,
                textcoords="offset points",
                ha="center",
                va="top" if label_offset_pt[1] < 0 else "bottom",
                color=text_color,
            ),
            cast,
        )
    return line


def region(ax, key, patch, cast, *, label=None, label_xy=None):
    """Style a Matplotlib patch as a region entity and add it to ``ax``.

    Use this for dust clouds and disks (``Ellipse``, ``Circle``,
    ``matplotlib.patches.Annulus``), a detector block (``Rectangle``), or a
    furniture pane.

    Args:
        ax: Axes to draw into.
        key: Cast key, for example ``"exozodi"`` or ``"local_zodi"``.
        patch: An unadded Matplotlib patch carrying only geometry.
        cast: The active ``Cast``.
        label: Optional text in the entity's color.
        label_xy: Label position in data coordinates; required with a label.

    Returns:
        The added patch.
    """
    patch.update(cast[key].patch_kw())
    ax.add_patch(patch)
    if label is not None:
        halo(
            ax.text(*label_xy, label, color=cast[key].color, ha="center", va="center"),
            cast,
        )
    return patch


def note(ax, xy, text, cast, *, transform=None, **text_kw):
    """Write an annotation in the shared note style (italic, neutral).

    Args:
        ax: Axes to write into.
        xy: Position, in data coordinates unless ``transform`` is given.
        text: The note.
        cast: The active ``Cast``.
        transform: Optional transform for ``xy``.
        **text_kw: Passed to ``ax.text`` (``ha``, ``va``, ``fontsize``).

    Returns:
        The ``Text`` artist.
    """
    kw = {
        "color": cast["annotation"].color,
        "fontstyle": cast["annotation"].fontstyle,
        "fontsize": cast.layout.small_pt,
    }
    kw.update(text_kw)
    if transform is not None:
        kw["transform"] = transform
    return halo(ax.text(*xy, text, **kw), cast)


def halo(text, cast, *, width_pt=None):
    """Outline a text artist in the background color so it reads over marks.

    The helpers apply this to every label they draw; call it on labels you
    place yourself over hatched regions, images or lines.

    Args:
        text: A Matplotlib ``Text`` (or ``Annotation``).
        cast: The active ``Cast``.
        width_pt: Outline width in points; None scales with the layout.

    Returns:
        The same text artist.
    """
    width = 0.3 * cast.layout.font_pt if width_pt is None else width_pt
    text.set_path_effects(
        [patheffects.withStroke(linewidth=width, foreground=cast.background)]
    )
    return text


# Hand-rolled shapes that eyepiece does not provide


def arrow(
    ax,
    start,
    end,
    kind,
    cast,
    *,
    source=None,
    color=None,
    label=None,
    label_frac=0.5,
    label_offset_pt=(0, 6),
    connectionstyle="arc3",
):
    """Draw one of the three arrow kinds, each with its own shape.

    - ``"ray"``: light propagating from ``start`` to ``end``. A solid shaft
      with a filled triangular head. Its color follows the emitting entity
      (``source``), because cause and effect share color.
    - ``"integration"``: the direction a line-of-sight integral runs. A
      dashed shaft with a bar at the start and an open head at the end.
    - ``"data"``: a transformation of data, not a physical path. A hollow
      block arrow.

    Args:
        ax: Axes to draw into.
        start: Tail position in data coordinates.
        end: Head position in data coordinates.
        kind: One of ``ARROW_KINDS``.
        cast: The active ``Cast``.
        source: For a ray, the cast key of the emitting entity; sets the
            color. None keeps the ray entity's color (starlight).
        color: Explicit color override; wins over ``source``.
        label: Optional text beside the shaft, in the arrow's color.
        label_frac: Label position along the shaft, 0 at the tail.
        label_offset_pt: Label offset in points, perpendicular to nothing in
            particular (plain x, y offset).
        connectionstyle: Matplotlib connection style, for curved ray and
            data paths. Integration arrows are always straight.

    Returns:
        A list of the artists drawn; the head-bearing patch is first.

    Raises:
        ValueError: If ``kind`` is not one of ``ARROW_KINDS``.
    """
    if kind not in ARROW_KINDS:
        msg = f"unknown arrow kind {kind!r}; known: {ARROW_KINDS}"
        raise ValueError(msg)
    entity = cast[kind]
    if color is None:
        color = cast[source].color if source is not None else entity.color
    scale = cast.layout.marker_pt
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    artists = []
    if kind == "ray":
        head = FancyArrowPatch(
            start,
            end,
            arrowstyle=(
                f"-|>,head_length={0.55 * scale / 7:.3f},"
                f"head_width={0.3 * scale / 7:.3f}"
            ),
            mutation_scale=scale * 1.6,
            color=color,
            lw=entity.lw,
            ls="-",
            shrinkA=0,
            shrinkB=0,
            connectionstyle=connectionstyle,
            zorder=4,
        )
        artists.append(ax.add_patch(head))
    elif kind == "integration":
        # A dashed chevron would break up, so the shaft (with its start bar)
        # is dashed and the head is a separate solid patch over its last
        # few percent. Integration arrows are therefore straight.
        head = FancyArrowPatch(
            end - 0.02 * (end - start),
            end,
            arrowstyle="->,head_length=0.4,head_width=0.2",
            mutation_scale=scale * 2.6,
            color=color,
            lw=entity.lw,
            ls="-",
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
        shaft = FancyArrowPatch(
            start,
            end,
            arrowstyle="]-,widthA=0.35,lengthA=0.0",
            mutation_scale=scale * 2.6,
            color=color,
            lw=entity.lw,
            ls=entity.ls,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
        artists.append(ax.add_patch(head))
        artists.append(ax.add_patch(shaft))
    else:
        width = 0.25 * scale / 7
        head = FancyArrowPatch(
            start,
            end,
            arrowstyle=(
                f"simple,head_length={1.2 * width:.3f},head_width={1.6 * width:.3f},"
                f"tail_width={0.6 * width:.3f}"
            ),
            mutation_scale=scale * 3.0,
            facecolor=to_rgba(color, entity.fill_alpha),
            edgecolor=color,
            lw=entity.lw,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=connectionstyle,
            zorder=4,
        )
        artists.append(ax.add_patch(head))
    if label is not None:
        pos = start + label_frac * (end - start)
        artists.append(
            halo(
                ax.annotate(
                    label,
                    pos,
                    xytext=label_offset_pt,
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if label_offset_pt[1] >= 0 else "top",
                    color=color,
                ),
                cast,
            )
        )
    return artists


def angle_arc(
    ax,
    vertex,
    start_deg,
    end_deg,
    cast,
    *,
    radius,
    label=None,
    color=None,
    sense=True,
    label_radius=1.35,
):
    """Draw a labeled angle arc from ``start_deg`` to ``end_deg``.

    Angles are measured counterclockwise from the axes' +x direction, in
    data coordinates, so the axes must have equal aspect. The arc runs from
    the start ray to the end ray; with ``sense`` an arrowhead at the end
    shows the direction in which the angle is measured, which is what a
    reader needs to fix a sign convention.

    Args:
        ax: Axes with equal aspect.
        vertex: Vertex in data coordinates.
        start_deg: Direction of the reference ray, degrees.
        end_deg: Direction of the measured ray, degrees. Smaller than
            ``start_deg`` means a clockwise angle.
        cast: The active ``Cast``.
        radius: Arc radius in data units.
        label: Text at the arc's midpoint, for example ``r"$i$"``.
        color: Arc and label color; None uses a bright neutral.
        sense: Whether to draw the arrowhead at the end of the arc.
        label_radius: Label distance as a multiple of ``radius``.

    Returns:
        A list of the artists drawn; the ``Arc`` is first.
    """
    color = cast.neutral(0.85) if color is None else color
    lo, hi = sorted((start_deg, end_deg))
    arc = Arc(
        vertex,
        2 * radius,
        2 * radius,
        theta1=lo,
        theta2=hi,
        color=color,
        lw=cast.layout.lw,
        zorder=4,
    )
    artists = [ax.add_patch(arc)]
    vx, vy = vertex
    if sense:
        step = math.copysign(
            min(6.0, 0.25 * abs(end_deg - start_deg)), end_deg - start_deg
        )
        tip = math.radians(end_deg)
        back = math.radians(end_deg - step)
        artists.append(
            ax.add_patch(
                FancyArrowPatch(
                    (vx + radius * math.cos(back), vy + radius * math.sin(back)),
                    (vx + radius * math.cos(tip), vy + radius * math.sin(tip)),
                    arrowstyle="-|>",
                    mutation_scale=cast.layout.marker_pt * 1.4,
                    color=color,
                    lw=cast.layout.lw,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=4,
                )
            )
        )
    if label is not None:
        mid = math.radians(0.5 * (start_deg + end_deg))
        artists.append(
            halo(
                ax.text(
                    vx + label_radius * radius * math.cos(mid),
                    vy + label_radius * radius * math.sin(mid),
                    label,
                    color=color,
                    ha="center",
                    va="center",
                ),
                cast,
            )
        )
    return artists


def scale_break(ax, xy, cast, *, along_deg=0.0, size_pt=None, gap_pt=None):
    """Draw a scale break: two slashes cutting a line at ``xy``.

    The break marks a distance that is not drawn to scale, for example
    between an astronomical scene and the telescope. Its size is fixed in
    points, so it looks the same whatever the data scale.

    Args:
        ax: Axes to draw into.
        xy: Center of the break, in data coordinates, on the line it cuts.
        cast: The active ``Cast``.
        along_deg: Direction of the line being cut, in degrees on the page.
        size_pt: Slash length in points; None scales with the layout.
        gap_pt: Separation of the two slashes in points; None scales with
            the layout.

    Returns:
        A list of the artists drawn: the background band and two slashes.
    """
    size = (size_pt or 1.6 * cast.layout.marker_pt) / 72.0
    gap = (gap_pt or 0.5 * cast.layout.marker_pt) / 72.0
    fig = ax.figure
    place = fig.dpi_scale_trans + transforms.ScaledTranslation(
        xy[0], xy[1], ax.transData
    )
    along = math.radians(along_deg)
    ux, uy = math.cos(along), math.sin(along)
    # Slashes lean 60 degrees from the line.
    slash = math.radians(along_deg + 60.0)
    sx, sy = 0.5 * size * math.cos(slash), 0.5 * size * math.sin(slash)
    artists = []
    band = Polygon(
        [
            (-0.5 * gap * ux - sx, -0.5 * gap * uy - sy),
            (0.5 * gap * ux - sx, 0.5 * gap * uy - sy),
            (0.5 * gap * ux + sx, 0.5 * gap * uy + sy),
            (-0.5 * gap * ux + sx, -0.5 * gap * uy + sy),
        ],
        closed=True,
        facecolor=cast.background,
        edgecolor="none",
        transform=place,
        zorder=6,
    )
    artists.append(ax.add_patch(band))
    for sign in (-0.5, 0.5):
        cx, cy = sign * gap * ux, sign * gap * uy
        line = Line2D(
            [cx - sx, cx + sx],
            [cy - sy, cy + sy],
            color=cast["scenery"].color,
            lw=cast.layout.lw,
            transform=place,
            zorder=7,
        )
        artists.append(ax.add_line(line))
    return artists


def badge(ax, cast, text="schematic, not to scale", *, loc="upper right"):
    """Put a status badge in a corner of a panel.

    Every diagram states on the graphic whether it is a schematic, an
    illustrative calculation or a simulation, so it stays honest when it is
    pasted into a slide without its caption.

    Args:
        ax: Axes to label.
        cast: The active ``Cast``.
        text: The status wording.
        loc: ``"upper right"``, ``"upper left"``, ``"lower right"`` or
            ``"lower left"``.

    Returns:
        The ``Text`` artist.
    """
    vertical, horizontal = loc.split()
    x = 0.99 if horizontal == "right" else 0.01
    y = 0.99 if vertical == "upper" else 0.01
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=horizontal,
        va="top" if vertical == "upper" else "bottom",
        fontsize=cast.layout.small_pt,
        fontstyle="italic",
        color=cast["annotation"].color,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": cast.background,
            "edgecolor": cast["scenery"].color,
            "lw": 0.6 * cast.layout.lw,
            "ls": "--",
        },
        zorder=8,
    )


def reference_plane(
    ax, center, width, depth, cast, *, skew=0.35, label=None, label_side="right"
):
    """Draw a plane seen obliquely, as a sheared parallelogram.

    Args:
        ax: Axes to draw into.
        center: Center of the plane in data coordinates.
        width: Horizontal extent in data units.
        depth: Vertical (foreshortened) extent in data units.
        cast: The active ``Cast``.
        skew: Horizontal shear of the far edge relative to the near edge, as
            a fraction of ``width``.
        label: Optional plane name, placed beside the far corner.
        label_side: ``"right"`` or ``"left"``.

    Returns:
        The added ``Polygon``.
    """
    cx, cy = center
    shift = 0.5 * skew * width
    corners = [
        (cx - 0.5 * width - shift, cy - 0.5 * depth),
        (cx + 0.5 * width - shift, cy - 0.5 * depth),
        (cx + 0.5 * width + shift, cy + 0.5 * depth),
        (cx - 0.5 * width + shift, cy + 0.5 * depth),
    ]
    plane = region(ax, "reference_plane", Polygon(corners, closed=True), cast)
    if label is not None:
        x, y = corners[2] if label_side == "right" else corners[3]
        halo(
            ax.text(
                x,
                y,
                label,
                color=cast["reference_plane"].color,
                ha="left" if label_side == "right" else "right",
                va="bottom",
            ),
            cast,
        )
    return plane


@dataclass
class PlaneCard:
    """A framed thumbnail of the quantity that lives on one plane.

    Attributes:
        ax: The inset axes holding the thumbnail.
        image: The ``AxesImage``; its norm is pinned at construction.
        label: The label ``Text``, or None.
    """

    ax: Any
    image: Any
    label: Any

    def update(self, data):
        """Replace the thumbnail data without changing its norm.

        Args:
            data: A 2D array of the same shape.
        """
        self.image.set_data(np.asarray(data, dtype=float))


def plane_card(
    ax,
    center,
    size,
    data,
    cast,
    *,
    role,
    norm="linear",
    vmin=None,
    vmax=None,
    label=None,
    label_pos="below",
):
    """Hold a face-on thumbnail of a plane's quantity at a place in a diagram.

    The thumbnail is an inset axes placed in the parent's data coordinates
    and framed in the reference-plane style, so a reader sees which plane
    the image belongs to. The colormap comes from the quantity's role and
    the norm is pinned from ``vmin`` and ``vmax``, so an animation that
    updates the card keeps one scale. Log and symmetric norms are drawn with
    eyepiece's ``imshow_log`` and ``imshow_diverging``.

    Args:
        ax: Parent axes, preferably with equal aspect.
        center: Card center in the parent's data coordinates.
        size: Card ``(width, height)`` in the parent's data units, or one
            number for a square card.
        data: 2D array to show.
        cast: The active ``Cast``.
        role: Image role from ``IMAGE_ROLES``.
        norm: ``"linear"``, ``"log"`` or ``"symmetric"`` (about zero).
        vmin: Lower bound (linear, log). None uses the data minimum.
        vmax: Upper bound (linear, log), or the symmetric half-range. None
            uses the data maximum, or the largest absolute value.
        label: Optional text naming the plane or quantity.
        label_pos: ``"below"`` or ``"above"``.

    Returns:
        A ``PlaneCard``.

    Raises:
        ValueError: If ``norm`` is not one of the three names.
    """
    width, height = (size, size) if np.isscalar(size) else size
    arr = np.asarray(data, dtype=float)
    bounds = [center[0] - 0.5 * width, center[1] - 0.5 * height, width, height]
    inset = ax.inset_axes(bounds, transform=ax.transData, zorder=5)
    cmap = image_cmap(role)
    if norm == "log":
        image = ep.imshow_log(
            arr, ax=inset, vmin=vmin, vmax=vmax, cmap=cmap, colorbar=False
        ).artists["image"]
    elif norm == "symmetric":
        vlim = float(np.nanmax(np.abs(arr))) if vmax is None else vmax
        image = ep.imshow_diverging(
            arr, ax=inset, vlim=vlim, cmap=cmap, colorbar=False
        ).artists["image"]
    elif norm == "linear":
        lo = float(np.nanmin(arr)) if vmin is None else vmin
        hi = float(np.nanmax(arr)) if vmax is None else vmax
        image = inset.imshow(
            arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest", origin="lower"
        )
    else:
        msg = f"unknown norm {norm!r}; use 'linear', 'log' or 'symmetric'"
        raise ValueError(msg)
    inset.set_aspect("auto")
    inset.set_xticks([])
    inset.set_yticks([])
    plane = cast["reference_plane"]
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(plane.color)
        spine.set_linestyle(plane.ls)
        spine.set_linewidth(plane.lw)
    text = None
    if label is not None:
        below = label_pos == "below"
        text = halo(
            ax.annotate(
                label,
                (center[0], center[1] + (-0.5 if below else 0.5) * height),
                xytext=(0, -3 if below else 3),
                textcoords="offset points",
                ha="center",
                va="top" if below else "bottom",
                color=plane.color,
                fontsize=cast.layout.small_pt,
            ),
            cast,
        )
    return PlaneCard(ax=inset, image=image, label=text)


def aperture(ax, center, diameter, cast, *, angle_deg=90.0, label=None):
    """Draw a collecting aperture edge-on, as a thick bar with end stops.

    Args:
        ax: Axes to draw into.
        center: Center of the aperture in data coordinates.
        diameter: Aperture diameter in data units.
        cast: The active ``Cast``.
        angle_deg: Direction of the bar on the page; 90 is vertical, facing
            light that travels along x.
        label: Optional text placed beyond the upper end.

    Returns:
        The ``Line2D`` of the bar (with its end-stop markers).
    """
    entity = cast["aperture"]
    ang = math.radians(angle_deg)
    dx, dy = 0.5 * diameter * math.cos(ang), 0.5 * diameter * math.sin(ang)
    (bar,) = ax.plot(
        [center[0] - dx, center[0] + dx],
        [center[1] - dy, center[1] + dy],
        color=entity.color,
        lw=entity.lw,
        solid_capstyle="butt",
        # End stops across the bar: a two-sided line marker rotated so it
        # stands perpendicular to the bar at any angle.
        marker=(2, 2, angle_deg),
        ms=1.6 * cast.layout.marker_pt,
        mew=0.6 * entity.lw,
        zorder=4,
    )
    if label is not None:
        halo(
            ax.annotate(
                label,
                (center[0] + dx, center[1] + dy),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=entity.color,
            ),
            cast,
        )
    return bar


def lens(ax, center, height, cast, *, thickness=0.18, label=None):
    """Draw a generic refractive element as a thin ellipse.

    Args:
        ax: Axes to draw into.
        center: Center in data coordinates.
        height: Clear height in data units.
        cast: The active ``Cast``.
        thickness: Width as a fraction of ``height``.
        label: Optional text placed above the element.

    Returns:
        The added ``Ellipse``.
    """
    patch = region(ax, "optics", Ellipse(center, thickness * height, height), cast)
    if label is not None:
        halo(
            ax.annotate(
                label,
                (center[0], center[1] + 0.5 * height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=cast["optics"].color,
            ),
            cast,
        )
    return patch


def lenslet_grid(
    ax, origin, shape, pitch, cast, *, rotation_deg=0.0, highlight=None, color=None
):
    """Draw a square lenslet array: cell outlines with an inscribed lens.

    Each cell is a square of side ``pitch`` with a circle inscribed, so a
    lenslet cell reads differently from a detector pixel (a plain square).

    Args:
        ax: Axes with equal aspect.
        origin: Lower-left corner of cell (0, 0) before rotation, in data
            coordinates. The array rotates about this point.
        shape: ``(n_rows, n_cols)``.
        pitch: Cell side in data units.
        cast: The active ``Cast``.
        rotation_deg: Rotation of the whole array, counterclockwise.
        highlight: Iterable of ``(row, col)`` cells drawn in ``color``.
        color: Highlight color; None uses the planet color.

    Returns:
        A dict mapping ``(row, col)`` to that cell's ``(square, circle)``
        patches.
    """
    entity = cast["lenslet"]
    highlight = set(highlight or ())
    color = cast["planet"].color if color is None else color
    rot = transforms.Affine2D().rotate_deg_around(origin[0], origin[1], rotation_deg)
    trans = rot + ax.transData
    cells = {}
    n_rows, n_cols = shape
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = origin[0] + col * pitch
            y0 = origin[1] + row * pitch
            on = (row, col) in highlight
            ink = color if on else entity.color
            square = Rectangle(
                (x0, y0),
                pitch,
                pitch,
                facecolor=to_rgba(ink, 0.25 if on else 0.0),
                edgecolor=ink,
                lw=entity.lw,
                transform=trans,
                zorder=3,
            )
            circle = Circle(
                (x0 + 0.5 * pitch, y0 + 0.5 * pitch),
                0.42 * pitch,
                facecolor="none",
                edgecolor=ink,
                lw=0.7 * entity.lw,
                transform=trans,
                zorder=3,
            )
            ax.add_patch(square)
            ax.add_patch(circle)
            cells[(row, col)] = (square, circle)
    return cells


def pixel_grid(
    ax,
    origin,
    shape,
    pitch,
    cast,
    *,
    values=None,
    role="readouts",
    vmin=None,
    vmax=None,
):
    """Draw a detector pixel grid, optionally filled with pixel values.

    Args:
        ax: Axes to draw into.
        origin: Lower-left corner of pixel (0, 0) in data coordinates.
        shape: ``(n_rows, n_cols)``.
        pitch: Pixel side in data units.
        cast: The active ``Cast``.
        values: Optional ``(n_rows, n_cols)`` array shown with ``role``'s
            colormap on a pinned linear norm.
        role: Image role for ``values``.
        vmin: Norm lower bound; None uses the data minimum.
        vmax: Norm upper bound; None uses the data maximum.

    Returns:
        A dict with ``"image"`` (the ``AxesImage`` or None), ``"grid"`` (the
        grid lines) and ``"frame"`` (the outer ``Rectangle``).
    """
    entity = cast["detector"]
    n_rows, n_cols = shape
    x0, y0 = origin
    x1, y1 = x0 + n_cols * pitch, y0 + n_rows * pitch
    image = None
    if values is not None:
        arr = np.asarray(values, dtype=float)
        image = ax.imshow(
            arr,
            extent=(x0, x1, y0, y1),
            origin="lower",
            cmap=image_cmap(role),
            vmin=float(np.nanmin(arr)) if vmin is None else vmin,
            vmax=float(np.nanmax(arr)) if vmax is None else vmax,
            interpolation="nearest",
            zorder=2,
        )
    grid = []
    for col in range(1, n_cols):
        x = x0 + col * pitch
        grid += ax.plot([x, x], [y0, y1], color=cast.neutral(0.35), lw=0.5 * entity.lw)
    for row in range(1, n_rows):
        y = y0 + row * pitch
        grid += ax.plot([x0, x1], [y, y], color=cast.neutral(0.35), lw=0.5 * entity.lw)
    frame = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor="none",
        edgecolor=entity.color,
        lw=1.5 * entity.lw,
        zorder=3,
    )
    ax.add_patch(frame)
    return {"image": image, "grid": grid, "frame": frame}


# Registry records

_SLUG = re.compile(r"^d\d\d(-[a-z0-9]+)+$")
MOTION_GROUNDS = ("viewpoint", "rate", "narration")


def _check_slug(slug):
    if not _SLUG.match(slug):
        msg = (
            f"slug {slug!r} must be lowercase words joined by hyphens after a "
            "dNN prefix, for example 'd02-orbit-planes'"
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class FigureSpec:
    """One still diagram, rendered in every venue from one builder.

    Attributes:
        slug: Output name, ``dNN-words``; the ``dNN`` must match the module.
        build: ``build(layout, cast) -> Figure``. Called once per venue
            inside the right style mode; it must not save or close the
            figure.
        caption: The caption for the documentation page: governing clause,
            physical assumptions, and whether the picture is an identity, a
            proposed convention or an example implementation.
        alt: Alternative text describing what the picture shows.
        status: Short honesty label written into the provenance stamp, for
            example ``"schematic, not to scale"`` or ``"illustrative
            calculation"``.
        params: Scientific inputs, recorded in the manifest.
        venues: Layout names to render, a subset of ``("doc", "slide")``.
            A talk-only still (one step of a slide relay) sets
            ``("slide",)`` so no unused documentation files are written.
    """

    slug: str
    build: Callable
    caption: str
    alt: str
    status: str = "schematic, not to scale"
    params: dict = field(default_factory=dict)
    venues: tuple = ("doc", "slide")

    def __post_init__(self):
        """Validate the slug and require a caption and alt text."""
        _check_slug(self.slug)
        if not self.caption.strip() or not self.alt.strip():
            msg = f"{self.slug}: caption and alt text are required"
            raise ValueError(msg)
        if not self.venues or not set(self.venues) <= {"doc", "slide"}:
            msg = f"{self.slug}: venues must be a nonempty subset of doc, slide"
            raise ValueError(msg)


@dataclass
class AnimationScene:
    """A built animation: a figure, a draw function, and its frames.

    Attributes:
        fig: The figure, built once with every artist present.
        draw: ``draw(fig, frame)``; mutates existing artists to show one
            frame. It must set the complete visible state from ``frame``
            alone, so any frame can be drawn in any order (the preview does
            exactly that).
        frames: A sequence of frame states (numbers, dicts, tuples). Each
            state is self-contained, not an increment.
    """

    fig: Any
    draw: Callable
    frames: Sequence


@dataclass(frozen=True)
class AnimationSpec:
    """One animation, rendered as documentation videos and a talk MP4.

    Attributes:
        slug: Output name, ``dNN-words``; must differ from every figure slug.
        build: ``build(layout, cast) -> AnimationScene``. In the
            documentation layout the scene may have at most
            ``layout.frame_budget`` frames; use ``layout.n_frames(n)``.
        ground: Why motion is warranted: ``"viewpoint"`` (the camera
            moves), ``"rate"`` (a rate is the concept) or ``"narration"`` (a
            live narration beat).
        caption: Caption for the documentation page.
        alt: Alternative text for the video.
        status: Honesty label for the provenance stamp.
        params: Scientific inputs, recorded in the manifest.
        fps: Playback rate; None uses eyepiece's preset for each venue.
        hold_s: Seconds to hold the first and the last frame in every video.
        preview_frames: Frame indices written by ``--preview``; None picks
            the first, middle and last frames. Choose frames at events.
        allow_rescale: Permit axis limits or color norms to change between
            frames. False makes a drifting scale fail the build.
    """

    slug: str
    build: Callable
    ground: str
    caption: str
    alt: str
    status: str = "schematic, not to scale"
    params: dict = field(default_factory=dict)
    fps: int | None = None
    hold_s: tuple = (1.0, 2.0)
    preview_frames: tuple | None = None
    allow_rescale: bool = False

    def __post_init__(self):
        """Validate the slug, the motion ground and the text fields."""
        _check_slug(self.slug)
        if self.ground not in MOTION_GROUNDS:
            msg = f"{self.slug}: ground must be one of {MOTION_GROUNDS}"
            raise ValueError(msg)
        if not self.caption.strip() or not self.alt.strip():
            msg = f"{self.slug}: caption and alt text are required"
            raise ValueError(msg)


# The vocabulary sheet


def cast_sheet(layout, cast):
    """Draw every cast entity and helper once, as a visual reference.

    The driver's ``--cast-sheet`` option previews this figure; the tests use
    it as a throwaway spec.

    Args:
        layout: ``DOC`` or ``SLIDE``.
        cast: The active ``Cast``.

    Returns:
        The figure.
    """
    fig, ax = figure(layout, doc_height_in=4.4)
    ax.set(xlim=(0, 12), ylim=(0, 7.3), aspect="equal")
    ax.axis("off")
    row = 6.2
    mark(ax, "star", (0.8, row), cast, label="star")
    mark(ax, "planet", (2.4, row), cast, label="planet")
    region(ax, "exozodi", Ellipse((4.4, row), 1.8, 0.8), cast)
    ax.text(4.4, row - 0.7, "exozodi", ha="center", color=cast["exozodi"].color)
    region(ax, "local_zodi", Ellipse((6.8, row), 1.8, 0.8), cast)
    ax.text(6.8, row - 0.7, "local zodi", ha="center", color=cast["local_zodi"].color)
    aperture(ax, (8.6, row), 1.0, cast, label="aperture")
    lens(ax, (10.2, row), 1.0, cast, label="optics")

    row = 3.9
    lenslet_grid(ax, (0.3, row - 0.6), (2, 2), 0.6, cast, highlight={(0, 1)})
    ax.text(0.9, row - 1.0, "lenslets", ha="center", color=cast["lenslet"].color)
    pixel_grid(
        ax,
        (2.2, row - 0.6),
        (3, 3),
        0.4,
        cast,
        values=np.arange(9.0).reshape(3, 3),
    )
    ax.text(2.8, row - 1.0, "pixels", ha="center", color=cast["detector"].color)
    region(ax, "detector", Rectangle((3.9, row - 0.5), 0.5, 1.0), cast)
    ax.text(4.15, row - 1.0, "detector", ha="center", color=cast["detector"].color)
    reference_plane(ax, (6.3, row), 2.0, 0.8, cast, label="plane")
    plane_card(
        ax,
        (9.0, row),
        1.2,
        np.outer(np.hanning(16), np.hanning(16)) + 1e-3,
        cast,
        role="intensity",
        norm="log",
        vmin=1e-3,
        vmax=1.0,
        label="plane card",
    )

    row = 1.4
    arrow(ax, (0.3, row), (2.6, row), "ray", cast, label="ray")
    arrow(ax, (3.2, row), (5.5, row), "integration", cast, label="integration")
    arrow(ax, (6.1, row), (8.4, row), "data", cast, label="data")
    ax.plot([8.9, 11.6], [row - 0.3, row - 0.3], **cast["scenery"].line_kw())
    ax.plot([8.9, 10.43], [row - 0.3, row + 0.99], **cast["scenery"].line_kw())
    scale_break(ax, (11.0, row - 0.3), cast)
    angle_arc(ax, (8.9, row - 0.3), 0.0, 40.0, cast, radius=1.0, label=r"$\theta$")
    badge(ax, cast, loc="lower right")
    note(ax, (0.3, 0.2), "annotation: italic neutral note", cast)
    return fig
