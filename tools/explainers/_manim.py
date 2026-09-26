"""Render Manim clips for the explainer registry, and the helpers they draw with.

Import this module only to build or test a clip: it imports Manim, which
needs system Cairo, Pango, pkg-config and ffmpeg (``pip install
".[explainers-manim]"``). Nothing else in the package imports it.

The clip path mirrors the Matplotlib path. A diagram module declares a
``ManimSpec`` (see ``explainers/__init__.py``) whose scene factory returns an
``ExplainerScene`` subclass. For each venue (documentation light and dark,
talk dark) the scene is rendered inside the same hwostyle mode and layout as
the stills, with the same entity cast, and every frame is piped straight to
ffmpeg with the exact arguments the Matplotlib recordings use, so the MP4 is
byte-reproducible and carries no metadata. Manim's own movie writer and its
partial-movie cache are never used.

Styling comes from eyepiece: the render profile (``snapshot_profile``) and
the Pango registration of the font Matplotlib resolves. The helpers here are
only what eyepiece's Manim renderer does not provide: the three arrow kinds
and a geometric vector, a status badge, the provenance stamp, Matplotlib
mathtext set as glyph outlines (no TeX), dashed polylines with gaps, and the
readability check of a held state.

Sizes are in points at the delivered size, as in the stills: a documentation
clip is 7.2 by 3.6 inches at the layout's animation dpi, and a talk clip is
16 by 9 inches at 1920 by 1080 pixels. ``Venue.font_size`` and
``Venue.stroke`` convert points to Manim's ``font_size`` and
``stroke_width``.
"""

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cairo
import eyepiece as ep
import eyepiece.anim
import manim
import manimpango
import matplotlib
import numpy as np
from eyepiece.manim._render import _ensure_font
from eyepiece.style import snapshot_profile
from matplotlib import font_manager
from matplotlib.colors import to_hex
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, text_to_path

from explainers import _common as ex
from explainers import _export as exporter

PACKAGE = Path(__file__).resolve().parent
FRAME_HEIGHT = 8.0  # Manim scene units across the frame height
STROKE_UNITS = 0.01  # scene units per unit of Manim stroke_width (Cairo)
EM_FONT_SIZE = 72.0  # a Manim font_size of 72 sets a one-unit em
LAYOUT_CHARS = 80  # longest line Pango lays out unwrapped at a one-unit em
# (venue, mode) pairs rendered for every clip, in manifest order.
VARIANTS = (("doc", "light"), ("doc", "dark"), ("slide", "dark"))
# Two separate labels closer than this read as one.
MIN_TEXT_GAP_PT = 1.0
# Constructions that need a TeX installation; none may appear in a scene.
TEX_CLASSES = (manim.SingleStringMathTex,)


# Venues


@dataclasses.dataclass(frozen=True)
class Venue:
    """One rendered variant: a layout, a style mode, pixels and a frame rate.

    Attributes:
        layout: ``ex.DOC`` or ``ex.SLIDE``.
        mode: ``"light"`` or ``"dark"``.
        pixel_width: Frame width in pixels.
        pixel_height: Frame height in pixels.
        fps: Frames per second.
    """

    layout: ex.Layout
    mode: str
    pixel_width: int
    pixel_height: int
    fps: int

    @property
    def name(self):
        """The layout name, ``"doc"`` or ``"slide"``."""
        return self.layout.name

    @property
    def frame_width(self):
        """Scene units across the frame width."""
        return FRAME_HEIGHT * self.pixel_width / self.pixel_height

    @property
    def pt_per_unit(self):
        """Points at the delivered size per scene unit."""
        return self.pixel_height / FRAME_HEIGHT * 72.0 / self.layout.anim_dpi

    def units(self, pt):
        """Scene units for a length in points."""
        return pt / self.pt_per_unit

    def font_size(self, pt):
        """Manim ``font_size`` that draws text at ``pt`` points."""
        return EM_FONT_SIZE * self.units(pt)

    def stroke(self, pt):
        """Manim ``stroke_width`` that draws a line ``pt`` points wide."""
        return self.units(pt) / STROKE_UNITS


def make_venue(name, mode, fps):
    """The venue for a layout name and mode at a frame rate.

    A documentation clip is the documentation layout's default size (7.2 by
    3.6 inches) at its animation dpi; a talk clip is 1920 by 1080.
    """
    layout = ex.LAYOUTS[name]
    if layout.is_slide:
        width, height = 1920, 1080
    else:
        width = round(layout.width_in * layout.anim_dpi)
        height = round(layout.height_in * layout.anim_dpi)
    return Venue(layout, mode, width, height, int(fps))


@dataclasses.dataclass(frozen=True)
class Style:
    """What a scene draws with: the venue, the entity cast and the font.

    Attributes:
        venue: The ``Venue``.
        cast: The ``ex.Cast`` of the venue's mode and layout.
        font: The font family Matplotlib resolves, registered with Pango.
    """

    venue: Venue
    cast: object
    font: str

    def color(self, value):
        """A Manim color from any Matplotlib color."""
        return manim.ManimColor(to_hex(value))

    def entity(self, key):
        """The Manim color of a cast entity."""
        return self.color(self.cast[key].color)

    def neutral(self, level):
        """A neutral between the background (0) and the text color (1)."""
        return self.color(self.cast.neutral(level))

    @property
    def text_color(self):
        """The text color of the mode."""
        return self.color(self.cast.text)

    @property
    def background(self):
        """The background color of the mode."""
        return self.color(self.cast.background)


@contextmanager
def activate(venue):
    """Activate the venue's style mode and layout, and yield its ``Style``.

    The mode, the layout's rc settings and the cast are those of the
    Matplotlib stills (``exporter.venue``); the font is the family eyepiece's
    render profile resolves, registered with Pango from the file Matplotlib
    draws it with.
    """
    with exporter.venue(venue.mode, venue.layout) as cast:
        profile = snapshot_profile(
            text_size_pt=venue.font_size(venue.layout.font_pt),
            stroke_width_pt=venue.stroke(venue.layout.lw),
        )
        _ensure_font(profile.font_family)
        yield Style(venue=venue, cast=cast, font=profile.font_family)


def font_file(family):
    """The font file Matplotlib (and so Pango) draws ``family`` from."""
    return Path(
        font_manager.findfont(
            font_manager.FontProperties(family=family), fallback_to_default=False
        )
    )


# Text


def _mark_text(mob, string, pt, role):
    """Record what a text mobject says and its nominal size, for the checks."""
    mob.explainer_text = string
    mob.explainer_pt = float(pt)
    mob.explainer_height = float(mob.height)
    mob.explainer_role = role
    return mob


def text(style, string, pt=None, *, color=None, italic=False, role="text"):
    """Pango text in the resolved font at ``pt`` points (default: body size).

    Args:
        style: The active ``Style``.
        string: The text; plain Unicode, no markup.
        pt: Size in points at the delivered size.
        color: Any Matplotlib color; None uses the text color.
        italic: Set in italic (notes and badges).
        role: ``"text"``, or ``"stamp"`` for the provenance line.

    Returns:
        A ``manim.Text``.
    """
    pt = style.venue.layout.font_pt if pt is None else pt
    # Pango lays text out at Manim's font_size / 4.8 in a 600-unit-wide box.
    # Below a font_size of a few tens it hints the glyph advances to a coarse
    # grid and letters drift apart; above about 80 characters per line at a
    # one-unit em it wraps. Lay out as large as the longest line allows, then
    # scale to the requested size.
    longest = max(len(line) for line in string.split("\n"))
    layout_size = EM_FONT_SIZE * min(1.0, LAYOUT_CHARS / max(longest, 1))
    mob = manim.Text(
        string,
        font=style.font,
        font_size=layout_size,
        color=style.text_color if color is None else style.color(color),
        slant=manim.ITALIC if italic else manim.NORMAL,
    )
    mob.scale(style.venue.font_size(pt) / layout_size)
    return _mark_text(mob, string, pt, role)


def _glyph_outline(vm, path, scale):
    """Append a Matplotlib path's contours to a VMobject, scaled to scene units."""
    for verts, code in path.iter_segments(simplify=False, curves=True):
        pts = np.column_stack(
            [np.reshape(verts, (-1, 2)) * scale, np.zeros(len(verts) // 2)]
        )
        if code == MplPath.MOVETO:
            vm.start_new_path(pts[0])
        elif code == MplPath.LINETO:
            vm.add_line_to(pts[0])
        elif code == MplPath.CURVE3:
            vm.add_quadratic_bezier_curve_to(pts[0], pts[1])
        elif code == MplPath.CURVE4:
            vm.add_cubic_bezier_curve_to(pts[0], pts[1], pts[2])
        elif code == MplPath.CLOSEPOLY:
            vm.close_path()
    return vm


class MathLabel(manim.VMobject):
    """Matplotlib mathtext set as glyph outlines, so no TeX is needed.

    The glyphs come from Matplotlib's own mathtext layout in the active
    style, so a symbol in a clip is the symbol in the matching still. The
    label's origin is its baseline start; ``advance`` is its advance width,
    so labels placed side by side share one baseline.
    """

    def __init__(self, style, tex, pt=None, *, color=None, italic=False, **kwargs):
        r"""Lay out ``tex`` (for example ``r"$R_Z(\Omega)$"``) at ``pt`` points.

        ``italic`` sets the text outside ``$...$`` in italic, as notes are.
        """
        super().__init__(**kwargs)
        pt = style.venue.layout.font_pt if pt is None else pt
        prop = font_manager.FontProperties(
            family=style.font, style="italic" if italic else "normal", size=100.0
        )
        em = style.venue.units(pt)
        path = TextPath((0.0, 0.0), tex, size=1.0, prop=prop)
        _glyph_outline(self, path, em)
        width, _, _ = text_to_path.get_text_width_height_descent(tex, prop, ismath=True)
        self.advance = em * width / 100.0
        self.set_fill(style.text_color if color is None else style.color(color), 1.0)
        self.set_stroke(width=0.0, opacity=0.0)
        _mark_text(self, tex, pt, "text")


def math_row(style, pieces, pt=None, *, gap_em=0.25, colors=None):
    """Mathtext pieces laid side by side on one baseline.

    Args:
        style: The active ``Style``.
        pieces: Mathtext strings, one per piece.
        pt: Size in points.
        gap_em: Space between pieces, in ems.
        colors: Optional color per piece.

    Returns:
        A ``VGroup`` of ``MathLabel`` pieces, in order.
    """
    pt = style.venue.layout.font_pt if pt is None else pt
    row = manim.VGroup()
    x = 0.0
    for k, tex in enumerate(pieces):
        piece = MathLabel(style, tex, pt, color=None if colors is None else colors[k])
        piece.shift(x * manim.RIGHT)
        x += piece.advance + gap_em * style.venue.units(pt)
        row.add(piece)
    return row


def place(mob, point, ha="center", va="center"):
    """Move ``mob`` so its box edge or center named by ``ha``/``va`` is at ``point``."""
    x, y = float(point[0]), float(point[1])
    left, bottom = mob.get_corner(manim.DL)[:2]
    right, top = mob.get_corner(manim.UR)[:2]
    ax = {"left": left, "center": 0.5 * (left + right), "right": right}[ha]
    ay = {"bottom": bottom, "center": 0.5 * (bottom + top), "top": top}[va]
    mob.shift(np.array([x - ax, y - ay, 0.0]))
    return mob


# Lines and arrows


def _runs(xy):
    """Finite runs of a polyline with NaN rows as gaps."""
    xy = np.asarray(xy, dtype=float)
    ok = np.all(np.isfinite(xy), axis=1)
    runs, start = [], None
    for k, good in enumerate(ok):
        if good and start is None:
            start = k
        if not good and start is not None:
            runs.append(xy[start:k])
            start = None
    if start is not None:
        runs.append(xy[start:])
    return [r for r in runs if len(r) >= 2]


def dash_runs(xy, pattern):
    """Cut a polyline into the drawn pieces of a dash pattern.

    Args:
        xy: ``(N, 2)`` points; NaN rows are gaps.
        pattern: On and off lengths in the polyline's units, alternating and
            starting with on; None draws it solid.

    Returns:
        A list of ``(M, 2)`` pieces.
    """
    runs = _runs(xy)
    if not pattern:
        return runs
    pieces = []
    for run in runs:
        seg = np.hypot(*np.diff(run, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        s, k = 0.0, 0
        while s < total - 1e-12:
            e = min(s + pattern[k % len(pattern)], total)
            if k % 2 == 0 and e > s:
                inner = run[(cum > s) & (cum < e)]
                ends = [
                    np.array(
                        [np.interp(t, cum, run[:, 0]), np.interp(t, cum, run[:, 1])]
                    )
                    for t in (s, e)
                ]
                pieces.append(np.vstack([ends[0], inner, ends[1]]))
            s, k = e, k + 1
    return pieces


class Poly(manim.VMobject):
    """A polyline in scene coordinates that can be redrawn in place.

    NaN rows split it; a dash pattern (in points) cuts it into dashes; a
    closed, undashed polyline can be filled. Redrawing keeps the mobject's
    identity, so an updater can move it every frame.
    """

    def __init__(
        self,
        style,
        *,
        color,
        width_pt,
        dash_pt=None,
        fill=None,
        fill_opacity=0.0,
        **kwargs,
    ):
        """Style a polyline; call ``redraw`` to give it points.

        Raises:
            ValueError: If it is both dashed and filled; the dashes are open
                pieces, so a filled dashed outline is two polylines.
        """
        if dash_pt is not None and fill is not None:
            msg = "a dashed polyline cannot be filled; draw the fill separately"
            raise ValueError(msg)
        super().__init__(**kwargs)
        self.style_ = style
        self.pattern = (
            None if dash_pt is None else [style.venue.units(p) for p in dash_pt]
        )
        self.set_stroke(
            style.color(color), width=style.venue.stroke(width_pt), opacity=1.0
        )
        if fill is not None:
            self.set_fill(style.color(fill), opacity=fill_opacity)
        else:
            self.set_fill(opacity=0.0)

    def redraw(self, xy, *, closed=False):
        """Replace the points with the polyline ``xy`` (scene x, y)."""
        xy = np.asarray(xy, dtype=float)
        if closed and len(xy) and np.all(np.isfinite(xy)):
            xy = np.vstack([xy, xy[:1]])
        self.clear_points()
        for piece in dash_runs(xy, self.pattern):
            pts = np.column_stack([piece, np.zeros(len(piece))])
            self.start_new_path(pts[0])
            self.add_points_as_corners(pts[1:])
        return self


def head_points(tip, direction, length, width):
    """The three corners of an arrowhead whose point is at ``tip``."""
    d = np.asarray(direction, dtype=float)
    norm = np.hypot(*d)
    d = d / norm if norm > 1e-12 else np.array([1.0, 0.0])
    n = np.array([-d[1], d[0]])
    tip = np.asarray(tip, dtype=float)
    base = tip - length * d
    return np.array([base + 0.5 * width * n, tip, base - 0.5 * width * n])


class Head(manim.VMobject):
    """An arrowhead that can be redrawn: filled (``-|>``) or open (``->``)."""

    def __init__(self, style, *, color, size_pt, width_pt, filled=True, **kwargs):
        """Style a head ``size_pt`` long; call ``redraw`` to place it."""
        super().__init__(**kwargs)
        self.style_ = style
        self.filled = filled
        self.head_len = style.venue.units(size_pt)
        self.head_wid = style.venue.units(0.6 * size_pt if filled else 0.9 * size_pt)
        self.set_stroke(
            style.color(color), width=style.venue.stroke(width_pt), opacity=1.0
        )
        self.set_fill(style.color(color), opacity=1.0 if filled else 0.0)

    def redraw(self, tip, direction):
        """Point the head along ``direction`` with its point at ``tip``."""
        corners = head_points(tip, direction, self.head_len, self.head_wid)
        pts = np.column_stack([corners, np.zeros(3)])
        self.clear_points()
        if self.filled:
            self.set_points_as_corners(np.vstack([pts, pts[:1]]))
        else:
            self.set_points_as_corners(pts)
        return self


class Vector(manim.VGroup):
    """A geometric vector (thin shaft, open head): not a light ray."""

    def __init__(self, style, *, color, width_pt, head_pt, **kwargs):
        """Style a vector; call ``redraw`` to place it."""
        self.shaft = Poly(style, color=color, width_pt=width_pt)
        self.head = Head(
            style, color=color, size_pt=head_pt, width_pt=width_pt, filled=False
        )
        super().__init__(self.shaft, self.head, **kwargs)

    def redraw(self, start, end):
        """Run the vector from ``start`` to ``end`` (scene x, y)."""
        start, end = np.asarray(start, float), np.asarray(end, float)
        self.shaft.redraw(np.array([start, end]))
        self.head.redraw(end, end - start)
        return self


def arrow(style, kind, start, end, *, source=None):
    """One of the three arrow kinds of the stills, with its own shape.

    - ``"ray"``: light from ``start`` to ``end``; solid shaft, filled head,
      in the color of the emitting entity (``source``, default starlight).
    - ``"integration"``: the direction a line-of-sight integral runs; dashed
      shaft, a bar at the start, an open head.
    - ``"data"``: a transformation of data; a hollow block arrow.

    Returns:
        A ``VGroup`` whose ``kind`` attribute names the kind.

    Raises:
        ValueError: If ``kind`` is not one of ``ex.ARROW_KINDS``.
    """
    if kind not in ex.ARROW_KINDS:
        msg = f"unknown arrow kind {kind!r}; known: {ex.ARROW_KINDS}"
        raise ValueError(msg)
    cast, layout = style.cast, style.venue.layout
    entity = cast[kind]
    color = cast[source].color if source is not None else entity.color
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = end - start
    u = d / np.hypot(*d)
    n = np.array([-u[1], u[0]])
    size = 1.6 * layout.marker_pt
    group = manim.VGroup()
    if kind == "ray":
        head = Head(style, color=color, size_pt=size, width_pt=entity.lw)
        head.redraw(end, d)
        shaft = Poly(style, color=color, width_pt=entity.lw)
        shaft.redraw(np.array([start, end - 0.9 * head.head_len * u]))
        group.add(shaft, head)
    elif kind == "integration":
        shaft = Poly(style, color=color, width_pt=entity.lw, dash_pt=(4.0, 3.0))
        shaft.redraw(np.array([start, end]))
        half = 0.5 * style.venue.units(size)
        bar = Poly(style, color=color, width_pt=entity.lw)
        bar.redraw(np.array([start + half * n, start - half * n]))
        head = Head(style, color=color, size_pt=size, width_pt=entity.lw, filled=False)
        head.redraw(end, d)
        group.add(shaft, bar, head)
    else:
        w = style.venue.units(0.35 * size)
        hl = style.venue.units(size)
        neck = end - hl * u
        outline = np.array(
            [
                start + 0.5 * w * n,
                neck + 0.5 * w * n,
                neck + w * n,
                end,
                neck - w * n,
                neck - 0.5 * w * n,
                start - 0.5 * w * n,
            ]
        )
        block = Poly(
            style,
            color=color,
            width_pt=entity.lw,
            fill=color,
            fill_opacity=entity.fill_alpha,
        )
        block.redraw(outline, closed=True)
        group.add(block)
    group.kind = kind
    return group


def badge(style, string, *, corner=manim.UL, margin_pt=6.0):
    """A status badge in a frame corner: italic text in a dashed box.

    Every clip keeps its status on every frame, as every still does, so a
    paused frame pasted into a slide stays honest.
    """
    cast, venue = style.cast, style.venue
    label = text(
        style,
        string,
        venue.layout.small_pt,
        color=cast["annotation"].color,
        italic=True,
    )
    pad = venue.units(0.4 * venue.layout.small_pt)
    left, bottom = label.get_corner(manim.DL)[:2] - pad
    right, top = label.get_corner(manim.UR)[:2] + pad
    corners = np.array([[left, bottom], [right, bottom], [right, top], [left, top]])
    back = Poly(
        style,
        color=cast.background,
        width_pt=0.6 * venue.layout.lw,
        fill=cast.background,
        fill_opacity=1.0,
    ).redraw(corners, closed=True)
    back.set_stroke(opacity=0.0)
    box = Poly(
        style,
        color=cast["scenery"].color,
        width_pt=0.6 * venue.layout.lw,
        dash_pt=(3.0, 2.0),
    ).redraw(corners, closed=True)
    group = manim.VGroup(back, box, label)
    m = venue.units(margin_pt)
    half = np.array([venue.frame_width / 2 - m, FRAME_HEIGHT / 2 - m, 0.0])
    group.move_to(corner * half, aligned_edge=corner)
    return group


def stamp(style, prov, status):
    """The provenance line of the stills, along the bottom-left edge."""
    venue = style.venue
    line = ep.provenance_text(
        script=prov.script, sha=prov.sha, date=prov.date, note=status
    )
    mob = text(
        style, line, venue.layout.stamp_pt, color=style.cast.neutral(0.45), role="stamp"
    )
    m = venue.units(0.6 * venue.layout.stamp_pt)
    mob.move_to(
        np.array([-venue.frame_width / 2 + m, -FRAME_HEIGHT / 2 + m, 0.0]),
        aligned_edge=manim.DL,
    )
    return mob


# Readability of a held state


def _visible(member):
    if isinstance(member, manim.VMobject):
        return member.has_points() and (
            member.get_fill_opacity() > 0
            or (member.get_stroke_opacity() > 0 and member.get_stroke_width() > 0)
        )
    return False


def _box(mob):
    (x0, y0), (x1, y1) = mob.get_corner(manim.DL)[:2], mob.get_corner(manim.UR)[:2]
    return x0, y0, x1, y1


def _gap(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def readability(mobjects, venue):
    """Drawn text and stroke sizes of a held state, at the delivered size.

    Floors: every text at least 95 percent of the layout's small type (the
    provenance stamp at least its stamp size), every stroke at least half the
    layout's base line width, nothing visible outside the frame, no two
    texts closer than ``MIN_TEXT_GAP_PT``, and no mobject that needs TeX.

    Args:
        mobjects: The scene's top-level mobjects.
        venue: The ``Venue`` the state is delivered at.

    Returns:
        Plain JSON: the smallest text (the stamp aside) and stroke in points,
        the text closest to its floor, the crowded text pairs, anything
        outside the frame, any TeX mobject, and ``passed``.
    """
    layout = venue.layout
    texts, strokes, outside, tex = [], [], [], []
    boxes = []
    glyphs = {}  # id of every glyph mobject -> the text it belongs to
    half_w = venue.frame_width / 2 + 1e-6
    half_h = FRAME_HEIGHT / 2 + 1e-6
    for top in mobjects:
        for member in top.get_family():
            if isinstance(member, TEX_CLASSES):
                tex.append(type(member).__name__)
            if not hasattr(member, "explainer_pt"):
                continue
            glyphs.update((id(g), member.explainer_text) for g in member.get_family())
            if not any(_visible(g) for g in member.family_members_with_points()):
                continue
            ratio = (
                member.height / member.explainer_height
                if member.explainer_height
                else 1.0
            )
            drawn = member.explainer_pt * ratio
            floor = (
                layout.stamp_pt
                if member.explainer_role == "stamp"
                else 0.95 * layout.small_pt
            )
            texts.append(
                (drawn - floor, drawn, member.explainer_text, member.explainer_role)
            )
            boxes.append((member.explainer_text, _box(member)))
    for top in mobjects:
        for member in top.get_family():
            if not _visible(member):
                continue
            left, bottom, right, upper = _box(member)
            if left < -half_w or right > half_w or bottom < -half_h or upper > half_h:
                outside.append(glyphs.get(id(member), type(member).__name__))
            if id(member) in glyphs:
                continue
            if member.get_stroke_opacity() > 0 and member.get_stroke_width() > 0:
                strokes.append(
                    member.get_stroke_width() * STROKE_UNITS * venue.pt_per_unit
                )
    crowded = sorted(
        {
            tuple(sorted((a, b)))
            for i, (a, box_a) in enumerate(boxes)
            for b, box_b in boxes[i + 1 :]
            if _gap(box_a, box_b) * venue.pt_per_unit < MIN_TEXT_GAP_PT
        }
    )
    worst = min(texts) if texts else (math.inf, math.inf, None, None)
    content = [t[1] for t in texts if t[3] != "stamp"]
    stroke_floor = 0.5 * layout.lw
    return {
        "min_text_margin_pt": round(float(worst[0]), 2),
        "min_text_pt": round(float(min(content)), 2) if content else None,
        "text_floor_pt": round(0.95 * layout.small_pt, 3),
        "tightest_text": worst[2],
        "n_text": len(texts),
        "min_stroke_pt": round(float(min(strokes)), 3) if strokes else None,
        "stroke_floor_pt": stroke_floor,
        "outside_frame": sorted(set(map(str, outside))),
        "crowded_text": [list(p) for p in crowded],
        "tex_mobjects": sorted(set(tex)),
        "passed": bool(
            worst[0] >= -1e-6
            and (not strokes or min(strokes) >= stroke_floor - 1e-6)
            and not outside
            and not crowded
            and not tex
        ),
    }


def printed(mobjects):
    """Every visible text string of a held state, sorted."""
    found = set()
    for top in mobjects:
        for member in top.get_family():
            if hasattr(member, "explainer_text") and any(
                _visible(g) for g in member.family_members_with_points()
            ):
                found.add(member.explainer_text)
    return sorted(found)


# The scene base


class ExplainerScene(manim.Scene):
    """A Manim scene drawn in an explainer venue's style.

    Subclasses implement ``construct`` and mark beats with ``section(name)``.
    At the end of every section the scene records the readability report
    and the printed strings of its held state, and (when asked) the rendered
    frame, so every section's final state is checked and can be previewed.

    Attributes:
        style: The ``Style`` (venue, cast, font).
        prov: The ``Provenance`` stamped on every frame.
        status: The honesty label of the clip.
        reports: Section name -> readability report.
        strings: Section name -> printed strings.
        frames: Section name -> RGBA frame (only with ``capture=True``).
    """

    def __init__(self, *, style, prov, status, capture=False, **kwargs):
        """Bind the scene to a style; ``kwargs`` go to ``manim.Scene``."""
        self.style = style
        self.prov = prov
        self.status = status
        self.capture = capture
        self.reports = {}
        self.strings = {}
        self.frames = {}
        self._current = None
        super().__init__(**kwargs)

    def setup(self):
        """Paint the background of the mode and add the provenance stamp."""
        self.camera.background_color = self.style.background
        self.add(stamp(self.style, self.prov, self.status))

    def section(self, name):
        """Close the previous section and start ``name`` (a beat boundary)."""
        self._close_section()
        self.next_section(name)
        self._current = name

    def _close_section(self):
        if self._current is None:
            return
        self.reports[self._current] = readability(self.mobjects, self.style.venue)
        self.strings[self._current] = printed(self.mobjects)
        if self.capture:
            self.renderer.update_frame(self, ignore_skipping=True)
            self.frames[self._current] = np.array(self.renderer.get_frame(), copy=True)
        self._current = None

    def tear_down(self):
        """Close the last section."""
        self._close_section()
        super().tear_down()


# Rendering and encoding


def ffmpeg_args(path, venue):
    """The ffmpeg command that encodes raw RGBA frames into ``path``.

    It is the command Matplotlib's ``FFMpegWriter`` runs for an eyepiece
    recording: the same writer (``eyepiece.anim``, with the even-size crop)
    and the same bit-exact output arguments as the Matplotlib clips, so both
    kinds of clip are encoded identically.
    """
    writer = eyepiece.anim._make_writer(path, venue.fps, exporter.BITEXACT)
    writer.outfile = str(path)
    return [
        eyepiece.anim._resolve_ffmpeg(),
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{venue.pixel_width}x{venue.pixel_height}",
        "-pix_fmt",
        "rgba",
        "-framerate",
        str(venue.fps),
        "-loglevel",
        "error",
        "-i",
        "pipe:",
        *writer.output_args,
    ]


class _Encoder:
    """Feed rendered frames to ffmpeg, or only count them."""

    def __init__(self, path, venue):
        self.frames = 0
        self.proc = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen(
                ffmpeg_args(path, venue),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

    def write_frame(self, frame, num_frames=1):
        if self.proc is not None:
            data = np.ascontiguousarray(frame[..., :4], dtype=np.uint8).tobytes()
            for _ in range(num_frames):
                self.proc.stdin.write(data)
        self.frames += num_frames

    def close(self):
        if self.proc is None:
            return
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            msg = f"ffmpeg failed: {err.strip()}"
            raise RuntimeError(msg)


def render(scene_class, venue, *, path=None, prov, status, skip=False, capture=False):
    """Render a scene in one venue; encode it to ``path`` when one is given.

    Args:
        scene_class: An ``ExplainerScene`` subclass.
        venue: The ``Venue``.
        path: MP4 to write, or None to render without encoding.
        prov: ``Provenance`` for the stamp.
        status: Honesty label for the stamp.
        skip: Evaluate only each animation's final state (fast; for tests).
        capture: Keep the rendered frame at the end of every section.

    Returns:
        The finished scene; its ``frame_count`` attribute is the number of
        encoded frames.
    """
    with tempfile.TemporaryDirectory() as media, activate(venue) as style:
        options = {
            "pixel_width": venue.pixel_width,
            "pixel_height": venue.pixel_height,
            "frame_height": FRAME_HEIGHT,
            "frame_width": venue.frame_width,
            "frame_rate": venue.fps,
            "background_color": to_hex(style.cast.background),
            "disable_caching": True,
            "dry_run": True,
            "media_dir": media,
            "verbosity": "ERROR",
            "progress_bar": "none",
        }
        with manim.tempconfig(options):
            scene = scene_class(
                style=style,
                prov=prov,
                status=status,
                capture=capture,
                skip_animations=skip,
            )
            encoder = _Encoder(None if skip else path, venue)
            scene.renderer.file_writer.write_frame = encoder.write_frame
            try:
                scene.render()
            finally:
                encoder.close()
            scene.frame_count = encoder.frames
    return scene


def provenance(module_file):
    """Provenance of a clip: its module plus the shared and Manim tooling."""
    files = [
        Path(module_file),
        PACKAGE / "__init__.py",
        PACKAGE / "_common.py",
        PACKAGE / "_export.py",
        PACKAGE / "_manim.py",
    ]
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    base = exporter.provenance(module_file)
    return exporter.Provenance(
        script=base.script,
        sha=digest[:12],
        date=base.date,
        sources={exporter._relative(f): exporter.sha256(f) for f in files},
    )


def output_path(spec, venue):
    """Repository-relative MP4 path of a clip in one venue."""
    if venue.layout.is_slide:
        return exporter.ANIMATIONS / f"{spec.slug}.mp4"
    return exporter.DOC_VIDEOS / f"explainer-{spec.slug}-{venue.mode}.mp4"


def variants(spec):
    """The venues a clip renders, in manifest order."""
    return [
        make_venue(name, mode, spec.fps[name])
        for name, mode in VARIANTS
        if name in spec.venues
    ]


def export_clip(spec, prov, *, root=exporter.ROOT):
    """Render one ``ManimSpec`` to every venue.

    Returns:
        ``(written, frames, reports)``: the paths relative to ``root``, the
        frame counts per venue, and the readability reports per venue and
        section.

    Raises:
        ValueError: A documentation clip runs longer than ``doc_max_s``.
        RuntimeError: A section fails the readability check.
    """
    scene_class = spec.scene()
    written, frames, reports = [], {}, {}
    for venue in variants(spec):
        rel = output_path(spec, venue)
        target = Path(root) / rel
        scene = render(scene_class, venue, path=target, prov=prov, status=spec.status)
        seconds = scene.frame_count / venue.fps
        if not venue.layout.is_slide and seconds > spec.doc_max_s + 1e-9:
            target.unlink(missing_ok=True)
            msg = (
                f"{spec.slug}: the documentation clip runs {seconds:.2f} s, "
                f"over {spec.doc_max_s} s"
            )
            raise ValueError(msg)
        failed = {k: r for k, r in scene.reports.items() if not r["passed"]}
        if failed:
            target.unlink(missing_ok=True)
            msg = f"{spec.slug} ({venue.name}, {venue.mode}): unreadable {failed}"
            raise RuntimeError(msg)
        written.append(rel)
        key = "talk" if venue.layout.is_slide else "doc"
        frames[key] = scene.frame_count
        reports[f"{venue.name}-{venue.mode}"] = {
            name: {k: r[k] for k in ("passed", "min_text_pt", "min_stroke_pt")}
            for name, r in scene.reports.items()
        }
    return written, frames, reports


def toolchain(font):
    """Versions of every renderer component, and the font file's hash."""
    path = font_file(font)
    return {
        "renderer": "manim (Cairo), frames piped to ffmpeg",
        "versions": {
            "manim": importlib.metadata.version("manim"),
            "manimpango": manimpango.__version__,
            "pango": manimpango.pango_version(),
            "pycairo": cairo.version,
            "cairo": cairo.cairo_version_string(),
            "eyepiece": importlib.metadata.version("eyepiece"),
            "matplotlib": matplotlib.__version__,
            "ffmpeg": exporter._ffmpeg_version(),
        },
        "font": {"family": font, "file": path.name, "sha256": exporter.sha256(path)},
    }


def export_module(module, clips, *, root=exporter.ROOT):
    """Render every clip of a module and return the manifest entries.

    Returns:
        ``(written, entries)``: the paths relative to ``root`` and the
        ``"manim"`` manifest entries.
    """
    prov = provenance(module.__file__)
    revision = exporter.git_revision(exporter.ROOT)
    written, entries = [], []
    for spec in clips:
        paths, frames, reports = export_clip(spec, prov, root=root)
        with activate(variants(spec)[0]) as style:
            chain = toolchain(style.font)
        written += paths
        entries.append(
            {
                "slug": spec.slug,
                "still": spec.still,
                "status": spec.status,
                "ground": spec.ground,
                "caption": spec.caption,
                "alt": spec.alt,
                "params": spec.params,
                "fps": {k: spec.fps[k] for k in spec.venues},
                "frames": frames,
                "source_hash": prov.sha,
                "source_sha256": prov.sources,
                "revision": revision,
                **chain,
                "readability": reports,
                "outputs": exporter._describe(root, paths),
            }
        )
    return written, entries


def preview_module(module, clips, *, root=exporter.ROOT):
    """Write every clip's section-end frames as PNGs to the preview directory."""
    from PIL import Image

    prov = provenance(module.__file__)
    written = []
    for spec in clips:
        scene_class = spec.scene()
        for venue in variants(spec):
            scene = render(
                scene_class,
                venue,
                prov=prov,
                status=spec.status,
                skip=True,
                capture=True,
            )
            for name, frame in scene.frames.items():
                rel = (
                    exporter.PREVIEW
                    / f"{spec.slug}-{name}-{venue.name}-{venue.mode}.png"
                )
                (Path(root) / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(Path(root) / rel)
                written.append(rel)
            report = {k: r["passed"] for k, r in scene.reports.items()}
            rel = (
                exporter.PREVIEW
                / f"{spec.slug}-{venue.name}-{venue.mode}-readability.json"
            )
            (Path(root) / rel).write_text(json.dumps(scene.reports, indent=2) + "\n")
            written.append(rel)
            if not all(report.values()):
                print(f"{spec.slug} {venue.name} {venue.mode}: readability {report}")
    return written
