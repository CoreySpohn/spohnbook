"""Numerical anchors of the surface-to-phasor explainer.

Every number printed on the strip is recomputed here by hand arithmetic or a
closed form, independently of the module's own arithmetic and of
physicaloptix: the chapter's worked fields E_0 = 0.2 and delta E = -0.1, the
intensities 0.04 and 0.01, the change -0.03 and its two terms -0.04 and
+0.01, the normal-reflection factor W = 2h, and the phase pi/4 of a
lambda/8 OPD under exp(+i 2 pi W/lambda). The worked numbers are also read
back from the chapter text, so the figure cannot drift from it.
"""

import cmath
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from explainers import _common as ex
from explainers import _export as exporter
from explainers import d10_phasor as d10

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "docs" / "conventions" / "optics-images.md"
CATALOG = ROOT / "docs" / "_data" / "handbook.yaml"


def build(layout=ex.DOC, mode="light"):
    with exporter.venue(mode, layout) as cast:
        return d10.FIGURES[0].build(layout, cast)


def texts(fig):
    return [t.get_text() for t in fig.findobj(Text) if t.get_text().strip()]


@pytest.fixture(scope="module")
def doc_fig():
    fig = build()
    yield fig
    plt.close(fig)


def test_worked_fields_are_the_chapters():
    # The chapter states: "For real `E_0=0.2` and `delta_E=-0.1`, the nominal
    # intensity is 0.04, the change is -0.03, and the final intensity is 0.01."
    text = CHAPTER.read_text()
    sentence = re.search(
        r"For real `E_0=([-\d.]+)` and `delta_E=([-\d.]+)`,([^\n]*)", text
    )
    assert sentence is not None
    assert float(sentence.group(1)) == d10.PARAMS["e0"] == 0.2
    assert float(sentence.group(2)) == d10.PARAMS["delta_e"] == -0.1
    rest = sentence.group(3)
    assert "nominal intensity is 0.04" in rest
    assert "the change is -0.03" in rest
    assert "final intensity is 0.01" in rest


def test_worked_arithmetic_by_hand():
    # Exact decimal arithmetic, in hundredths: E_0 = 20, delta E = -10.
    e0, de = Fraction(2, 10), Fraction(-1, 10)
    e = e0 + de
    assert e == Fraction(1, 10)
    assert e0**2 == Fraction(4, 100)
    assert e**2 == Fraction(1, 100)
    assert e**2 - e0**2 == Fraction(-3, 100)
    # Delta I = 2 Re(conj(E_0) delta E) + |delta E|^2, both terms real here.
    assert 2 * e0 * de == Fraction(-4, 100)
    assert de**2 == Fraction(1, 100)
    assert 2 * e0 * de + de**2 == e**2 - e0**2
    w = d10.worked()
    for key, value in (
        ("e", e),
        ("i0", e0**2),
        ("i1", e**2),
        ("di", e**2 - e0**2),
        ("cross", 2 * e0 * de),
        ("square", de**2),
    ):
        assert w[key] == pytest.approx(float(value), abs=1e-15)


def test_normal_reflection_doubles_the_displacement():
    # Hecht 2017, Sec. 9.4.2, text before eq. 9.44: OPD = 2 d cos(theta),
    # so 2 h at normal incidence.
    assert d10.h_waves() == Fraction(1, 16)
    assert d10.opd_waves() == Fraction(1, 8)


def test_phase_arrow_is_plus_pi_over_four_from_physicaloptix():
    # W = lambda/8 under exp(+i 2 pi W/lambda): phase 2 pi / 8 = pi/4,
    # counterclockwise, unit length.
    arrow = d10.phase_arrow()
    assert abs(arrow) == pytest.approx(1.0, abs=1e-12)
    assert cmath.phase(arrow) == pytest.approx(math.pi / 4, abs=1e-12)
    assert arrow.imag > 0


def test_printed_numbers_are_exactly_the_checked_ones(doc_fig):
    printed = texts(doc_fig)
    for label in (
        "$0.04$",
        "$0.01$",
        "$-0.03$",
        "$E_0=0.2$",
        r"$\delta E=-0.1$",
        "$E=0.1$",
        r"$h=\lambda/16$",
        r"$W=\lambda/8$",
        r"$\phi=\pi/4$",
    ):
        assert label in printed
    # No other decimal number appears on the documentation still.
    numbers = sorted(
        {m for t in printed for m in re.findall(r"-?\d+\.\d+", t)}
        - {"0.04", "0.01", "-0.03", "0.2", "-0.1", "0.1"}
    )
    assert numbers == []


def test_slide_equation_matches_the_hand_terms():
    fig = build(ex.SLIDE, "dark")
    try:
        equation = [t for t in texts(fig) if "Delta I =" in t]
        assert len(equation) == 1
        assert equation[0].endswith("= -0.04 + 0.01 = -0.03$")
    finally:
        plt.close(fig)


def test_bars_are_the_squared_arrow_lengths(doc_fig):
    ax = doc_fig.axes[4]
    heights = [p.get_height() for p in ax.patches if isinstance(p, Rectangle)]
    assert heights[:3] == pytest.approx([0.2**2, 0.1**2, 0.1**2 - 0.2**2])
    assert heights[2] < 0


def _arrow_ends(ax):
    ends = []
    for patch in ax.patches:
        if isinstance(patch, FancyArrowPatch) and patch._posA_posB is not None:
            ends.append(tuple(map(tuple, np.round(patch._posA_posB, 12))))
    return ends


def test_drawn_arrows_have_the_stated_geometry(doc_fig):
    phase_ax, sum_ax = doc_fig.axes[2], doc_fig.axes[3]
    c = math.cos(math.pi / 4)
    ends = _arrow_ends(phase_ax)
    assert ((0.0, 0.0), (1.0, 0.0)) in ends
    assert ((0.0, 0.0), (round(c, 12), round(c, 12))) in ends
    # Head to tail: E_0 from the origin to 0.2, delta E back from 0.2 to
    # 0.1 (lifted), and the sum from the origin to 0.1 (lowered).
    spans = sorted((round(a[0], 12), round(b[0], 12)) for a, b in _arrow_ends(sum_ax))
    assert spans == [(0.0, 0.1), (0.0, 0.2), (0.2, 0.1)]


def test_surface_moves_away_from_the_incoming_light():
    # Light travels toward +x; the moved face lies at +x of the rest line,
    # the path-increasing direction, and the rest line is x = 0.
    assert d10.face_x(0.0) == pytest.approx(d10.H_DRAW)
    assert d10.H_DRAW > 0
    assert d10.face_x(d10.HALF_Y) == pytest.approx(0.0)


def test_caption_cites_catalogued_locators():
    caption = d10.FIGURES[0].caption
    # Hecht eq. 9.44 is the Michelson dark-fringe condition; the path change of
    # twice the mirror displacement is stated in the text just before it.
    assert (
        "{ref}`Hecht 2017, Sec. 9.4.2, text before eq. 9.44 <source-hecht2017>`"
        in caption
    )
    catalog = CATALOG.read_text()
    assert "id: hecht2017" in catalog
    assert "source: hecht2017" in catalog
    for label in ("optics-coherent-phase", "optics-signed-intensity"):
        assert f"({label})=" in CHAPTER.read_text()
        assert f"{{ref}}`{label}`" in caption


def test_prose_is_plain():
    spec = d10.FIGURES[0]
    dashes = (chr(0x2014), chr(0x2013))
    for prose in (spec.caption, spec.alt):
        assert not any(d in prose for d in dashes)
        # Contractions; the possessives "chapter's" and "book's" are allowed.
        assert not re.search(r"n't\b|'(re|ll|ve|m|d)\b", prose)
    assert spec.slug.startswith("d10-")
    assert d10.ANIMATIONS == []


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("slide", [False, True])
def test_no_label_carries_a_stroked_halo(mode, slide):
    """Labels sit on plain backing boxes; a stroked halo blobs the glyphs."""
    fig = build(ex.SLIDE if slide else ex.DOC, mode)
    try:
        stroked = [t.get_text() for t in fig.findobj(Text) if t.get_path_effects()]
        assert stroked == []
    finally:
        plt.close(fig)
