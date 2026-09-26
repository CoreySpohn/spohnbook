"""Contracts of the radiometric-collection explainer figures.

The reference-experiment figure prints numbers. Each one is recomputed here
from the exact SI value of the Planck constant and the definition of the
jansky with 40-digit decimal arithmetic, never from hwoutils, and the test
reads the text the builder actually drew. The figure's parameters must be
those of the reference case code. The schematic figures print no numbers.

The "where each factor lives" piece is checked for its geometry (straight
lines through the lens center carry the sky cell onto pixel p, and a
direction one pixel outside the cell lands on the neighbor), for the Airy
profile that reaches into p, for the relay (its face-on panel is the one the
wavelength figure carries), and for plain labels with no stroked halo.
"""

import re
import sys
from decimal import ROUND_HALF_EVEN, Decimal, getcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from matplotlib.text import Text
from scipy.special import j0, j1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from explainers import _common as ex  # noqa: E402
from explainers import _export as exporter  # noqa: E402
from explainers import d03_radiometric_collection as d03  # noqa: E402

H_SI = Decimal("6.62607015e-34")  # J s, exact (SI 2019)
JY_SI = Decimal("1e-26")  # W m-2 Hz-1 per Jy, by definition


def _independent():
    """Printed values of the reference experiment, by hand arithmetic."""
    getcontext().prec = 40
    density = JY_SI / (H_SI * Decimal(700))  # photon s-1 m-2 nm-1
    photons = density * 1 * 1 * 1  # 1 m2, 1 nm, 1 s
    return {
        "density": density,
        "per_pixel": photons / 4,
        "variance": {q: Decimal(q) * photons for q in ("0.0", "0.5", "1.0")},
    }


def _two_dp(value):
    q = Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    return f"{q:,.2f}"


def _texts(slug, layout, mode="light"):
    spec = next(s for s in d03.FIGURES if s.slug == slug)
    with exporter.venue(mode, layout) as cast:
        fig = spec.build(layout, cast)
        try:
            return [
                t.get_text()
                for t in fig.findobj(Text)
                if t.get_text().strip() and t.get_visible()
            ]
        finally:
            plt.close(fig)


def _numbers(texts):
    """Numeric tokens in the drawn text, ignoring exponents and subscripts."""
    tokens = []
    for text in texts:
        stripped = re.sub(r"[\^_]\{?-?\d+\}?", "", text)
        tokens += re.findall(r"\d[\d,]*(?:\.\d+)?", stripped)
    return tokens


def test_anchor_matches_published_value():
    # The chapter prints 21,559.8597091736 photon s-1 m-2 nm-1.
    assert f"{_independent()['density']:.10f}" == "21559.8597091736"


def test_module_numbers_match_independent_arithmetic():
    nums = d03.reference_numbers()
    ind = _independent()
    np.testing.assert_allclose(nums["density"], float(ind["density"]), rtol=1e-12)
    np.testing.assert_allclose(nums["per_pixel"], float(ind["per_pixel"]), rtol=1e-12)
    for q, value in nums["variance"].items():
        np.testing.assert_allclose(
            value, float(ind["variance"][f"{q:.1f}"]), rtol=1e-12
        )


def test_parameters_are_those_of_the_reference_case():
    source = (ROOT / "tests" / "test_radiometry_boundary.py").read_text()
    assert "AREA_M2, BANDWIDTH_NM, EXPOSURE_S, N_PIXELS = 1.0, 1.0, 1.0, 4" in source
    assert '@pytest.mark.parametrize("qe", [0.0, 0.5, 1.0])' in source
    assert "jy_to_photons_per_nm_per_m2(1.0, 700.0)" in source
    r = d03.REFERENCE
    assert (r["area_m2"], r["bandwidth_nm"], r["exposure_s"], r["n_pixels"]) == (
        1.0,
        1.0,
        1.0,
        4,
    )
    assert (r["flux_jy"], r["wavelength_nm"]) == (1.0, 700.0)
    assert r["qe_values"] == (0.0, 0.5, 1.0)


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_reference_figure_prints_exactly_the_checked_numbers(layout):
    ind = _independent()
    expected = {
        _two_dp(ind["density"]),  # density, and the photon rate over 1 m2 and 1 nm
        _two_dp(ind["per_pixel"]),
        *(_two_dp(v) for v in ind["variance"].values()),
    }
    assert expected == {"21,559.86", "5,389.96", "0.00", "10,779.93"}
    texts = _texts("d03-four-pixel-reference", layout)
    tokens = _numbers(texts)
    for value in expected:
        assert value in tokens, value
    assert tokens.count("5,389.96") == 4  # one label per pixel
    # Parameters printed as labels: 700 nm, 1 Jy, 1 nm, 1 m2, 1 s, 4 pixels,
    # and the three QE values.
    parameters = {"700", "1", "4", "0.0", "0.5", "1.0"}
    stray = set(tokens) - expected - parameters
    assert not stray, stray
    joined = "\n".join(texts)
    for label in ("1 Jy at 700 nm", "= 1 nm", "= 1 m", "= 1 s", "4 pixels"):
        assert label in joined, label
    # Each QE row pairs its q with its own value, in one drawn string.
    for q, value in ind["variance"].items():
        row = f"q = {q}:  {_two_dp(value)}"
        assert row in texts, row
    # The density carries its unit and measure on the next line.
    density = (
        r"$\Phi_{\lambda,\mathrm{nm}}$ = "
        + _two_dp(ind["density"])
        + "\n"
        + r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$"
    )
    assert any(t.startswith(density) for t in texts), density
    # The collected rate and the per-pixel count carry their units.
    assert any(_two_dp(ind["density"]) + "\n" + r"photon s$^{-1}$" in t for t in texts)
    assert texts.count(_two_dp(ind["per_pixel"]) + "\nphoton") == 4


@pytest.mark.parametrize(
    "slug",
    [
        "d03-radiometric-collection",
        "d03-collection-where",
        "d03-collection-wavelength",
        "d03-collection-electrons",
    ],
)
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_schematics_print_no_numbers(slug, layout):
    assert _numbers(_texts(slug, layout)) == []


def test_overview_names_every_factor_of_the_count_integral():
    joined = "\n".join(_texts("d03-radiometric-collection", ex.DOC))
    for symbol in (
        r"$A$",
        r"\Phi_\lambda",
        r"I_\lambda",
        r"\Omega_p",
        r"T_{\rm opt}",
        r"P_p",
        r"q_p",
        r"\mu_{e,p}",
        r"\int_{\rm live}",
        r"\lambda_b^-",
        r"\lambda_b^+",
    ):
        assert symbol in joined, symbol
    # Rate arriving and count accumulated are separate labels with their units.
    assert "arriving in $p$" in joined
    assert "accumulated in $p$" in joined
    assert r"photon s$^{-1}$ nm$^{-1}$" in joined
    assert "[electron]" in joined
    # The spatial response is chromatic, and the extended source has its own
    # replacement term.
    assert r"$P_p(\lambda)$" in joined
    assert r"$\int I_\lambda P_p\,d\Omega$ replaces" in joined
    assert r"$\Phi_\lambda T_{\rm opt}P_p q_p$" in joined
    assert "one wavelength" in joined


def test_psf_window_holds_less_than_all_light():
    frac = d03.psf_pixel_fractions()
    total = frac.sum()

    # Closed-form Airy encircled energy, 1 - J0^2 - J1^2 at pi r (r in
    # lambda/D): the 7 by 7 window contains the r = 3.5 disk and lies inside
    # the r = 3.5 sqrt(2) disk.
    def encircled(r):
        x = np.pi * r
        return 1.0 - j0(x) ** 2 - j1(x) ** 2

    assert encircled(3.5) < total < encircled(3.5 * np.sqrt(2.0)) < 1.0
    center = frac[3, 3]
    assert center == frac.max()
    assert total - center > 0.4  # much of the point source lands outside p


def test_captions_name_their_clauses():
    labels = {
        "d03-radiometric-collection": "radiometry-response-ownership",
        "d03-collection-where": "radiometry-pixel-brightness",
        "d03-collection-wavelength": "radiometry-response-ownership",
        "d03-collection-electrons": "radiometry-detector-counts",
        "d03-four-pixel-reference": "photon-electron-reference-experiment",
    }
    specs = {s.slug: s for s in d03.FIGURES}
    assert set(specs) == set(labels)
    for slug, label in labels.items():
        assert f"`{label}`" in specs[slug].caption
        assert chr(0x2014) not in specs[slug].caption + specs[slug].alt


# Relay piece 1: where each factor lives


def _build(slug, layout, mode="light"):
    spec = next(s for s in d03.FIGURES if s.slug == slug)
    with exporter.venue(mode, layout) as cast:
        return spec.build(layout, cast)


def _by_gid(ax, gid):
    return [a for a in ax.get_children() if a.get_gid() == gid]


def _rows(fig):
    """The point-source row, the extended-source row and the face-on panel."""
    axes = fig.get_axes()
    titles = [a.get_title(loc="left") for a in axes]
    point = axes[next(i for i, t in enumerate(titles) if t.startswith("point"))]
    extended = axes[next(i for i, t in enumerate(titles) if t.startswith("extended"))]
    face = axes[next(i for i, t in enumerate(titles) if t.startswith("face-on"))]
    return point, extended, face


def _pixel_p(ax):
    (rect,) = _by_gid(ax, "pixel-p")
    y = rect.get_y()
    return rect.get_x(), y, y + rect.get_height()


def _lens_center(ax):
    (lens,) = [a for a in ax.patches if type(a) is Ellipse]  # an Arc is an Ellipse
    return lens.center


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_where_names_every_factor_at_its_place(layout):
    texts = _texts("d03-collection-where", layout)
    joined = "\n".join(texts)
    for symbol in (
        r"$A$",
        r"\Phi_\lambda",
        r"T_{\rm opt}(\lambda)",
        r"$P_p(\lambda,\boldsymbol{\theta})$",
        r"$q_p(\lambda)$",
        r"I_\lambda",
        r"$\Omega_p$",
        r"P_p(\lambda,\boldsymbol{\theta})\,d\Omega$ replaces $\Phi_\lambda P_p$",
        r"$A\,\Phi_\lambda\,T_{\rm opt}\,P_p(\lambda,\boldsymbol{\theta})\,q_p$",
    ):
        assert symbol in joined, symbol
    for label in ("aperture", "optics", "detector QE", "sky cell of $p$", "outside"):
        assert label in joined, label
    assert texts.count(r"pixel $p$") == 2  # one per row
    assert "schematic, not to scale" in texts


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_where_cell_maps_onto_pixel_p_through_the_lens_center(layout):
    fig = _build("d03-collection-where", layout)
    try:
        _, ax, _ = _rows(fig)
        px, p_lo, p_hi = _pixel_p(ax)
        lx, ly = _lens_center(ax)
        (cell,) = _by_gid(ax, "sky-cell")
        cell_lo, cell_hi = cell.get_y(), cell.get_y() + cell.get_height()
        cell_x = cell.get_x() + cell.get_width()
        # The dashed cell and pixel p are centered on the lens axis.
        assert abs(0.5 * (cell_lo + cell_hi) - ly) < 1e-9
        assert abs(0.5 * (p_lo + p_hi) - ly) < 1e-9
        lines = _by_gid(ax, "cell-edge")
        assert len(lines) == 2
        landed = set()
        for line in lines:
            xs, ys = (np.asarray(v, dtype=float) for v in line.get_data())
            # Starts on a cell edge, passes the lens center, ends on the pixel.
            assert xs[0] == pytest.approx(cell_x)
            assert ys[0] == pytest.approx(cell_lo) or ys[0] == pytest.approx(cell_hi)
            assert (xs[1], ys[1]) == pytest.approx((lx, ly))
            assert xs[2] == pytest.approx(px)
            # Straight: equal slope on both sides of the pivot (similar
            # triangles, so the angle is kept).
            before = (ys[1] - ys[0]) / (xs[1] - xs[0])
            after = (ys[2] - ys[1]) / (xs[2] - xs[1])
            assert after == pytest.approx(before, rel=1e-12)
            # Inverted: the upper cell edge lands on the lower pixel edge.
            want = p_lo if ys[0] > ly else p_hi
            assert ys[2] == pytest.approx(want, abs=1e-12)
            landed.add(round(ys[2], 9))
        assert landed == {round(p_lo, 9), round(p_hi, 9)}
    finally:
        plt.close(fig)


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_where_omega_cone_has_its_vertex_at_the_lens_center(layout):
    fig = _build("d03-collection-where", layout)
    try:
        _, ax, _ = _rows(fig)
        lx, ly = _lens_center(ax)
        (cell,) = _by_gid(ax, "sky-cell")
        (cone,) = _by_gid(ax, "omega-cone")
        xy = np.asarray(cone.get_xy(), dtype=float)[:3]
        right = cell.get_x() + cell.get_width()
        top, bottom = cell.get_y() + cell.get_height(), cell.get_y()
        assert sorted(map(tuple, xy.round(12))) == sorted(
            [
                (round(right, 12), round(bottom, 12)),
                (round(right, 12), round(top, 12)),
                (round(lx, 12), round(ly, 12)),
            ]
        )
        texts = [t.get_text() for t in ax.texts]
        assert r"$\Omega_p$" in texts
    finally:
        plt.close(fig)


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_where_outside_direction_lands_on_the_neighbor(layout):
    fig = _build("d03-collection-where", layout)
    try:
        _, ax, _ = _rows(fig)
        px, p_lo, p_hi = _pixel_p(ax)
        pitch = p_hi - p_lo
        lx, ly = _lens_center(ax)
        (cell,) = _by_gid(ax, "sky-cell")
        (dot,) = _by_gid(ax, "outside-point")
        ox, oy = (float(v[0]) for v in dot.get_data())
        # Outside the dashed cell, on the far side from the landing pixel:
        # one pitch (in angle) from the cell center, half beyond its edge.
        cell_half = 0.5 * cell.get_height()
        cell_x = cell.get_x() + cell.get_width()
        assert (oy - ly) / (lx - ox) == pytest.approx(2.0 * cell_half / (lx - cell_x))
        assert oy > cell.get_y() + cell.get_height()
        # Its line through the lens center meets the detector plane at the
        # center of the pixel below p: one pixel pitch in angle.
        hit = ly + (ly - oy) * (px - lx) / (lx - ox)
        assert hit == pytest.approx(ly - pitch, abs=1e-9)
        # The drawn ray runs from the dot to that point, straight through
        # the lens center.
        (ray,) = [
            a for a in _by_gid(ax, "outside-ray") if isinstance(a, FancyArrowPatch)
        ]
        (x0, y0), (x1, y1) = ray._posA_posB
        assert (x0, y0) == pytest.approx((ox, oy))
        assert (x1, y1) == pytest.approx((px, ly - pitch))
        on_line = y0 + (y1 - y0) * (lx - x0) / (x1 - x0)
        assert on_line == pytest.approx(ly, abs=1e-9)
    finally:
        plt.close(fig)


def _airy_closed_form(r):
    """[2 J1(pi r) / (pi r)]^2 with r in lambda/D, written out here."""
    x = np.pi * np.asarray(r, dtype=float)
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = (2.0 * j1(x[nz]) / x[nz]) ** 2
    return out


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_where_profiles_are_airy_cuts_centered_where_the_rays_land(layout):
    fig = _build("d03-collection-where", layout)
    try:
        point, extended, _ = _rows(fig)
        for ax, gid, shift in (
            (point, "profile-point", 0.0),
            (extended, "profile-outside", -1.0),
        ):
            _, p_lo, p_hi = _pixel_p(ax)
            pitch = p_hi - p_lo  # one lambda/D per pixel
            center = 0.5 * (p_lo + p_hi) + shift * pitch
            (line,) = _by_gid(ax, gid)
            xs, ys = (np.asarray(v, dtype=float) for v in line.get_data())
            height = xs - xs.min()
            shape = _airy_closed_form((ys - center) / pitch)
            np.testing.assert_allclose(height / height.max(), shape, atol=2e-3)
            in_p = (ys >= p_lo) & (ys <= p_hi)
            if shift == 0.0:
                assert ys[np.argmax(xs)] == pytest.approx(center, abs=0.02 * pitch)
            else:
                # The neighbor's image reaches into p: at the shared edge,
                # half a lambda/D from its center, the Airy intensity is
                # [2 J1(pi/2) / (pi/2)]^2, about 0.52 of its peak.
                edge = float(_airy_closed_form(0.5))
                assert 0.51 < edge < 0.53
                # Inside p the drawn tail is highest at that shared edge,
                # within one sample of it, and close to the edge value.
                step = ys[1] - ys[0]
                k = np.argmax(np.where(in_p, height, -np.inf))
                assert ys[k] - p_lo <= step + 1e-12
                assert height[k] / height.max() == pytest.approx(edge, abs=0.03)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_where_ends_on_the_panel_the_wavelength_figure_carries(layout):
    where = _build("d03-collection-where", layout)
    carried = _build("d03-collection-wavelength", layout)
    try:
        _, _, face = _rows(where)
        left = carried.get_axes()[0]
        # Both panels name the same response; only the title prefix differs.
        assert "$P_p" in face.get_title(loc="left")
        assert "$P_p" in left.get_title(loc="left")
        (a,) = face.get_images()
        (b,) = left.get_images()
        np.testing.assert_array_equal(a.get_array(), b.get_array())
        assert a.get_extent() == b.get_extent()
        assert (a.norm.vmin, a.norm.vmax) == (b.norm.vmin, b.norm.vmax)
        assert a.get_cmap().name == b.get_cmap().name
        # Pixel p is outlined at the same place in both.
        outline = [
            (r.get_x(), r.get_y(), r.get_width())
            for ax in (face, left)
            for r in ax.patches
            if isinstance(r, Rectangle)
            and r.get_facecolor()[3] == 0.0
            and r.get_width() == 1.0
        ]
        assert outline == [(-0.5, -0.5, 1.0)] * 2
        # The piece is the first of the relay: nothing on it is carried.
        texts = [t.get_text() for t in where.findobj(Text)]
        assert not any("previous" in t for t in texts)
    finally:
        plt.close(where)
        plt.close(carried)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_where_labels_carry_no_stroked_halo(layout, mode):
    fig = _build("d03-collection-where", layout, mode)
    try:
        stroked = [
            t.get_text()
            for t in fig.findobj(Text)
            if t.get_text().strip() and t.get_path_effects()
        ]
        assert stroked == []
    finally:
        plt.close(fig)
