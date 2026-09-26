"""Numbers and geometry printed on the dust-geometry explainer (d06).

Every expected value here is computed by hand from the chapter's
definitions (half-ray from the observer, scattering angle between the
incident propagation direction and the direction toward the observer,
illumination angle its supplement, near half toward the observer), not from
the plotting library that draws it. The figure checks read the printed text
and marker positions back from the built figure.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from explainers import _common as ex
from explainers import _export as exporter
from explainers import d06_dust_geometry as d06


def _need_viz():
    # The figure tests need the skyscapes viz views; the hand geometry above
    # them does not.
    pytest.importorskip("skyscapes.viz")


def _hand_zodi():
    """Scattering and elongation for the local-zodi grain, by hand."""
    beta = math.radians(30.0)
    dlon = math.radians(135.0)
    # Look direction: Sun on -x from the observer at (1, 0, 0).
    n = (
        -math.cos(beta) * math.cos(dlon),
        math.cos(beta) * math.sin(dlon),
        math.sin(beta),
    )
    grain = (1.0 + 1.0 * n[0], 1.0 * n[1], 1.0 * n[2])
    r = math.sqrt(sum(g * g for g in grain))
    cos_theta = -sum(g * k for g, k in zip(grain, n, strict=True)) / r
    theta = math.degrees(math.acos(cos_theta))
    elong = math.degrees(math.acos(-n[0]))
    return theta, elong


def test_parameters_are_the_captioned_ones():
    assert d06.ZODI["ecliptic_lat_deg"] == 30.0
    assert d06.ZODI["solar_lon_deg"] == 135.0
    assert d06.ZODI["observer_distance_AU"] == 1.0
    assert d06.ZODI["grain_distance_AU"] == 1.0
    assert d06.INCL_DEG == 60.0
    assert d06.DISK["dist_pc"] == 10.0
    assert d06.GRAIN_RADIUS_AU == 3.0
    assert (d06.DISK["rmin_AU"], d06.DISK["rmax_AU"]) == (1.5, 5.0)
    assert d06.DISK["g_HG"] == 0.4
    assert d06.DISK["wavelength_nm"] == 550.0


def test_zodi_grain_angles_by_hand():
    theta, elong = _hand_zodi()
    assert round(theta) == 154
    assert round(180.0 - theta) == 26
    assert round(elong) == 128
    nums = d06.zodi_numbers()
    assert nums["theta"] == pytest.approx(theta, abs=1e-12)
    assert nums["elongation"] == pytest.approx(elong, abs=1e-12)


def test_elongation_is_not_the_longitude_difference():
    # The 135 degrees is the ecliptic longitude difference (the projected
    # arc); the 3-D angle to the Sun is cos e = cos(beta) cos(dlon).
    e = math.degrees(
        math.acos(math.cos(math.radians(30.0)) * math.cos(math.radians(135.0)))
    )
    assert round(e) == 128
    assert d06.zodi_elongation_deg() == pytest.approx(e, abs=1e-12)
    assert d06.zodi_elongation_deg() == pytest.approx(_hand_zodi()[1], abs=1e-12)
    cap = d06.TWO_SYSTEMS_CAPTION
    assert r"|\lambda-\lambda_\odot|=135^\circ" in cap
    assert "is $128^\\circ$" in cap
    assert "differs from the Sun" in cap and "ecliptic longitude" in cap


def test_disk_minor_axis_angles_by_hand():
    # A near-half grain on the minor axis sits at r (0, cos i, sin i) in a
    # frame with +z toward the observer; the star-to-grain direction dotted
    # with +z is sin i, so Theta = 90 - i.
    for incl in (0.0, 30.0, 60.0, 80.0):
        i = math.radians(incl)
        k_in = (0.0, math.cos(i), math.sin(i))
        theta = math.degrees(math.acos(k_in[2]))
        assert theta == pytest.approx(90.0 - incl, abs=1e-9)
        assert d06.disk_minor_axis_theta(incl) == pytest.approx(theta, abs=1e-9)
        assert d06.disk_minor_axis_theta(incl, near=False) == pytest.approx(
            180.0 - theta, abs=1e-9
        )


def test_library_frame_matches_the_chapter():
    # The chapter's proposed profile: near half toward +z (the observer).
    _need_viz()
    from skyscapes.viz import _geometry

    for incl in (20.0, 60.0, 80.0):
        _, near, _ = _geometry.disk_axes_sky(incl, d06.PA_DEG)
        i = math.radians(incl)
        assert near[2] == pytest.approx(math.sin(i), abs=1e-6)
        assert abs(near[1]) == pytest.approx(math.cos(i), abs=1e-6)
        assert near[0] == pytest.approx(0.0, abs=1e-6)


def _texts(fig):
    out = []
    for a in fig.findobj(lambda o: hasattr(o, "get_text")):
        out.append(a.get_text())
    return out


def _gid(fig, gid):
    return [o for o in fig.findobj() if o.get_gid() == gid]


@pytest.fixture(scope="module")
def doc_still():
    _need_viz()
    with exporter.venue("light", ex.DOC) as cast:
        fig = d06.build_two_systems(ex.DOC, cast)
        yield fig
        plt.close(fig)


def _panel(fig, xlabel_start):
    (ax,) = [a for a in fig.axes if a.get_xlabel().startswith(xlabel_start)]
    return ax


def _in(ax, gid):
    return [o for o in ax.findobj() if o.get_gid() == gid]


def test_printed_zodi_inset(doc_still):
    theta, _ = _hand_zodi()
    ax = _panel(doc_still, "$x$, Sun to telescope")
    (value,) = _in(ax, "inset/scattering_angle/value")
    assert f"= {round(theta)}" in value.get_text()
    (alpha,) = _in(ax, "inset/illumination_angle/value")
    assert f"= {round(180.0 - theta)}" in alpha.get_text()


def test_drawn_sense_of_the_solar_longitude(doc_still):
    # From the ecliptic north pole with the observer on +x, longitude grows
    # counterclockwise and the Sun is at 180 degrees. Read the drawn sightline
    # back and compute its lambda - lambda_sun by hand.
    ax = _panel(doc_still, "$x$, Sun to telescope")
    (line,) = _in(ax, "sightline")
    x, y = line.get_data()
    lam = math.degrees(math.atan2(y[1] - y[0], x[1] - x[0]))
    dlam = (lam - 180.0 + 180.0) % 360.0 - 180.0
    assert dlam == pytest.approx(-135.0, abs=1e-9)
    assert d06.drawn_solar_lon_deg() == pytest.approx(dlam, abs=1e-9)
    (label,) = _in(ax, "look_angle/label")
    assert "|" in label.get_text() and "= 135" in label.get_text()
    texts = _texts(doc_still)
    assert any("= -135" in t and "symmetric" in t for t in texts)


def test_printed_zodi_notes(doc_still):
    texts = _texts(doc_still)
    assert any("solar elongation is 128" in t for t in texts)
    assert any(t == "Earth orbit, 1 AU" for t in texts)
    assert any("position angle" in t and "orientation" in t for t in texts)


def test_dust_path_is_labeled(doc_still):
    # The leader of the dust-path label ends on the highlighted chord.
    ax = _panel(doc_still, "$z$, toward the telescope")
    (label,) = _in(ax, "d06/path_label")
    assert "path through" in label.get_text() and "dust layer" in label.get_text()
    (path,) = _in(ax, "sightline/path")
    z = np.asarray(path.get_xdata(), float)[:2]
    u = np.asarray(path.get_ydata(), float)[0]
    assert min(z) < label.xy[0] < max(z)
    assert label.xy[1] == pytest.approx(u)


def test_printed_disk_angles(doc_still):
    ax = _panel(doc_still, "$z$, toward the telescope")
    (value,) = _in(ax, "inset/scattering_angle/value")
    (alpha,) = _in(ax, "inset/illumination_angle/value")
    assert "Theta" in value.get_text() and "= 30" in value.get_text()
    assert "alpha" in alpha.get_text() and "= 150" in alpha.get_text()
    (incl,) = _in(ax, "inclination/label")
    assert "= 60" in incl.get_text()
    assert any(t == "telescope,\n10 pc away" for t in _texts(doc_still))


def test_side_view_grain_position(doc_still):
    # Near-half midplane grain 3 AU out on the minor axis at i = 60 degrees:
    # (z, u) = 3 (sin 60, cos 60) = (2.598, 1.5) AU.
    ax = _panel(doc_still, "$z$, toward the telescope")
    (grain,) = _in(ax, "grain")
    z, u = grain.get_offsets()[0]
    assert z == pytest.approx(3.0 * math.sqrt(3.0) / 2.0, abs=1e-6)
    assert u == pytest.approx(1.5, abs=1e-6)


def test_image_axes_name_no_sky_basis(doc_still):
    ax = _panel(doc_still, "along the line of nodes")
    assert (
        ax.get_ylabel().replace("\n", " ").startswith("along the projected minor axis")
    )
    labels = " ".join(a.get_xlabel() + a.get_ylabel() for a in doc_still.axes)
    assert "RA" not in labels and "Dec" not in labels
    # The sign along the line of nodes is not used: the image is symmetric.
    image = d06.still_image()
    assert np.allclose(image, image[:, ::-1], rtol=1e-6, atol=1e-12)


def test_sightline_marker_on_the_sky(doc_still):
    # 3 AU on the minor axis, foreshortened by cos 60, at 10 pc: 1 AU at
    # 1 pc subtends 1 arcsec, so the offset is 3 * 0.5 / 10 = 0.15 arcsec.
    ax = _panel(doc_still, "along the line of nodes")
    (marker,) = _in(ax, "sightline")
    x, y = marker.get_offsets()[0]
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(0.15, abs=1e-6)


def test_phase_function_is_normalized():
    # Integral over 4 pi sr of the HG form is one; p(0)/p(180) for g = 0.4 is
    # ((1 + g) / (1 - g))^3 = (1.4 / 0.6)^3.
    theta = np.linspace(0.0, 180.0, 20001)
    mu = np.cos(np.radians(theta))
    p = d06.hg_phase(theta, 0.4)
    total = 2.0 * math.pi * np.trapezoid(p, -mu)
    assert total == pytest.approx(1.0, rel=1e-6)
    ratio = d06.hg_phase(0.0, 0.4) / d06.hg_phase(180.0, 0.4)
    assert ratio == pytest.approx((1.4 / 0.6) ** 3, rel=1e-12)


def test_sweep_caption_phase_values_by_hand():
    # HG at g = 0.4: p = (1 - g^2) / (4 pi (1 + g^2 - 2 g cos Theta)^1.5).
    # Face-on both minor-axis grains sit at 90 degrees; at i = 80 the near
    # grain is at 10 and the far grain at 170 degrees.
    g = 0.4

    def hg(theta_deg):
        mu = math.cos(math.radians(theta_deg))
        return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * mu) ** 1.5)

    assert f"{hg(90.0):.3f}" == "0.054"
    assert f"{hg(10.0):.2f}" == "0.29"
    assert f"{hg(170.0):.3f}" == "0.025"
    cap = d06.SWEEP_CAPTION
    assert "0.054 to 0.29" in cap and "0.054 to 0.025" in cap
    assert "dims the far half" in cap and "brighten the image" not in cap
    assert d06.hg_phase(10.0, g) == pytest.approx(hg(10.0), rel=1e-12)


def _overlaps(fig, artists):
    """Pairs of the named texts whose rendered boxes intersect.

    Only the glyph boxes count (``Text.get_window_extent``), not an
    annotation's leader line.
    """
    renderer = fig.canvas.get_renderer()
    boxes = {k: Text.get_window_extent(a, renderer) for k, a in artists.items()}
    keys = sorted(boxes)
    return [
        (a, b)
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
        if boxes[a].overlaps(boxes[b])
    ]


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_sweep_labels_stay_clear(layout):
    # The contract from review: at every checked inclination, no label of
    # the side view or the view glyph sits on another.
    _need_viz()
    with exporter.venue("dark", layout) as cast:
        scene = d06.build_sweep(layout, cast)
        try:
            incls = np.array([f["incl"] for f in scene.frames])
            side = _panel(scene.fig, "$z$, toward the telescope")
            (view,) = [a for a in scene.fig.axes if _in(a, "d06/view_telescope")]
            for target in (0.0, 10.0, 20.0, 30.0, 60.0, 69.0, 80.0):
                k = int(np.argmin(abs(incls - target)))
                scene.draw(scene.fig, scene.frames[k])
                scene.fig.canvas.draw()
                named = {}
                for gid in (
                    "label/sky_plane",
                    "label/near_side",
                    "label/far_side",
                    "inclination/label",
                    "scattering_angle/label",
                    "d06/telescope_label",
                    "d06/path_label",
                ):
                    (art,) = _in(side, gid)
                    if art.get_text():
                        named[gid] = art
                assert _overlaps(scene.fig, named) == [], (target, named.keys())
                glyph = {
                    t.get_text(): t
                    for t in view.texts
                    if t.get_text() and t.get_visible()
                }
                assert _overlaps(scene.fig, glyph) == [], target
        finally:
            plt.close(scene.fig)


def _stroked(fig):
    return [
        t.get_text()
        for t in fig.findobj(Text)
        if any(
            type(e).__name__ in ("withStroke", "Stroke") for e in t.get_path_effects()
        )
    ]


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_no_stroked_text_halos(layout, mode):
    # No label carries a stroke path effect, in the still or in any drawn
    # frame of the animation; labels over marks sit on a plain box instead.
    _need_viz()
    with exporter.venue(mode, layout) as cast:
        fig = d06.build_two_systems(layout, cast)
        try:
            assert _stroked(fig) == []
        finally:
            plt.close(fig)
        scene = d06.build_sweep(layout, cast)
        try:
            for frame in (scene.frames[0], scene.frames[-1], scene.frames[5]):
                scene.draw(scene.fig, frame)
                assert _stroked(scene.fig) == []
        finally:
            plt.close(scene.fig)


def test_near_half_is_brighter():
    image = d06.still_image()
    assert np.all(np.isfinite(image))
    assert image.min() >= 0.0
    assert image.max() == pytest.approx(1.0)
    c = image.shape[0] // 2
    near = image[c + 5 :, c].max()  # +y rows (origin lower)
    far = image[: c - 5, c].max()
    assert near > 2.0 * far


def test_sweep_frames_share_one_scale():
    for layout in (ex.DOC, ex.SLIDE):
        incls = d06.sweep_incls(layout)
        assert incls[0] == 0.0
        assert incls[-1] == 80.0
        assert np.all(np.diff(incls) > 0)
        if layout.frame_budget is not None:
            assert len(incls) <= layout.frame_budget
    images = d06.disk_images(d06.sweep_incls(ex.DOC))
    assert max(float(im.max()) for im in images) == pytest.approx(1.0)
    assert all(float(im.min()) >= 0.0 for im in images)


def test_animation_draws_any_frame():
    _need_viz()
    with exporter.venue("dark", ex.DOC) as cast:
        scene = d06.build_sweep(ex.DOC, cast)
        try:
            for frame in (scene.frames[-1], scene.frames[0], scene.frames[10]):
                scene.draw(scene.fig, frame)
            (value,) = _gid(scene.fig, "inset/scattering_angle/value")
            incl = scene.frames[10]["incl"]
            assert f"= {90.0 - incl:.0f}" in value.get_text()
        finally:
            plt.close(scene.fig)
