"""Numerical anchors of the angular-grids explainer.

Every number printed on the strip or in its caption is recomputed here by
hand arithmetic or a closed form: the first dark ring of a clear disk at
j_1,1 / pi lambda/D (Hecht 2017, Sec. 10.2.5, eq. 10.58), which is
lambda / lambda_ref times that in u_ref; the source at u_ref = alpha D /
lambda_ref on the fixed grid and alpha D / lambda on the native grid; and
the angular pixel du lambda / D in milliarcseconds. The propagated images
are then checked against those anchors: the dashed ring sits on an
intensity minimum and the source marker on the image centroid.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("physicaloptix")
pytest.importorskip("eyepiece")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explainers import _common as ex
from explainers import _export as exporter
from explainers import d13_angular_grids as d13
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from matplotlib.patches import Circle
from matplotlib.text import Text

P = d13.PARAMS
LAMS = P["wavelengths_nm"]
# First zero of J1 (Abramowitz and Stegun Table 9.5), and radians to mas.
J1_FIRST_ZERO = 3.8317059702
MAS_PER_RAD = 180.0 / math.pi * 3600.0 * 1000.0


def source_alpha_rad():
    return P["source_u_ref"] * P["reference_wavelength_nm"] * 1e-9 / P["diameter_m"]


def hand_source_u(row, lam):
    if row == "fixed":
        return P["source_u_ref"]
    return source_alpha_rad() * P["diameter_m"] / (lam * 1e-9)


def hand_ring(row, lam):
    zero = J1_FIRST_ZERO / math.pi
    return zero if row == "native" else zero * lam / P["reference_wavelength_nm"]


def hand_pixel_mas(row, lam):
    ref = P["reference_wavelength_nm"] if row == "fixed" else lam
    return P["du"] * ref * 1e-9 / P["diameter_m"] * MAS_PER_RAD


def distances(row, k):
    u = d13.compute()["focal_coords"]
    uu, vv = np.meshgrid(u, u)
    return np.hypot(uu - hand_source_u(row, LAMS[k]), vv), uu, vv


@pytest.fixture(scope="module")
def rendered():
    with exporter.venue("light", ex.DOC) as cast:
        fig = d13.build_strip(ex.DOC, cast)
    yield fig
    plt.close(fig)


def test_ring_radius_is_the_first_airy_zero_scaled_to_each_grid():
    assert d13.first_zero_lod() == pytest.approx(1.2196699, abs=1e-6)
    expected = {"fixed": (1.22, 1.83, 2.44), "native": (1.22, 1.22, 1.22)}
    for row, printed in expected.items():
        for lam, text in zip(LAMS, printed, strict=True):
            assert d13.ring_radius(row, lam) == pytest.approx(
                hand_ring(row, lam), rel=1e-9
            )
            assert f"{hand_ring(row, lam):.2f}" == f"{text:.2f}"


def test_first_zero_of_an_independent_disk_transform_is_at_1_22():
    # A plain numpy transform of a sharp disk, azimuthally averaged: the
    # first minimum of the intensity lies at j_1,1 / pi lambda/D.
    n = 256
    x = (np.arange(n) - n / 2 + 0.5) / n
    xx, yy = np.meshgrid(x, x)
    disk = (np.hypot(xx, yy) <= 0.5).astype(float)
    radii = np.linspace(1.0, 1.5, 51)
    angles = np.linspace(0.0, math.pi / 2, 7)
    profile = []
    for r in radii:
        values = []
        for a in angles:
            kx, ky = r * math.cos(a), r * math.sin(a)
            values.append(
                abs(np.sum(disk * np.exp(-2j * math.pi * (kx * xx + ky * yy)))) ** 2
            )
        profile.append(np.mean(values))
    assert radii[int(np.argmin(profile))] == pytest.approx(1.22, abs=0.015)


def test_dashed_ring_sits_on_the_image_minimum():
    # In every panel the mean power in a band at the ring radius is below
    # the band inside it (the core) and the band outside it (the first
    # bright ring, at 1.635 lambda/D, 1.34 ring radii).
    power = d13.compute()["power"]
    for row in d13.ROW_KEYS:
        for k, lam in enumerate(LAMS):
            dist, _, _ = distances(row, k)
            ring = hand_ring(row, lam)
            image = power[row][k]
            inner = image[(dist > 0.55 * ring) & (dist < 0.8 * ring)].mean()
            at = image[(dist > 0.9 * ring) & (dist < 1.1 * ring)].mean()
            outer = image[(dist > 1.25 * ring) & (dist < 1.45 * ring)].mean()
            assert at < inner
            assert at < outer, (row, lam)


def test_source_positions_by_hand_and_in_the_image():
    power = d13.compute()["power"]
    expected = {"fixed": (4.0, 4.0, 4.0), "native": (4.0, 8.0 / 3.0, 2.0)}
    for row, positions in expected.items():
        for k, lam in enumerate(LAMS):
            assert hand_source_u(row, lam) == pytest.approx(positions[k], rel=1e-12)
            assert d13.source_u(row, lam) == pytest.approx(positions[k], rel=1e-9)
            # The power-weighted centroid of the core sits on the marker.
            dist, uu, vv = distances(row, k)
            core = dist < 0.6 * hand_ring(row, lam)
            weights = power[row][k] * core
            cx = float((weights * uu).sum() / weights.sum())
            cy = float((weights * vv).sum() / weights.sum())
            assert cx == pytest.approx(positions[k], abs=0.06), (row, lam)
            assert cy == pytest.approx(0.0, abs=1e-6)


def test_positive_angle_lands_on_positive_x():
    # The coherent profile: a positive x phase ramp moves the response
    # toward +x, so the brightest pixel of every panel has x > 0.
    u = d13.compute()["focal_coords"]
    for row in d13.ROW_KEYS:
        for image in d13.compute()["power"][row]:
            _, c = np.unravel_index(np.argmax(image), image.shape)
            assert u[c] > 1.0


def test_pixel_angular_spacing_in_milliarcseconds():
    printed = {"fixed": ("8.6", "8.6", "8.6"), "native": ("8.6", "12.9", "17.2")}
    for row, texts in printed.items():
        for lam, text in zip(LAMS, texts, strict=True):
            assert d13.pixel_mas(row, lam) == pytest.approx(
                hand_pixel_mas(row, lam), rel=1e-9
            )
            assert f"{hand_pixel_mas(row, lam):.1f}" == text
    assert source_alpha_rad() * MAS_PER_RAD == pytest.approx(68.755, abs=1e-3)
    assert d13.source_alpha_arcsec() * 1e3 == pytest.approx(
        source_alpha_rad() * MAS_PER_RAD, rel=1e-9
    )


def test_the_grids_coincide_at_the_reference_wavelength():
    # At lambda = lambda_ref the stored coordinates are identical, so the
    # two rows must hold the same array in the first column.
    power = d13.compute()["power"]
    assert LAMS[0] == P["reference_wavelength_nm"]
    assert np.allclose(power["fixed"][0], power["native"][0], rtol=1e-10, atol=0)


def test_each_panel_holds_nearly_all_the_source_power():
    # The fixed-row dimming is spreading over more fixed pixels, not loss:
    # every panel still holds more than 95 percent of the unit pupil power.
    power = d13.compute()["power"]
    for row in d13.ROW_KEYS:
        for image in power[row]:
            assert 0.95 < image.sum() <= 1.0 + 1e-9


def test_panel_labels_and_marks_carry_the_hand_values(rendered):
    texts = {t.get_gid(): t.get_text() for t in rendered.findobj(Text) if t.get_gid()}
    marks = {
        a.get_gid(): a
        for a in rendered.findobj()
        if (a.get_gid() or "").startswith(("ring-", "source-"))
    }
    for row in d13.ROW_KEYS:
        for k, lam in enumerate(LAMS):
            label = texts[f"label-{row}-{k}"]
            assert f"pixel {hand_pixel_mas(row, lam):.1f} mas" in label
            assert f"={hand_source_u(row, lam):.2f}$" in label
            unit = {"fixed": r"\lambda_{\rm ref}/D", "native": r"\lambda/D"}[row]
            assert f"dark ring radius ${hand_ring(row, lam):.2f}\\,{unit}$" in label
            assert "$r=" not in label
            ring = marks[f"ring-{row}-{k}"]
            assert isinstance(ring, Circle)
            assert ring.get_radius() == pytest.approx(hand_ring(row, lam), rel=1e-9)
            assert ring.center == pytest.approx((hand_source_u(row, lam), 0.0))
            xs, ys = marks[f"source-{row}-{k}"].get_data()
            assert xs[0] == pytest.approx(hand_source_u(row, lam), rel=1e-9)
            assert ys[0] == 0.0


def test_images_are_raw_pixels_on_one_shared_log_norm(rendered):
    power = d13.compute()["power"]
    images = [a for a in rendered.findobj(AxesImage)]
    assert len(images) == 6
    norms = {id(im.norm) for im in images}
    assert len(norms) == 1
    norm = images[0].norm
    assert isinstance(norm, LogNorm)
    assert (norm.vmin, norm.vmax) == (P["power_floor"], P["power_ceiling"])
    expected = [power[row][k] for row in d13.ROW_KEYS for k in range(3)]
    for im, data in zip(images, expected, strict=True):
        assert im.get_interpolation() == "nearest"
        assert im.origin == "lower"
        drawn = np.asarray(im.get_array(), dtype=float)
        assert np.allclose(drawn, np.clip(data, P["power_floor"], None))


def test_no_text_carries_a_stroked_halo(rendered):
    stroked = [t.get_text() for t in rendered.findobj(Text) if t.get_path_effects()]
    assert stroked == []
    with exporter.venue("dark", ex.SLIDE) as cast:
        fig = d13.build_strip(ex.SLIDE, cast)
    try:
        assert [t for t in fig.findobj(Text) if t.get_path_effects()] == []
    finally:
        plt.close(fig)


def test_caption_numbers_match_the_hand_values():
    caption = d13.STRIP_CAPTION
    assert f"{source_alpha_rad() * MAS_PER_RAD:.1f} mas" in caption
    native = [f"{hand_pixel_mas('native', lam):.1f}" for lam in LAMS]
    assert f"{native[0]}, {native[1]} and {native[2]} mas" in caption
    rings = [f"{hand_ring('fixed', lam):.2f}" for lam in LAMS]
    assert f"{rings[0]}, {rings[1]} and {rings[2]} in u_ref" in caption
    positions = [hand_source_u("native", lam) for lam in LAMS]
    assert f"{positions[0]:g}, {positions[1]:.2f} and {positions[2]:g}" in caption
    assert f"D = {P['diameter_m']:g} m" in caption
    assert "8.6 mas at every wavelength" in caption
