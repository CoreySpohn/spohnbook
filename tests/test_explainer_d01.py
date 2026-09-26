"""Contracts of the observation-overview explainer figures.

The overview is conceptual: it prints no number, so no brightness ratio can
slip onto it. Its sky panels share one orientation (east to the left, north
up), its raw frame is drawn as raw pixels, the planet sits at the same sky
position in every panel, and the highlighted variants dim everything outside
the chosen region. The one numerical input a reader could check, the orbital
period behind the drawn orbit, is recomputed from Kepler's third law.
"""

import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage
from matplotlib.text import Text

DOCS = Path(__file__).resolve().parents[1] / "docs"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

pytest.importorskip("skyscapes.viz")

from explainers import _common as ex  # noqa: E402
from explainers import _export as exporter  # noqa: E402
from explainers import d01_observation_overview as d01  # noqa: E402


def _build(builder, mode="light", layout=ex.DOC):
    with exporter.venue(mode, layout) as cast:
        return builder(layout, cast), cast


def _texts(fig):
    return [
        t.get_text()
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()
    ]


@pytest.fixture(scope="module")
def overview():
    fig, cast = _build(d01.build_overview)
    yield fig, cast
    plt.close(fig)


def test_overview_prints_no_number(overview):
    fig, _ = overview
    texts = _texts(fig)
    assert texts, "the overview carries direct labels"
    with_digits = [t for t in texts if re.search(r"\d", t)]
    assert with_digits == []


@pytest.mark.parametrize(
    "builder",
    [
        d01.build_relay_1,
        d01.build_relay_2,
        d01.build_relay_3,
        d01.build_relay_4,
        d01.build_relay_5,
    ],
)
def test_relay_stills_print_no_number(builder):
    fig, _ = _build(builder, "dark", ex.SLIDE)
    try:
        assert [t for t in _texts(fig) if re.search(r"\d", t)] == []
    finally:
        plt.close(fig)


def test_status_is_on_the_graphic(overview):
    fig, _ = overview
    joined = " | ".join(_texts(fig))
    assert "schematic" in joined
    assert "not to scale" in joined
    assert "synthetic" in joined
    assert "not a flight design" in joined
    assert "posterior draws" in joined


def test_every_image_is_drawn_as_raw_pixels(overview):
    fig, _ = overview
    images = fig.findobj(AxesImage)
    assert len(images) >= 1
    assert {im.get_interpolation() for im in images} == {"nearest"}


def test_display_mapping_puts_east_left_and_north_up():
    # Hand arithmetic: 25 pixels, center 12, 0.04 arcsec per pixel.
    assert d01.FRAME["n_pix"] == 25
    assert d01.FRAME["pixel_arcsec"] == pytest.approx(0.04)
    row, col = d01.sky_to_display_pixel(np.array([0.04, 0.0]))
    assert (row, col) == pytest.approx((12.0, 11.0))
    row, col = d01.sky_to_display_pixel(np.array([0.0, 0.08]))
    assert (row, col) == pytest.approx((14.0, 12.0))


def test_orbit_panel_has_east_to_the_left(overview):
    fig, _ = overview
    orbit_axes = [a for a in fig.findobj(plt.Axes) if a.get_xlim()[0] > a.get_xlim()[1]]
    assert orbit_axes, "a sky panel draws the RA offset increasing to the left"


def test_frame_places_the_planet_at_its_sky_position():
    comps = d01.frame_components()
    peak = np.unravel_index(np.argmax(comps["planet"]), comps["planet"].shape)
    xy = d01.sky_positions_arcsec(d01.T_EPOCH_JD)[0]
    row, col = d01.sky_to_display_pixel(xy)
    assert peak == (round(row), round(col))
    # The planet sits well outside the leakage core, on the frame.
    center = 0.5 * (d01.FRAME["n_pix"] - 1)
    assert math.hypot(row - center, col - center) > 4.0
    assert 0 <= round(row) < d01.FRAME["n_pix"]
    assert 0 <= round(col) < d01.FRAME["n_pix"]


def test_drawn_epoch_is_a_visit_and_matches_the_orbit_panel():
    assert d01.T_EPOCH_JD in d01.T_VISITS_JD
    _, measured, sigma = d01.orbit_draws()
    k = d01.T_VISITS_JD.index(d01.T_EPOCH_JD)
    truth = d01.sky_positions_arcsec(d01.T_EPOCH_JD)[0]
    assert np.all(np.abs(measured[k] - truth) < 4.0 * sigma)


def test_drawn_planet_is_on_the_far_side():
    # Far side: displaced away from the observer (z < 0), so the side view
    # shows it left of the sky plane and it is lit gibbous.
    assert d01.planet_xyz_au(d01.T_EPOCH_JD)[2] < 0.0


def test_orbit_lies_in_the_disk_midplane():
    # In the chapter basis (X, Y, Z) = (north, east, toward observer), a
    # plane containing the line of nodes (at the position angle from north
    # toward east) and tilted by the inclination from the sky plane has one
    # of two normals; the orbit's normal must be one of them.
    incl = math.radians(d01.SCENE["disk_incl_deg"])
    pa = math.radians(d01.SCENE["disk_pa_deg"])
    candidates = [
        np.array(
            [
                s * math.sin(incl) * math.sin(pa),
                -s * math.sin(incl) * math.cos(pa),
                math.cos(incl),
            ]
        )
        for s in (1.0, -1.0)
    ]
    r1 = d01.planet_xyz_au(0.0)
    r2 = d01.planet_xyz_au(0.25 * d01.period_d())
    normal = np.cross(r1, r2)
    normal /= np.linalg.norm(normal)
    best = max(abs(float(np.dot(normal, c))) for c in candidates)
    assert best == pytest.approx(1.0, abs=1e-4)


def test_period_matches_keplers_third_law():
    # P = a^1.5 years for a solar-mass host, a in AU; a = 3 AU gives
    # 5.196 years, about 1898 days.
    a = d01.SCENE["a_AU"]
    expected_d = a**1.5 * 365.25
    assert d01.period_d() == pytest.approx(expected_d, rel=2e-3)


def test_unknown_highlight_is_rejected():
    with (
        exporter.venue("light", ex.DOC) as cast,
        pytest.raises(ValueError, match="unknown highlight"),
    ):
        d01.build_overview(ex.DOC, cast, highlight="spectra")


def _luminance_distance(color, background):
    return float(np.max(np.abs(np.array(to_rgb(color)) - np.array(to_rgb(background)))))


@pytest.mark.parametrize("key", sorted(d01.HIGHLIGHTS))
def test_highlight_dims_outside_and_keeps_inside(key):
    fig, cast = _build(
        lambda layout, cast: d01.build_overview(layout, cast, highlight=key)
    )
    try:
        texts = {
            t.get_text(): t
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip()
        }
        # A label of a part the highlight keeps stays at full color; a label of
        # a part it drops fades toward the background.
        kept_label = {
            "geometry": "target system, side view",
            "dust": "local zodiacal dust",
            "radiometry": "raw frame, synthetic",
            "optics": "raw frame, synthetic",
            "inference": "records, one per visit",
            "tutorial": "raw frame, synthetic",
        }[key]
        dropped_label = (
            "records, one per visit" if key != "inference" else "local zodiacal dust"
        )
        full = _luminance_distance(texts[kept_label].get_color(), cast.background)
        faded = _luminance_distance(texts[dropped_label].get_color(), cast.background)
        assert faded < 0.5 * full
        if key == "tutorial":
            assert "full strength: this tutorial" in texts
            # The read-out arrow feeds the lit raw frame, so it is lit too.
            readout = _luminance_distance(
                texts["read out"].get_color(), cast.background
            )
            assert readout > 0.5 * full
            # The tutorial line is not styled as a chapter locator.
            assert texts["full strength: this tutorial"].get_bbox_patch() is None
        else:
            assert "this chapter" in texts
            assert texts[d01.CHAPTERS[key]].get_fontweight() == "bold"
            others = [n for k, n in d01.CHAPTERS.items() if k != key]
            assert all(texts[n].get_fontweight() == "normal" for n in others)
    finally:
        plt.close(fig)


def test_no_label_carries_a_stroked_halo(overview):
    # Stroked halos turn every glyph into an outline path in the vector
    # exports; labels use backing boxes instead.
    fig, _ = overview
    assert [t.get_text() for t in fig.findobj(Text) if t.get_path_effects()] == []


def test_records_card_uses_the_records_chapter_symbols(overview):
    # The records chapter names the reporting event D, the east and north
    # offsets (xi, eta) and their covariance C; the card indexes them by
    # visit k and must not introduce another covariance symbol.
    chapter = (DOCS / "conventions" / "inference-records.md").read_text()
    assert "$D$ a reporting event" in chapter
    assert "$(\\xi,\\eta,f)$" in chapter
    assert "$C_{\\xi f}$" in chapter
    fig, _ = overview
    joined = " | ".join(_texts(fig))
    for symbol in ("$D_k$", "$C_k$", "$(\\xi_k,\\eta_k)$", "$t_k$"):
        assert symbol in joined
    assert "K_k" not in joined


def test_chapter_locators_are_short_and_on_the_map(overview):
    fig, _ = overview
    texts = _texts(fig)
    for name in d01.CHAPTERS.values():
        assert name in texts
        assert len(name.split()) == 1
    assert set(d01.CHAPTERS) == set(d01.HIGHLIGHTS) - {"tutorial"}


def test_four_visits_feed_the_orbit_panel():
    _, measured, sigma = d01.orbit_draws()
    assert len(d01.T_VISITS_JD) == 4
    assert measured.shape == (4, 2)
    assert sigma == d01.DRAWS["astrometry_sigma_arcsec"]


def _distance_to_polyline(point, track):
    # Closest approach of a point to a closed polyline, by hand geometry.
    a = track
    b = np.roll(track, -1, axis=0)
    ab = b - a
    t = np.clip(np.sum((point - a) * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return float(np.min(np.hypot(*(point - closest).T)))


def test_posterior_draws_pass_through_the_measured_positions():
    # Posterior-like: every drawn track passes within a few sigma of every
    # measured position (checked geometrically, independent of the
    # propagator), and the weights behind them are not degenerate.
    tracks, measured, sigma = d01.orbit_draws()
    assert tracks.shape[0] == d01.DRAWS["n"]
    for track in tracks:
        for point in measured:
            assert _distance_to_polyline(point, track) < 4.0 * sigma
    _, _, _, ess = d01.posterior_draws()
    # Enough independent weight behind the resampled draws.
    assert ess >= 2 * d01.DRAWS["n"]


def test_posterior_draws_gather_on_the_data_and_spread_elsewhere():
    # A coherent cloud: the draws' scatter at the measured arc is small
    # compared with their scatter on the unobserved half of the orbit.
    tracks, measured, sigma = d01.orbit_draws()
    near = []
    far = []
    center = measured.mean(axis=0)
    opposite = -center
    # Spread across draws of the track point nearest to each direction.
    angles = np.arctan2(tracks[..., 1], tracks[..., 0])
    for direction, bucket in (
        (math.atan2(center[1], center[0]), near),
        (math.atan2(opposite[1], opposite[0]), far),
    ):
        for track, ang in zip(tracks, angles, strict=True):
            k = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - direction))))))
            bucket.append(track[k])
    near_spread = float(np.max(np.std(np.array(near), axis=0)))
    far_spread = float(np.max(np.std(np.array(far), axis=0)))
    assert near_spread < 2.0 * sigma
    assert far_spread > 1.5 * near_spread


def test_posterior_draws_are_deterministic():
    first = d01.posterior_draws.__wrapped__()
    second = d01.posterior_draws.__wrapped__()
    for key in first[0]:
        np.testing.assert_array_equal(first[0][key], second[0][key])


# The chapter's observer profile, recomputed by hand: Kepler's equation by
# Newton iteration and r = R_Z(Omega) R_X(i) R_Z(omega_p) [r cos nu, r sin nu, 0]
# in the basis (X, Y, Z) = (north, east, toward observer); offsets
# (xi, eta) = (Y, X) / d.
AU_PER_PC_ARCSEC = 1.0  # 1 AU at 1 pc subtends 1 arcsec, by the parsec's definition


def _chapter_offsets_arcsec(t_d):
    sc = d01.SCENE
    a, e = sc["a_AU"], sc["e"]
    big_w, inc, w = (math.radians(sc[k]) for k in ("W_deg", "i_deg", "w_deg"))
    n = 2.0 * math.pi / d01.period_d()
    mean = math.radians(sc["M0_deg"]) + n * np.asarray(t_d, float)
    ecc = mean.copy()
    for _ in range(50):
        ecc = ecc - (ecc - e * np.sin(ecc) - mean) / (1.0 - e * np.cos(ecc))
    x_p = a * (np.cos(ecc) - e)
    y_p = a * math.sqrt(1.0 - e * e) * np.sin(ecc)

    def rz(t):
        return np.array(
            [
                [math.cos(t), -math.sin(t), 0.0],
                [math.sin(t), math.cos(t), 0.0],
                [0, 0, 1],
            ]
        )

    def rx(t):
        return np.array(
            [
                [1, 0, 0],
                [0.0, math.cos(t), -math.sin(t)],
                [0.0, math.sin(t), math.cos(t)],
            ]
        )

    rot = rz(big_w) @ rx(inc) @ rz(w)
    xyz = rot @ np.stack([x_p, y_p, np.zeros_like(x_p)])
    scale = AU_PER_PC_ARCSEC / sc["dist_pc"]
    return np.stack([xyz[1] * scale, xyz[0] * scale], axis=-1), xyz


def test_sky_positions_follow_the_chapter_profile():
    t = np.linspace(0.0, d01.period_d(), 37)
    expected, xyz = _chapter_offsets_arcsec(t)
    np.testing.assert_allclose(d01.sky_positions_arcsec(t), expected, atol=2e-4)
    np.testing.assert_allclose(d01.planet_xyz_au(t[5]), xyz[:, 5], atol=2e-3)


def test_ascending_node_lies_at_omega_east_of_north():
    # The ascending node is the sky-plane crossing with dZ/dt > 0; in the
    # chapter profile its position angle, from north toward east, is Omega.
    # Start before t = 0: the scene's epoch puts the planet on the node.
    t = np.linspace(-0.25 * d01.period_d(), 0.75 * d01.period_d(), 20001)
    _, xyz = _chapter_offsets_arcsec(t)
    k = int(np.flatnonzero((xyz[2, :-1] < 0.0) & (xyz[2, 1:] >= 0.0))[0])
    got = d01.sky_positions_arcsec(t[k : k + 1])[0]
    pa = math.degrees(math.atan2(got[0], got[1])) % 360.0
    assert pa == pytest.approx(d01.SCENE["W_deg"], abs=0.2)
    assert "north" in d01.PROFILE and "east" in d01.PROFILE


def test_disk_image_major_axis_lies_along_the_node_line():
    # Second moments of the chapter-profile disk image, indexed [north, east]:
    # the projected midplane's long axis lies at the node angle east of north.
    img = d01.disk_image()
    n = img.shape[0]
    c = 0.5 * (n - 1)
    north, east = np.indices(img.shape) - c
    w = img / img.sum()
    cov = np.array(
        [
            [np.sum(w * north * north), np.sum(w * north * east)],
            [np.sum(w * north * east), np.sum(w * east * east)],
        ]
    )
    vec = np.linalg.eigh(cov)[1][:, 1]
    pa = math.degrees(math.atan2(vec[1], vec[0])) % 180.0
    assert pa == pytest.approx(d01.SCENE["disk_pa_deg"], abs=0.5)


def _axes_texts(ax):
    return {t.get_text() for t in ax.findobj(Text) if t.get_visible()}


def test_compass_on_sky_panels_and_not_on_raw_frames(overview):
    fig, _ = overview
    frames = [
        a
        for a in fig.findobj(plt.Axes)
        if any(im.get_cmap().name == ex.image_cmap("readouts").name for im in a.images)
    ]
    assert frames
    for ax in frames:
        assert not {"E", "N"} & _axes_texts(ax)
    skies = [a for a in fig.findobj(plt.Axes) if a.get_xlim()[0] > a.get_xlim()[1]]
    assert skies
    for ax in skies:
        assert {"E", "N"} <= _axes_texts(ax)
    for builder, sky_side, frame_side in (
        (d01.build_relay_1, 1, None),
        (d01.build_relay_2, 0, None),
        (d01.build_relay_3, None, 1),
        (d01.build_relay_4, None, 0),
        (d01.build_relay_5, 1, None),
    ):
        relay, _ = _build(builder)
        try:
            axes = relay.axes
            if sky_side is not None:
                assert {"E", "N"} <= _axes_texts(axes[sky_side])
            if frame_side is not None:
                assert not {"E", "N"} & _axes_texts(axes[frame_side])
        finally:
            plt.close(relay)


def test_relay_has_five_steps_and_rings_the_current_visit():
    names = [spec.slug for spec in d01.FIGURES if spec.slug.startswith("d01-relay-")]
    assert names == [f"d01-relay-{k}" for k in range(1, 6)]
    captions = {spec.slug: spec.caption for spec in d01.FIGURES}
    assert "one standard deviation" in captions["d01-relay-5"]
    assert "ringed" in captions["d01-relay-5"]
    fig, _ = _build(d01.build_relay_5)
    try:
        assert "current visit" in _texts(fig)
    finally:
        plt.close(fig)


def test_caption_states_the_prior_the_draws_use():
    # Every number the caption quotes about the prior and the sampler.
    assert d01.DRAWS["prior_e_max"] == 0.6
    assert d01.DRAWS["prior_a_frac"] == 0.15
    assert d01.DRAWS["prior_i_deg"] == 15.0
    assert d01.DRAWS["prior_W_deg"] == 15.0
    assert d01.DRAWS["n_prior"] == 6_000_000
    assert d01.DRAWS["n"] == 30
    text = d01._PRIOR_TEXT
    for phrase in (
        "uniform on [0, 0.6]",
        "uniform on the circle",
        "15 percent, 15 degrees and 15",
        "six million",
        "thirty draws",
        "systematic resampling",
    ):
        assert phrase in text
