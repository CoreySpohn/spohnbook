"""Independent checks of every anchor the barycenter and RV explainer draws.

Expected values are computed here by hand from the worked signed example
(a circular relative orbit with i = 90 deg, Omega = omega_p = M_0 = 0, so the
chapter's rotation reduces to r = a (cos nu, 0, sin nu) in (north, east,
toward observer)) and from hwoutils constants, not from the module's
helpers, and compared with what the module computes and plots.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hwoutils import constants as const
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from explainers import _common as ex
from explainers import _export as exporter
from explainers import d11_barycenter_rv as d11

Q = d11.MASS_RATIO
F_P = Q / (1.0 + Q)  # Mp / (M* + Mp)
F_S = 1.0 / (1.0 + Q)  # M* / (M* + Mp)


def hand_relative(frac):
    """r / a and v / (n a) for the worked example, written out by hand."""
    nu = 2.0 * math.pi * frac
    r = np.array([math.cos(nu), 0.0, math.sin(nu)])
    v = np.array([-math.sin(nu), 0.0, math.cos(nu)])
    return r, v


def page(vec):
    """Side view from the east: x = toward observer (Z), y = north (X)."""
    return np.array([vec[2], vec[0]])


def _by_gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


# The worked number


def test_earth_sun_k_is_the_chapter_worked_value():
    m_s, m_e, au = const.Msun2kg, const.Mearth2kg, const.AU2m
    n = math.sqrt(const.G_si * (m_s + m_e) / au**3)
    k = m_e / (m_s + m_e) * n * au
    assert round(k, 5) == 0.08946
    assert d11.EARTH_SUN_K == pytest.approx(k, rel=1e-12)
    assert d11.K_TEXT == "+0.08946 m/s"


def test_earth_sun_rv_sign_by_finite_difference_in_m_per_s():
    """The fixture: difference the star's Z position, v_r = -dZ_star/dt > 0."""
    m_s, m_e, au = const.Msun2kg, const.Mearth2kg, const.AU2m
    period = 2.0 * math.pi * math.sqrt(au**3 / (const.G_si * (m_s + m_e)))

    def z_star(t):
        # r_star = -Mp/(M*+Mp) r and Z = a sin(2 pi t / P) at t0 = 0.
        return -m_e / (m_s + m_e) * au * math.sin(2.0 * math.pi * t / period)

    dt = 10.0  # seconds
    v_r = -(z_star(dt) - z_star(-dt)) / (2.0 * dt)
    assert v_r > 0.0
    assert v_r == pytest.approx(0.08946, abs=5e-6)


def test_printed_mass_ratio_matches_hwoutils():
    assert const.Mearth2kg / const.Msun2kg == pytest.approx(3.0e-6, rel=0.01)
    assert "3.0" in d11.Q_TEXT and "10^{-6}" in d11.Q_TEXT


# Reflex physics against hand arithmetic


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.25, 0.4, 0.75, 0.9])
def test_relative_state_is_the_worked_example(frac):
    r, v = d11.relative_state(frac)
    r_hand, v_hand = hand_relative(frac)
    assert np.allclose(r, r_hand, atol=1e-12)
    assert np.allclose(v, v_hand, atol=1e-12)


def test_worked_example_states():
    r0, v0 = hand_relative(0.0)
    assert r0[0] > 0 and abs(r0[2]) < 1e-12  # north of the star, Z = 0
    assert v0[2] > 0  # moving toward the observer
    assert d11.relative_state(0.25)[0][2] == pytest.approx(1.0)  # at +Z
    assert d11.relative_state(0.75)[0][2] == pytest.approx(-1.0)  # at -Z


@pytest.mark.parametrize("frac", [0.0, 0.13, 0.25, 0.6, 0.75])
def test_barycenter_is_the_mass_weighted_origin(frac):
    s = d11.barycentric_states(frac)
    r_hand, _ = hand_relative(frac)
    # M* r_star + Mp r_p = 0, with masses in the drawn ratio.
    assert np.allclose(1.0 * s["r_star"] + Q * s["r_p"], 0.0, atol=1e-12)
    assert np.allclose(1.0 * s["v_star"] + Q * s["v_p"], 0.0, atol=1e-12)
    assert np.allclose(s["r_p"] - s["r_star"], r_hand, atol=1e-12)
    assert np.allclose(s["r_star"], -F_P * r_hand, atol=1e-12)


@pytest.mark.parametrize("frac", np.linspace(0.0, 1.0, 13))
def test_rv_is_minus_zdot_star_and_the_chapter_bracket(frac):
    _, v_hand = hand_relative(frac)
    v_r_over_k = -(-F_P * v_hand[2]) / F_P  # -dZ_star/dt in units of K
    nu = 2.0 * math.pi * frac
    bracket = math.cos(nu + 0.0) + 0.0 * math.cos(0.0)  # e = 0, omega_p = 0
    assert d11.stellar_rv_over_k(frac) == pytest.approx(v_r_over_k, abs=1e-12)
    assert d11.stellar_rv_over_k(frac) == pytest.approx(bracket, abs=1e-12)


def test_rv_at_the_three_states():
    assert d11.stellar_rv_over_k(0.0) == pytest.approx(1.0)
    assert d11.stellar_rv_over_k(0.25) == pytest.approx(0.0, abs=1e-12)
    assert d11.stellar_rv_over_k(0.75) == pytest.approx(0.0, abs=1e-12)


def test_side_view_puts_observer_right_and_north_up():
    assert np.allclose(d11.side_xy(d11.TOWARD_OBSERVER), [1.0, 0.0], atol=1e-12)
    assert np.allclose(d11.side_xy(d11.NORTH), [0.0, 1.0], atol=1e-12)


# The drawn still


@pytest.fixture(scope="module", params=[ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def states_fig(request):
    with exporter.venue("light", request.param) as cast:
        fig = d11.build_states(request.param, cast)
        fig.canvas.draw()
        yield fig
        plt.close(fig)


@pytest.mark.parametrize("col", [0, 1, 2])
def test_drawn_bodies_sit_on_their_reflex_orbits(states_fig, col):
    frac = d11.STATE_FRACS[col]
    r_hand, _ = hand_relative(frac)
    (star,) = _by_gid(states_fig, f"d11-star-{col}")
    (planet,) = _by_gid(states_fig, f"d11-planet-{col}")
    star_xy = np.array([star.get_xdata()[0], star.get_ydata()[0]])
    assert np.allclose(star_xy, page(-F_P * r_hand), atol=1e-9)
    assert np.allclose(planet.get_center(), page(F_S * r_hand), atol=1e-9)
    # Star and planet on opposite sides of the barycenter at the origin.
    assert np.dot(star_xy, planet.get_center()) < 0


def test_drawn_velocities_at_quadrature(states_fig):
    """At t0 the planet moves toward the observer (right), the star away."""
    (vs,) = _by_gid(states_fig, "d11-v-star-0")
    (vp,) = _by_gid(states_fig, "d11-v-planet-0")
    (a, b) = vs._posA_posB
    assert b[0] - a[0] < 0 and abs(b[1] - a[1]) < 1e-9
    (a, b) = vp._posA_posB
    assert b[0] - a[0] > 0 and abs(b[1] - a[1]) < 1e-9
    # Both arrows share one scale: lengths in the ratio Mp : M*.
    ls = np.hypot(*np.subtract(*vs._posA_posB[::-1]))
    lp = np.hypot(*np.subtract(*vp._posA_posB[::-1]))
    assert ls / lp == pytest.approx(Q)


@pytest.mark.parametrize("col", [1, 2])
def test_star_moves_across_the_line_of_sight_at_plus_and_minus_z(states_fig, col):
    (vs,) = _by_gid(states_fig, f"d11-v-star-{col}")
    (a, b) = vs._posA_posB
    assert abs(b[0] - a[0]) < 1e-9 and abs(b[1] - a[1]) > 0.1


@pytest.mark.parametrize(("col", "lit_toward_observer"), [(1, False), (2, True)])
def test_lit_half_faces_the_star(states_fig, col, lit_toward_observer):
    """At +Z the observer (right) sees the unlit half; at -Z the lit half."""
    (planet,) = _by_gid(states_fig, f"d11-planet-{col}")
    lit = [
        p
        for p in states_fig.axes[col].patches
        if p.get_zorder() == 9 and hasattr(p, "get_xy")
    ]
    centroid = np.mean(lit[0].get_xy(), axis=0)
    assert (centroid[0] > planet.get_center()[0]) == lit_toward_observer


def test_drawn_relative_vector_runs_star_to_planet(states_fig):
    for col in range(3):
        (rel,) = _by_gid(states_fig, f"d11-relative-{col}")
        (star,) = _by_gid(states_fig, f"d11-star-{col}")
        (planet,) = _by_gid(states_fig, f"d11-planet-{col}")
        a, b = (np.asarray(p) for p in rel._posA_posB)
        assert np.allclose(a, [star.get_xdata()[0], star.get_ydata()[0]])
        d_rel = _unit(b - a)
        d_hand = _unit(np.asarray(planet.get_center()) - a)
        assert np.allclose(d_rel, d_hand, atol=1e-9)


def _unit(v):
    return v / np.linalg.norm(v)


def test_drawn_rv_curve_and_state_markers(states_fig):
    (curve,) = _by_gid(states_fig, "d11-rv-curve")
    x, y = curve.get_xdata(), curve.get_ydata()
    assert np.allclose(y, np.cos(2.0 * np.pi * x), atol=1e-12)
    expected = [(0.0, 1.0), (0.25, 0.0), (0.75, 0.0)]
    for k, (fx, fy) in enumerate(expected):
        (m,) = _by_gid(states_fig, f"d11-rv-state-{k}")
        assert m.get_xdata()[0] == pytest.approx(fx)
        assert m.get_ydata()[0] == pytest.approx(fy, abs=1e-12)
    # Recession is labeled on the positive side.
    (rec,) = _by_gid(states_fig, "d11-rv-recession")
    assert rec.get_position()[1] > 0 and "recession" in rec.get_text()


def test_printed_numbers_and_titles(states_fig):
    texts = [t.get_text() for t in states_fig.findobj(Text)]
    joined = "\n".join(texts)
    assert "+0.08946 m/s" in joined
    assert f"= {d11.MASS_RATIO:g}" in joined
    titles = [ax.get_title() for ax in states_fig.axes[:3]]
    assert "+K" in titles[0] and "=0" in titles[1] and "=0" in titles[2]
    assert "quadrature" in titles[0]
    assert "+Z" in titles[1] and "-Z" in titles[2]
    (status,) = _by_gid(states_fig, "d11-status")
    assert "exaggerated" in status.get_text() and "pending" in status.get_text()


# The clock animation


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_clock_frames_are_equal_time_steps_through_the_states(layout):
    frames = d11.clock_frames(layout)
    n = len(frames)
    assert n % 4 == 0
    if layout.frame_budget is not None:
        assert n <= layout.frame_budget
    fracs = np.array([f["frac"] for f in frames])
    assert fracs[0] == 0.0
    assert np.allclose(np.diff(fracs), 1.0 / n)
    for state in d11.STATE_FRACS:
        assert np.any(np.isclose(fracs, state))


def test_clock_draw_sets_state_from_the_frame_alone():
    with exporter.venue("dark", ex.DOC) as cast:
        scene = d11.build_clock(ex.DOC, cast)
        frames = scene.frames

        def snapshot():
            (dot,) = _by_gid(scene.fig, "d11-rv-now")
            (star,) = _by_gid(scene.fig, "d11-star")
            return (
                dot.get_xdata()[0],
                dot.get_ydata()[0],
                star.get_xdata()[0],
                star.get_ydata()[0],
            )

        scene.draw(scene.fig, frames[7])
        first = snapshot()
        scene.draw(scene.fig, frames[3])
        scene.draw(scene.fig, frames[7])
        assert snapshot() == first
        frac = frames[7]["frac"]
        assert first[0] == pytest.approx(frac)
        assert first[1] == pytest.approx(math.cos(2.0 * math.pi * frac))
        assert np.allclose(first[2:], page(-F_P * hand_relative(frac)[0]))
        plt.close(scene.fig)


# Registry and text rules


def test_specs_are_consistent():
    slugs = [s.slug for s in d11.FIGURES + d11.ANIMATIONS]
    assert all(s.startswith("d11-") for s in slugs)
    assert len(set(slugs)) == len(slugs)
    (anim,) = d11.ANIMATIONS
    assert anim.ground == "rate"
    for spec in d11.FIGURES + d11.ANIMATIONS:
        assert "exaggerated" in spec.status and "pending" in spec.status
        assert "exaggerated" in spec.caption
        assert rf"M_p/M_\star={d11.MASS_RATIO:g}" in spec.caption
    # The caption and alt text print the same worked number as the figure.
    number = f"{d11.EARTH_SUN_K:+.5f}"
    assert number in d11.STATES_CAPTION
    assert number.lstrip("+") in d11.STATES_ALT


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_no_label_carries_a_stroked_halo(mode, layout):
    with exporter.venue(mode, layout) as cast:
        figs = [spec.build(layout, cast) for spec in d11.FIGURES]
        figs += [spec.build(layout, cast).fig for spec in d11.ANIMATIONS]
        for fig in figs:
            stroked = [t.get_text() for t in fig.findobj(Text) if t.get_path_effects()]
            assert stroked == []
            plt.close(fig)


def test_module_source_is_ascii():
    src = Path(d11.__file__).read_text(encoding="utf-8")
    assert src.isascii()


# Review round: fourth RV state, line-of-sight projection, wording


def test_half_period_marker_is_minus_k(states_fig):
    (m,) = _by_gid(states_fig, "d11-rv-state-3")
    assert m.get_xdata()[0] == pytest.approx(0.5)
    assert m.get_ydata()[0] == pytest.approx(-1.0)
    (label,) = _by_gid(states_fig, "d11-rv-approach")
    assert label.get_position()[1] < 0 and "approach" in label.get_text()


def test_titles_name_nearest_and_farthest_with_lit_side(states_fig):
    titles = [ax.get_title() for ax in states_fig.axes[:3]]
    assert "nearest the observer" in titles[1] and "dark side to us" in titles[1]
    assert "farthest" in titles[2] and "lit side to us" in titles[2]
    texts = [t.get_text() for t in states_fig.findobj(Text)] + titles
    # Time labels use mathtext t_0, never a bare "t0".
    assert all("t0" not in t for t in texts)
    assert any("t_0" in t for t in titles)


def test_centimeter_gloss_matches_the_worked_number(states_fig):
    assert round(d11.EARTH_SUN_K * 100) == 9
    joined = "\n".join(t.get_text() for t in states_fig.findobj(Text))
    assert "(about 9 cm/s)" in joined
    assert "(about 9 cm/s)" in d11.STATES_CAPTION


def test_caption_states_the_omega_star_sign_flip():
    cap = d11.STATES_CAPTION
    assert r"\omega_\star=\omega_p+\pi" in cap
    assert "introduces a minus sign" in cap


def test_clock_line_of_sight_projection_matches_the_curve():
    with exporter.venue("light", ex.DOC) as cast:
        scene = d11.build_clock(ex.DOC, cast)
        (los,) = _by_gid(scene.fig, "d11-v-star-los")
        (dot,) = _by_gid(scene.fig, "d11-rv-now")
        for frame in scene.frames:
            scene.draw(scene.fig, frame)
            x, y = los.get_xdata(), los.get_ydata()
            assert y[0] == y[1]  # along the line of sight (page x is Z)
            # Page length is VEL_SCALE * dZ_star/dt in units of n a, and
            # v_r = -dZ_star/dt: the segment points opposite the curve's sign.
            v_r_over_k = dot.get_ydata()[0]
            assert x[1] - x[0] == pytest.approx(-d11.VEL_SCALE * F_P * v_r_over_k)
        plt.close(scene.fig)


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_clock_labels_never_collide(layout):
    """Moving labels in the orbit panel never overlap each other or the
    observer arrow, at any frame."""
    with exporter.venue("light", layout) as cast:
        scene = d11.build_clock(layout, cast)
        side = scene.fig.axes[0]
        (obs,) = _by_gid(scene.fig, "d11-to-observer")
        for frame in scene.frames[:: max(1, len(scene.frames) // 40)]:
            scene.draw(scene.fig, frame)
            scene.fig.canvas.draw()
            rend = scene.fig.canvas.get_renderer()
            boxes = [
                t.get_window_extent(rend)
                for t in side.texts
                if t.get_text() and t.get_visible()
            ]
            arrow = obs.get_window_extent(rend)
            for i, a in enumerate(boxes):
                assert not a.overlaps(arrow)
                assert not any(a.overlaps(b) for b in boxes[i + 1 :])
        plt.close(scene.fig)
