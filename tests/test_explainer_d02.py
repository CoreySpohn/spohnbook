"""Independent checks of every anchor the orbit-geometry explainer draws.

Each expected value is computed here from closed forms or hand-written
matrices, not from the module's helpers, and compared with what the module
plots: the side-view construction, the node, periapsis and angular momentum,
the three element arcs, the illumination angles and phase-function values
printed on the epoch figure, the lit fraction of the planet glyph, the
cameras, and the frame contracts of the two animations.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from explainers import _common as ex
from explainers import _export as exporter
from explainers import d02_orbit_geometry as d02

DEG = math.pi / 180.0
ORB = d02.ORBIT
INC, OM, W = ORB["i_deg"] * DEG, ORB["Omega_deg"] * DEG, ORB["omega_deg"] * DEG
A, E = ORB["a"], ORB["e"]


def lambert(alpha):
    return (math.sin(alpha) + (math.pi - alpha) * math.cos(alpha)) / math.pi


def chapter_rotation():
    """R_Z(Omega) R_X(i) R_Z(omega_p), written out by hand."""
    co, so = math.cos(OM), math.sin(OM)
    ci, si = math.cos(INC), math.sin(INC)
    cw, sw = math.cos(W), math.sin(W)
    rz_om = np.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])
    rx_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    rz_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    return rz_om @ rx_i @ rz_w


def hand_position(nu):
    r = A * (1 - E**2) / (1 + E * math.cos(nu))
    return chapter_rotation() @ np.array([r * math.cos(nu), r * math.sin(nu), 0.0])


# Basis and rotation


def test_basis_is_right_handed_north_east_toward_observer():
    assert np.allclose(np.cross(d02.NORTH, d02.EAST), d02.TOWARD_OBSERVER)


def test_orbit_rotation_is_the_chapter_product():
    assert np.allclose(d02.orbit_rotation(), chapter_rotation(), atol=1e-14)
    for nu in np.linspace(0.0, 2 * math.pi, 7):
        assert np.allclose(d02.position(nu), hand_position(nu), atol=1e-14)


# Side-view construction


def test_side_construction_projection_and_angles():
    r, beta = d02.SIDE["r"], d02.SIDE["beta_deg"] * DEG
    z, rho, alpha = d02.side_state()
    assert z == pytest.approx(r * math.cos(beta))
    assert rho == pytest.approx(r * math.sin(beta))
    assert alpha == pytest.approx(math.pi - beta)
    # Distant-observer limit of the finite-observer angle (planet-centered).
    r_vec = np.array([rho, 0.0, z])  # vertical first, line of sight third
    r_obs = np.array([0.0, 0.0, 1e9])
    to_obs = r_obs - r_vec
    cos_a = (-r_vec) @ to_obs / (np.linalg.norm(r_vec) * np.linalg.norm(to_obs))
    assert math.acos(cos_a) == pytest.approx(alpha, abs=1e-8)
    assert math.cos(alpha) == pytest.approx(-z / r)


def test_side_construction_arc_directions():
    # The alpha arc at the planet runs from the observer direction (0 deg)
    # to the direction back to the star, beta - 180 deg.
    z, rho, alpha = d02.side_state()
    back = math.degrees(math.atan2(-rho, -z))
    assert back == pytest.approx(d02.SIDE["beta_deg"] - 180.0)
    assert abs(back) == pytest.approx(math.degrees(alpha))


# Nodes, periapsis, angular momentum and the three arcs


def test_ascending_node_lies_on_the_sky_plane_moving_toward_observer():
    nu_node = -W
    node = d02.ascending_node()
    assert node[2] == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(node, hand_position(nu_node))
    # True anomaly increases with time, so dZ/dnu > 0 means dZ/dt > 0.
    step = 1e-6
    dz = hand_position(nu_node + step)[2] - hand_position(nu_node - step)[2]
    assert dz > 0
    desc = d02.descending_node()
    assert desc[2] == pytest.approx(0.0, abs=1e-14)
    dz = hand_position(math.pi - W + step)[2] - hand_position(math.pi - W - step)[2]
    assert dz < 0


def test_node_longitude_is_measured_from_north_toward_east():
    node = d02.ascending_node()
    angle = math.atan2(node @ d02.EAST, node @ d02.NORTH) % (2 * math.pi)
    assert angle == pytest.approx(OM)
    assert np.allclose(node / np.linalg.norm(node), d02.node_direction())


def test_inclination_is_the_angle_from_plus_z_to_angular_momentum():
    # Angular momentum from r x v, with v by central difference in anomaly.
    nu, step = 0.3, 1e-6
    r = hand_position(nu)
    v = (hand_position(nu + step) - hand_position(nu - step)) / (2 * step)
    h = np.cross(r, v)
    h /= np.linalg.norm(h)
    assert np.allclose(d02.angular_momentum_direction(), h, atol=1e-8)
    assert math.acos(h[2]) == pytest.approx(INC)


def test_argument_of_periapsis_is_measured_from_node_along_motion():
    n = d02.node_direction()
    p = d02.periapsis() / np.linalg.norm(d02.periapsis())
    h = d02.angular_momentum_direction()
    angle = math.atan2(np.cross(n, p) @ h, n @ p) % (2 * math.pi)
    assert angle == pytest.approx(W)
    assert np.linalg.norm(d02.periapsis()) == pytest.approx(A * (1 - E))


def test_drawn_arcs_end_on_the_vectors_they_measure_to():
    h = d02.angular_momentum_direction()
    i_arc = d02.arc_points(
        d02.TOWARD_OBSERVER, np.cross(d02.TOWARD_OBSERVER, h), INC, 1.0
    )
    assert np.allclose(i_arc[0], d02.TOWARD_OBSERVER)
    assert np.allclose(i_arc[-1], h)
    om_arc = d02.arc_points(d02.NORTH, d02.TOWARD_OBSERVER, OM, 1.0)
    assert np.allclose(om_arc[0], d02.NORTH)
    assert np.allclose(om_arc[-1], d02.node_direction())
    assert np.allclose(om_arc[:, 2], 0.0)  # the Omega arc stays in the sky plane
    w_arc = d02.arc_points(d02.node_direction(), h, W, 1.0)
    p = d02.periapsis() / np.linalg.norm(d02.periapsis())
    assert np.allclose(w_arc[-1], p)
    assert np.allclose(w_arc @ h, 0.0)  # the omega arc stays in the orbit plane


# Illumination at the three epochs and the printed numbers


@pytest.mark.parametrize(
    ("u_deg", "alpha_deg"),
    [(270.0, 90.0 - ORB["i_deg"]), (0.0, 90.0), (90.0, 90.0 + ORB["i_deg"])],
)
def test_epoch_illumination_angles(u_deg, alpha_deg):
    r_vec, alpha, phi = d02.epoch_state(u_deg)
    # Z = r sin(i) sin(u) from the chapter's rotation.
    r = np.linalg.norm(r_vec)
    assert r_vec[2] == pytest.approx(
        r * math.sin(INC) * math.sin(u_deg * DEG), abs=1e-14
    )
    assert math.degrees(alpha) == pytest.approx(alpha_deg)
    assert phi == pytest.approx(lambert(alpha_deg * DEG))


def test_lambert_limits():
    assert d02.lambert_phase(0.0) == pytest.approx(1.0)
    assert d02.lambert_phase(math.pi / 2) == pytest.approx(1.0 / math.pi)
    assert d02.lambert_phase(math.pi) == pytest.approx(0.0, abs=1e-15)


def test_epoch_figure_prints_the_closed_form_values():
    with exporter.venue("light", ex.DOC) as cast:
        fig = d02.build_epochs(ex.DOC, cast)
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        texts += [ax.get_title() for ax in fig.axes]
        plt.close(fig)
    for alpha_deg in (90.0 - ORB["i_deg"], 90.0, 90.0 + ORB["i_deg"]):
        want_alpha = f"{alpha_deg:.0f} deg"
        want_phi = f"{lambert(alpha_deg * DEG):.2f}"
        assert any(want_alpha in t and want_phi in t for t in texts), (
            want_alpha,
            want_phi,
        )
    # The hand-computed values themselves.
    assert f"{lambert(35 * DEG):.2f}" == "0.84"
    assert f"{lambert(90 * DEG):.2f}" == "0.32"
    assert f"{lambert(145 * DEG):.2f}" == "0.02"


@pytest.mark.parametrize("alpha_deg", [0.0, 35.0, 90.0, 120.0, 145.0, 180.0])
def test_lit_glyph_area_is_the_illuminated_fraction(alpha_deg):
    alpha = alpha_deg * DEG
    poly = d02.lit_polygon((0.3, -0.2), (1.0, 2.0), alpha, 1.0, n=4001)
    x, y = poly.T
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area / math.pi == pytest.approx((1 + math.cos(alpha)) / 2, abs=1e-5)


def test_lit_limb_faces_the_star():
    toward = np.array([-1.0, 0.0])
    poly = d02.lit_polygon((0.0, 0.0), toward, 100 * DEG, 1.0, n=65)
    assert (poly @ toward).max() == pytest.approx(1.0)
    assert (poly @ toward).min() > 0.0  # a crescent on the star side only


# Cameras


def test_observer_view_puts_north_up_and_east_left():
    assert np.allclose(d02.project(d02.NORTH, 90.0), [0.0, 1.0])
    assert np.allclose(d02.project(d02.EAST, 90.0), [-1.0, 0.0])
    assert np.allclose(d02.project(d02.TOWARD_OBSERVER, 90.0), [0.0, 0.0])


def test_side_view_puts_observer_right_and_north_up():
    assert np.allclose(d02.project(d02.TOWARD_OBSERVER, 0.0), [1.0, 0.0])
    assert np.allclose(d02.project(d02.NORTH, 0.0), [0.0, 1.0])


@pytest.mark.parametrize("view", [0.0, 32.0, 60.0, 90.0, d02.OBLIQUE_VIEW])
def test_every_camera_is_orthonormal_so_page_scale_is_fixed(view):
    right, up, back = d02.camera(view)
    basis = np.array([right, up, back])
    assert np.allclose(basis @ basis.T, np.eye(3))
    assert np.allclose(np.cross(right, up), back)
    g, el = (view, 0.0) if np.ndim(view) == 0 else view
    g, el = g * DEG, el * DEG
    # The camera direction: turned g from east toward the observer, raised el
    # toward north.
    want = np.array(
        [math.sin(el), math.cos(el) * math.cos(g), math.cos(el) * math.sin(g)]
    )
    assert np.allclose(back, want)
    if el == 0.0:
        # A camera turned about north only keeps north straight up.
        assert np.allclose(up, d02.NORTH)


# Time


def test_kepler_solution_and_its_inverse():
    frac = np.linspace(0.0, 0.999, 37)
    nu = d02.nu_from_time_fraction(frac)
    ecc = 2 * np.arctan(np.sqrt((1 - E) / (1 + E)) * np.tan(nu / 2))
    mean = (ecc - E * np.sin(ecc)) % (2 * math.pi)
    assert np.allclose(mean, 2 * math.pi * frac, atol=1e-12)
    assert np.allclose(d02.time_fraction_from_nu(nu), frac, atol=1e-12)


# Animation contracts: never move the camera and the orbit together


def test_viewpoint_sweep_freezes_the_orbit_state():
    for layout in (ex.DOC, ex.SLIDE):
        frames = d02.sweep_frames(layout)
        assert len({f["u_deg"] for f in frames}) == 1
        views = [f["view_deg"] for f in frames]
        assert views[0] == 0.0 and views[-1] == 90.0
        assert np.all(np.diff(views) > 0)
    assert len(d02.sweep_frames(ex.DOC)) <= 30


def test_orbit_clock_steps_are_equal_in_time_and_carry_no_camera():
    for layout in (ex.DOC, ex.SLIDE):
        frames = d02.clock_frames(layout)
        assert all(set(f) == {"frac"} for f in frames)
        steps = np.diff([f["frac"] for f in frames]) % 1.0
        assert np.allclose(steps, 1.0 / len(frames))
    assert len(d02.clock_frames(ex.DOC)) <= 30


def test_clock_first_frame_starts_behind_the_sky_plane():
    frac = d02.clock_frames(ex.DOC)[0]["frac"]
    r_vec = d02.position(d02.nu_from_time_fraction(frac))
    u = math.atan2(r_vec[2] / math.sin(INC), (r_vec @ d02.node_direction()))
    assert math.degrees(u) % 360 == pytest.approx(270.0)


def _scale(ax):
    """Pixels per data unit along x and along y."""
    p0 = ax.transData.transform((0.0, 0.0))
    px = ax.transData.transform((1.0, 0.0)) - p0
    py = ax.transData.transform((0.0, 1.0)) - p0
    return abs(px[0]), abs(py[1])


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_elements_panels_share_one_page_scale(layout):
    with exporter.venue("light", layout) as cast:
        fig = d02.build_elements(layout, cast)
        fig.canvas.draw()
        space, sky = fig.axes[0], fig.axes[1]
        sx, sy = _scale(space)
        kx, _ky = _scale(sky)
        plt.close(fig)
    assert sx == pytest.approx(sy, rel=1e-6)
    assert kx == pytest.approx(sx, rel=0.05)


# The drawn artists: arcs, arrowheads and rays as they appear in the figures


def _artists(fig, gid):
    found = [
        a for ax in fig.axes for a in [*ax.lines, *ax.patches] if a.get_gid() == gid
    ]
    assert len(found) == 1, (gid, len(found))
    return found[0]


def _ends(patch):
    """Tail and head of a FancyArrowPatch, in data coordinates."""
    tail, head = patch._posA_posB  # no public getter exists
    return np.asarray(tail), np.asarray(head)


def _proj(vec, view):
    right, up, _ = d02.camera(view)
    return np.array([vec @ right, vec @ up])


def _unit_node():
    return np.array([math.cos(OM), math.sin(OM), 0.0])


def _unit_h():
    return chapter_rotation() @ np.array([0.0, 0.0, 1.0])


def _unit_peri():
    return chapter_rotation() @ np.array([1.0, 0.0, 0.0])


def _bisector(a, b):
    m = a / np.linalg.norm(a) + b / np.linalg.norm(b)
    return m / np.linalg.norm(m)


# (gid, start vector, end vector, the vector halfway through the measured
# sense): each arc must start on its reference, end on its target, and pass
# the bisector on the way, so an arc drawn about the wrong axis or backwards
# fails.
ELEMENT_ARCS = (
    ("d02-arc-i", np.array([0.0, 0.0, 1.0]), _unit_h(), None),
    (
        "d02-arc-Omega",
        np.array([1.0, 0.0, 0.0]),
        _unit_node(),
        np.array([math.cos(OM / 2), math.sin(OM / 2), 0.0]),
    ),
    ("d02-arc-omega", _unit_node(), _unit_peri(), None),
)


@pytest.fixture(scope="module", params=[ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def elements_fig(request):
    with exporter.venue("dark", request.param) as cast:
        fig = d02.build_elements(request.param, cast)
    yield fig
    plt.close(fig)


@pytest.mark.parametrize(("gid", "start", "end", "mid"), ELEMENT_ARCS)
def test_drawn_element_arcs_start_end_and_sense(elements_fig, gid, start, end, mid):
    view = d02.OBLIQUE_VIEW
    pts = _artists(elements_fig, gid).get_xydata()
    radius = np.linalg.norm(pts[0]) / np.linalg.norm(_proj(start, view))
    assert np.allclose(pts[0], radius * _proj(start, view), atol=1e-9)
    assert np.allclose(pts[-1], radius * _proj(end, view), atol=1e-9)
    mid = _bisector(start, end) if mid is None else mid
    assert np.allclose(pts[len(pts) // 2], radius * _proj(mid, view), atol=1e-9)
    tail, head = _ends(_artists(elements_fig, f"{gid}-head"))
    assert np.allclose(head, pts[-1])  # the arrowhead sits on the measured end
    assert np.linalg.norm(tail - pts[-1]) < np.linalg.norm(tail - pts[0])


def test_drawn_sky_omega_arc_runs_north_to_node_counterclockwise(elements_fig):
    pts = _artists(elements_fig, "d02-sky-arc-Omega").get_xydata()
    # Observer's view: north up, east left, so north-toward-east is
    # counterclockwise on the image.
    angles = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
    assert angles[0] == pytest.approx(math.pi / 2)
    assert np.all(np.diff(angles) > 0)
    assert angles[-1] - angles[0] == pytest.approx(OM)
    node = _proj(_unit_node(), 90.0)
    assert np.allclose(pts[-1] / np.linalg.norm(pts[-1]), node)
    _, head = _ends(_artists(elements_fig, "d02-sky-arc-Omega-head"))
    assert np.allclose(head, pts[-1])


def test_drawn_angular_momentum_vector(elements_fig):
    tail, head = _ends(_artists(elements_fig, "d02-h"))
    assert np.allclose(tail, 0.0)
    want = _proj(_unit_h(), d02.OBLIQUE_VIEW)
    assert np.allclose(head / np.linalg.norm(head), want / np.linalg.norm(want))


@pytest.mark.parametrize(
    ("prefix", "view"), [("d02", d02.OBLIQUE_VIEW), ("d02-sky", 90.0)]
)
def test_drawn_motion_arrows_point_forward_in_time(elements_fig, prefix, view):
    for k, u_head in enumerate(d02.MOTION_U_DEG):
        tail, head = _ends(_artists(elements_fig, f"{prefix}-motion-{k}"))
        # Argument of latitude increases with time, so the head is at the
        # larger u; nu = u - omega_p.
        want_head = _proj(hand_position(u_head * DEG - W), view)
        want_tail = _proj(hand_position((u_head - d02.MOTION_SPAN_DEG) * DEG - W), view)
        assert d02.MOTION_SPAN_DEG > 0
        assert np.allclose(head, want_head)
        assert np.allclose(tail, want_tail)


@pytest.fixture(scope="module", params=[ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def construction_fig(request):
    with exporter.venue("light", request.param) as cast:
        fig = d02.build_construction(request.param, cast)
    yield fig
    plt.close(fig)


def test_drawn_construction_rays(construction_fig):
    z, rho, _ = d02.side_state()
    beta = d02.SIDE["beta_deg"] * DEG
    tail, head = _ends(_artists(construction_fig, "d02-ray-r"))
    assert np.allclose(tail, 0.0)
    assert math.atan2(head[1], head[0]) == pytest.approx(beta)
    # The starlight arrow stops short of the planet disk.
    assert np.hypot(head[0] - z, head[1] - rho) > d02.PLANET_R
    tail, head = _ends(_artists(construction_fig, "d02-ray-observer"))
    assert tail[1] == pytest.approx(rho) and head[1] == pytest.approx(rho)
    assert head[0] > tail[0] > z + d02.PLANET_R  # leaves the disk toward +Z


def test_drawn_construction_arcs(construction_fig):
    z, rho, alpha = d02.side_state()
    beta_deg = d02.SIDE["beta_deg"]
    arc = _artists(construction_fig, "d02-arc-beta")
    assert np.allclose(arc.center, (0.0, 0.0))
    assert (arc.theta1, arc.theta2) == pytest.approx((0.0, beta_deg))
    _, head = _ends(_artists(construction_fig, "d02-arc-beta-head"))
    assert math.degrees(math.atan2(head[1], head[0])) == pytest.approx(beta_deg)
    arc = _artists(construction_fig, "d02-arc-alpha")
    assert np.allclose(arc.center, (z, rho))
    assert arc.theta2 - arc.theta1 == pytest.approx(math.degrees(alpha))
    _, head = _ends(_artists(construction_fig, "d02-arc-alpha-head"))
    # The alpha arc ends pointing back toward the star.
    toward_star = math.degrees(math.atan2(-rho, -z))
    got = math.degrees(math.atan2(head[1] - rho, head[0] - z))
    assert got == pytest.approx(toward_star)


@pytest.mark.parametrize("builder", ["build_clock", "build_epochs"])
def test_paired_panels_share_one_page_scale(builder):
    with exporter.venue("light", ex.DOC) as cast:
        obj = getattr(d02, builder)(ex.DOC, cast)
        fig = obj.fig if hasattr(obj, "fig") else obj
        fig.canvas.draw()
        scales = [_scale(ax)[0] for ax in fig.axes if ax.get_aspect() == 1.0]
        plt.close(fig)
    assert len(scales) >= 2
    assert max(scales) == pytest.approx(min(scales), rel=0.03)
