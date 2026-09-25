"""Numerical anchors of the optical-planes explainer.

Every number printed on the figures, and every claim their captions make
about the propagated example, is checked here against a closed form, hand
arithmetic, or an independent NumPy model: the normalization is the
aperture integral squared, the shift theorem places a speckle pair at
k lambda/D for a ripple of k cycles per aperture, a small ripple sends
(pi a0/lambda)^2 times the coronagraphic point-source response into each
speckle, the speckle field has phase +pi/2 under exp(+i 2 pi W/lambda), a
cancelled ripple removes the pair, a reversed ripple restores the intensity
with the opposite field sign, and a positive OPD tilt moves the image toward
positive x. The arrays under test come from the diagram module's own
propagation, so the tests pin what the figures show, not how physicaloptix
computes it.
"""

import itertools
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
from explainers import d04_optical_planes as d04

P = d04.PARAMS


def peak_near(image, u_lod, v_lod, radius_lod=1.0):
    """Largest value within ``radius_lod`` of ``(u, v)``, and its location."""
    u = d04.optics()["focal_coords"]
    uu, vv = np.meshgrid(u, u)
    near = np.hypot(uu - u_lod, vv - v_lod) <= radius_lod
    index = np.unravel_index(np.argmax(np.where(near, image, -np.inf)), image.shape)
    return float(image[index]), (float(uu[index]), float(vv[index])), index


def dft(field, coords, u, v):
    """Continuous transform exp(-i 2 pi (u x + v y)) of a sampled field at (u, v)."""
    dx = coords[1] - coords[0]
    x, y = np.meshgrid(coords, coords)
    return np.sum(field * np.exp(-2j * np.pi * (u * x + v * y))) * dx * dx


def test_grids_match_the_declared_sampling():
    o = d04.optics()
    u = o["focal_coords"]
    # Half-pixel-offset focal grid: 48 samples at 0.5 lambda/D span
    # -11.75 to +11.75, so no sample sits on the axis.
    assert u.size == P["nfoc"]
    assert u[0] == pytest.approx(-(P["nfoc"] - 1) / 2 * P["pixscale_lod"])
    assert np.diff(u) == pytest.approx(P["pixscale_lod"])
    assert o["aperture"].shape == (P["npup"], P["npup"])
    assert o["wide_coords"].size == round(P["thumbnail_width_d"] * P["npup"])


def test_gray_disks_hold_their_geometric_areas():
    # A gray-pixel disk sums to the circle's area in cells: pi r^2 N^2.
    o = d04.optics()
    n = P["npup"]
    for key, radius in (("aperture", 0.5), ("stop", 0.4)):
        expected = math.pi * radius**2 * n**2
        assert o[key].sum() == pytest.approx(expected, rel=2e-3)


def test_normalization_is_the_on_axis_peak_not_the_brightest_sample():
    # I_peak = |F[A](0)|^2 = (sum A dx^2)^2. The brightest sample of the
    # half-pixel-offset grid, at (0.25, 0.25) lambda/D, is about 0.73 of it.
    o = d04.optics()
    coords = o["pupil_coords"]
    dx = coords[1] - coords[0]
    on_axis = abs(o["aperture"].sum() * dx * dx) ** 2
    assert o["peak"] == pytest.approx(on_axis, rel=1e-12)
    off = abs(dft(o["aperture"], coords, 0.25, 0.25)) ** 2 / on_axis
    assert off == pytest.approx(0.728, abs=0.005)
    # The same ratio through the production camera.
    import jax

    with jax.enable_x64(True):
        sampled = float(o["camera"](o["entrance"]).intensity().max())
    assert sampled / o["peak"] == pytest.approx(off, rel=1e-6)


def test_opd_card_amplitude_is_the_labeled_one_nanometer():
    # The DM card scale is labeled "+-1 nm": the static ripple a0 cos(2 pi k x).
    o = d04.optics()
    opd = d04.propagate(d04.static_coeffs(0.0))["opd"]
    inside = o["aperture"] > 0.0
    x = o["pupil_coords"][None, :] * np.ones((P["npup"], 1))
    expected = P["static_ripple_nm"] * np.cos(
        2 * math.pi * P["static_ripple_cycles_per_d"] * x
    )
    assert np.allclose(opd, expected, atol=1e-12)
    assert np.max(np.abs(opd[inside])) <= P["static_ripple_nm"]
    assert np.max(np.abs(opd[inside])) > 0.99 * P["static_ripple_nm"]


def test_mask_phase_winds_charge_times_around_the_center():
    # The card says "6 turns of 2 pi": the phase advances by 6 x 2 pi
    # around a circle about the optical axis.
    phase = d04.optics()["mask_phase"]
    u = d04.optics()["focal_coords"]
    angles = np.linspace(0.0, 2 * math.pi, 721)
    radius = 8.0
    cols = [int(np.argmin(np.abs(u - radius * math.cos(a)))) for a in angles]
    rows = [int(np.argmin(np.abs(u - radius * math.sin(a)))) for a in angles]
    samples = phase[rows, cols]
    steps = np.angle(np.exp(1j * np.diff(samples)))
    assert round(steps.sum() / (2 * math.pi)) == P["vortex_charge"]


def test_lyot_thumbnail_extends_the_path_array_and_shows_rejected_light():
    # The wide thumbnail is the same operator: its central D-wide block is
    # the tapped field. The D-wide array holds about 4 percent of the
    # entrance energy (the caption's number), under 1 percent lies inside
    # the pupil edge, and the ring just outside the edge is where the light is.
    o = d04.optics()
    state = d04.propagate(d04.static_coeffs(0.0))
    n = P["npup"]
    m = o["wide_coords"].size
    lo = (m - n) // 2
    assert np.allclose(
        state["lyot_wide"][lo : lo + n, lo : lo + n], state["lyot_field"]
    )
    entrance = np.sum(o["aperture"] ** 2)
    in_array = np.sum(np.abs(state["lyot_field"]) ** 2) / entrance
    assert 0.03 < in_array < 0.05
    xw, yw = np.meshgrid(o["wide_coords"], o["wide_coords"])
    r = np.hypot(xw, yw)
    energy = np.abs(state["lyot_wide"]) ** 2
    inside = energy[r <= P["aperture_radius_d"]].sum() / entrance
    ring = energy[(r > 0.5) & (r <= 1.0)].sum() / entrance
    assert inside < 0.01
    assert ring > 10 * inside


def test_speckle_pair_sits_at_k_lambda_over_d():
    # Shift theorem: a cosine ripple of k cycles per D puts a pair at +-k.
    image = d04.propagate(d04.static_coeffs(0.0))["intensity"]
    k = P["static_ripple_cycles_per_d"]
    for sign in (1.0, -1.0):
        _, (u, v), _ = peak_near(image, sign * k, 0.0, radius_lod=2.0)
        assert abs(u - sign * k) <= P["pixscale_lod"]
        assert abs(v) <= P["pixscale_lod"]
    # The pair is the brightest structure in the image.
    assert np.max(image) == pytest.approx(peak_near(image, k, 0.0, 2.0)[0], rel=0.05)


def numpy_vortex_response(u_s, v_s, source_u, *, vortex=True):
    """Independent single-scale vortex coronagraph, in NumPy.

    A point source at ``(source_u, 0)`` lambda/D passes the gray aperture, a
    focal-plane transform on a 0.125 lambda/D grid out to 32 lambda/D, the
    charge-6 phase (optional), the inverse transform, the Lyot stop, and a
    last transform evaluated at ``(u_s, v_s)``. Returned relative to the
    unocculted on-axis peak.
    """
    o = d04.optics()
    coords = o["pupil_coords"]
    dx = coords[1] - coords[0]
    x, _ = np.meshgrid(coords, coords)
    field = o["aperture"] * np.exp(2j * np.pi * source_u * x)
    u = (np.arange(512) - 256 + 0.5) * 0.125
    du = u[1] - u[0]
    forward = np.exp(-2j * np.pi * np.outer(u, coords)) * dx
    focal = forward @ field @ forward.T
    if vortex:
        uu, vv = np.meshgrid(u, u)
        focal = focal * np.exp(1j * P["vortex_charge"] * np.arctan2(vv, uu))
    backward = np.exp(2j * np.pi * np.outer(coords, u)) * du
    lyot = (backward @ focal @ backward.T) * o["stop"]
    peak = (o["aperture"].sum() * dx * dx) ** 2
    return dft(lyot, coords, u_s, v_s) / math.sqrt(peak)


def test_speckle_brightness_matches_the_first_order_estimate():
    # First order: a ripple a0 cos(2 pi k x) is two tilts of amplitude
    # i (2 pi/lambda)(a0/2), so each speckle is (pi a0/lambda)^2 times the
    # coronagraphic response to a point source at k lambda/D. That response
    # factors into the Lyot-stop peak (r_L/r_A)^4, the stopped PSF's loss at
    # the sample offset, and the vortex's off-axis throughput, all computed
    # here in NumPy. The residual few percent is the cross term with the
    # sampled vortex's own leakage.
    k = P["static_ripple_cycles_per_d"]
    image = d04.propagate(d04.static_coeffs(0.0))["intensity"]
    model, (u_s, v_s), _ = peak_near(image, k, 0.0, 2.0)
    first_order = (math.pi * P["static_ripple_nm"] / P["wavelength_nm"]) ** 2
    response = abs(numpy_vortex_response(u_s, v_s, k)) ** 2
    assert model == pytest.approx(first_order * response, rel=0.03)

    o = d04.optics()
    coords = o["pupil_coords"]
    stop_peak = (o["stop"].sum() / o["aperture"].sum()) ** 2
    offset = (
        abs(dft(o["stop"], coords, u_s - k, v_s)) ** 2
        / abs(dft(o["stop"], coords, 0.0, 0.0)) ** 2
    )
    no_vortex = abs(numpy_vortex_response(u_s, v_s, k, vortex=False)) ** 2
    throughput = response / no_vortex
    assert no_vortex == pytest.approx(stop_peak * offset, rel=1e-6)
    assert 0.85 < throughput < 0.97
    # The printed difference unit, 1e-5 of the unocculted peak.
    assert 0.5 < model / P["delta_unit"] < 2.0


def test_speckle_field_phase_is_plus_half_pi_under_the_coherent_profile():
    # exp(+i 2 pi W/lambda) linearizes to +i (2 pi/lambda) W, and the
    # vortex's exp(i 6 theta) is 1 on the x axis, so the speckle field has
    # phase +pi/2 at c = 0. Its sign is the convention-sensitive visual of
    # the animation's phase panel.
    k = P["static_ripple_cycles_per_d"]
    state = d04.propagate(d04.static_coeffs(0.0))
    for sign in (1.0, -1.0):
        _, _, index = peak_near(state["intensity"], sign * k, 0.0, 2.0)
        assert np.angle(state["image_field"][index]) == pytest.approx(
            math.pi / 2, abs=0.1
        )


def test_positive_opd_tilt_moves_the_image_toward_positive_x():
    # Through the same path the figures use: W = t x with t = 8 lambda puts
    # the source at +8 lambda/D, outside the vortex's inner working region.
    o = d04.optics()
    x = o["pupil_coords"][None, :] * np.ones((P["npup"], 1))
    state = d04.run(8.0 * P["wavelength_nm"] * x)
    _, (u, v), _ = peak_near(state["intensity"], 0.0, 0.0, radius_lod=20.0)
    assert u == pytest.approx(8.0, abs=P["pixscale_lod"])
    assert abs(v) <= P["pixscale_lod"]


def test_opposite_command_cancels_the_pair_and_the_change_is_negative():
    k = P["static_ripple_cycles_per_d"]
    nominal = d04.propagate(d04.static_coeffs(0.0))["intensity"]
    cancelled = d04.propagate(d04.static_coeffs(-1.0))["intensity"]
    for sign in (1.0, -1.0):
        _, _, index = peak_near(nominal, sign * k, 0.0, 2.0)
        assert cancelled[index] < 1e-2 * nominal[index]
        assert cancelled[index] - nominal[index] < 0.0


def test_reversed_ripple_restores_intensity_and_flips_the_field():
    # W -> -W flips the first-order field, so |E|^2 is unchanged except for
    # the cross term with the sampled vortex's small leakage, while the phase
    # at the speckle moves by pi. The animation caption states a 4 percent
    # bound, relative to the speckle peak, checked here.
    k = P["static_ripple_cycles_per_d"]
    nominal = d04.propagate(d04.static_coeffs(0.0))
    reversed_ = d04.propagate(d04.static_coeffs(-2.0))
    for sign in (1.0, -1.0):
        _, _, index = peak_near(nominal["intensity"], sign * k, 0.0, 2.0)
        assert reversed_["intensity"][index] == pytest.approx(
            nominal["intensity"][index], rel=0.03
        )
        turn = np.angle(reversed_["image_field"][index] / nominal["image_field"][index])
        assert abs(abs(turn) - math.pi) < 0.1
    change = reversed_["intensity"] - nominal["intensity"]
    assert np.max(np.abs(change)) < 0.04 * np.max(nominal["intensity"])


def test_perturbation_difference_signs_follow_the_command():
    # The still labels the old x pair "< 0" and the new y pair "> 0".
    k = P["static_ripple_cycles_per_d"]
    k2 = P["probe_ripple_cycles_per_d"]
    nominal = d04.propagate(d04.static_coeffs(0.0))["intensity"]
    perturbed = d04.propagate(d04.probe_coeffs())["intensity"]
    delta = perturbed - nominal
    for sign in (1.0, -1.0):
        _, _, index = peak_near(nominal, sign * k, 0.0, 2.0)
        assert delta[index] < 0.0
        _, _, index = peak_near(perturbed, 0.0, sign * k2, 2.0)
        assert delta[index] > 0.0


def test_thumbnails_render_with_origin_lower_and_no_transpose():
    # x right and y up: every card draws its array unchanged with
    # origin="lower". The DM card's x cosine would show a transpose.
    state = d04.propagate(d04.static_coeffs(0.0))
    expected = {
        "dm": d04.wide_opd(state),
        "lyot": np.abs(state["lyot_wide"]),
        "image": d04.floored(state["intensity"]),
        "detector": state["counts"],
    }
    with exporter.venue("light", ex.DOC) as cast:
        fig = plt.figure(figsize=(ex.DOC.width_in, 3.6))
        handles = d04.draw_train(fig, ex.DOC, cast, state, bottom_in=0.1, height_in=3.4)
        try:
            for key, data in expected.items():
                image = handles["cards"][key].image
                assert image.origin == "lower"
                drawn = np.ma.filled(image.get_array().astype(float), np.nan)
                assert drawn.shape == data.shape
                both = np.isfinite(data)
                assert np.allclose(drawn[both], data[both])
        finally:
            plt.close(fig)
    # The x cosine varies along columns: the drawn DM array must too.
    dm = expected["dm"]
    middle = dm.shape[0] // 2
    assert np.nanstd(dm[middle, :]) > 10 * np.nanstd(dm[:, middle])


def test_frame_schedules_fit_budgets_and_hit_the_named_states():
    for layout in (ex.DOC, ex.SLIDE):
        frames = d04.frame_schedule(layout)
        if layout.frame_budget is not None:
            assert len(frames) <= layout.frame_budget
        stops = [f["stop"] for f in frames]
        assert [s for i, s in enumerate(stops) if i == 0 or s != stops[i - 1]] == [
            *d04.TOUR_STOPS,
            "sweep",
        ]
        sweep = [f["c"] for f in frames if f["stop"] == "sweep"]
        assert sweep[0] == 0.0
        assert sweep[-1] == d04.SWEEP_END
        assert -1.0 in sweep
        assert all(b <= a for a, b in itertools.pairwise(sweep))


def test_readout_values_are_the_command_in_nanometers():
    # The panel readout prints c * a0 with two decimals.
    values = [f["c"] * P["static_ripple_nm"] for f in d04.frame_schedule(ex.DOC)]
    assert f"{min(values):+.2f}" == "-2.00"
    assert f"{max(values):+.2f}" == "+0.00"
