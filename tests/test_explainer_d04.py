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


# The light at every plane


@pytest.fixture(scope="module")
def light():
    state = d04.propagate(d04.static_coeffs(0.0))
    return state, d04.plane_fields(state)


@pytest.fixture(scope="module")
def light_figure():
    with exporter.venue("light", ex.DOC) as cast:
        fig = d04.build_light(ex.DOC, cast)
    yield fig
    plt.close(fig)


def axes_by_gid(fig):
    found = fig.findobj(matplotlib.axes.Axes)
    return {ax.get_gid(): ax for ax in found if ax.get_gid()}


def test_light_figure_is_a_new_slug_and_the_old_ones_stay():
    slugs = [spec.slug for spec in d04.FIGURES]
    assert slugs[:2] == ["d04-optical-planes", "d04-opd-perturbation"]
    assert "d04-light-every-plane" in slugs
    assert [spec.slug for spec in d04.ANIMATIONS] == ["d04-plane-tour"]
    # The existing specs keep the original parameter set.
    for spec in d04.FIGURES[:2]:
        assert spec.params is d04.PARAMS


def test_entrance_field_is_the_aperture_times_the_ripple_phase(light):
    # E = A exp(+i 2 pi W/lambda): the phase is at most 2 pi a0/lambda,
    # printed as 0.011 rad.
    state, fields = light
    o = d04.optics()
    expected = o["aperture"] * np.exp(2j * math.pi * state["opd"] / P["wavelength_nm"])
    assert np.allclose(state["entrance_field"], expected, atol=1e-12)
    peak = 2 * math.pi * 1.0 / 550.0
    assert f"{peak:.3f}" == "0.011"
    assert d04.entrance_phase_peak() == pytest.approx(peak)
    inside = o["aperture"] > 0
    phase = np.angle(state["entrance_field"][inside])
    assert np.max(np.abs(phase)) <= peak + 1e-12
    assert np.max(np.abs(phase)) > 0.99 * peak
    # The drawn entrance column is the same field padded to 2 D.
    n = P["npup"]
    lo = (fields["entrance"].shape[0] - n) // 2
    assert fields["entrance"].shape[0] == 2 * n
    assert np.allclose(fields["entrance"][lo : lo + n, lo : lo + n], expected)


def test_focal_field_is_the_continuous_transform_of_the_entrance_field(light):
    # F[E](u) = sum E exp(-i 2 pi u x) dx^2, relative to sqrt of the
    # unocculted peak, checked in NumPy at a few display samples.
    state, fields = light
    o = d04.optics()
    u = d04.fpm_coords()
    peak = (o["aperture"].sum() * (1.0 / P["npup"]) ** 2) ** 2
    for i, j in ((48, 48), (40, 60), (72, 30)):
        expected = dft(state["entrance_field"], o["pupil_coords"], u[j], u[i])
        assert fields["focal_before"][i, j] == pytest.approx(
            expected / math.sqrt(peak), rel=1e-9, abs=1e-12
        )


def test_display_grid_of_the_vortex_plane_matches_the_image_extent():
    # 96 samples at 0.25 lambda/D span the image grid's 24 lambda/D, so the
    # focal panels share one page scale; the ticks at -10, 0, 10 fit inside.
    u = d04.fpm_coords()
    assert np.diff(u) == pytest.approx(0.25)
    assert u.size == 96
    assert ep_extent(u) == pytest.approx(ep_extent(d04.optics()["focal_coords"]))
    assert ep_extent(u)[1] == pytest.approx(12.0)
    assert max(d04.LIGHT_TICKS["focal"]) < 12.0
    # The pupil panels span 2 D, from -1 to 1, the outer ticks.
    assert ep_extent(d04.optics()["wide_coords"])[1] == pytest.approx(1.0)
    assert max(d04.LIGHT_TICKS["pupil"]) == pytest.approx(1.0)


def ep_extent(coords):
    step = coords[1] - coords[0]
    return (coords[0] - step / 2, coords[-1] + step / 2)


def test_airy_rings_alternate_in_sign_before_the_vortex(light):
    # The caption says the field before the vortex is essentially the real
    # Airy pattern: phase 0 in the core and pi in the first bright ring,
    # between the zeros at 1.22 and 2.23 lambda/D (2 J1(pi r)/(pi r) < 0).
    from scipy.special import j1

    _, fields = light
    u = d04.fpm_coords()
    uu, vv = np.meshgrid(u, u)
    r = np.hypot(uu, vv)
    airy = 2 * j1(np.pi * r) / (np.pi * r)
    phase = np.angle(fields["focal_before"])
    core = r < 0.8
    ring = (r > 1.45) & (r < 2.0)
    assert np.all(airy[ring] < 0)
    assert np.all(np.abs(phase[core]) < 0.05)
    assert np.all(np.abs(np.abs(phase[ring]) - math.pi) < 0.05)


def test_vortex_changes_only_the_phase_by_six_theta(light):
    _, fields = light
    before, after = fields["focal_before"], fields["focal_after"]
    assert np.allclose(np.abs(after), np.abs(before), rtol=1e-12)
    u = d04.fpm_coords()
    uu, vv = np.meshgrid(u, u)
    turn = np.angle(after / before)
    expected = np.angle(np.exp(6j * np.arctan2(vv, uu)))
    assert np.allclose(np.angle(np.exp(1j * (turn - expected))), 0.0, atol=1e-9)
    # Six windings around a circle about the axis.
    angles = np.linspace(0.0, 2 * math.pi, 721)
    samples = [
        turn[
            int(np.argmin(np.abs(u - 5 * math.sin(a)))),
            int(np.argmin(np.abs(u - 5 * math.cos(a)))),
        ]
        for a in angles
    ]
    steps = np.angle(np.exp(1j * np.diff(samples)))
    assert round(steps.sum() / (2 * math.pi)) == 6


def test_stop_multiplies_and_the_image_is_the_transform_of_what_passes(light):
    state, fields = light
    o = d04.optics()
    stop = d04.pad_to_thumbnail(o["stop"])
    assert np.allclose(fields["lyot_after"], fields["lyot_before"] * stop)
    xw, yw = np.meshgrid(o["wide_coords"], o["wide_coords"])
    assert np.all(fields["lyot_after"][np.hypot(xw, yw) > 0.41] == 0)
    # The image field is the NumPy transform of the stopped Lyot field.
    peak = (o["aperture"].sum() * (1.0 / P["npup"]) ** 2) ** 2
    u = o["focal_coords"]
    _, _, (i, j) = peak_near(state["intensity"], 6.0, 0.0, 2.0)
    for a, b in ((i, j), (24, 24), (10, 35)):
        expected = dft(fields["lyot_after"], o["wide_coords"], u[b], u[a])
        assert fields["image"][a, b] == pytest.approx(
            expected / math.sqrt(peak), rel=1e-8, abs=1e-14
        )
    assert np.allclose(np.abs(fields["image"]) ** 2, state["intensity"], rtol=1e-9)


def test_energy_through_the_stop_matches_the_first_order_ripple_share(light):
    # Caption: about 4e-5 of the entrance energy passes; the ripple's share
    # is (2 pi a0/lambda)^2/2 = 6.5e-5 and the stop keeps 0.64 of the
    # aperture area; the leakage without the ripple is under 5 percent.
    _, fields = light
    o = d04.optics()
    entrance = np.sum(o["aperture"] ** 2)
    passed = np.sum(np.abs(fields["lyot_after"]) ** 2) / entrance
    share = (2 * math.pi * 1.0 / 550.0) ** 2 / 2
    area = (0.4 / 0.5) ** 2
    assert f"{share:.1e}" == "6.5e-05"
    assert f"{area:.2f}" == "0.64"
    assert f"{passed:.0e}" == "4e-05"
    assert 0.9 < passed / (share * area) < 1.0
    clean = d04.plane_fields(d04.propagate((0.0, 0.0)))
    leak = np.sum(np.abs(clean["lyot_after"]) ** 2) / entrance
    assert leak < 0.05 * passed


def test_amplitude_floor_matches_the_intensity_floor():
    # Amplitude 1e-5 is intensity 1e-10, the image scale's floor.
    assert d04.amplitude_floor() ** 2 == pytest.approx(P["intensity_floor"])
    assert d04.amplitude_floor() == pytest.approx(1e-5)


def test_light_panels_draw_the_named_quantities(light, light_figure):
    state, fields = light
    axes = axes_by_gid(light_figure)
    floor = d04.amplitude_floor()
    for key in d04.LIGHT_COLUMNS:
        (amp,) = axes[f"amplitude-{key}"].get_images()
        (phase,) = axes[f"phase-{key}"].get_images()
        for image in (amp, phase):
            assert image.get_interpolation() == "nearest"
            assert image.origin == "lower"
        drawn = np.ma.filled(amp.get_array().astype(float), np.nan)
        if key == "image":
            assert np.allclose(drawn, d04.floored(state["intensity"]))
            assert amp.norm.vmin == pytest.approx(1e-10)
            assert amp.norm.vmax == pytest.approx(1e-4)
        else:
            assert np.allclose(drawn, np.clip(np.abs(fields[key]), floor, None))
            assert amp.norm.vmin == pytest.approx(1e-5)
            assert amp.norm.vmax == pytest.approx(1.0)
        # Phase appears exactly where |E|^2 reaches the threshold.
        drawn_phase = np.ma.filled(phase.get_array().astype(float), np.nan)
        lit = np.abs(fields[key]) ** 2 >= P["phase_min_intensity"]
        assert np.array_equal(np.isfinite(drawn_phase), lit)
        assert np.allclose(drawn_phase[lit], np.angle(fields[key][lit]))
        assert phase.norm.vmin == pytest.approx(-math.pi)
        assert phase.norm.vmax == pytest.approx(math.pi)


def test_light_keys_print_the_scales_they_draw(light_figure):
    axes = axes_by_gid(light_figure)
    expected = {
        "key-amplitude": ([1e-4, 1e-2, 1.0], (1e-5, 1.0)),
        "key-phase": ([-math.pi, 0.0, math.pi], (-math.pi, math.pi)),
        "key-intensity": ([1e-10, 1e-7, 1e-4], (1e-10, 1e-4)),
    }
    for gid, (ticks, limits) in expected.items():
        ax = axes[gid]
        assert np.allclose(ax.get_yticks(), ticks)
        assert np.allclose(ax.get_ylim(), limits)


def test_light_figure_text_has_no_stroke_and_names_the_printed_numbers(light_figure):
    texts = [t for t in light_figure.findobj(matplotlib.text.Text) if t.get_text()]
    assert all(not t.get_path_effects() for t in texts)
    words = " ".join(t.get_text() for t in texts)
    assert "$\\leq$0.011" in words
    assert "e^{i6\\theta}" in words
    assert all(t.get_fontsize() >= 6.0 for t in texts)


def test_gap_labels_mark_the_inverse_transform_into_the_lyot_plane(light_figure):
    # The model reaches the Lyot plane with cmft_bwd, the inverse of the
    # clause's exp(-i 2 pi u x) transform; the other two gaps are forward.
    gaps = dict(zip(d04.LIGHT_COLUMNS[1:], d04.LIGHT_GAPS, strict=True))
    assert gaps["lyot_before"] == "$\\mathcal{F}^{-1}$"
    assert gaps["focal_before"] == gaps["image"] == "$\\mathcal{F}$"
    words = [t.get_text() for t in light_figure.findobj(matplotlib.text.Text)]
    assert words.count("$\\mathcal{F}^{-1}$") == 1
    assert words.count("$\\mathcal{F}$") == 2
    assert "FT" not in words


def test_vortex_operand_is_the_train_figures_mask_thumbnail(light_figure):
    (image,) = axes_by_gid(light_figure)["operand-vortex"].get_images()
    assert image.get_interpolation() == "nearest"
    assert image.origin == "lower"
    assert np.array_equal(image.get_array(), d04.optics()["mask_phase"])
    assert image.norm.vmin == pytest.approx(-math.pi)
    assert image.norm.vmax == pytest.approx(math.pi)


def test_train_keeps_the_element_names_and_starlight(light_figure):
    words = [t.get_text() for t in light_figure.findobj(matplotlib.text.Text)]
    for name in d04.LIGHT_ELEMENTS.values():
        assert name in words
    assert "starlight" in words
    assert "$|E|$ rel. incident" in words
    assert "$|E|$ rel. $\\sqrt{I_{\\rm peak}}$" in words
    assert "amplitude\n$|E|$" in words
    assert "phase\n[rad]" in words


def test_amplitude_map_is_monotone_in_lightness_in_both_modes():
    # Background at the floor, the pupil-role cyan in the middle, and a top
    # that continues the same direction of lightness.
    for mode in ("light", "dark"):
        with exporter.venue(mode, ex.DOC) as cast:
            cmap = d04.amplitude_cmap(cast)
            pupil = ex.image_cmap("pupil")
            lum = [
                float(np.dot(cmap(v)[:3], (0.2126, 0.7152, 0.0722)))
                for v in np.linspace(0, 1, 11)
            ]
            assert np.allclose(cmap(0.0), pupil(0.0))
            assert np.allclose(cmap(0.5)[:3], pupil(1.0)[:3], atol=0.01)
            steps = np.diff(lum)
            assert np.all(steps < 0) or np.all(steps > 0)


def test_notes_are_enlarged_and_the_bound_fits_its_panel(light_figure):
    small = ex.DOC.small_pt
    notes = [
        t
        for t in light_figure.findobj(matplotlib.text.Text)
        if t.get_fontstyle() == "italic"
        and t.get_text().startswith(("gray", "the", "solid"))
    ]
    assert len(notes) == 3
    assert all(t.get_fontsize() == pytest.approx(1.08 * small) for t in notes)
    axes = axes_by_gid(light_figure)
    panel = axes["phase-entrance"]
    (bound,) = [t for t in panel.texts if "0.011" in t.get_text()]
    # At the enlarged note size, inside its panel.
    assert bound.get_fontsize() == pytest.approx(1.08 * small)
    light_figure.canvas.draw()
    renderer = light_figure.canvas.get_renderer()
    box = bound.get_bbox_patch().get_window_extent(renderer)
    frame = panel.get_window_extent(renderer)
    assert box.x0 >= frame.x0 - 1.0
    assert box.x1 <= frame.x1 + 1.0
