"""Numbers printed on the experiment, record and model explainer diagrams.

The record cards print values measured from the drawn pixels. These tests
read the pixels and the text back from the rendered figure and recompute
every printed number by hand arithmetic: the aperture is the 3 by 3 block
of pixels around the center (the only pixel centers within 1.5 pixels), the
flux is the background-subtracted sum over it, and its standard deviation is
the per-pixel noise times the square root of 9. The forecast figure must
use the tutorial's own epochs, noise level and reference day, and must keep
every forecast epoch after the last acquired epoch.
"""

import math
import re
import sys
from pathlib import Path

import hwostyle
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from explainers import _common as ex  # noqa: E402
from explainers import _export as exporter  # noqa: E402
from explainers import d08_experiment_record_model as d08  # noqa: E402

TUTORIAL = ROOT / "docs" / "examples" / "fit-and-forecast.md"


def _number(text):
    """The first decimal number in a label."""
    return float(re.search(r"-?\d+(\.\d+)?", text).group())


@pytest.fixture(scope="module")
def overview():
    with exporter.venue("light", ex.DOC) as cast:
        fig = d08.build_overview(ex.DOC, cast)
        ax = fig.axes[0]
        images = sorted(ax.get_images(), key=lambda im: -im.get_extent()[2])
        texts = {t.get_gid(): t for t in ax.texts if t.get_gid()}
        patches = {p.get_gid(): p for p in ax.patches if p.get_gid()}
        out = {
            "arrays": [im.get_array().data.copy() for im in images],
            "extents": [im.get_extent() for im in images],
            "clims": [im.get_clim() for im in images],
            "interps": [im.get_interpolation() for im in images],
            "cmaps": [im.get_cmap().name for im in images],
            "texts": {k: t.get_text() for k, t in texts.items()},
            "colors": {k: t.get_color() for k, t in texts.items()},
            "outlines": {
                k: (p.get_x(), p.get_y(), p.get_width(), p.get_height())
                for k, p in patches.items()
                if k.startswith("aperture-outline")
            },
            "all_text": [t.get_text() for t in ax.texts],
            "flows": [k for k in patches if k in ("noise-draw", "truth-lane")],
        }
        plt.close(fig)
    return out


def _aperture_sum(pixels):
    # Pixel centers lie on integer offsets from the central pixel (4, 4) of
    # the 9 by 9 cutout. Offsets with |dx|, |dy| <= 1 are at most sqrt(2) =
    # 1.414 from the center; the next nearest, (2, 0), is at 2 > 1.5.
    block = [pixels[4 + dy][4 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    assert len(block) == 9
    return sum(v - 20.0 for v in block)


def test_two_raw_cutouts_on_one_pinned_readout_scale(overview):
    assert len(overview["arrays"]) == 2
    assert all(a.shape == (9, 9) for a in overview["arrays"])
    assert overview["clims"][0] == overview["clims"][1]
    assert overview["interps"] == ["nearest", "nearest"]
    assert overview["cmaps"] == ["magma", "magma"]


def test_aperture_outline_is_the_summed_three_by_three_block(overview):
    for k, (x0, x1, y0, y1) in enumerate(overview["extents"], start=1):
        pitch = (x1 - x0) / 9.0
        ox, oy, w, h = overview["outlines"][f"aperture-outline-visit{k}"]
        assert w == pytest.approx(3.0 * pitch)
        assert h == pytest.approx(3.0 * pitch)
        assert ox + 0.5 * w == pytest.approx(0.5 * (x0 + x1))
        assert oy + 0.5 * h == pytest.approx(0.5 * (y0 + y1))


def _settings_are_fixed(texts, colors, key):
    assert _number(texts[f"record-{key}-L"]) == pytest.approx(30.0)
    assert _number(texts[f"record-{key}-sigma"]) == pytest.approx(2.0 * math.sqrt(9))
    # Settings are drawn apart from the measured fields.
    assert colors[f"record-{key}-L"] != colors[f"record-{key}-D"]
    assert colors[f"record-{key}-sigma"] != colors[f"record-{key}-F"]


def test_detection_card_matches_drawn_pixels(overview):
    texts, colors = overview["texts"], overview["colors"]
    flux = _aperture_sum(overview["arrays"][0])
    assert flux > 30.0
    assert texts["record-visit1-D"].startswith("1")
    assert _number(texts["record-visit1-F"]) == pytest.approx(round(flux, 1))
    _settings_are_fixed(texts, colors, "visit1")


def test_nondetection_card_reports_the_event_not_a_flux(overview):
    texts, colors = overview["texts"], overview["colors"]
    flux = _aperture_sum(overview["arrays"][1])
    assert flux <= 30.0
    assert texts["record-visit2-D"].startswith("0")
    assert texts["record-visit2-F"] == "not reported"
    _settings_are_fixed(texts, colors, "visit2")


def test_forced_photometry_is_an_alternative_below_threshold(overview):
    texts, colors = overview["texts"], overview["colors"]
    flux = _aperture_sum(overview["arrays"][1])
    assert texts["record-forced-D"].startswith("0")
    printed = _number(texts["record-forced-F"])
    assert printed == pytest.approx(round(flux, 1))
    assert printed < 30.0
    _settings_are_fixed(texts, colors, "forced")
    assert any("replaces visit 2 record" in t for t in overview["all_text"])


def test_boundary_is_drawn(overview):
    joined = " ".join(overview["all_text"])
    assert "truth or scores" in joined and "inference" in joined
    assert "simulation-only scoring" in joined
    assert "a decision rule" in joined
    assert sorted(overview["flows"]) == ["noise-draw", "truth-lane"]


def test_captions_quote_the_printed_numbers(overview):
    forced = _number(overview["texts"]["record-forced-F"])
    detected = _number(overview["texts"]["record-visit1-F"])
    caption, alt = d08.OVERVIEW_CAPTION, d08.OVERVIEW_ALT
    assert f"F = {forced:.1f}" in caption
    assert "background of 20" in caption
    assert "noise of 2" in caption
    assert "sums the 9 dotted pixels" in caption
    assert "sigma = 6.0" in caption
    assert "the noise draw" in caption
    assert "does not assert" in caption
    assert f"F = {detected:.1f} electrons" in alt
    assert f"F = {forced:.1f} electrons" in alt


def _tutorial_value(pattern):
    match = re.search(pattern, TUTORIAL.read_text())
    assert match, pattern
    return match.group(1)


def _floats(text):
    return tuple(float(v) for v in re.findall(r"-?\d+\.?\d*", text))


def test_forecast_setup_is_the_tutorials():
    assert (
        _floats(_tutorial_value(r"t_obs = jnp\.array\(\[([^\]]*)\]\)")) == d08.T_OBS_D
    )
    assert (
        _floats(_tutorial_value(r"t_future = jnp\.array\(\[([^\]]*)\]\)"))
        == d08.T_FUTURE_D
    )
    assert float(_tutorial_value(r"sigma_mas = ([\d.]+)")) == d08.SIGMA_MAS
    assert float(_tutorial_value(r"t_ref_jd = ([\d.]+)")) == d08.T_REF_JD
    assert int(_tutorial_value(r"PRNGKey\((\d+)\)")) == d08.TUTORIAL_SEED
    orbit = _floats(_tutorial_value(r"T_d, e, i_deg, W_deg, w_deg = ([^\n]*)"))
    assert orbit == (
        d08.ORBIT["period_d"],
        d08.ORBIT["eccentricity"],
        d08.ORBIT["inclination_deg"],
        d08.ORBIT["node_deg"],
        d08.ORBIT["arg_periastron_deg"],
    )
    assert (
        float(_tutorial_value(r"tp_jd, Ms_kg, dist_pc = ([\d.]+)"))
        == d08.ORBIT["periastron_jd"]
    )
    assert float(_tutorial_value(r"Msun2kg, ([\d.]+)")) == d08.ORBIT["distance_pc"]
    assert min(d08.T_FUTURE_D) > max(d08.T_OBS_D)


@pytest.fixture(scope="module")
def forecast():
    data = d08.forecast_data()
    with exporter.venue("light", ex.DOC) as cast:
        fig = d08.build_forecast(ex.DOC, cast)
        fig.canvas.draw()
        axes = fig.axes[:2]
        out = {
            "data": data,
            "ticks": [
                float(t.get_text().replace("+", "")) for t in axes[1].get_xticklabels()
            ],
            "xlabel": axes[1].get_xlabel(),
            "headers": {
                t.get_gid(): (t.get_position()[0], t.get_text())
                for t in axes[0].texts
                if t.get_gid()
            },
            "gids": {},
            "measured": hwostyle.roles.measured,
            "truth": cast["scenery"].color,
        }
        for ax in axes:
            for artist in [*ax.lines, *ax.patches, *ax.collections]:
                if artist.get_gid():
                    out["gids"][artist.get_gid()] = artist
        plt.close(fig)
    return out


def test_forecast_figure_prints_the_tutorial_epochs(forecast):
    assert tuple(forecast["ticks"]) == (*d08.T_OBS_D, *d08.T_FUTURE_D)
    assert forecast["xlabel"] == f"Days from reference epoch (JD {d08.T_REF_JD:.0f})"
    assert (
        forecast["headers"]["header-acquired"][1]
        == "acquired: 5 epochs, 5 mas per axis"
    )
    assert forecast["headers"]["header-forecast"][1] == "forecast: no data yet"


def test_acquired_and_forecast_sides_are_separated(forecast):
    t_last = max(d08.T_OBS_D)
    gids = forecast["gids"]
    assert forecast["headers"]["header-acquired"][0] < t_last
    assert forecast["headers"]["header-forecast"][0] > t_last
    for k in (0, 1):
        # axvspan returns a rectangle in data x and axes y.
        shade = gids[f"forecast-shade-{k}"]
        assert shade.get_x() == pytest.approx(t_last)
        assert shade.get_x() + shade.get_width() > max(d08.T_FUTURE_D)
        assert tuple(gids[f"last-acquired-{k}"].get_xdata()) == (t_last, t_last)
        acquired = gids[f"acquired-{k}"]
        truth = gids[f"truth-future-{k}"]
        assert tuple(acquired.get_xdata()) == d08.T_OBS_D
        assert tuple(truth.get_xdata()) == d08.T_FUTURE_D
        assert max(acquired.get_xdata()) <= t_last < min(truth.get_xdata())
        assert matplotlib.colors.same_color(acquired.get_color(), forecast["measured"])
        assert matplotlib.colors.same_color(truth.get_color(), forecast["truth"])


def test_forecast_intervals_are_the_drawn_percentiles(forecast):
    future = forecast["data"]["future"]
    for k in (0, 1):
        for j, epoch in enumerate(d08.T_FUTURE_D):
            line = forecast["gids"][f"forecast-interval-{k}-{j}"]
            values = sorted(future[k][:, j])
            n = len(values)
            assert n == 2000
            lo, hi = line.get_ydata()
            # Linear-interpolation percentile by hand: rank p (n - 1).
            for p, drawn in ((0.05, lo), (0.95, hi)):
                r = p * (n - 1)
                i = int(r)
                expect = values[i] + (r - i) * (values[i + 1] - values[i])
                assert drawn == pytest.approx(expect)
            assert tuple(line.get_xdata()) == (epoch, epoch)


def test_caption_claims_about_the_forecasts(forecast):
    future = forecast["data"]["future"]
    truth = forecast["data"]["truth_future"]
    outside = []
    for k in (0, 1):
        for j in range(3):
            v = sorted(future[k][:, j])
            lo, hi = v[int(0.05 * 1999)], v[int(0.95 * 1999) + 1]
            if not lo <= truth[k][j] <= hi:
                outside.append((k, j, truth[k][j] > hi))
    # Only the north offset at day 400, above its interval.
    assert outside == [(1, 2, True)]
    assert "one of the six forecast truths" in d08.FORECAST_CAPTION
    assert "north truth at day 400 lies just above" in d08.FORECAST_ALT
    # The day 400 east forecast is bimodal: few draws near the median.
    east = future[0][:, 2]
    near = lambda c: int(sum(abs(x - c) < 15.0 for x in east))  # noqa: E731
    median = sorted(east)[1000]
    peak = max(near(c) for c in range(-150, 160, 5))
    assert near(median) < 0.5 * peak
    assert "bimodal" in d08.FORECAST_CAPTION
