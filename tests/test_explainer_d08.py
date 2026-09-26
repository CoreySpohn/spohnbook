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


# The three-piece relay: boundary, record under a reporting law, campaign loop.

# The three relay slugs, their builders and the short names used below.
SLUGS = {
    "boundary": ("d08-information-boundary", d08.build_boundary),
    "reporting": ("d08-record-reporting-law", d08.build_reporting_law),
    "campaign": ("d08-campaign-timeline", d08.build_campaign),
}
PIECES = {name: build for name, (_, build) in SLUGS.items()}


def test_split_slugs_are_registered_with_their_builders():
    specs = {spec.slug: spec for spec in d08.FIGURES}
    for slug, build in SLUGS.values():
        assert specs[slug].build is build
        assert specs[slug].venues == ("doc", "slide")
    # The overview and the forecast keep their slugs.
    assert "d08-experiment-record-model" in specs
    assert "d08-fit-forecast-timeline" in specs


def _ends(patch):
    """Tail and head of an arrow patch, in data coordinates."""
    (x0, y0), (x1, y1) = patch._posA_posB
    return (x0, y0), (x1, y1)


def _collect(fig):
    ax = fig.axes[0]
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    return {
        "texts": {t.get_gid(): t.get_text() for t in texts if t.get_gid()},
        "text_pos": {t.get_gid(): t.get_position() for t in texts if t.get_gid()},
        "all_text": [t.get_text() for t in texts],
        "patches": {p.get_gid(): p for p in ax.patches if p.get_gid()},
        "lines": {ln.get_gid(): ln for ln in ax.lines if ln.get_gid()},
        "arrows": [
            p for p in ax.patches if isinstance(p, matplotlib.patches.FancyArrowPatch)
        ],
        "images": sorted(ax.get_images(), key=lambda im: -im.get_extent()[2]),
        "ax": ax,
    }


@pytest.fixture(scope="module")
def pieces():
    out = {}
    with exporter.venue("light", ex.DOC) as cast:
        for name, build in PIECES.items():
            fig = build(ex.DOC, cast)
            fig.canvas.draw()
            out[name] = _collect(fig)
            out[name]["fig"] = fig
    yield out
    for piece in out.values():
        plt.close(piece["fig"])


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
@pytest.mark.parametrize("name", list(PIECES), ids=[s for s, _ in SLUGS.values()])
def test_split_labels_carry_no_stroked_halo(name, layout, mode):
    with exporter.venue(mode, layout) as cast:
        fig = PIECES[name](layout, cast)
        stroked = [
            t.get_text()
            for t in fig.findobj(matplotlib.text.Text)
            if t.get_path_effects()
        ]
        size = tuple(fig.get_size_inches())
        plt.close(fig)
    assert stroked == []
    assert size[0] == pytest.approx(layout.width_in)


def test_boundary_only_the_record_crosses_into_the_model_zone(pieces):
    # The model and policy zone lies right of the second divider and above
    # the simulation-only evaluation band.
    p = pieces["boundary"]
    in_zone = lambda xy: xy[0] > 10.4 and xy[1] > 3.25  # noqa: E731
    blocked = {p["patches"]["blocked-truth"], p["patches"]["blocked-scores"]}
    entering = [
        a for a in p["arrows"] if in_zone(_ends(a)[1]) and not in_zone(_ends(a)[0])
    ]
    allowed = [a for a in entering if a not in blocked]
    assert len(allowed) == 1
    glyph = p["patches"]["record-glyph"]
    tail = _ends(allowed[0])[0]
    right_edge = glyph.get_x() + glyph.get_width()
    assert tail[0] == pytest.approx(right_edge, abs=0.1)
    assert glyph.get_y() <= tail[1] <= glyph.get_y() + glyph.get_height()
    # The scores arrow is the one crossed arrow into the zone; the truth arrow
    # is cut where it would leave the simulated world.
    assert p["patches"]["blocked-scores"] in entering
    # The crossed truth arrow aims at the inference model, its head across
    # the zone line, and is struck through just after it leaves the world.
    (tx, _), (hx, hy) = _ends(p["patches"]["blocked-truth"])
    assert tx < 4.6
    box = p["patches"]["inference-box"]
    assert hx > 10.4
    assert hx == pytest.approx(box.get_x(), abs=0.2)
    assert box.get_y() <= hy <= box.get_y() + box.get_height()
    (cx,) = p["lines"]["blocked-truth-cross"].get_xdata()
    assert 4.6 < cx < 5.5


def test_boundary_truth_reaches_only_detector_and_evaluation(pieces):
    p = pieces["boundary"]
    truth = p["patches"]["truth-record"]
    noise = p["patches"]["noise-draw"]
    lane = p["patches"]["truth-lane"]
    evaluation = p["patches"]["evaluation-box"]
    right = truth.get_x() + truth.get_width()
    for arrow in (noise, lane):
        (x0, _), _ = _ends(arrow)
        assert x0 == pytest.approx(right, abs=0.15)
    _, (nx, ny) = _ends(noise)
    assert 6.1 <= nx <= 6.36 and ny < 5.2  # the detector's lower edge
    _, (lx, _) = _ends(lane)
    assert lx == pytest.approx(evaluation.get_x(), abs=0.15)
    joined = " ".join(p["all_text"])
    assert "no path: truth never" in joined
    assert "no path:\nscores" in joined
    divider = p["texts"]["evaluation-divider-label"]
    assert divider.replace("\n", "") == "below: simulation-only scoring"


def test_record_glyph_uses_the_chapter_symbols(pieces):
    joined = " ".join(pieces["boundary"]["all_text"])
    for symbol in (r"$R_k$", r"$t_k$", r"$D_k$", r"$(\xi_k, \eta_k)$", r"$C_k$"):
        assert symbol in joined
    assert "if reported:" in joined
    assert "when reported" in d08.BOUNDARY_CAPTION


def _inside(patch, texts):
    x0, y0 = patch.get_x(), patch.get_y()
    x1, y1 = x0 + patch.get_width(), y0 + patch.get_height()
    return sorted(
        t.get_text()
        for t in texts
        if x0 <= t.get_position()[0] <= x1 and y0 <= t.get_position()[1] <= y1
    )


def test_carried_elements_are_drawn_identically(pieces):
    # The carried panel of a relay is the same constructor at the same size.
    def card(name, gid):
        p = pieces[name]
        patch = p["patches"][gid]
        texts = p["ax"].texts
        return (patch.get_width(), patch.get_height(), _inside(patch, texts))

    boundary, reporting = (
        card("boundary", "record-glyph"),
        card("reporting", "record-glyph"),
    )
    assert boundary[:2] == reporting[:2]
    # The same card, with the example's fields swapped in.
    assert set(boundary[2]) - set(reporting[2]) == {r"$(\xi_k, \eta_k)$,  $C_k$"}
    assert set(reporting[2]) - set(boundary[2]) == {r"$F_k$,  $\sigma^2$"}
    swap = pieces["reporting"]["texts"]["swap-note"].replace("\n", " ")
    assert "flux F in place of the offsets" in swap
    assert "in place of $C_k$" in swap
    assert card("boundary", "policy-box") == card("campaign", "policy-box")
    for name in ("reporting", "campaign"):
        assert "information-" in pieces[name]["texts"]["carried-tag"]
        assert "boundary figure" in pieces[name]["texts"]["carried-tag"]


@pytest.fixture(scope="module")
def reporting(pieces):
    p = pieces["reporting"]
    ax = p["ax"]
    fluxes = {}
    for k in (1, 2):
        (marker,) = [ln for ln in ax.lines if ln.get_gid() == f"flux-visit{k}"]
        fluxes[k] = marker
    bars = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
    ]
    return {
        **p,
        "pixels": [im.get_array().data.copy() for im in p["images"]],
        "markers": fluxes,
        "bars": bars,
    }


def _page_to_flux(reporting):
    # The axis scale read back from the drawn 0 and 100 tick labels.
    x0 = reporting["text_pos"]["f-tick-0"][0]
    x100 = reporting["text_pos"]["f-tick-100"][0]
    assert reporting["texts"]["f-tick-0"] == "0"
    assert reporting["texts"]["f-tick-100"] == "100"
    return lambda x: 100.0 * (x - x0) / (x100 - x0)


def test_reporting_markers_sit_at_the_hand_computed_flux(reporting):
    to_flux = _page_to_flux(reporting)
    assert len(reporting["pixels"]) == 2
    for k, pixels in enumerate(reporting["pixels"], start=1):
        flux = _aperture_sum(pixels)
        (x,) = reporting["markers"][k].get_xdata()
        assert to_flux(x) == pytest.approx(flux, abs=1e-9)
        label = reporting["texts"][f"flux-label-visit{k}"]
        assert f"$F_{k}$" in label
        assert _number(label.split("=")[1]) == pytest.approx(round(flux, 1))
    f1 = _aperture_sum(reporting["pixels"][0])
    f2 = _aperture_sum(reporting["pixels"][1])
    assert f1 > 30.0 >= f2


def test_reporting_threshold_and_sigma_are_drawn_to_scale(reporting):
    to_flux = _page_to_flux(reporting)
    line = reporting["lines"]["threshold-line"]
    xs = set(line.get_xdata())
    assert len(xs) == 1
    assert to_flux(xs.pop()) == pytest.approx(30.0)
    assert _number(reporting["texts"]["threshold-label"].split("=")[1]) == 30.0
    # Every error bar spans F -/+ sigma, with sigma = 2 e- times sqrt(9).
    sigma = 2.0 * math.sqrt(9)
    assert _number(reporting["texts"]["sigma-label"].split("=")[1]) == pytest.approx(
        sigma
    )
    spans = []
    for coll in reporting["bars"]:
        for seg in coll.get_segments():
            (xa, ya), (xb, yb) = seg
            if ya == pytest.approx(yb) and xa != xb:
                spans.append(abs(to_flux(xb) - to_flux(xa)))
    assert len(spans) == 2
    assert spans == pytest.approx([2.0 * sigma, 2.0 * sigma])
    # The detection is filled and solid; the visit 2 flux, computed but not
    # released under report-only-on-detection, is hollow and dashed.
    solid, hollow = reporting["markers"][1], reporting["markers"][2]
    assert matplotlib.colors.same_color(solid.get_markerfacecolor(), solid.get_color())
    assert not matplotlib.colors.same_color(
        hollow.get_markerfacecolor(), hollow.get_markeredgecolor()
    )
    styles = [c.get_linestyle() for c in reporting["bars"]]
    assert sum(ls[0][1] is not None for ls in styles) == 1
    assert "released only" in reporting["texts"]["flux-note-visit2"]
    assert "only forced photometry releases it" in d08.REPORTING_CAPTION
    # The detection lies right of the threshold, the nondetection left of it.
    x_l = line.get_xdata()[0]
    assert reporting["markers"][1].get_xdata()[0] > x_l
    assert reporting["markers"][2].get_xdata()[0] < x_l


def test_reporting_cards_follow_each_law(reporting):
    texts = reporting["texts"]
    f1 = _aperture_sum(reporting["pixels"][0])
    f2 = _aperture_sum(reporting["pixels"][1])
    assert texts["record-visit1-D"].startswith("1")
    assert _number(texts["record-visit1-F"]) == pytest.approx(round(f1, 1))
    assert texts["record-visit2-D"].startswith("0")
    assert texts["record-visit2-F"] == "not reported"
    assert texts["record-forced-D"].startswith("0")
    assert _number(texts["record-forced-F"]) == pytest.approx(round(f2, 1))
    assert _number(texts["record-forced-F"]) < 30.0
    assert "report only" in texts["law-report-on-detection"]
    assert "forced photometry" in texts["law-forced"]
    assert "different experiment" in texts["law-forced"]
    # Subscripts on the cards match the axis labels.
    names = set(reporting["all_text"])
    assert {"$D_1$", "$F_1$", "$D_2$", "$F_2$"} <= names
    # Both law headers sit above the wide visit 1 card, over its two halves.
    patches, pos = reporting["patches"], reporting["text_pos"]
    wide = patches["card-visit1"]
    top = wide.get_y() + wide.get_height()
    for gid, card in (
        ("law-report-on-detection", "card-visit2"),
        ("law-forced", "card-forced"),
    ):
        x, y = pos[gid]
        assert y > top
        c = patches[card]
        assert c.get_x() < x < c.get_x() + c.get_width()
        assert wide.get_x() < x < wide.get_x() + wide.get_width()
    # The carried card is linked to each record card it describes.
    glyph = patches["record-glyph"]
    for key in ("visit1", "visit2", "forced"):
        line = reporting["lines"][f"record-link-{key}"]
        ys = list(line.get_ydata())
        assert ys[0] in (glyph.get_y(), glyph.get_y() + glyph.get_height())
        c = patches[f"card-{key}"]
        assert ys[-1] == pytest.approx(
            c.get_y() + (c.get_height() if key != "visit1" else 0.0)
        )
    # The forced-photometry card is the dashed one.
    dashed = [
        p
        for p in reporting["ax"].patches
        if isinstance(p, matplotlib.patches.Rectangle)
        and p.get_linestyle() in ("--", "dashed")
    ]
    assert len(dashed) == 1


def test_reporting_captions_quote_the_printed_numbers(reporting):
    f1 = round(_aperture_sum(reporting["pixels"][0]), 1)
    f2 = round(_aperture_sum(reporting["pixels"][1]), 1)
    caption, alt = d08.REPORTING_CAPTION, d08.REPORTING_ALT
    assert f"$F$ = {f1:.1f} e" in caption
    assert f"$F$ = {f2:.1f} e" in caption
    assert "$\\sigma$ = 6.0" in caption
    assert "$L$ = 30" in caption
    assert "background of 20" in caption and "noise of 2" in caption
    assert "sums the 9 dotted pixels" in caption
    assert f"F sub 1 = {f1:.1f} electrons" in alt
    assert f"F sub 2 = {f2:.1f} electrons" in alt
    assert "sigma = 6.0 electrons" in alt
    assert "L = 30 electrons" in alt


def test_campaign_events_follow_the_chapter_order(pieces):
    # Admission, acquisition, completion, release, inference, then policy.
    p = pieces["campaign"]
    patches, lines = p["patches"], p["lines"]

    def x_of_line(gid):
        (x,) = set(lines[gid].get_xdata())
        return x

    reserved = patches["reserved-k"]
    exposure = patches["exposure-k"]
    r0, r1 = reserved.get_x(), reserved.get_x() + reserved.get_width()
    e0, e1 = exposure.get_x(), exposure.get_x() + exposure.get_width()
    # Occupied facility time and science exposure are different quantities.
    assert r0 < e0 < e1 < r1
    order = [
        x_of_line("event-decide-k"),
        r0,
        e0,
        e1,
        x_of_line("event-complete-k"),
        patches["event-release-k"].get_x(),
        x_of_line("event-infer-k"),
        x_of_line("event-decide-k1"),
        patches["reserved-k1"].get_x(),
    ]
    assert order == sorted(order)
    # Actual cost and the reserved interval are distinct: completion falls
    # after the exposure and strictly before the reservation ends.
    assert e1 < x_of_line("event-complete-k") < r1
    revise = patches["event-revise-k"].get_x()
    assert revise > x_of_line("event-decide-k1")
    assert x_of_line("event-infer-later") > revise


def test_campaign_arrows_run_forward_in_time_except_the_crossed_one(pieces):
    p = pieces["campaign"]
    causal = {gid: a for gid, a in p["patches"].items() if gid.startswith("causal-")}
    assert len(causal) == 7
    for gid, arrow in causal.items():
        (x0, _), (x1, _) = _ends(arrow)
        assert x1 >= x0, gid
    # Release happens after completion: its arrow leaves at or after the tick.
    (tail_x, _), _ = _ends(causal["causal-release-k"])
    complete = p["lines"]["event-complete-k"].get_xdata()[0]
    assert tail_x >= complete
    # A dotted marker at the decision for visit k+1 separates before from after.
    marker = p["lines"]["time-marker-decide-k1"]
    assert set(marker.get_xdata()) == {p["lines"]["event-decide-k1"].get_xdata()[0]}
    assert "may use only records already released" in d08.CAMPAIGN_CAPTION
    assert "only then" not in d08.CAMPAIGN_CAPTION
    blocked = p["patches"]["blocked-revision"]
    (bx0, by0), (bx1, by1) = _ends(blocked)
    assert bx1 < bx0 and by1 > by0
    assert bx0 > marker.get_xdata()[0] > bx1 - 0.3
    # It points from the revised product back at the decision for visit k+1.
    decide = p["lines"]["event-decide-k1"].get_xdata()[0]
    assert bx1 == pytest.approx(decide, abs=0.3)
    assert "no path back" in " ".join(p["all_text"])
