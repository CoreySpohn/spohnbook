"""Numbers on the detector acquisition explainer, checked by hand.

Every number the figures print is recomputed here from the radiometry
chapter's stated primitives with plain arithmetic, independently of the
diagram module's own budget code and of the production libraries, and the
primitives themselves are read from the chapter text so the figure cannot
drift from the page it illustrates.
"""

import math
import re
import subprocess
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from explainers import _common as ex
from explainers import _export as exporter
from explainers import d07_detector_acquisition as d07

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = (ROOT / "docs/conventions/radiometry-detectors.md").read_text()
EXAMPLE = CHAPTER.split("(radiometry-count-example)=")[1].split(
    "(radiometry-etc-forecast)="
)[0]


def _num(pattern, text=EXAMPLE):
    match = re.search(pattern, text)
    assert match, f"the chapter no longer states {pattern!r}"
    return float(match.group(1))


# The chapter's primitives and stated results, read from its text.
PLANET = _num(r"planet/background rates of (\d+)/")
BACKGROUND = _num(r"planet/background rates of \d+/(\d+) photon/s")
QE = _num(r"constant QE (\.?\d+)")
LIVE = _num(r"Observe for (\d+) live seconds")
FRAME = _num(r"ten (\d+)-second frames")
N_FRAMES = 10  # "ten", checked in test_chapter_primitives
DARK = _num(r"dark current (\.?\d+) electron/s")
CIC = _num(r"CIC (\.?\d+) electron/frame")
READ = _num(r"read noise (\d+) electron RMS/read")
READ_TIME = _num(r"followed by a (\.?\d+)-second nonoverlapping read")
N_PIX = 4  # "four pixels", checked in test_chapter_primitives


def _texts(fig):
    """Every text string drawn in a figure, math markup included."""
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text)]


def _render(build, layout=ex.DOC, mode="light"):
    with exporter.venue(mode, layout) as cast:
        fig = build(layout, cast)
        fig.canvas.draw()
        texts = _texts(fig)
        plt.close(fig)
    return texts


def test_chapter_primitives_match_the_module():
    assert "An aperture contains four pixels" in EXAMPLE
    assert "in ten 10-second frames" in EXAMPLE
    assert N_FRAMES * FRAME == LIVE
    c = d07.COUNT
    assert c["n_pixels"] == N_PIX
    assert c["planet_photon_rate_per_s"] == PLANET
    assert c["background_photon_rate_per_s"] == BACKGROUND
    assert c["qe"] == QE
    assert c["n_frames"] == N_FRAMES
    assert c["frame_time_s"] == FRAME
    assert c["read_time_s"] == READ_TIME
    assert c["dark_e_per_pixel_s"] == DARK
    assert c["cic_e_per_pixel_frame"] == CIC
    assert c["read_noise_e_rms_per_pixel_read"] == READ


# The hand budget, row by row: name, scaling category, mean, variance.
# E = mu_e + d t_live + c n_f and Var = E + sigma_r^2 n_f, per pixel, summed
# over the four aperture pixels (the chapter's formula).
HAND_ROWS = [
    ("planet photoelectrons", "live time", PLANET * QE * LIVE, PLANET * QE * LIVE),
    (
        "background photoelectrons",
        "live time",
        BACKGROUND * QE * LIVE,
        BACKGROUND * QE * LIVE,
    ),
    ("dark charge", "live time", N_PIX * DARK * LIVE, N_PIX * DARK * LIVE),
    (
        "clock-induced charge (CIC)",
        "frames",
        N_PIX * CIC * N_FRAMES,
        N_PIX * CIC * N_FRAMES,
    ),
    ("read noise", "reads", 0.0, N_PIX * READ**2 * N_FRAMES),
]


def _render_fig(build, layout, mode):
    """Render a still and return its figure (the caller closes it)."""
    with exporter.venue(mode, layout) as cast:
        fig = build(layout, cast)
        fig.canvas.draw()
    return fig


def test_budget_by_hand_matches_the_chapter_and_the_module():
    values = [(m, v) for _, _, m, v in HAND_ROWS]
    assert values == [(1000, 1000), (4000, 4000), (4, 4), (0.8, 0.8), (0, 160)]
    assert [g for _, g, _, _ in HAND_ROWS] == ["live time"] * 3 + ["frames", "reads"]
    mean = sum(m for _, _, m, _ in HAND_ROWS)
    var = sum(v for _, _, _, v in HAND_ROWS)
    # The chapter's own stated results.
    assert mean == pytest.approx(_num(r"raw mean is ([\d.]+) electrons"))
    assert var == pytest.approx(_num(r"variance is ([\d.]+) electron"))
    planet = HAND_ROWS[0][2]
    snr = planet / math.sqrt(var)
    assert snr == pytest.approx(_num(r"= ?(13\.\d+)"), abs=5e-5)
    elapsed = N_FRAMES * (FRAME + READ_TIME)
    assert elapsed == pytest.approx(_num(r"occupies ([\d.]+) seconds"))
    # The module's rows, in full: name, category, mean and variance.
    b = d07.count_budget()
    assert len(b["rows"]) == len(HAND_ROWS)
    for (name, grows, _calc, m, v), (h_name, h_grows, h_m, h_v) in zip(
        b["rows"], HAND_ROWS, strict=True
    ):
        assert (name, grows) == (h_name, h_grows)
        assert (m, v) == pytest.approx((h_m, h_v))
    assert b["rows"][-1][2] == "4 px x 2$^2$ x 10 reads"
    assert (b["mean"], b["variance"], b["snr"]) == pytest.approx((mean, var, snr))
    assert (b["live_s"], b["elapsed_s"]) == pytest.approx((LIVE, elapsed))


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_schedule_table_rows_carry_their_own_values(layout):
    fig = _render_fig(
        d07.build_acquisition, layout, "dark" if layout.is_slide else "light"
    )
    table = fig.axes[1]
    texts = [t for t in table.texts if t.get_text()]
    plt.close(fig)

    def column_x(header):
        (cell,) = [t for t in texts if t.get_text() == header]
        return cell.get_position()[0]

    x_mean, x_var = column_x("mean (e)"), column_x(r"variance (e$^2$)")

    def row_of(name):
        """The row's cells, keyed by their x position."""
        (anchor,) = [t for t in texts if t.get_text() == name]
        y = anchor.get_position()[1]
        return {
            t.get_position()[0]: t.get_text()
            for t in texts
            if abs(t.get_position()[1] - y) < 1e-9
        }

    for name, grows, mean, var in HAND_ROWS:
        row = row_of(name)
        assert d07.GROWS_DISPLAY[grows] in row.values(), (name, row)
        assert row.get(x_mean) == d07._fmt(mean), (name, row)
        assert row.get(x_var) == d07._fmt(var), (name, row)
        if not layout.is_slide and name == "read noise":
            assert "4 px x 2$^2$ x 10 reads" in row.values()
    assert "reads (variance only)" in row_of("read noise").values()
    total = row_of(r"total, $\Sigma_p E_p$ over 4 pixels")
    assert (total.get(x_mean), total.get(x_var)) == ("5004.8", "5164.8")


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_schedule_still_prints_the_hand_numbers(layout):
    texts = _render(
        d07.build_acquisition, layout, "dark" if layout.is_slide else "light"
    )
    joined = "\n".join(texts)
    assert "10 x 10 s = 100 s" in joined
    assert "10 x (10 s + 0.05 s) = 100.5 s" in joined
    assert "$t_f$ = 10 s" in joined
    assert "10 frame values" in joined
    assert "grows with live time" in texts
    assert "once per frame, at its read" in texts
    assert "schematic, not to scale; reads drawn wider" in texts
    assert "independent Poisson" in joined and "independent Gaussian reads" in joined
    assert "one of the 4\naperture pixels" in texts
    assert "clock-induced charge (CIC)" in joined.replace(
        "clock-\ninduced", "clock-induced"
    )
    assert "quantum\nefficiency (QE)" in joined
    if layout.is_slide:
        assert d07.HEADLINE in texts
        assert "SNR" not in joined
    else:
        assert "read\n0.05 s" in texts
        assert f"{1000 / math.sqrt(5164.8):.3f}" == "13.915"
        assert "= 13.915" in joined
        for label in ("= 0.5", "= 0.01 e/s", "= 0.02 e/frame", "= 2 e RMS"):
            assert any(label in t for t in texts), label
        for calc in (
            "20/s (aperture) x 0.5 x 100 s",
            "80/s (aperture) x 0.5 x 100 s",
            "4 px x 0.01/s x 100 s",
            "4 px x 0.02 x 10 frames",
        ):
            assert calc in texts, calc


def test_reference_still_matches_an_independent_decimal_anchor():
    getcontext().prec = 40
    anchor = Decimal("1e-26") / (Decimal("6.62607015e-34") * Decimal(700))
    assert f"{anchor:.10f}" == "21559.8597091736"
    ref = d07.reference_budget()
    assert ref["density"] == pytest.approx(float(anchor), rel=1e-12)
    # 1 m^2 x 1 nm x 1 s, then QE 0.5; Poisson variance equals the mean.
    assert ref["photons"] == pytest.approx(float(anchor), rel=1e-12)
    assert ref["variance"] == pytest.approx(0.5 * float(anchor), rel=1e-12)
    joined = "\n".join(_render(d07.build_reference))
    assert "21 559.86 photon s" in joined
    assert "21 559.86 photons" in joined
    assert joined.count("10 779.93") == 2
    assert "dark charge: 0 here" in joined
    assert "no CIC here" in joined and "no read noise here" in joined
    assert "computed, not sampled" in joined
    assert "drawn" not in joined
    assert "photon spectral flux density" in joined
    assert "one of the 4" in joined


def test_simulation_is_seeded_and_consistent_with_the_model():
    spec = d07.ANIMATIONS[0]
    assert spec.params["seed"] == d07.SEED
    assert str(d07.SEED) in spec.status and "selected" in spec.status
    # The caption discloses the selection rule and its result.
    for phrase in ("selected as representative", "seeds 0 to 399", "0.5", "1.1"):
        assert phrase in spec.caption, phrase
    a, b = d07.simulate(), d07.simulate()
    assert np.array_equal(a["values"], b["values"])
    # Expected rate in the wells, expected frame value and its variance.
    rate = (PLANET + BACKGROUND) * QE + N_PIX * DARK
    assert a["rate"] == pytest.approx(50.04)
    assert rate == pytest.approx(50.04)
    assert a["frame_mean"] == pytest.approx(rate * FRAME + N_PIX * CIC)
    assert a["frame_mean"] == pytest.approx(500.48)
    frame_var = a["frame_mean"] + N_PIX * READ**2
    assert d07.frame_variance() == pytest.approx(frame_var)
    assert N_FRAMES * frame_var == pytest.approx(5164.8)
    assert len(a["values"]) == N_FRAMES
    # Each frame value is its arrival count plus a small CIC and read term.
    for arrivals, value in zip(a["arrivals"], a["values"], strict=True):
        assert np.all((arrivals >= 0) & (arrivals <= FRAME))
        assert abs(value - len(arrivals)) < 5 * math.sqrt(N_PIX) * READ + 5
    # The disclosed selection result holds for this seed.
    z_total = (a["values"].sum() - 5004.8) / math.sqrt(5164.8)
    assert z_total == pytest.approx(-0.5, abs=0.05)
    z_frames = (a["values"] - a["frame_mean"]) / math.sqrt(frame_var)
    assert np.all(np.abs(z_frames) < 1.1)


def test_clock_separates_live_and_elapsed_time():
    assert d07._clock(10.0) == pytest.approx((10.0, 0))
    assert d07._clock(10.05) == pytest.approx((10.0, 1))
    assert d07._clock(20.05) == pytest.approx((20.0, 1))
    assert d07._clock(20.10) == pytest.approx((20.0, 2))
    assert d07._clock(100.5) == pytest.approx((100.0, 10))


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_animation_scales_are_pinned_and_it_ends_at_the_schedule(layout):
    with exporter.venue("light", layout) as cast:
        scene = d07.build_count_animation(layout, cast)
        frames = list(scene.frames)
        if not layout.is_slide:
            assert len(frames) <= layout.frame_budget
        axes = scene.fig.axes
        limits = []
        for frame in (frames[0], frames[len(frames) // 2], frames[-1]):
            scene.draw(scene.fig, frame)
            scene.fig.canvas.draw()
            limits.append([(a.get_xlim(), a.get_ylim()) for a in axes])
            # Every frame keeps elapsed = live + reads x read time.
            live, reads = d07._clock(frame["t"])
            assert frame["t"] == pytest.approx(live + reads * READ_TIME)
        assert limits[0] == limits[1] == limits[2]
        assert frames[-1]["t"] == pytest.approx(100.5)
        texts = _texts(scene.fig)
        plt.close(scene.fig)
    joined = "\n".join(texts)
    assert "elapsed 100.50 s = live 100.00 s + 10 reads x 0.05 s" in joined
    # The on-screen recorded sum is the sum of the simulated frame values.
    total = d07.simulate()["values"].sum()
    match = re.search(r"recorded sum\s+([\d.]+) e", joined)
    assert match and float(match.group(1)) == pytest.approx(total, abs=0.05)
    sd = math.sqrt(5164.8)
    assert f"expected  5004.8 $\\pm$ {sd:4.1f} e" in joined
    assert "50.04 e/s during live time" in joined
    assert "500.48 e per read" in joined
    assert "one simulated realization, selected as representative" in texts
    assert f"seed {d07.SEED}" not in joined


def test_no_internal_references_in_this_diagram():
    files = [
        ROOT / "tools/explainers/d07_detector_acquisition.py",
        Path(__file__),
    ]
    notes = ROOT / "talks/notes/d07.md"
    if notes.exists():
        files.append(notes)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_internal_refs.py"), *map(str, files)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
