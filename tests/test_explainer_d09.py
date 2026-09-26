"""Pixels on the roll-versus-rotation explainer, checked by hand.

The chapter's worked examples are read from its text and recomputed here
with plain arithmetic on the pixel profile x = (c - c_x) s, y = (r - c_y) s:
a +90-degree telescope roll sends detector coordinates to
d = R(-90) s = (y, -x), and an active +90-degree rotation sends content to
R(+90) p = (-y, x). No rotation helper, from the diagram module or from a
library, enters the hand values. A separate test checks that the library
sign conventions the caption cites agree with those hand values.
"""

import re
import subprocess
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
from explainers import d09_roll_rotation as d09

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = (ROOT / "docs/conventions/optics-images.md").read_text()
SECTION = CHAPTER.split("(optics-roll-rotation)=")[1].split("(optics-angular-grids)=")[
    0
]
DETECTION = (
    (ROOT / "docs/examples/scene-to-detection.md")
    .read_text()
    .split("## Detection")[1]
    .split("```")[0]
)


def _hand_pixel(x, y, n):
    """(row, column) of physical (x, y), in units of s, on an n-by-n grid."""
    c0 = (n - 1) / 2  # geometric origin; 1.5 for n = 4, 2 for n = 5
    col, row = x + c0, y + c0
    assert col == int(col) and row == int(row), "a worked source sits on a center"
    return int(row), int(col)


def _roll90(x, y):
    """d = R(-90) s, written out: cos = 0, sin(-90) = -1."""
    return y, -x


def _active90(x, y):
    """R(+90) p, written out: cos = 0, sin(90) = 1."""
    return -y, x


def test_chapter_states_the_worked_pixels():
    text = " ".join(SECTION.split())
    assert (
        "a source at `(x,y)=(1.5s,0.5s)` occupies `(r,c)=(2,3)`. At telescope roll "
        "+90 degrees it becomes `(0.5s,-1.5s)` and occupies `(0,2)`." in text
    )
    assert (
        "An **active** +90-degree image rotation instead sends the same physical "
        "point to `(-0.5s,1.5s)`, at `(3,1)`." in text
    )
    assert (
        "On a 5-by-5 grid, `(x,y)=(s,0)` starts at `(2,3)` and reaches `(1,2)` "
        "after +90-degree telescope roll." in text
    )
    assert r"\mathbf{d}=R(-\theta)\mathbf{s}" in SECTION
    assert r"\mathbf{s}=(x_{\rm sky},y_{\rm sky})=(E,N)" in SECTION


def test_worked_pixels_by_hand():
    # 4-by-4 example.
    assert _hand_pixel(1.5, 0.5, 4) == (2, 3)
    assert _roll90(1.5, 0.5) == (0.5, -1.5)
    assert _hand_pixel(*_roll90(1.5, 0.5), 4) == (0, 2)
    assert _active90(1.5, 0.5) == (-0.5, 1.5)
    assert _hand_pixel(*_active90(1.5, 0.5), 4) == (3, 1)
    # 5-by-5 example.
    assert _hand_pixel(1.0, 0.0, 5) == (2, 3)
    assert _hand_pixel(*_roll90(1.0, 0.0), 5) == (1, 2)
    # The chapter's pixel numbers, parsed from its text, match the hand values.
    found = re.findall(r"\((\d),(\d)\)`", " ".join(SECTION.split()))
    pixels = [(int(a), int(b)) for a, b in found]
    assert pixels == [(2, 3), (0, 2), (3, 1), (2, 3), (1, 2)]


def test_module_geometry_matches_the_hand_values():
    got = d09.cases()
    assert got["zero"]["pixel"] == (2, 3)
    assert got["roll"]["pixel"] == (0, 2)
    assert got["active"]["pixel"] == (3, 1)
    np.testing.assert_allclose(got["roll"]["xy"], _roll90(*d09.SOURCE_XY), atol=1e-12)
    np.testing.assert_allclose(
        got["active"]["xy"], _active90(*d09.SOURCE_XY), atol=1e-12
    )
    assert d09.pixel_of(d09.detector_xy((1.0, 0.0), 90.0), 5) == (1, 2)
    assert d09.optical_origin(4) == 1.5
    assert d09.optical_origin(5) == 2.0


def test_library_signs_agree_with_the_chapter():
    # The caption cites these conventions from the scene-to-detection example;
    # this checks the contract (where the pixel ends up), not the mechanism.
    text = " ".join(DETECTION.split())
    assert "ccw_rotation_matrix(-telescope_pa_deg)" in text
    assert "a positive angle turns the scene clockwise on the readout" in text
    assert "`rotate_image` from hwoutils turns an image counter-clockwise" in text
    transforms = pytest.importorskip("hwoutils.transforms")
    rolled = np.asarray(transforms.ccw_rotation_matrix(-90.0)) @ np.array([1.5, 0.5])
    np.testing.assert_allclose(rolled, _roll90(1.5, 0.5), atol=1e-6)
    image = np.zeros((4, 4))
    image[2, 3] = 1.0
    out = np.asarray(transforms.rotate_image(image, 90.0))
    assert np.unravel_index(np.argmax(out), out.shape) == (3, 1)


def _render_texts(build, layout, mode="light"):
    with exporter.venue(mode, layout) as cast:
        fig = build(layout, cast)
        fig.canvas.draw()
        texts = [t.get_text() for t in fig.findobj(Text)]
        plt.close(fig)
    return texts


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_still_prints_the_hand_pixels_and_the_pending_status(layout):
    joined = "\n".join(_render_texts(d09.build_still, layout))
    for pixel in ((2, 3), (0, 2), (3, 1)):
        assert f"(r, c) = {pixel}" in joined
    assert "image-coordinate profile pending" in joined
    assert "sky turns clockwise" in joined
    assert "content turns\ncounterclockwise" in joined


def test_highlighted_pixel_contains_the_fixed_source_on_the_sky():
    # In the rolled sky panel the shaded pixel, carried by the roll
    # transform, must sit where the fixed source is: chart (1.5 s, 0.5 s).
    with exporter.venue("light", ex.DOC) as cast:
        fig = d09.build_still(ex.DOC, cast)
        fig.canvas.draw()
        ax = fig.axes[1]
        lit = [p for p in ax.patches if p.get_gid() == "d09-lit-pixel"]
        assert len(lit) == 1
        to_data = lit[0].get_transform() - ax.transData
        # get_transform() already includes the patch's own unit-square map.
        corners = to_data.transform(lit[0].get_path().vertices[:4])
        plt.close(fig)
    center = corners.mean(axis=0)
    np.testing.assert_allclose(center, d09.SOURCE_XY, atol=1e-9)


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_animation_runs_zero_to_ninety_with_pinned_scales(layout):
    with exporter.venue("light", layout) as cast:
        scene = d09.build_roll_animation(layout, cast)
        frames = list(scene.frames)
        if not layout.is_slide:
            assert len(frames) <= layout.frame_budget
        assert frames[0]["theta"] == 0.0
        assert frames[-1]["theta"] == pytest.approx(90.0)
        thetas = [f["theta"] for f in frames]
        assert thetas == sorted(thetas)
        limits = []
        for frame in (frames[-1], frames[0], frames[len(frames) // 2], frames[-1]):
            scene.draw(scene.fig, frame)
            scene.fig.canvas.draw()
            limits.append([(a.get_xlim(), a.get_ylim()) for a in scene.fig.axes])
        texts = [t.get_text() for t in scene.fig.findobj(Text)]
        plt.close(scene.fig)
    assert all(lim == limits[0] for lim in limits)
    joined = "\n".join(texts)
    assert "stored array: source in (r, c) = (0, 2)" in joined


def test_animation_pixels_follow_the_hand_rotation():
    # At every frame angle the stored pixel is the one holding
    # (1.5 cos t + 0.5 sin t, -1.5 sin t + 0.5 cos t), written out by hand.
    for frame in d09.roll_states(ex.SLIDE):
        t = np.deg2rad(frame["theta"])
        x = 1.5 * np.cos(t) + 0.5 * np.sin(t)
        y = -1.5 * np.sin(t) + 0.5 * np.cos(t)
        expected = (int(np.floor(y + 2.0)), int(np.floor(x + 2.0)))
        assert d09.pixel_of(d09.detector_xy(d09.SOURCE_XY, frame["theta"])) == expected


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
    with exporter.venue(mode, layout) as cast:
        fig = d09.build_still(layout, cast)
        try:
            assert _stroked(fig) == []
        finally:
            plt.close(fig)
        scene = d09.build_roll_animation(layout, cast)
        try:
            for frame in (scene.frames[0], scene.frames[-1]):
                scene.draw(scene.fig, frame)
                assert _stroked(scene.fig) == []
        finally:
            plt.close(scene.fig)


def test_captions_name_the_clause_and_the_pending_profile():
    for spec in (*d09.FIGURES, *d09.ANIMATIONS):
        assert spec.slug.startswith("d09-")
        assert "{ref}`optics-roll-rotation`" in spec.caption
        assert "pending" in spec.caption
        assert "decision-image-coordinates-and-psflet-origin" in spec.caption
        assert "pending" in spec.status
        for text in (spec.caption, spec.alt):
            assert chr(0x2014) not in text  # no em-dashes
            assert "n't" not in text  # no contractions
    assert d09.ANIMATIONS[0].ground == "viewpoint"


def test_no_internal_references_in_this_diagram():
    files = [ROOT / "tools/explainers/d09_roll_rotation.py", Path(__file__)]
    notes = ROOT / "talks/notes/d09.md"
    if notes.exists():
        files.append(notes)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_internal_refs.py"), *map(str, files)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
