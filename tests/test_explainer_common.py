"""Contracts of the shared explainer-figure tooling.

These tests render a throwaway spec (the vocabulary sheet and a three-frame
animation) into a temporary directory. They check what diagram authors rely
on: every venue's files exist at the documented paths, the talk still is
exactly 1920 by 1080, the manifest hashes are the files' hashes, rebuilds
are byte-identical, one module never writes another's files, drifting
animation scales fail the build, and every cast entity is identifiable
without color.
"""

import hashlib
import json
import struct
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from explainers import _common as ex
from explainers import _export as exporter

EXPECTED_STILLS = {
    "docs/conventions/figures/explainer-d99-probe-light.png",
    "docs/conventions/figures/explainer-d99-probe-light.pdf",
    "docs/conventions/figures/explainer-d99-probe-dark.png",
    "docs/conventions/figures/explainer-d99-probe-dark.pdf",
    "talks/stills/d99-probe.png",
    "talks/stills/d99-probe.pdf",
}
EXPECTED_ANIMATION = {
    "docs/_static/explainers/explainer-d99-probe-anim-light.mp4",
    "docs/_static/explainers/explainer-d99-probe-anim-dark.mp4",
    "talks/animations/d99-probe-anim.mp4",
}


def _sweep(layout, cast, *, drift=False):
    fig, ax = ex.figure(layout, doc_height_in=2.5)
    ax.set(xlim=(0, 4), ylim=(-1.2, 1.2))
    (planet,) = ax.plot([0.0], [0.0], **cast["planet"].marker_kw())
    ex.badge(ax, cast)

    def draw(fig, x):
        planet.set_data([x], [np.sin(x)])
        if drift:
            ax.set_ylim(-1.2, 1.2 + x)

    return ex.AnimationScene(fig=fig, draw=draw, frames=np.linspace(0.0, 4.0, 3))


def _module(name, figures=(), animations=()):
    module = types.ModuleType(f"explainers.{name}")
    module.__file__ = exporter.PACKAGE / "_common.py"
    module.FIGURES = list(figures)
    module.ANIMATIONS = list(animations)
    return module


PROBE = ex.FigureSpec(
    slug="d99-probe",
    build=ex.cast_sheet,
    caption="Every shared entity and helper, drawn once.",
    alt="A sheet of the shared diagram entities.",
    params={"sample": np.arange(3)},
)
PROBE_ANIM = ex.AnimationSpec(
    slug="d99-probe-anim",
    build=_sweep,
    ground="narration",
    caption="A planet marker moving along a fixed curve.",
    alt="A dot moving across a fixed frame.",
    hold_s=(0.0, 0.0),
)


def _png_size(path):
    with open(path, "rb") as handle:
        header = handle.read(24)
    return struct.unpack(">II", header[16:24])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _files(root):
    return {
        p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()
    }


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_figure_export_writes_every_venue(tmp_path):
    prov = exporter.provenance(exporter.PACKAGE / "_common.py")
    written = exporter.export_figure(PROBE, prov, root=tmp_path)
    assert {p.as_posix() for p in written} == EXPECTED_STILLS
    assert _files(tmp_path) == EXPECTED_STILLS


def test_slide_still_is_1920_by_1080_and_doc_still_is_column_width(tmp_path):
    prov = exporter.provenance(exporter.PACKAGE / "_common.py")
    exporter.export_figure(PROBE, prov, root=tmp_path)
    assert _png_size(tmp_path / "talks/stills/d99-probe.png") == (1920, 1080)
    doc = tmp_path / "docs/conventions/figures/explainer-d99-probe-light.png"
    width, _ = _png_size(doc)
    assert width == round(ex.DOC.width_in * ex.DOC.dpi)


def test_rebuild_is_byte_identical(tmp_path):
    prov = exporter.provenance(exporter.PACKAGE / "_common.py")
    exporter.export_figure(PROBE, prov, root=tmp_path / "a")
    exporter.export_figure(PROBE, prov, root=tmp_path / "b")
    for rel in EXPECTED_STILLS:
        assert _sha(tmp_path / "a" / rel) == _sha(tmp_path / "b" / rel), rel


def test_manifest_records_each_output_hash(tmp_path):
    module = _module("d99_probe", figures=[PROBE], animations=[PROBE_ANIM])
    written = exporter.build_module(module, root=tmp_path)
    manifest_rel = "docs/conventions/figures/explainer-manifests/d99_probe.json"
    assert manifest_rel in {p.as_posix() for p in written}
    manifest = json.loads((tmp_path / manifest_rel).read_text())
    outputs = [
        o for e in manifest["figures"] + manifest["animations"] for o in e["outputs"]
    ]
    assert {o["file"] for o in outputs} == EXPECTED_STILLS | EXPECTED_ANIMATION
    for entry in outputs:
        assert entry["sha256"] == _sha(tmp_path / entry["file"]), entry["file"]
    assert manifest["figures"][0]["params"] == {"sample": [0, 1, 2]}
    assert manifest["animations"][0]["frames"]["doc"] == 3


def test_animation_videos_are_deterministic(tmp_path):
    prov = exporter.provenance(exporter.PACKAGE / "_common.py")
    written, _ = exporter.export_animation(PROBE_ANIM, prov, root=tmp_path / "a")
    exporter.export_animation(PROBE_ANIM, prov, root=tmp_path / "b")
    assert {p.as_posix() for p in written} == EXPECTED_ANIMATION
    for rel in EXPECTED_ANIMATION:
        assert (tmp_path / "a" / rel).stat().st_size > 0
        assert _sha(tmp_path / "a" / rel) == _sha(tmp_path / "b" / rel)


def test_building_one_module_leaves_other_modules_untouched(tmp_path):
    other = {
        "docs/conventions/figures/explainer-d98-other-light.svg": b"other",
        "docs/conventions/figures/explainer-manifests/d98_other.json": b"{}",
        "talks/stills/d98-other.png": b"other",
    }
    for rel, content in other.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(content)
    exporter.build_module(_module("d99_probe", figures=[PROBE]), root=tmp_path)
    for rel, content in other.items():
        assert (tmp_path / rel).read_bytes() == content
    touched = _files(tmp_path) - set(other)
    assert all("d99" in rel for rel in touched)


def test_slug_outside_the_module_prefix_is_refused(tmp_path):
    stray = ex.FigureSpec(slug="d01-stray", build=ex.cast_sheet, caption="c", alt="a")
    with pytest.raises(ValueError, match="d99"):
        exporter.build_module(_module("d99_probe", figures=[stray]), root=tmp_path)
    assert not _files(tmp_path)


def test_documentation_frame_budget_is_enforced(tmp_path):
    def long_sweep(layout, cast):
        scene = _sweep(layout, cast)
        scene.frames = np.linspace(0.0, 4.0, ex.DOC.frame_budget + 1)
        return scene

    spec = ex.AnimationSpec(
        slug="d99-long", build=long_sweep, ground="rate", caption="c", alt="a"
    )
    prov = exporter.provenance(exporter.PACKAGE / "_common.py")
    with pytest.raises(ValueError, match="budget"):
        exporter.export_animation(spec, prov, root=tmp_path)


def test_drifting_animation_scale_fails_the_build(tmp_path):
    spec = ex.AnimationSpec(
        slug="d99-drift",
        build=lambda layout, cast: _sweep(layout, cast, drift=True),
        ground="rate",
        caption="c",
        alt="a",
    )
    prov = exporter.provenance(exporter.PACKAGE / "_common.py")
    with pytest.raises(RuntimeError, match="scales changed"):
        exporter.export_animation(spec, prov, root=tmp_path)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE])
def test_cast_entities_differ_without_color(mode, layout):
    with exporter.venue(mode, layout) as cast:
        assert tuple(cast) == ex.ENTITY_KEYS
        signatures = {key: cast[key].signature() for key in cast}
    keys = list(signatures)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert signatures[a] != signatures[b], (a, b)


def test_arrow_kinds_are_drawn_differently():
    with exporter.venue("light", ex.DOC) as cast:
        _fig, ax = ex.figure(ex.DOC)
        drawn = {}
        for kind in ex.ARROW_KINDS:
            patches = [
                a
                for a in ex.arrow(ax, (0, 0), (1, 0), kind, cast)
                if hasattr(a, "get_arrowstyle")
            ]
            drawn[kind] = frozenset(
                (type(p.get_arrowstyle()).__name__, p.get_linestyle()) for p in patches
            )
    assert len(set(drawn.values())) == len(ex.ARROW_KINDS)


# Manim clips: the registry needs no Manim; rendering does


import explainers  # noqa: E402

CLIP = dict(
    slug="d99-probe-clip",
    scene=lambda: None,
    still="d99-probe",
    ground="narration",
    caption="A probe clip.",
    alt="A dot moving.",
)


def _clip_module(name="d99_probe", clips=(), figures=(PROBE,)):
    module = _module(name, figures=figures)
    module.MANIM = list(clips)
    return module


def test_manim_spec_validates_its_fields():
    assert explainers.ManimSpec(**CLIP).fps == {"doc": 15, "slide": 30}
    for bad in (
        {"slug": "D99 clip"},
        {"ground": "decoration"},
        {"caption": " "},
        {"venues": ("poster",)},
        {"fps": {"doc": 15}},
    ):
        with pytest.raises(ValueError):
            explainers.ManimSpec(**{**CLIP, **bad})


def test_manim_registry_is_checked_against_the_module():
    clip = explainers.ManimSpec(**CLIP)
    assert explainers.manim_specs(_clip_module(clips=[clip])) == [clip]
    assert explainers.manim_specs(_module("d99_probe", figures=[PROBE])) == []
    with pytest.raises(ValueError, match="d99"):
        stray = explainers.ManimSpec(**{**CLIP, "slug": "d01-stray-clip"})
        explainers.manim_specs(_clip_module(clips=[stray]))
    with pytest.raises(ValueError, match="still"):
        orphan = explainers.ManimSpec(**{**CLIP, "still": "d99-missing"})
        explainers.manim_specs(_clip_module(clips=[orphan]))
    with pytest.raises(ValueError, match="unique"):
        twin = explainers.ManimSpec(**{**CLIP, "slug": "d99-probe"})
        explainers.manim_specs(_clip_module(clips=[twin]))


def test_importing_the_diagrams_never_imports_manim():
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'tools'); import explainers, "
        "explainers._export; from explainers import d02_orbit_geometry as m; "
        "import explainers.__init__; explainers.manim_specs(m); "
        "print('manim' in sys.modules)"
    )
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_no_explainer_source_uses_tex():
    """Math goes through Matplotlib mathtext; nothing may need a TeX install."""
    import re

    banned = re.compile(
        r"\b(MathTex|Tex|SingleStringMathTex|DecimalNumber|Integer|Variable|"
        r"Matrix|BulletedList)\s*\("
    )
    sources = sorted(exporter.PACKAGE.glob("*.py"))
    assert len(sources) > 3
    hits = [p.name for p in sources if banned.search(p.read_text())]
    assert hits == []


def test_driver_records_manim_entries_beside_the_module_build(tmp_path):
    sys.path.insert(0, str(exporter.PACKAGE.parent))
    import build_explainer_figures as driver

    manifest = tmp_path / "d99_probe.json"
    manifest.write_text(json.dumps({"module": "d99_probe", "figures": []}))
    entries = [{"slug": "d99-probe-clip", "params": {"x": np.arange(2)}}]
    driver._record_manim(manifest, entries)
    data = json.loads(manifest.read_text())
    assert data["module"] == "d99_probe"
    assert data["manim"] == [{"slug": "d99-probe-clip", "params": {"x": [0, 1]}}]


def _manim_module():
    import os

    if os.environ.get("SPOHNBOOK_REQUIRE_MANIM"):
        import eyepiece.manim  # noqa: F401
        import manim  # noqa: F401
    else:
        pytest.importorskip("manim")
        pytest.importorskip("eyepiece.manim")
    from explainers import _manim

    return _manim


def _probe_scene(em):
    import manim

    class Probe(em.ExplainerScene):
        def construct(self):
            s = self.style
            self.section("move")
            dot = manim.Dot(color=s.entity("planet"))
            label = em.MathLabel(s, r"$R_Z(\Omega)$", 10)
            em.place(label, (0.0, 1.0))
            self.add(dot, label, em.badge(s, "probe status"))
            self.play(dot.animate.shift(manim.RIGHT), run_time=0.6)
            self.wait(0.4)

    return Probe


PROBE_PROV = exporter.Provenance(
    script="tools/explainers/probe.py", sha="0" * 12, date="2026-01-01", sources={}
)


@pytest.mark.manim
def test_manim_probe_renders_byte_identical_videos(tmp_path):
    em = _manim_module()
    venue = em.Venue(ex.DOC, "light", 320, 180, 10)
    scene_class = _probe_scene(em)
    shas = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.mp4"
        scene = em.render(scene_class, venue, path=path, prov=PROBE_PROV, status="p")
        assert scene.frame_count == 10
        assert path.stat().st_size > 0
        shas.append(_sha(path))
    assert shas[0] == shas[1]


@pytest.mark.manim
def test_manim_clips_use_the_matplotlib_encoder_command(tmp_path):
    from matplotlib.animation import FFMpegWriter

    em = _manim_module()
    venue = em.Venue(ex.SLIDE, "dark", 1920, 1080, 30)
    path = tmp_path / "clip.mp4"
    args = em.ffmpeg_args(path, venue)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    writer = FFMpegWriter(
        fps=30,
        extra_args=["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", *exporter.BITEXACT],
    )
    writer.fig, writer.dpi, writer.outfile = fig, 120, str(path)
    writer.frame_format = "rgba"
    want = writer._args()  # the command a Matplotlib recording runs
    want[0] = args[0]
    assert args == want


@pytest.mark.manim
def test_manim_readability_flags_small_crowded_and_outside_text():
    import manim

    em = _manim_module()
    venue = em.Venue(ex.DOC, "light", 1080, 540, 10)

    class Check(em.ExplainerScene):
        def construct(self):
            s = self.style
            fine = em.text(s, "fine", ex.DOC.font_pt)
            self.section("fine")
            self.add(fine)
            self.section("tiny")
            tiny = em.text(s, "tiny", 0.5 * ex.DOC.small_pt).shift(2 * manim.UP)
            self.add(tiny)
            self.section("crowded")
            self.remove(tiny)
            twin = em.text(s, "twin", ex.DOC.font_pt)
            self.add(twin.next_to(fine, manim.RIGHT, buff=0))
            self.section("outside")
            self.remove(twin)
            self.add(em.text(s, "away", ex.DOC.font_pt).shift(9 * manim.RIGHT))
            self.wait(0.1)

    scene = em.render(Check, venue, prov=PROBE_PROV, status="p", skip=True)
    reports = scene.reports
    assert reports["fine"]["passed"]
    assert not reports["tiny"]["passed"]
    assert reports["crowded"]["crowded_text"] == [["fine", "twin"]]
    assert reports["outside"]["outside_frame"] == ["away"]
