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
    "docs/conventions/figures/explainer-d99-probe-light.svg",
    "docs/conventions/figures/explainer-d99-probe-light.pdf",
    "docs/conventions/figures/explainer-d99-probe-dark.png",
    "docs/conventions/figures/explainer-d99-probe-dark.svg",
    "docs/conventions/figures/explainer-d99-probe-dark.pdf",
    "talks/stills/d99-probe.png",
    "talks/stills/d99-probe.pdf",
}
EXPECTED_ANIMATION = {
    "docs/conventions/figures/explainer-d99-probe-anim-light.html",
    "docs/conventions/figures/explainer-d99-probe-anim-dark.html",
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


def test_animation_players_are_deterministic_and_carry_alt_text(tmp_path):
    prov = exporter.provenance(exporter.PACKAGE / "_common.py")
    written, _ = exporter.export_animation(PROBE_ANIM, prov, root=tmp_path / "a")
    exporter.export_animation(PROBE_ANIM, prov, root=tmp_path / "b")
    assert {p.as_posix() for p in written} == EXPECTED_ANIMATION
    for rel in EXPECTED_ANIMATION:
        assert (tmp_path / "a" / rel).stat().st_size > 0
        if rel.endswith(".html"):
            text = (tmp_path / "a" / rel).read_text()
            assert f'alt="{PROBE_ANIM.alt}"' in text
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
