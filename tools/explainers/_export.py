"""Render explainer specs to every venue and record a per-module manifest.

A figure becomes five documentation files (light and dark, PNG, SVG and
PDF, at the documentation column width) and two talk files (dark, 1920 by
1080, PNG and PDF). An animation becomes two documentation players (light
and dark self-contained HTML fragments, at most 30 frames at 100 dpi) and
one talk MP4 (dark, 1920 by 1080). Every file carries an eyepiece provenance
stamp, and each diagram module gets its own manifest, so modules built in
parallel never write the same file.

Vector exports are deterministic: the SVG date is dropped, the PDF creation
date is dropped, the SVG hash salt is fixed, and the stamp date is the date
of the source commit rather than today. The HTML players get a fixed element
id instead of a random one. Rebuilding an unchanged module therefore
reproduces its files byte for byte.
"""

import dataclasses
import hashlib
import html
import importlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import eyepiece as ep
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

from explainers import _common as ex

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
DOC_FIGURES = Path("docs/conventions/figures")
MANIFESTS = DOC_FIGURES / "explainer-manifests"
STILLS = Path("talks/stills")
ANIMATIONS = Path("talks/animations")
PREVIEW = Path(".explainer-preview")

# (layout, mode, suffixes, directory, name pattern) for every still variant.
STILL_VARIANTS = (
    (ex.DOC, "light", ("png", "svg", "pdf"), DOC_FIGURES, "explainer-{slug}-light"),
    (ex.DOC, "dark", ("png", "svg", "pdf"), DOC_FIGURES, "explainer-{slug}-dark"),
    (ex.SLIDE, "dark", ("png", "pdf"), STILLS, "{slug}"),
)
# Output arguments that keep an MP4 free of encoder timestamps and metadata.
BITEXACT = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1"]
PACKAGES = ("eyepiece", "hwostyle", "hwoutils", "matplotlib", "numpy")


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def git_revision(path):
    """Short commit and dirty flag of the repository containing ``path``."""
    return {
        "commit": _git(path, "rev-parse", "--short", "HEAD"),
        "dirty": bool(_git(path, "status", "--porcelain")),
    }


def sha256(path):
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Provenance:
    """What the provenance stamp says about the code that drew a figure.

    Attributes:
        script: Repository-relative path of the diagram module.
        sha: First 12 hex digits of the SHA-256 over the module and the
            shared tooling sources.
        date: Commit date of the source revision, so rebuilding the same
            revision writes the same stamp.
        sources: Repository-relative path to full SHA-256 for each source.
    """

    script: str
    sha: str
    date: str
    sources: dict


def _relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def provenance(module_file):
    """Provenance for a diagram module and the shared tooling it uses."""
    files = [Path(module_file), PACKAGE / "_common.py", PACKAGE / "_export.py"]
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    date = _git(ROOT, "log", "-1", "--format=%cs") or "undated"
    return Provenance(
        script=_relative(module_file),
        sha=digest[:12],
        date=date,
        sources={_relative(f): sha256(f) for f in files},
    )


@contextmanager
def venue(mode, layout):
    """Activate a style mode and a layout, and yield the matching cast.

    Everything built and saved inside the block uses the mode's colors and
    the layout's type sizes.
    """
    hwostyle.use(mode)
    with matplotlib.rc_context(layout.rc()):
        yield ex.make_cast(layout)


def _check_size(fig, layout, slug):
    width, height = fig.get_size_inches()
    ok_width = abs(width - layout.width_in) < 1e-6
    ok_height = not layout.is_slide or abs(height - layout.height_in) < 1e-6
    if not (ok_width and ok_height):
        msg = (
            f"{slug}: the {layout.name} figure is {width:g} x {height:g} in; "
            f"create it with ex.figure(layout) so it is {layout.width_in:g} in "
            "wide (and 16 x 9 in on a slide)"
        )
        raise ValueError(msg)


def _stamp(fig, prov, status, layout):
    """Stamp provenance on the figure and return the metadata payload."""
    engine = fig.get_layout_engine()
    if isinstance(engine, matplotlib.layout_engine.ConstrainedLayoutEngine):
        bottom = 1.8 * layout.stamp_pt / (72.0 * fig.get_size_inches()[1])
        engine.set(rect=(0, bottom, 1, 1 - bottom))
    kw = {"script": prov.script, "sha": prov.sha, "date": prov.date, "note": status}
    ep.stamp(fig, size=layout.stamp_pt, **kw)
    return ep.provenance_fields(**kw)


def _save(fig, path, layout, fields):
    metadata = ep.file_metadata(fields, path.suffix)
    if path.suffix == ".svg":
        metadata["Date"] = None
    elif path.suffix == ".pdf":
        metadata["CreationDate"] = None
    kw = {"metadata": metadata, "bbox_inches": None}
    if path.suffix == ".png":
        kw["dpi"] = layout.dpi
    return ep.save_fig(fig, path.name, dir=path.parent, **kw)


def export_figure(spec, prov, *, root=ROOT):
    """Render one ``FigureSpec`` to every still variant.

    Args:
        spec: The figure spec.
        prov: ``Provenance`` of the module that declares it.
        root: Repository root the output paths are relative to.

    Returns:
        The written paths, relative to ``root``.
    """
    written = []
    for layout, mode, suffixes, directory, pattern in STILL_VARIANTS:
        with venue(mode, layout) as cast:
            fig = spec.build(layout, cast)
            try:
                _check_size(fig, layout, spec.slug)
                fields = _stamp(fig, prov, spec.status, layout)
                for suffix in suffixes:
                    rel = directory / f"{pattern.format(slug=spec.slug)}.{suffix}"
                    _save(fig, Path(root) / rel, layout, fields)
                    written.append(rel)
            finally:
                plt.close(fig)
    return written


def _build_scene(spec, layout, cast):
    scene = spec.build(layout, cast)
    frames = list(scene.frames)
    if not frames:
        msg = f"{spec.slug}: the animation has no frames"
        raise ValueError(msg)
    budget = layout.frame_budget
    if budget is not None and len(frames) > budget:
        plt.close(scene.fig)
        msg = (
            f"{spec.slug}: {len(frames)} frames exceed the {layout.name} budget "
            f"of {budget}; size the frame list with layout.n_frames(n)"
        )
        raise ValueError(msg)
    _check_size(scene.fig, layout, spec.slug)
    return scene, frames


def _record(scene, frames, path, *, fps, dpi, holds, allow_rescale, slug):
    """Record the frames into ``path``; a drifting scale fails the build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ep.record(
            scene.fig,
            path,
            fps=fps,
            dpi=dpi,
            extra_ffmpeg_args=BITEXACT,
            allow_rescale=allow_rescale,
        ) as rec:
            for index, frame in enumerate(frames):
                scene.draw(scene.fig, frame)
                rec.frame()
                if index == 0 and holds[0]:
                    rec.hold(holds[0])
            if holds[1]:
                rec.hold(holds[1])
    drift = [w for w in caught if "data scales changed" in str(w.message)]
    for w in caught:
        if w not in drift:
            warnings.warn(w.message, w.category, stacklevel=2)
    if drift:
        path.unlink(missing_ok=True)
        msg = f"{slug}: {drift[0].message}"
        raise RuntimeError(msg)


def _html_id(slug, mode):
    return re.sub(r"[^A-Za-z0-9]", "_", f"explainer_{slug}_{mode}")


def _finalize_player(path, slug, mode, alt):
    """Give the player a fixed id, alt text, and a width that fits the column.

    Matplotlib names the player's elements with a random id, which would
    change the file on every build and collide if two players shared one.
    The frame image also gets the theme's ``dark-light`` class, which stops
    the dark theme from dimming it.
    """
    text = path.read_text()
    ids = set(re.findall(r"_anim_img([0-9a-f]{32})", text))
    if len(ids) != 1:
        msg = f"{path.name}: expected one player id, found {len(ids)}"
        raise RuntimeError(msg)
    new = _html_id(slug, mode)
    text = text.replace(ids.pop(), new)
    tag = f'<img id="_anim_img{new}">'
    replacement = (
        f'<img id="_anim_img{new}" class="dark-light" '
        f'alt="{html.escape(alt, quote=True)}" style="max-width: 100%; height: auto;">'
    )
    if tag not in text:
        msg = f"{path.name}: player image tag not found"
        raise RuntimeError(msg)
    path.write_text(text.replace(tag, replacement))


def export_animation(spec, prov, *, root=ROOT):
    """Render one ``AnimationSpec`` to the documentation players and the MP4.

    Args:
        spec: The animation spec.
        prov: ``Provenance`` of the module that declares it.
        root: Repository root the output paths are relative to.

    Returns:
        ``(written, frame_counts)``: the paths relative to ``root``, and the
        frame counts per venue.
    """
    written = []
    counts = {}
    for mode in ("light", "dark"):
        with venue(mode, ex.DOC) as cast:
            scene, frames = _build_scene(spec, ex.DOC, cast)
            try:
                _stamp(scene.fig, prov, spec.status, ex.DOC)
                rel = DOC_FIGURES / f"explainer-{spec.slug}-{mode}.html"
                _record(
                    scene,
                    frames,
                    Path(root) / rel,
                    fps=spec.fps or ep.PRESETS["jshtml"]["fps"],
                    dpi=ex.DOC.anim_dpi,
                    holds=(0, 0),
                    allow_rescale=spec.allow_rescale,
                    slug=spec.slug,
                )
                _finalize_player(Path(root) / rel, spec.slug, mode, spec.alt)
                written.append(rel)
                counts["doc"] = len(frames)
            finally:
                plt.close(scene.fig)
    with venue("dark", ex.SLIDE) as cast:
        scene, frames = _build_scene(spec, ex.SLIDE, cast)
        try:
            _stamp(scene.fig, prov, spec.status, ex.SLIDE)
            fps = spec.fps or ep.PRESETS["talk"]["fps"]
            holds = tuple(round(s * fps) for s in spec.hold_s)
            rel = ANIMATIONS / f"{spec.slug}.mp4"
            _record(
                scene,
                frames,
                Path(root) / rel,
                fps=fps,
                dpi=ex.SLIDE.anim_dpi,
                holds=holds,
                allow_rescale=spec.allow_rescale,
                slug=spec.slug,
            )
            written.append(rel)
            counts["talk"] = len(frames) + sum(holds)
        finally:
            plt.close(scene.fig)
    return written, counts


# Preview


def preview_builder(build, name, *, root=ROOT, prov=None, status="preview"):
    """Write quick PNGs of a figure builder in every venue, for review.

    Args:
        build: ``build(layout, cast) -> Figure``.
        name: File stem inside the preview directory.
        root: Repository root.
        prov: Optional ``Provenance`` to stamp with.
        status: Stamp note.

    Returns:
        The written paths, relative to ``root``.
    """
    written = []
    for layout, mode, *_ in STILL_VARIANTS:
        with venue(mode, layout) as cast:
            fig = build(layout, cast)
            try:
                if prov is not None:
                    _stamp(fig, prov, status, layout)
                rel = PREVIEW / f"{name}-{layout.name}-{mode}.png"
                (Path(root) / rel).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    Path(root) / rel,
                    dpi=110 if layout.name == "doc" else 80,
                    facecolor=fig.get_facecolor(),
                )
                written.append(rel)
            finally:
                plt.close(fig)
    return written


def preview_animation(spec, prov, *, root=ROOT):
    """Write chosen frames of an animation as PNGs, without encoding."""
    written = []
    for layout, mode in ((ex.DOC, "light"), (ex.DOC, "dark"), (ex.SLIDE, "dark")):
        with venue(mode, layout) as cast:
            scene, frames = _build_scene(spec, layout, cast)
            try:
                _stamp(scene.fig, prov, spec.status, layout)
                n = len(frames)
                indices = spec.preview_frames or (0, n // 2, n - 1)
                for index in sorted({min(int(i), n - 1) for i in indices}):
                    scene.draw(scene.fig, frames[index])
                    rel = PREVIEW / f"{spec.slug}-f{index:03d}-{layout.name}-{mode}.png"
                    (Path(root) / rel).parent.mkdir(parents=True, exist_ok=True)
                    scene.fig.savefig(
                        Path(root) / rel,
                        dpi=100 if layout.name == "doc" else 80,
                        facecolor=scene.fig.get_facecolor(),
                    )
                    written.append(rel)
            finally:
                plt.close(scene.fig)
    return written


# Modules and manifests


def discover():
    """Names of the diagram modules, ``dNN_topic``, in order."""
    return sorted(p.stem for p in PACKAGE.glob("d[0-9][0-9]_*.py"))


def load(name):
    """Import one diagram module by name."""
    return importlib.import_module(f"explainers.{name}")


def _specs(module):
    name = module.__name__.rsplit(".", 1)[-1]
    prefix = name[:3]
    figures = list(getattr(module, "FIGURES", []))
    animations = list(getattr(module, "ANIMATIONS", []))
    slugs = [s.slug for s in [*figures, *animations]]
    for slug in slugs:
        if not slug.startswith(f"{prefix}-"):
            msg = f"{name}: slug {slug!r} must start with {prefix!r}-"
            raise ValueError(msg)
    if len(set(slugs)) != len(slugs):
        msg = f"{name}: slugs must be unique across FIGURES and ANIMATIONS"
        raise ValueError(msg)
    return name, figures, animations


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _describe(root, paths):
    return [
        {
            "file": Path(rel).as_posix(),
            "sha256": sha256(Path(root) / rel),
            "bytes": (Path(root) / rel).stat().st_size,
        }
        for rel in paths
    ]


def _ffmpeg_version():
    binary = shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]) or shutil.which(
        "ffmpeg"
    )
    if binary is None:
        return None
    first = subprocess.run(
        [binary, "-version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    return first[0] if first else None


def build_module(module, *, root=ROOT, animations=True, preview=False):
    """Build every figure and animation one diagram module declares.

    Args:
        module: The imported diagram module.
        root: Repository root for outputs.
        animations: Whether to render animations. When False the manifest
            keeps the animation entries recorded by the previous build.
        preview: Write review PNGs to the preview directory instead, and
            leave the published outputs and the manifest untouched.

    Returns:
        A list of the written paths, relative to ``root``.
    """
    name, figures, anims = _specs(module)
    prov = provenance(module.__file__)
    if preview:
        written = []
        for spec in figures:
            written += preview_builder(
                spec.build, spec.slug, root=root, prov=prov, status=spec.status
            )
        for spec in anims if animations else []:
            written += preview_animation(spec, prov, root=root)
        return written

    manifest_path = Path(root) / MANIFESTS / f"{name}.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
    revision = {
        "spohnbook": git_revision(ROOT),
        "eyepiece": git_revision(Path(ep.__file__).resolve().parent),
        "hwostyle": git_revision(Path(hwostyle.__file__).resolve().parent),
    }
    written = []
    figure_entries = []
    for spec in figures:
        paths = export_figure(spec, prov, root=root)
        written += paths
        figure_entries.append(
            {
                "slug": spec.slug,
                "status": spec.status,
                "caption": spec.caption,
                "alt": spec.alt,
                "params": spec.params,
                "outputs": _describe(root, paths),
            }
        )
    if animations:
        animation_entries = []
        for spec in anims:
            paths, counts = export_animation(spec, prov, root=root)
            written += paths
            animation_entries.append(
                {
                    "slug": spec.slug,
                    "status": spec.status,
                    "ground": spec.ground,
                    "caption": spec.caption,
                    "alt": spec.alt,
                    "params": spec.params,
                    "frames": counts,
                    "revision": revision["spohnbook"],
                    "outputs": _describe(root, paths),
                }
            )
    else:
        animation_entries = previous.get("animations", [])

    manifest = {
        "module": name,
        "script": prov.script,
        "build_command": f"python tools/build_explainer_figures.py --only {name[:3]}",
        "source_hash": prov.sha,
        "source_sha256": prov.sources,
        "stamp_date": prov.date,
        "revisions": revision,
        "package_versions": {p: importlib.metadata.version(p) for p in PACKAGES},
        "ffmpeg": _ffmpeg_version() if animations and anims else None,
        "layouts": {k: dataclasses.asdict(v) for k, v in ex.LAYOUTS.items()},
        "figures": figure_entries,
        "animations": animation_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_jsonable) + "\n")
    written.append(MANIFESTS / f"{name}.json")
    return written
