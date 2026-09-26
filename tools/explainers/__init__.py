"""Shared tooling for the handbook's physical explainer diagrams.

Each diagram lives in one module named ``dNN_<topic>.py`` in this package
(``d02_orbit.py``, for example). A module declares two registries and
nothing else is required of it::

    from explainers import _common as ex

    FIGURES = [
        ex.FigureSpec(
            slug="d02-orbit-planes",
            build=build_planes,  # build_planes(layout, cast) -> Figure
            caption="...",
            alt="...",
            status="schematic, not to scale",
            params={"inclination_deg": 60.0},
        ),
    ]
    ANIMATIONS = [
        ex.AnimationSpec(
            slug="d02-orbit-viewpoint",
            build=build_viewpoint,  # (layout, cast) -> ex.AnimationScene
            ground="viewpoint",
            caption="...",
            alt="...",
        ),
    ]

Every slug starts with the module's ``dNN`` prefix, so one module can never
write another module's files. The driver ``tools/build_explainer_figures.py``
discovers the modules, renders each figure as a documentation still (light
and dark) and a talk still (dark, 1920 by 1080), renders each animation as a
compact documentation player (light and dark) and a talk MP4, and records a
per-module manifest with the SHA-256 of every output.

Embedding a still in a MyST page (the path is relative to the page; from a
page in ``docs/conventions/`` it is ``figures/...``, from ``docs/`` it is
``conventions/figures/...``)::

    ```{figure} figures/explainer-d02-orbit-planes-light.png
    :class: only-light
    :name: fig-explainer-d02-orbit-planes
    :alt: <the alt text from the spec>

    <the caption from the spec>
    ```

    ```{figure} figures/explainer-d02-orbit-planes-dark.png
    :class: only-dark
    :alt: <the alt text from the spec>

    <the caption from the spec>
    ```

Embedding a documentation animation. The build writes a light and a dark
MP4 to docs/_static/explainers/, which Sphinx copies to the site; a raw HTML
video element plays it, and the theme hides whichever mode is inactive. From
a page one directory below docs/ (conventions/ or examples/)::

    ```{raw} html
    <video class="only-light" controls loop muted playsinline
      preload="metadata" style="width: 100%; height: auto;"
      aria-label="<the alt text from the spec>">
      <source type="video/mp4"
        src="../_static/explainers/explainer-d02-orbit-viewpoint-light.mp4">
    </video>
    <video class="only-dark" controls loop muted playsinline
      preload="metadata" style="width: 100%; height: auto;"
      aria-label="<the alt text from the spec>">
      <source type="video/mp4"
        src="../_static/explainers/explainer-d02-orbit-viewpoint-dark.mp4">
    </video>
    ```

    <the caption from the spec, as a paragraph>

Place the still before the animation and keep the page understandable
without the animation.

Manim clips. A module may also declare ``MANIM``, a list of ``ManimSpec``
records, for a talk-first clip whose lesson is a continuous transformation
(one object rotated, rescaled or carried into another representation).
Matplotlib draws states and Manim draws transformations: every still, and
every animation that indexes states on fixed axes, stays in Matplotlib. A
clip names the Matplotlib figure that documents it (``still``), and the
page must be understandable from that still alone::

    MANIM = [
        ManimSpec(
            slug="d02-three-rotations-clip",
            scene=three_rotations_scene,  # () -> an ExplainerScene subclass
            still="d02-three-rotations",
            ground="narration",
            caption="...",
            alt="...",
        ),
    ]

The scene factory imports Manim (through ``explainers._manim``) only when
it is called, so importing a diagram module, the documentation build and
the Matplotlib tests never need Manim. The build writes a light and a dark
documentation MP4 (at most ``doc_max_s`` seconds) to
docs/_static/explainers/ and a talk MP4 to talks/animations/, embedded like
the Matplotlib animations above, and records them under ``"manim"`` in the
module's manifest. ``--no-manim`` keeps the recorded entries.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

MANIM_VENUES = ("doc", "slide")


@dataclass(frozen=True)
class ManimSpec:
    """One Manim clip, rendered as documentation videos and a talk MP4.

    Attributes:
        slug: Output name, ``dNN-words``; unique across the module's
            ``FIGURES``, ``ANIMATIONS`` and ``MANIM``.
        scene: ``scene() -> type``, returning an ``ExplainerScene`` subclass.
            Only calling it imports Manim.
        still: Slug of the module's ``FigureSpec`` that documents this clip.
        ground: Why motion is warranted, one of ``MOTION_GROUNDS``.
        caption: Caption for the documentation page.
        alt: Alternative text for the video.
        status: Honesty label for the provenance stamp.
        params: Scientific inputs, recorded in the manifest.
        venues: Layout names to render, a subset of ``("doc", "slide")``.
        doc_max_s: Longest documentation clip, in seconds; a longer render
            fails the build.
        fps: Frames per second for each venue.
    """

    slug: str
    scene: Callable
    still: str
    ground: str
    caption: str
    alt: str
    status: str = "schematic, not to scale"
    params: dict = field(default_factory=dict)
    venues: tuple = MANIM_VENUES
    doc_max_s: float = 12.0
    fps: dict = field(default_factory=lambda: {"doc": 15, "slide": 30})

    def __post_init__(self):
        """Validate the slug, the ground, the venues and the text fields."""
        from explainers import _common as ex

        ex._check_slug(self.slug)
        if self.ground not in ex.MOTION_GROUNDS:
            msg = f"{self.slug}: ground must be one of {ex.MOTION_GROUNDS}"
            raise ValueError(msg)
        if not self.caption.strip() or not self.alt.strip():
            msg = f"{self.slug}: caption and alt text are required"
            raise ValueError(msg)
        if not self.venues or not set(self.venues) <= set(MANIM_VENUES):
            msg = f"{self.slug}: venues must be a nonempty subset of doc, slide"
            raise ValueError(msg)
        if not set(self.venues) <= set(self.fps):
            msg = f"{self.slug}: fps needs a rate for every venue"
            raise ValueError(msg)


def manim_specs(module):
    """The ``MANIM`` records of a diagram module, validated against the module.

    Every slug carries the module's ``dNN`` prefix and is unique across the
    three registries, and every clip's ``still`` names one of the module's
    figures.

    Args:
        module: An imported diagram module.

    Returns:
        The list of ``ManimSpec`` records (empty when the module has none).

    Raises:
        ValueError: On a foreign prefix, a repeated slug or an unknown still.
    """
    name = module.__name__.rsplit(".", 1)[-1]
    prefix = name[:3]
    clips = list(getattr(module, "MANIM", []))
    figures = [s.slug for s in getattr(module, "FIGURES", [])]
    animations = [s.slug for s in getattr(module, "ANIMATIONS", [])]
    slugs = figures + animations + [c.slug for c in clips]
    if len(set(slugs)) != len(slugs):
        msg = f"{name}: slugs must be unique across FIGURES, ANIMATIONS and MANIM"
        raise ValueError(msg)
    for clip in clips:
        if not clip.slug.startswith(f"{prefix}-"):
            msg = f"{name}: slug {clip.slug!r} must start with {prefix!r}-"
            raise ValueError(msg)
        if clip.still not in figures:
            msg = f"{name}: {clip.slug} names still {clip.still!r}, not a figure here"
            raise ValueError(msg)
    return clips
