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
"""
