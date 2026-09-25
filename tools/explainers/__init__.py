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

    ```{figure} figures/explainer-d02-orbit-planes-light.svg
    :class: only-light
    :name: fig-explainer-d02-orbit-planes
    :alt: <the alt text from the spec>

    <the caption from the spec>
    ```

    ```{figure} figures/explainer-d02-orbit-planes-dark.svg
    :class: only-dark
    :alt: <the alt text from the spec>

    <the caption from the spec>
    ```

Embedding a documentation animation. The file is a self-contained HTML
fragment with its frames embedded, so the page needs no execution, no
ffmpeg and no sidecar files; the theme hides whichever mode is inactive::

    ```{raw} html
    :file: figures/explainer-d02-orbit-viewpoint-light.html
    :class: only-light
    ```

    ```{raw} html
    :file: figures/explainer-d02-orbit-viewpoint-dark.html
    :class: only-dark
    ```

Place the still before the animation and keep the page understandable
without the animation.
"""
