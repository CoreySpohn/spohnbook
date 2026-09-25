# Talk assets

This directory holds the slide versions of the handbook's physical explainer
diagrams. Each asset is rendered from the same diagram module and the same
scientific inputs as the figure on the documentation page, so a slide and the
page it came from cannot drift apart.

## Layout

| Path | Contents |
|---|---|
| `stills/<slug>.png`, `stills/<slug>.pdf` | Dark still at 1920 by 1080 pixels (16:9), with slide-sized type. |
| `animations/<slug>.mp4` | Dark animation at 1920 by 1080 pixels, holding its first and last frames. |
| `notes/dNN.md` | Speaker notes for diagram `dNN`: the lesson, the assumptions and the source locators. |

The documentation versions live beside the pages: light and dark stills in
`docs/conventions/figures/` and light and dark videos in
`docs/_static/explainers/`, each named `explainer-<slug>-<mode>`.
Each diagram module records every output, with its SHA-256, in
`docs/conventions/figures/explainer-manifests/<module>.json`.

## Rebuilding

From the repository root:

```sh
python tools/build_explainer_figures.py --only d02      # one diagram module
python tools/build_explainer_figures.py --preview       # review PNGs only
python tools/build_explainer_figures.py --no-anim       # stills only
```

Previews go to the ignored `.explainer-preview/` directory. Animations need
ffmpeg.

## Using an asset in a talk

Every still and every animation carries its status (schematic, illustrative
calculation or simulation) and a provenance stamp on the graphic, so it stays
honest when it is pasted into a slide without its caption. Read the diagram's
speaker notes before presenting it: they state what the picture does not
establish.

## Index

| Diagram | Lesson | Stills | Animations | Notes |
|---|---|---|---|---|
| D01 | The physical observation, from system to measured product | | | |
| D02 | Star, planet, observer and orbital reference planes | | | |
| D03 | What an aperture and a pixel collect | | | |
| D04 | Optical planes and the quantities that live on them | | | |
| D05 | How a lenslet IFS turns a sky cube into detector traces | | | |
| D06 | Local zodiacal light and an exozodiacal disk | | | |
| D07 | A pixel's acquisition and readout schedule | | | |
| D08 | The physical experiment, acquired record and assumed model | | | |
