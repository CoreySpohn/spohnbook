"""Structural checks on the handbook: legacy links, catalog and evidence states."""

import os
import re
import sys
import unicodedata
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
sys.path.insert(0, str(DOCS))

# Every page and section anchor published before the durability reorganization.
# A reader holding one of these links must still land on the same content.
LEGACY_ANCHORS = {
    "index": ["the-spohn-book", "fig-library-graph"],
    "start-here": [
        "start-here",
        "fig-start-pipeline",
        "things-every-library-assumes",
        "read-next",
    ],
    "map": [
        "library-map",
        "fig-map-graph",
        "foundation",
        "geometry",
        "scene",
        "data",
        "simulation",
        "wavefront",
        "exposure-time",
        "analysis",
        "inference",
        "planning",
        "orchestration",
        "visualization",
    ],
    "library-conventions": [
        "library-conventions",
        "repository-layout",
        "naming",
        "code-style",
        "jax-and-equinox",
        "testing-and-evidence",
        "documentation",
        "data",
        "release-and-ci",
        "fig-release-flow",
        "keeping-the-library-public-safe",
    ],
    "conventions/index": [
        "hwo-scientific-conventions-handbook",
        "how-to-read-this-handbook",
        "three-distinctions-to-preserve",
        "contract-vocabulary-and-status",
        "decision-register",
        "what-travels-across-a-boundary",
        "contributing-to-this-handbook",
    ],
    "conventions/geometry-time": [
        "geometry-and-time-from-a-physical-orbit-to-a-reported-observation",
        "start-with-the-observer",
        "fig-geometry",
        "choose-a-basis-before-interpreting-orbital-angles",
        "external-reference-profile-savransky-2019",
        "keep-origin-mass-and-orbital-phase-together",
        "derive-radial-velocity-from-motion-then-choose-its-units",
        "worked-signed-example",
        "public-quantities-and-uncertainty-transforms",
        "a-time-value-answers-several-different-questions",
        "fig-time",
        "named-acceptance-fixtures",
        "coverage-and-disposition",
    ],
    "conventions/radiometry-detectors": [
        "radiometry-detector-counts-and-spectral-measurements",
        "fig-pipeline",
        "quantities-and-measures",
        "from-jy-to-photons",
        "from-brightness-to-pixels",
        "contrast-magnitudes-and-zodiacal-light",
        "reference-planes-and-response-ownership",
        "detector-expectation-variance-and-cadence",
        "worked-count-example",
        "etc-forecasts-and-the-reported-experiment",
        "spectral-response-and-covariance",
        "exchange-contract-and-acceptance-fixtures",
        "coverage-and-gates",
    ],
    "conventions/optics-images": [
        "optical-fields-image-coordinates-and-ifs-products",
        "symbols-and-product-metadata",
        "physical-directions-and-stored-pixels",
        "fig-pixels",
        "east-north-telescope-roll-and-active-image-rotation",
        "native-and-fixed-angular-grids",
        "density-integrated-pixels-and-photometric-normalization",
        "coherent-phase-opd-and-the-adjoint-measure",
        "signed-intensity-changes-and-absolute-light",
        "lenslet-collection-psflet-placement-and-extracted-spectra",
        "independent-acceptance-fixtures-and-adoption-stages",
    ],
    "conventions/inference-records": [
        "measurements-probability-and-records",
        "define-the-experiment-before-the-uncertainty",
        "fig-measurement",
        "the-simplest-information-test",
        "covariance-describes-an-ordered-dimensional-vector",
        "a-posterior-needs-coordinates-and-a-normalization-history",
        "records-distinguish-acquisition-interpretation-and-use",
        "a-reproducible-campaign-preserves-causality",
        "named-fixtures-and-coverage",
    ],
    "conventions/figures-and-color": [
        "figures-and-color-one-encoding-per-quantity",
        "the-principle",
        "fig-color-roles",
        "images-colormap-by-quantity",
        "curves-and-markers-color-by-role",
        "time-wavelength-and-ensembles",
        "named-acceptance-fixtures",
        "coverage",
    ],
    "conventions/figures": [
        "figure-atlas",
        "atlas",
        "figure-1-direction-before-formula",
        "fig-atlas-geometry",
        "figure-2-a-number-is-not-an-epoch",
        "fig-atlas-time",
        "figure-3-the-physical-quantity-changes-at-each-boundary",
        "fig-atlas-pipeline",
        "figure-4-coordinate-and-measure-changes-are-separate",
        "fig-atlas-pixels",
        "figure-5-the-record-defines-the-inference-problem",
        "fig-atlas-measurement",
        "figure-6-integration-follows-the-dependencies",
        "fig-atlas-roadmap",
        "rebuild-and-provenance",
        "contributing-figures",
    ],
    "examples/index": ["examples"],
    "examples/orbit-astrometry": ["orbit-astrometry"],
    "examples/exposure-time-and-contrast": ["exposure-time-and-contrast"],
    "examples/fit-and-forecast": ["fit-and-forecast"],
    "examples/psf-and-speckles": ["psf-and-speckles"],
    "examples/scene-to-detection": [
        "scene-to-image-to-detection",
        "the-coronagraph-sets-the-scale",
        "the-scene",
        "the-optical-path",
        "the-image",
        "detection",
        "the-image-against-the-declared-contrast",
    ],
}

LEGACY_CASES = [(page, anchor) for page, ids in LEGACY_ANCHORS.items() for anchor in ids]


def _slug(text):
    """Section id the way docutils derives it from a heading."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"^[^a-z]+", "", text)


def _source_targets(page):
    """Anchors a Markdown source page defines: headings, labels, figure names."""
    text = (DOCS / f"{page}.md").read_text()
    targets = {_slug(h) for h in re.findall(r"^#{1,6}\s+(.+)$", text, re.M)}
    targets |= set(re.findall(r"^\(([\w-]+)\)=\s*$", text, re.M))
    targets |= set(re.findall(r"^:name:\s*([\w-]+)\s*$", text, re.M))
    targets |= set(re.findall(r'<span id="([\w-]+)"></span>', text))
    if page == "map":
        from build_map import LAYER_NAMES

        targets |= {_slug(title) for title in LAYER_NAMES.values()}
    return targets


@pytest.mark.parametrize(("page", "anchor"), LEGACY_CASES)
def test_legacy_anchor_still_defined_in_source(page, anchor):
    assert (DOCS / f"{page}.md").exists(), f"legacy page {page} was removed"
    assert anchor in _source_targets(page), f"{page}#{anchor} no longer has a target"


def test_rendered_legacy_anchors_resolve():
    """Check the built HTML when SPOHNBOOK_HTML_DIR names a finished build."""
    html_dir = os.environ.get("SPOHNBOOK_HTML_DIR")
    if not html_dir:
        pytest.skip("set SPOHNBOOK_HTML_DIR to a built site to check rendered anchors")
    missing = []
    for page, anchor in LEGACY_CASES:
        path = Path(html_dir) / f"{page}.html"
        if not path.exists() or f'id="{anchor}"' not in path.read_text():
            missing.append(f"{page}.html#{anchor}")
    assert not missing, missing
