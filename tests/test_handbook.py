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

LEGACY_CASES = [
    (page, anchor) for page, ids in LEGACY_ANCHORS.items() for anchor in ids
]


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


# ---------------------------------------------------------------------------
# Catalog structure
# ---------------------------------------------------------------------------


def _mini_docs(tmp_path):
    """A docs tree with one chapter carrying two labels and a decision page."""
    docs = tmp_path / "docs"
    (docs / "conventions").mkdir(parents=True)
    (docs / "profiles").mkdir()
    (docs / "conventions" / "chapter.md").write_text(
        "# Chapter\n\n(clause-borrowed)=\n## Borrowed\n\ntext\n\n"
        "(clause-derived)=\n## Derived\n\ntext\n"
    )
    (docs / "profiles" / "decisions.md").write_text(
        "# Decisions\n\n(decision-some-choice)=\n## Some choice\n"
    )
    return docs


def _mini_catalog():
    return {
        "schema_version": 1,
        "sources": [
            {
                "id": "author2000",
                "authors": "Author, A.",
                "year": 2000,
                "title": "A paper",
                "venue": "A journal",
                "doi": "10.0000/example",
            }
        ],
        "clauses": [
            {
                "id": "clause-borrowed",
                "page": "conventions/chapter",
                "citations": [
                    {"source": "author2000", "locator": "Eq. 3", "supports": "x"}
                ],
            },
            {
                "id": "clause-derived",
                "page": "conventions/chapter",
                "derivation": "follows from the stated definitions",
            },
        ],
        "profiles": [
            {
                "id": "example-profile-v1",
                "status": "proposed",
                "title": "Example",
                "clauses": ["clause-borrowed", "clause-derived"],
            }
        ],
        "cases": [
            {
                "id": "example-case",
                "profile": "example-profile-v1",
                "claim": "a precise claim",
                "evidence": "code-verification",
                "packages": ["numpy"],
                "tests": ["tests/test_x.py::test_a", "tests/test_x.py::test_b"],
            }
        ],
    }


def test_valid_catalog_has_no_errors(tmp_path):
    from handbook import validate_catalog

    assert validate_catalog(_mini_catalog(), _mini_docs(tmp_path)) == []


def test_author_derived_clause_needs_no_citation(tmp_path):
    from handbook import validate_catalog

    catalog = _mini_catalog()
    assert "citations" not in catalog["clauses"][1]
    assert validate_catalog(catalog, _mini_docs(tmp_path)) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda c: c["sources"].append(dict(c["sources"][0])), "duplicate source"),
        (lambda c: c["clauses"].append(dict(c["clauses"][1])), "duplicate clause"),
        (
            lambda c: c["clauses"][0]["citations"][0].update(source="nobody1999"),
            "unknown source",
        ),
        (lambda c: c["clauses"][0]["citations"][0].pop("locator"), "no locator"),
        (lambda c: c["clauses"][0]["citations"][0].update(locator=" "), "no locator"),
        (lambda c: c["clauses"][1].update(id="clause-missing"), "no label"),
        (lambda c: c["clauses"][1].update(page="conventions/nope"), "no page"),
        (lambda c: c["clauses"][1].pop("derivation"), "no basis"),
        (lambda c: c["profiles"][0]["clauses"].append("clause-x"), "unknown clause"),
        (lambda c: c["profiles"][0].update(status="verified"), "status"),
        (lambda c: c["profiles"][0].update(status="adopted"), "decision"),
        (lambda c: c["cases"][0].update(profile="other-v1"), "unknown profile"),
        (lambda c: c["cases"][0].update(tests=[]), "no tests"),
        (lambda c: c["cases"][0].update(evidence="proof"), "evidence kind"),
        (lambda c: c["sources"][0].pop("doi"), "needs a doi"),
        (lambda c: c.update(schema_version=2), "schema_version"),
        (
            lambda c: c["clauses"][1].update(decisions=["decision-absent"]),
            "unknown decision",
        ),
    ],
)
def test_malformed_catalog_is_rejected(tmp_path, mutate, expected):
    from handbook import validate_catalog

    catalog = _mini_catalog()
    mutate(catalog)
    errors = validate_catalog(catalog, _mini_docs(tmp_path))
    assert any(expected in e for e in errors), errors


def test_adopted_profile_with_recorded_decision_is_accepted(tmp_path):
    from handbook import validate_catalog

    catalog = _mini_catalog()
    catalog["profiles"][0].update(status="adopted", decision="decision-some-choice")
    assert validate_catalog(catalog, _mini_docs(tmp_path)) == []


def test_repository_catalog_is_valid():
    from handbook import load_catalog, validate_catalog

    assert validate_catalog(load_catalog(DOCS), DOCS) == []


def test_every_cited_source_is_cited_on_its_clause_page():
    """A catalog citation must also appear in the chapter prose it supports."""
    from handbook import load_catalog

    catalog = load_catalog(DOCS)
    missing = []
    for clause in catalog["clauses"]:
        text = (DOCS / f"{clause['page']}.md").read_text()
        for citation in clause.get("citations", []):
            if f"<source-{citation['source']}>" not in text:
                missing.append(f"{clause['id']} -> {citation['source']}")
    assert not missing, missing


def test_initial_profile_stays_proposed():
    from handbook import load_catalog

    profiles = {p["id"]: p for p in load_catalog(DOCS)["profiles"]}
    assert profiles["photon-electron-reference-v1"]["status"] == "proposed"


# ---------------------------------------------------------------------------
# Evidence states
# ---------------------------------------------------------------------------

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _ledger(outcomes, case="example-case", commit=COMMIT[:7]):
    return {
        "library": "spohnbook",
        "commit": commit,
        "recorded": "2026-01-01T00:00:00Z",
        "tests": [
            {
                "nodeid": nodeid,
                "case": case,
                "evidence": "code-verification",
                "outcome": outcome,
            }
            for nodeid, outcome in outcomes.items()
        ],
    }


def _manifest(catalog, ledger, **overrides):
    from handbook import catalog_digest, json_digest, profile_digest

    manifest = {
        "schema_version": 1,
        "book": {"commit": COMMIT, "dirty": False},
        "catalog_sha256": catalog_digest(catalog),
        "profiles": {
            p["id"]: profile_digest(catalog, p["id"]) for p in catalog["profiles"]
        },
        "ledger": {"sha256": json_digest(ledger)},
        "packages": {"numpy": {"version": "2.0.0"}},
    }
    manifest.update(overrides)
    return manifest


ENV = {"numpy": "2.0.0"}
BOTH_PASS = {"tests/test_x.py::test_a": "passed", "tests/test_x.py::test_b": "passed"}


def _status(ledger, manifest, catalog=None, environment=ENV):
    from handbook import evidence_rows

    catalog = catalog or _mini_catalog()
    (row,) = evidence_rows(catalog, ledger, manifest, environment=environment)
    return row


def test_no_ledger_or_manifest_is_unassessed():
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    assert _status(None, None)["status"] == "unassessed"
    assert _status(ledger, None)["status"] == "unassessed"
    assert _status(None, _manifest(catalog, ledger))["status"] == "unassessed"


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ({"tests/test_x.py::test_a": "passed"}, "incomplete"),
        (
            {"tests/test_x.py::test_a": "passed", "tests/test_x.py::test_b": None},
            "incomplete",
        ),
        (
            {"tests/test_x.py::test_a": "passed", "tests/test_x.py::test_b": "skipped"},
            "incomplete",
        ),
        (
            {"tests/test_x.py::test_a": "passed", "tests/test_x.py::test_b": "xfailed"},
            "incomplete",
        ),
        (
            {"tests/test_x.py::test_a": "passed", "tests/test_x.py::test_b": "failed"},
            "failed",
        ),
        (
            {"tests/test_x.py::test_a": "error", "tests/test_x.py::test_b": "passed"},
            "failed",
        ),
        (BOTH_PASS, "passed"),
    ],
)
def test_outcomes_map_to_case_status(outcomes, expected):
    catalog = _mini_catalog()
    ledger = _ledger(outcomes)
    assert _status(ledger, _manifest(catalog, ledger))["status"] == expected


def test_unrelated_passing_tests_do_not_satisfy_a_binding():
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS, case="another-case")
    ledger["tests"].append(
        {
            "nodeid": "tests/test_y.py::test_c",
            "case": "example-case",
            "evidence": "code-verification",
            "outcome": "passed",
        }
    )
    assert _status(ledger, _manifest(catalog, ledger))["status"] == "incomplete"


@pytest.mark.parametrize(
    "change",
    ["dependency", "profile", "catalog", "commit", "ledger-bytes"],
)
def test_identity_mismatch_is_stale(change):
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    manifest = _manifest(catalog, ledger)
    environment = dict(ENV)
    if change == "dependency":
        environment["numpy"] = "2.1.0"
    elif change == "profile":
        catalog = _mini_catalog()
        catalog["profiles"][0]["clauses"].pop()
    elif change == "catalog":
        catalog = _mini_catalog()
        catalog["sources"][0]["title"] = "A revised title"
    elif change == "commit":
        ledger = _ledger(BOTH_PASS, commit="fedcba9")
    elif change == "ledger-bytes":
        ledger = _ledger(BOTH_PASS)
        ledger["recorded"] = "2026-02-02T00:00:00Z"
    row = _status(ledger, manifest, catalog=catalog, environment=environment)
    assert row["status"] == "stale", row


def test_stale_takes_precedence_over_a_recorded_pass():
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    manifest = _manifest(catalog, ledger)
    row = _status(ledger, manifest, environment={"numpy": "9.9.9"})
    assert row["status"] == "stale"
    assert any("numpy" in r for r in row["reasons"])


@pytest.mark.parametrize("missing", ["catalog_sha256", "profiles", "ledger", "book"])
def test_missing_identity_is_incomplete(missing):
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    manifest = _manifest(catalog, ledger)
    del manifest[missing]
    assert _status(ledger, manifest)["status"] == "incomplete"


def test_uncommitted_source_is_incomplete():
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    manifest = _manifest(catalog, ledger, book={"commit": COMMIT, "dirty": True})
    assert _status(ledger, manifest)["status"] == "incomplete"


def test_passed_case_keeps_profile_proposed_and_validation_unassessed():
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    row = _status(ledger, _manifest(catalog, ledger))
    assert row["status"] == "passed"
    assert row["profile_status"] == "proposed"
    assert row["validation"] == "not assessed"


def test_support_view_never_prints_verified_or_validated_for_a_case():
    from handbook import render_evidence_table

    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    rows = [_status(ledger, _manifest(catalog, ledger))]
    text = render_evidence_table(rows).lower()
    assert "passed" in text
    assert "verified" not in text.replace("verification", "")
    assert "validated" not in text


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_evidence_for_another_book_commit_is_stale():
    from handbook import evidence_rows

    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    manifest = _manifest(catalog, ledger)
    (row,) = evidence_rows(catalog, ledger, manifest, environment=ENV, commit="f" * 40)
    assert row["status"] == "stale"
    assert any("commit" in r for r in row["reasons"])
    (row,) = evidence_rows(catalog, ledger, manifest, environment=ENV, commit=COMMIT)
    assert row["status"] == "passed"


@pytest.mark.parametrize("source", ["local-editable", "local", "vcs", "url"])
def test_case_package_not_from_the_index_is_incomplete(source):
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    manifest = _manifest(catalog, ledger)
    manifest["packages"]["numpy"]["source"] = source
    row = _status(ledger, manifest)
    assert row["status"] == "incomplete"
    assert any(source in r for r in row["reasons"])


def test_case_without_packages_is_rejected(tmp_path):
    from handbook import validate_catalog

    catalog = _mini_catalog()
    catalog["cases"][0]["packages"] = []
    errors = validate_catalog(catalog, _mini_docs(tmp_path))
    assert any("packages" in e for e in errors), errors


@pytest.mark.parametrize(
    ("version", "tags", "expected"),
    [
        ("0.0.1", ["v0.0.1"], "edition"),
        ("0.0.1", [], "development"),
        ("0.1.dev3+gabc", [], "development"),
        ("0.0.1", ["v0.0.2"], "development"),
    ],
)
def test_edition_claim_requires_the_tag_at_the_built_commit(version, tags, expected):
    from handbook import edition_state

    assert edition_state(version, tags) == expected


def test_locked_manifest_that_differs_from_the_lock_is_incomplete():
    catalog = _mini_catalog()
    ledger = _ledger(BOTH_PASS)
    manifest = _manifest(
        catalog,
        ledger,
        resolution="locked",
        lock={"mismatches": ["numpy: lock 1.0, installed 2.0.0"]},
    )
    assert _status(ledger, manifest)["status"] == "incomplete"
