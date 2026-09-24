"""Checks on the build manifest: complete identities, change detection, privacy."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "docs"))

import record_build as rb  # noqa: E402
from handbook import evidence_rows, load_catalog  # noqa: E402

PILOT = "jy-to-ideal-detector-shot-variance"


def _ledger(tmp_path, commit, outcome="passed"):
    catalog = load_catalog(ROOT / "docs")
    (case,) = [c for c in catalog["cases"] if c["id"] == PILOT]
    ledger = {
        "library": "spohnbook",
        "commit": commit[:7],
        "recorded": "2026-01-01T00:00:00Z",
        "tests": [
            {
                "nodeid": n,
                "case": PILOT,
                "evidence": "code-verification",
                "outcome": outcome,
            }
            for n in case["tests"]
        ],
    }
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    return path, ledger


def _lock(tmp_path, text="pytest==8.0.0 \\\n    --hash=sha256:00\n"):
    path = tmp_path / "requirements-docs.txt"
    path.write_text(text)
    return path


def _manifest(tmp_path, **kwargs):
    commit = rb.git_identity(ROOT)["commit"]
    ledger_path, _ = _ledger(tmp_path, commit)
    options = {
        "repo_root": ROOT,
        "ledger_path": ledger_path,
        "lock_path": _lock(tmp_path),
        "command": ["pytest", "--vv-evidence", str(ledger_path)],
        "resolution": "locked",
        "inputs": [],
    }
    options.update(kwargs)
    return rb.build_manifest(**options)


def test_manifest_records_every_required_identity(tmp_path):
    manifest = _manifest(tmp_path)
    assert rb.missing_identities(manifest) == []
    assert len(manifest["book"]["commit"]) == 40
    assert isinstance(manifest["book"]["dirty"], bool)
    assert manifest["python"]["version"]
    assert manifest["platform"]["system"]
    assert manifest["packages"]["pytest"]["version"]
    assert manifest["numerical_precision"]["tests"]["jax_enable_x64"] is True
    assert manifest["resolution"] == "locked"


@pytest.mark.parametrize(
    "field", ["book", "catalog_sha256", "ledger", "lock", "packages"]
)
def test_missing_identity_is_reported(tmp_path, field):
    manifest = _manifest(tmp_path)
    del manifest[field]
    assert rb.missing_identities(manifest)


def test_changed_lock_changes_its_identity(tmp_path):
    first = _manifest(tmp_path)["lock"]["sha256"]
    other = tmp_path / "other"
    other.mkdir()
    second = _manifest(tmp_path, lock_path=_lock(other, "pytest==8.0.1\n"))["lock"]
    assert second["sha256"] != first


def test_lock_pins_are_compared_with_the_environment():
    pins = rb.parse_lock("Foo_Bar==1.0 \\\n    --hash=sha256:ab\nbaz==2.0\n# c\n")
    assert pins == {"foo-bar": "1.0", "baz": "2.0"}
    mismatches = rb.lock_mismatches(
        pins, {"foo-bar": {"version": "1.0"}, "baz": {"version": "2.1"}}
    )
    assert mismatches == ["baz: lock 2.0, installed 2.1"]
    assert rb.lock_mismatches({"qux": "1"}, {}) == ["qux: lock 1, not installed"]


def test_changed_input_bytes_change_their_identity(tmp_path):
    data = tmp_path / "input.zip"
    data.write_bytes(b"first")
    first = rb.file_sha256(data)
    data.write_bytes(b"second")
    assert rb.file_sha256(data) != first


def test_absent_input_is_recorded_as_absent_not_archived(tmp_path):
    record = rb.input_identity(
        {
            "name": "x",
            "owner": "lib",
            "upstream": "https://example.org/x.zip",
            "registry_hash": "md5:00",
        },
        tmp_path / "missing.zip",
    )
    assert record["sha256"] is None
    assert record["status"] == "not present"
    assert record["archived"] is False


def test_manifest_contains_no_private_paths(tmp_path):
    manifest = _manifest(tmp_path)
    text = json.dumps(manifest)
    for root in rb.private_roots(ROOT):
        assert root not in text
    assert rb.private_path_hits(manifest, rb.private_roots(ROOT)) == []


def test_private_path_is_detected_and_canonicalized():
    home = str(Path.home())
    obj = {"command": ["python", f"{home}/secret/ledger.json"]}
    assert rb.private_path_hits(obj, [home])
    clean = rb.canonicalize(obj, {home: "~"})
    assert rb.private_path_hits(clean, [home]) == []
    assert clean["command"][1] == "~/secret/ledger.json"


def test_manifest_and_ledger_join_for_the_pilot_case(tmp_path):
    catalog = load_catalog(ROOT / "docs")
    commit = rb.git_identity(ROOT)["commit"]
    ledger_path, ledger = _ledger(tmp_path, commit)
    manifest = _manifest(tmp_path, ledger_path=ledger_path)
    manifest["book"]["dirty"] = False  # the check below concerns identity joins
    environment = {k: v["version"] for k, v in manifest["packages"].items()}
    rows = {
        r["case"]: r
        for r in evidence_rows(catalog, ledger, manifest, environment=environment)
    }
    assert rows[PILOT]["status"] == "passed"

    newer = dict(environment, optixstuff="999.0.0")
    rows = {
        r["case"]: r
        for r in evidence_rows(catalog, ledger, manifest, environment=newer)
    }
    assert rows[PILOT]["status"] == "stale"

    other_ledger = dict(ledger, recorded="2026-02-02T00:00:00Z")
    rows = {
        r["case"]: r
        for r in evidence_rows(catalog, other_ledger, manifest, environment=environment)
    }
    assert rows[PILOT]["status"] == "stale"
