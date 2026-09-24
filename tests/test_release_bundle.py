"""The release bundle: complete, checksummed, and bound to its own commit."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_release_bundle as mrb  # noqa: E402
import record_build as rb  # noqa: E402


def _inputs(tmp_path, commit):
    html = tmp_path / "html"
    html.mkdir()
    (html / "index.html").write_text("<html></html>")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "ledger.json").write_text(json.dumps({"commit": commit[:7]}))
    manifest = {
        "book": {"commit": commit, "dirty": False},
        "inputs": [{"name": "x", "sha256": "ab", "archived": False}],
    }
    (evidence / "build-manifest.json").write_text(json.dumps(manifest))
    (evidence / "sphinx.log").write_text("build succeeded")
    return html, evidence


def test_bundle_contains_every_part_and_verifies(tmp_path):
    commit = rb.git_identity(ROOT)["commit"]
    html, evidence = _inputs(tmp_path, commit)
    bundle = mrb.make_bundle("HEAD", html, evidence, tmp_path / "out")
    names = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    for required in (
        "source.tar.gz",
        "html/index.html",
        "handbook.yaml",
        "CITATION.cff",
        "requirements-docs.txt",
        "evidence/ledger.json",
        "evidence/build-manifest.json",
        "evidence/sphinx.log",
        "inputs.json",
        "README.txt",
        "SHA256SUMS",
    ):
        assert required in names, required
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
    inputs = json.loads((bundle / "inputs.json").read_text())
    assert inputs["bytes_included"] is False
    assert Path(str(bundle) + ".tar.gz").exists()


def test_manifest_for_another_commit_is_refused(tmp_path):
    html, evidence = _inputs(tmp_path, "f" * 40)
    with pytest.raises(ValueError, match="commit"):
        mrb.make_bundle("HEAD", html, evidence, tmp_path / "out")


def test_uncommitted_build_is_refused(tmp_path):
    commit = rb.git_identity(ROOT)["commit"]
    html, evidence = _inputs(tmp_path, commit)
    path = evidence / "build-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["book"]["dirty"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="uncommitted"):
        mrb.make_bundle("HEAD", html, evidence, tmp_path / "out")


def test_bundle_with_private_paths_is_refused(tmp_path):
    commit = rb.git_identity(ROOT)["commit"]
    html, evidence = _inputs(tmp_path, commit)
    (evidence / "sphinx.log").write_text(f"warning in {Path.home()}/lib/x.py")
    with pytest.raises(ValueError, match="private"):
        mrb.make_bundle("HEAD", html, evidence, tmp_path / "out")
