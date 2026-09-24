#!/usr/bin/env python3
"""Assemble the archival bundle of an edition from its locked build.

The bundle holds the source at the release reference, the rendered HTML, the
source catalog, the citation metadata, the dependency lock, the evidence ledger,
the build manifest and logs, the identity of every input data file, a README and
a SHA-256 checksum of every file, as a directory and a .tar.gz of it. The tool
refuses a build manifest recorded for another commit or for uncommitted source,
so a bundle cannot pair one edition's text with another build's evidence.

It publishes nothing and mints no identifier.

Usage: make_release_bundle.py --ref v0.0.1 --html docs/_build/html
       --evidence docs/_evidence --output dist-bundle
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README = """Release bundle of The Spohn Book, {ref} (commit {commit}).

source.tar.gz          the repository at {ref}, as git archive produces it
html/                  the rendered site
handbook.yaml          the source catalog
CITATION.cff           citation metadata
requirements-docs.txt  the dependency lock the build used
evidence/              the evidence ledger, the build manifest and the logs
inputs.json            identities of the input data; the bytes are not included
SHA256SUMS             checksum of every file; verify with sha256sum -c SHA256SUMS

Reproduce the build with the procedure on the book's "Reproducing an edition"
page. No DOI is recorded in this bundle unless CITATION.cff carries one.
"""


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, check=True
    ).stdout


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bundle(ref, html_dir, evidence_dir, output_dir):
    """Write the bundle directory and its .tar.gz; return the directory.

    Raises:
        ValueError: if the manifest is missing, names another commit, or records
            uncommitted source.
    """
    commit = _git("rev-parse", f"{ref}^{{commit}}").decode().strip()
    manifest_path = Path(evidence_dir) / "build-manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"no build manifest in {evidence_dir}")
    manifest = json.loads(manifest_path.read_text())
    book = manifest.get("book") or {}
    if book.get("commit") != commit:
        raise ValueError(
            f"the manifest records commit {book.get('commit')}, not {ref} ({commit})"
        )
    if book.get("dirty") is not False:
        raise ValueError("the manifest records a build from uncommitted source")

    name = f"spohnbook-{ref.lstrip('v') if ref != 'HEAD' else commit[:12]}"
    bundle = Path(output_dir) / name
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    (bundle / "source.tar.gz").write_bytes(
        _git("archive", "--format=tar.gz", f"--prefix={name}/", commit)
    )
    shutil.copytree(html_dir, bundle / "html")
    shutil.copytree(evidence_dir, bundle / "evidence")
    for path, target in (
        ("docs/_data/handbook.yaml", "handbook.yaml"),
        ("CITATION.cff", "CITATION.cff"),
        ("requirements-docs.txt", "requirements-docs.txt"),
    ):
        (bundle / target).write_bytes(_git("show", f"{commit}:{path}"))
    inputs = {
        "bytes_included": False,
        "note": "Identities only. Retrieve each file from its upstream address and "
        "compare its SHA-256; this bundle does not archive the bytes.",
        "inputs": manifest.get("inputs", []),
    }
    (bundle / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (bundle / "README.txt").write_text(README.format(ref=ref, commit=commit))
    files = sorted(
        p for p in bundle.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    )
    sums = [f"{_sha256(p)}  {p.relative_to(bundle).as_posix()}" for p in files]
    (bundle / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with tarfile.open(str(bundle) + ".tar.gz", "w:gz") as archive:
        archive.add(bundle, arcname=name)
    return bundle


def main(argv=None):
    """Build the bundle from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--html", type=Path, default=ROOT / "docs/_build/html")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/_evidence")
    parser.add_argument("--output", type=Path, default=ROOT / "dist-bundle")
    args = parser.parse_args(argv)
    try:
        bundle = make_bundle(args.ref, args.html, args.evidence, args.output)
    except ValueError as err:
        print(f"refusing to build the bundle: {err}", file=sys.stderr)
        return 1
    print(bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
