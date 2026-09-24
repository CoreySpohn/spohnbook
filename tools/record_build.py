#!/usr/bin/env python3
"""Record the identity of a documentation build and its evidence run.

The manifest names the book commit, the Python and platform, every installed
distribution with its version and origin, the lock, catalog, profile and ledger
digests, the numerical precision of the tests, the input data bytes the
examples read, and the command that ran. The evidence page joins a ledger to
the manifest only when these identities match, so a manifest recorded for other
bytes cannot certify a later build.

Paths under the user's home directory or the checkout are rewritten before the
manifest is written, and the tool refuses to write one that still contains them.

Usage: record_build.py --output PATH --ledger PATH [--lock PATH]
       [--resolution locked|latest] [--command TEXT]
"""

import argparse
import ast
import getpass
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from handbook import (  # noqa: E402
    catalog_digest,
    json_digest,
    load_catalog,
    profile_digest,
)

SCHEMA_VERSION = 1

# Input data the executed examples read, found by building the pages with empty
# caches. Each entry is located in its owning library's download cache.
INPUTS = [
    {
        "name": "eac1_optimal_order_6_1d",
        "owner": "yippy",
        "kind": "yield input package (coronagraph response)",
        "used_by": [
            "examples/exposure-time-and-contrast",
            "examples/scene-to-detection",
        ],
    },
]


def _normalize(name):
    return name.lower().replace("_", "-").replace(".", "-")


def git_identity(repo_root):
    """The checked-out commit and whether tracked or untracked files changed."""

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def installed_packages():
    """Every installed distribution: version, installer and origin, without paths."""
    packages = {}
    for dist in metadata.distributions():
        name = _normalize(dist.metadata["Name"] or "")
        if not name:
            continue
        record = {"version": dist.version, "source": "index"}
        installer = dist.read_text("INSTALLER")
        if installer:
            record["installer"] = installer.strip()
        direct = dist.read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            url = info.get("url", "")
            if "vcs_info" in info:
                record["source"] = "vcs"
                record["url"] = url
                record["commit"] = info["vcs_info"].get("commit_id")
            elif url.startswith("file:"):
                editable = info.get("dir_info", {}).get("editable", False)
                record["source"] = "local-editable" if editable else "local"
            else:
                record["source"] = "url"
                record["url"] = url
        packages[name] = record
    return dict(sorted(packages.items()))


def parse_lock(text):
    """Pinned versions from a requirements lock, keyed by normalized name."""
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip().rstrip("\\").strip()
        if "==" in line and not line.startswith("-"):
            name, version = line.split("==", 1)
            name = name.split("[", 1)[0].strip()
            pins[_normalize(name)] = version.split(";", 1)[0].strip()
    return pins


def lock_mismatches(pins, packages):
    """Differences between the lock's pins and the installed versions."""
    problems = []
    for name, version in sorted(pins.items()):
        installed = (packages.get(name) or {}).get("version")
        if installed is None:
            problems.append(f"{name}: lock {version}, not installed")
        elif installed != version:
            problems.append(f"{name}: lock {version}, installed {installed}")
    return problems


def file_sha256(path):
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_identity(entry, path):
    """Identity of one input file; recording a hash does not archive the bytes."""
    record = {
        key: entry.get(key)
        for key in ("name", "owner", "kind", "upstream", "registry_hash", "used_by")
    }
    record["archived"] = False
    record["license"] = entry.get("license", "not recorded; see the owning library")
    if path is not None and Path(path).exists():
        record["sha256"] = file_sha256(path)
        record["bytes"] = Path(path).stat().st_size
        record["status"] = "present"
    else:
        record["sha256"] = None
        record["status"] = "not present"
    return record


def _locate(entry, fetch=False):
    """Upstream URL, registry hash and cached path of a declared input.

    With ``fetch``, the owning library downloads the file first (verifying its
    registry hash), so the manifest can be recorded before the pages execute.
    """
    if entry["owner"] == "yippy":
        from yippy import datasets

        name = entry["name"]
        if fetch:
            datasets.fetch_yip(name)
        filename = f"{name}.zip"
        entry = dict(
            entry,
            upstream=datasets._DATA_BASE_URL + filename,
            registry_hash=datasets.CATALOG[name]["md5"],
        )
        return entry, Path(datasets.cache_dir()) / filename
    return entry, None


def input_identities(inputs, fetch=False):
    """Identity records for every declared input."""
    records = []
    for entry in inputs:
        try:
            located, path = _locate(entry, fetch=fetch)
        except ImportError:
            located, path = entry, None
        records.append(input_identity(located, path))
    return records


def test_precision(repo_root):
    """The JAX_ENABLE_X64 constant the test configuration sets."""
    conftest = Path(repo_root) / "tests" / "conftest.py"
    for node in ast.parse(conftest.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "JAX_ENABLE_X64" for t in node.targets
        ):
            return {"jax_enable_x64": ast.literal_eval(node.value)}
    return {"jax_enable_x64": None}


def private_roots(repo_root):
    """Strings that identify this machine's user or checkout location."""
    roots = {str(Path.home()), str(Path(repo_root).resolve())}
    try:
        roots.add(f"/{getpass.getuser()}/")
    except (KeyError, OSError):
        pass
    return sorted(roots, key=len, reverse=True)


def canonicalize(obj, replacements):
    """Rewrite private path prefixes inside every string of a JSON-like object."""
    if isinstance(obj, dict):
        return {k: canonicalize(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [canonicalize(v, replacements) for v in obj]
    if isinstance(obj, str):
        for old, new in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
            obj = obj.replace(old, new)
    return obj


def private_path_hits(obj, roots):
    """Private strings still present in the object."""
    text = json.dumps(obj)
    return [root for root in roots if root in text]


def build_manifest(
    repo_root,
    ledger_path,
    lock_path,
    command,
    resolution,
    inputs=None,
    fetch_inputs=False,
):
    """Assemble the manifest dictionary for one build."""
    repo_root = Path(repo_root)
    catalog = load_catalog(repo_root / "docs")
    packages = installed_packages()
    ledger = json.loads(Path(ledger_path).read_text())
    lock_text = Path(lock_path).read_text()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "book": dict(
            git_identity(repo_root),
            version=(packages.get("spohnbook") or {}).get("version"),
        ),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "libc": " ".join(platform.libc_ver()).strip() or None,
        },
        "resolution": resolution,
        "packages": packages,
        "lock": {
            "path": Path(lock_path).name,
            "sha256": file_sha256(lock_path),
            "matches_environment": None,
            "mismatches": [],
        },
        "catalog_sha256": catalog_digest(catalog),
        "profiles": {
            p["id"]: profile_digest(catalog, p["id"]) for p in catalog["profiles"]
        },
        "ledger": {
            "path": Path(ledger_path).name,
            "sha256": json_digest(ledger),
            "file_sha256": file_sha256(ledger_path),
            "commit": ledger.get("commit"),
        },
        "numerical_precision": {
            "tests": test_precision(repo_root),
            "examples": "each example sets its precision in its first code cell",
        },
        "inputs": input_identities(
            INPUTS if inputs is None else inputs, fetch=fetch_inputs
        ),
        "command": list(command),
    }
    mismatches = lock_mismatches(parse_lock(lock_text), packages)
    manifest["lock"]["mismatches"] = mismatches
    manifest["lock"]["matches_environment"] = not mismatches
    replacements = {str(repo_root.resolve()): "<checkout>", str(Path.home()): "~"}
    return canonicalize(manifest, replacements)


def missing_identities(manifest):
    """Required identities that are absent from a manifest."""
    missing = []
    book = manifest.get("book") or {}
    if not book.get("commit"):
        missing.append("book commit")
    if not isinstance(book.get("dirty"), bool):
        missing.append("book dirty state")
    for key in ("catalog_sha256", "resolution"):
        if not manifest.get(key):
            missing.append(key)
    if not manifest.get("profiles"):
        missing.append("profile digests")
    if not (manifest.get("ledger") or {}).get("sha256"):
        missing.append("ledger digest")
    if not (manifest.get("lock") or {}).get("sha256"):
        missing.append("lock digest")
    if not manifest.get("packages"):
        missing.append("installed packages")
    if not (manifest.get("python") or {}).get("version"):
        missing.append("python version")
    if not (manifest.get("platform") or {}).get("system"):
        missing.append("platform")
    for record in manifest.get("inputs", []):
        if record.get("status") != "present" or not record.get("sha256"):
            missing.append(f"input identity of {record.get('name')} (not present)")
    return missing


def manifest_problems(manifest, roots):
    """Every reason the manifest cannot serve as complete evidence identity."""
    problems = [f"missing {m}" for m in missing_identities(manifest)]
    problems += [f"private path {h!r}" for h in private_path_hits(manifest, roots)]
    lock = manifest.get("lock") or {}
    if manifest.get("resolution") == "locked" and lock.get("mismatches"):
        problems.append(
            "resolution is locked but the environment differs from the lock: "
            + "; ".join(lock["mismatches"][:5])
        )
    return problems


def main(argv=None):
    """Write the manifest; exit nonzero if it is incomplete or inconsistent.

    A manifest that still contains a private path is never written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--lock", type=Path, default=ROOT / "requirements-docs.txt")
    parser.add_argument("--resolution", choices=["locked", "latest"], default="locked")
    parser.add_argument("--command", default="", help="the commands this run executed")
    parser.add_argument(
        "--fetch-inputs",
        action="store_true",
        help="download declared input data through its owning library first",
    )
    args = parser.parse_args(argv)
    manifest = build_manifest(
        ROOT,
        args.ledger,
        args.lock,
        [args.command] if args.command else [],
        args.resolution,
        fetch_inputs=args.fetch_inputs,
    )
    roots = private_roots(ROOT)
    problems = manifest_problems(manifest, roots)
    if private_path_hits(manifest, roots):
        print("refusing to write a manifest with private paths", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if problems:
        print(
            "build manifest is incomplete:\n  " + "\n  ".join(problems), file=sys.stderr
        )
        return 1
    if manifest["book"]["dirty"]:
        print("warning: the checkout has uncommitted changes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
