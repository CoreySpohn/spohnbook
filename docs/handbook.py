"""Load and check the handbook catalog, and render its generated pages.

The catalog (``docs/_data/handbook.yaml``) binds stable clause labels to their
sources or derivations, lists the convention profiles, and binds reference
cases to the tests that exercise them. Chapter prose and the decision register
stay in Markdown; this module only checks the catalog's structure and renders
the tables that are derived from it. A structurally valid catalog says that a
citation and its locator exist, not that the source supports the prose.

Case statuses are computed from an ``ineedvalidation`` evidence ledger and a
build manifest written by ``tools/record_build.py``. Nothing here reads the
network or a private workspace.
"""

import hashlib
import json
import os
import re
import subprocess
from importlib import metadata
from pathlib import Path

import yaml

SCHEMA_VERSION = 1
SECTIONS = ("sources", "clauses", "profiles", "cases")
PROFILE_STATUSES = ("proposed", "adopted", "deprecated")
EVIDENCE_KINDS = (
    "code-verification",
    "solution-verification",
    "cross-code-benchmark",
    "validation",
    "uncertainty-quantification",
)
DECISIONS_PAGE = "profiles/decisions"
PASSED = "passed"
INCOMPLETE_OUTCOMES = (None, "skipped", "xfailed")


def load_catalog(docs_dir: Path) -> dict:
    """Read the catalog YAML from ``docs_dir/_data/handbook.yaml``."""
    return yaml.safe_load((Path(docs_dir) / "_data" / "handbook.yaml").read_text())


def json_digest(obj) -> str:
    """SHA-256 of an object's canonical JSON form."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def catalog_digest(catalog: dict) -> str:
    """Identity of the whole catalog."""
    return json_digest(catalog)


def profile_digest(catalog: dict, profile_id: str) -> str:
    """Identity of one profile: its entry, its clauses and their sources."""
    profile = _by_id(catalog.get("profiles", []))[profile_id]
    clauses = _by_id(catalog.get("clauses", []))
    sources = _by_id(catalog.get("sources", []))
    bound = [clauses[c] for c in profile.get("clauses", []) if c in clauses]
    cited = sorted(
        {cit["source"] for clause in bound for cit in clause.get("citations", [])}
    )
    return json_digest(
        {
            "profile": profile,
            "clauses": bound,
            "sources": [sources[s] for s in cited if s in sources],
        }
    )


def _by_id(items):
    return {
        item["id"]: item for item in items if isinstance(item, dict) and "id" in item
    }


def _page_targets(docs_dir, page):
    path = Path(docs_dir) / f"{page}.md"
    if not path.exists():
        return None
    return set(re.findall(r"^\(([\w-]+)\)=\s*$", path.read_text(), re.M))


def _duplicates(items, section):
    seen, errors = set(), []
    for item in items:
        ident = item.get("id") if isinstance(item, dict) else None
        if not ident:
            errors.append(f"{section}: entry without an id")
        elif ident in seen:
            errors.append(f"{section}: duplicate {section[:-1]} id {ident!r}")
        seen.add(ident)
    return errors


def validate_catalog(catalog: dict, docs_dir: Path) -> list[str]:
    """Return every structural error in the catalog; empty means valid."""
    errors = []
    if catalog.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for section in SECTIONS:
        if not isinstance(catalog.get(section), list):
            errors.append(f"missing section {section!r}")
    if errors:
        return errors
    for section in SECTIONS:
        errors += _duplicates(catalog[section], section)

    decision_targets = _page_targets(docs_dir, DECISIONS_PAGE) or set()
    sources = _by_id(catalog["sources"])
    for source in catalog["sources"]:
        for field in ("authors", "year", "title"):
            if not source.get(field):
                errors.append(f"source {source.get('id')!r}: missing {field}")
        if not (source.get("doi") or source.get("url") or source.get("isbn")):
            errors.append(f"source {source.get('id')!r}: needs a doi, url or isbn")

    for clause in catalog["clauses"]:
        cid = clause.get("id")
        targets = _page_targets(docs_dir, clause.get("page", ""))
        if targets is None:
            errors.append(f"clause {cid!r}: no page {clause.get('page')!r}")
        elif cid not in targets:
            errors.append(f"clause {cid!r}: no label ({cid})= on {clause['page']}")
        citations = clause.get("citations", [])
        decisions = clause.get("decisions", [])
        if not (citations or clause.get("derivation") or decisions):
            errors.append(
                f"clause {cid!r}: no basis (citations, derivation or decisions)"
            )
        for citation in citations:
            if citation.get("source") not in sources:
                errors.append(
                    f"clause {cid!r}: unknown source {citation.get('source')!r}"
                )
            if not str(citation.get("locator", "")).strip():
                cited = citation.get("source")
                errors.append(f"clause {cid!r}: citation of {cited!r} has no locator")
        for decision in decisions:
            if decision not in decision_targets:
                errors.append(f"clause {cid!r}: unknown decision {decision!r}")

    clauses = _by_id(catalog["clauses"])
    for profile in catalog["profiles"]:
        pid = profile.get("id")
        if profile.get("status") not in PROFILE_STATUSES:
            errors.append(f"profile {pid!r}: status must be one of {PROFILE_STATUSES}")
        if profile.get("status") == "adopted" and (
            profile.get("decision") not in decision_targets
        ):
            errors.append(f"profile {pid!r}: adopted without a recorded decision")
        for clause_id in profile.get("clauses", []):
            if clause_id not in clauses:
                errors.append(f"profile {pid!r}: unknown clause {clause_id!r}")

    profiles = _by_id(catalog["profiles"])
    for case in catalog["cases"]:
        kid = case.get("id")
        if case.get("profile") not in profiles:
            errors.append(f"case {kid!r}: unknown profile {case.get('profile')!r}")
        if not case.get("claim"):
            errors.append(f"case {kid!r}: missing claim")
        if case.get("evidence") not in EVIDENCE_KINDS:
            errors.append(
                f"case {kid!r}: evidence kind must be one of {EVIDENCE_KINDS}"
            )
        if not case.get("tests"):
            errors.append(f"case {kid!r}: no tests bound")
    return errors


def _installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _identity_problems(case, catalog, ledger, manifest, environment):
    """Return (missing, mismatched) identity descriptions for one case."""
    missing, mismatched = [], []
    book = manifest.get("book") or {}
    commit = book.get("commit")
    if not commit:
        missing.append("book commit")
    elif book.get("dirty") is not False:
        missing.append("committed source (the build had uncommitted changes)")
    if not manifest.get("catalog_sha256"):
        missing.append("catalog identity")
    elif manifest["catalog_sha256"] != catalog_digest(catalog):
        mismatched.append("catalog changed since the evidence was recorded")
    recorded_profile = (manifest.get("profiles") or {}).get(case["profile"])
    if not recorded_profile:
        missing.append(f"profile identity for {case['profile']}")
    elif recorded_profile != profile_digest(catalog, case["profile"]):
        mismatched.append(f"profile {case['profile']} changed")
    ledger_sha = (manifest.get("ledger") or {}).get("sha256")
    if not ledger_sha:
        missing.append("ledger identity")
    elif ledger_sha != json_digest(ledger):
        mismatched.append("ledger is not the one this manifest recorded")
    ledger_commit = ledger.get("commit")
    if not ledger_commit:
        missing.append("ledger commit")
    elif commit and not commit.startswith(ledger_commit):
        mismatched.append(f"ledger commit {ledger_commit} is not the build commit")
    packages = manifest.get("packages") or {}
    for name in case.get("packages", []):
        recorded = (packages.get(name) or {}).get("version")
        current = (
            environment.get(name)
            if environment is not None
            else _installed_version(name)
        )
        if not recorded:
            missing.append(f"version of {name}")
        elif recorded != current:
            mismatched.append(f"{name} {recorded} recorded, {current} present")
    return missing, mismatched


def _test_status(case, ledger):
    """Status from the bound tests' outcomes, and the reasons behind it."""
    records = {
        t.get("nodeid"): t
        for t in ledger.get("tests", [])
        if t.get("case") == case["id"]
    }
    failed, incomplete = [], []
    for nodeid in case["tests"]:
        record = records.get(nodeid)
        outcome = record.get("outcome") if record else None
        if record is None:
            incomplete.append(f"{nodeid} has no record for this case")
        elif outcome in INCOMPLETE_OUTCOMES:
            incomplete.append(f"{nodeid} {outcome or 'not run'}")
        elif outcome != PASSED:
            failed.append(f"{nodeid} {outcome}")
    if failed:
        return "failed", failed + incomplete
    if incomplete:
        return "incomplete", incomplete
    return PASSED, []


def evidence_rows(
    catalog: dict,
    ledger: dict | None,
    manifest: dict | None,
    *,
    environment: dict | None = None,
) -> list[dict]:
    """Compute one status row per case binding.

    Args:
        catalog: the loaded catalog.
        ledger: an ``ineedvalidation`` evidence file, or None.
        manifest: a build manifest from ``tools/record_build.py``, or None.
        environment: package versions present now; None reads the installed ones.

    Returns:
        Rows with the case, its profile and profile status, the claim, the
        computed status (unassessed, passed, failed, incomplete or stale), the
        reasons, the validation state and the evidence identity.
    """
    profiles = _by_id(catalog["profiles"])
    rows = []
    for case in catalog["cases"]:
        profile = profiles[case["profile"]]
        row = {
            "case": case["id"],
            "claim": case["claim"],
            "profile": case["profile"],
            "profile_status": profile["status"],
            "evidence": case["evidence"],
            "page": case.get("page"),
            "validation": "not assessed",
            "reasons": [],
            "identity": None,
        }
        if ledger is None or manifest is None:
            row["status"] = "unassessed"
            row["reasons"] = ["no evidence ledger and build manifest for this build"]
            rows.append(row)
            continue
        packages = manifest.get("packages") or {}
        row["identity"] = {
            "commit": (manifest.get("book") or {}).get("commit"),
            "recorded": ledger.get("recorded"),
            "packages": {
                name: (packages.get(name) or {}).get("version")
                for name in case.get("packages", [])
            },
        }
        missing, mismatched = _identity_problems(
            case, catalog, ledger, manifest, environment
        )
        if mismatched:
            row["status"], row["reasons"] = "stale", mismatched + missing
        elif missing:
            row["status"] = "incomplete"
            row["reasons"] = [f"missing {m}" for m in missing]
        else:
            row["status"], row["reasons"] = _test_status(case, ledger)
        if case["evidence"] == "validation":
            row["validation"] = row["status"]
        rows.append(row)
    return rows


def render_evidence_table(rows: list[dict]) -> str:
    """Markdown table of case results; a result never names a profile verified."""
    lines = [
        "| Case | Claim | Profile and its state | Evidence kind | Case result | "
        "Measured-data validation | Evidence identity |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        case = f"`{row['case']}`"
        if row.get("page"):
            case = f"{{doc}}`{row['case']} </{row['page']}>`"
        result = f"**{row['status']}**"
        if row["reasons"]:
            result += ": " + "; ".join(row["reasons"])
        identity = "none"
        if row["identity"]:
            ident = row["identity"]
            versions = ", ".join(f"{k} {v}" for k, v in ident["packages"].items())
            commit = (ident["commit"] or "unknown")[:12]
            identity = f"book {commit}; {versions}; recorded {ident['recorded']}"
        lines.append(
            f"| {case} | {row['claim']} | `{row['profile']}` ({row['profile_status']}) "
            f"| {row['evidence']} | {result} | {row['validation']} | {identity} |"
        )
    return "\n".join(lines) + "\n"


def _format_source(source):
    parts = [f"{source['authors']} ({source['year']}). {source['title']}."]
    if source.get("venue"):
        parts.append(f"{source['venue']}.")
    if source.get("doi"):
        parts.append(f"[doi:{source['doi']}](https://doi.org/{source['doi']})")
    elif source.get("url"):
        parts.append(f"<{source['url']}>")
    if source.get("isbn"):
        parts.append(f"ISBN {source['isbn']}.")
    return " ".join(parts)


def render_references(catalog: dict) -> str:
    """Bibliography with, for each source, the clauses citing it and where."""
    citing = {}
    for clause in catalog["clauses"]:
        for citation in clause.get("citations", []):
            citing.setdefault(citation["source"], []).append((clause, citation))
    lines = []
    for source in sorted(catalog["sources"], key=lambda s: (s["authors"], s["year"])):
        lines += [f"(source-{source['id']})=", _format_source(source), ""]
        uses = citing.get(source["id"], [])
        if uses:
            cited = "; ".join(
                f"{{ref}}`{clause['id']} <{clause['id']}>` ({citation['locator']})"
                for clause, citation in uses
            )
            lines += [f"Cited at: {cited}.", ""]
    return "\n".join(lines)


def render_clause_bases(catalog: dict) -> str:
    """Table of every catalogued clause and what its statements rest on."""
    lines = [
        "| Clause | Sources and locators | Derivation | Decisions |",
        "|---|---|---|---|",
    ]
    for clause in catalog["clauses"]:
        cites = "; ".join(
            f"{{ref}}`{c['source']} <source-{c['source']}>`, {c['locator']}"
            for c in clause.get("citations", [])
        )
        decisions = ", ".join(
            f"{{ref}}`{d} <{d}>`" for d in clause.get("decisions", [])
        )
        lines.append(
            f"| {{ref}}`{clause['id']} <{clause['id']}>` | {cites or 'none'} | "
            f"{clause.get('derivation', 'none')} | {decisions or 'none'} |"
        )
    return "\n".join(lines) + "\n"


def render_profiles(catalog: dict) -> str:
    """Table of convention profiles with their state and clauses."""
    cases = {}
    for case in catalog["cases"]:
        cases.setdefault(case["profile"], []).append(case["id"])
    lines = [
        "| Profile | State | Decision | Clauses | Reference cases |",
        "|---|---|---|---|---|",
    ]
    for profile in catalog["profiles"]:
        clauses = ", ".join(f"{{ref}}`{c} <{c}>`" for c in profile["clauses"])
        decision = profile.get("decision")
        decision = f"{{ref}}`{decision} <{decision}>`" if decision else "none recorded"
        refs = ", ".join(f"`{c}`" for c in cases.get(profile["id"], [])) or "none"
        lines.append(
            f"| `{profile['id']}`: {profile.get('title', '')} | {profile['status']} | "
            f"{decision} | {clauses} | {refs} |"
        )
    return "\n".join(lines) + "\n"


def _git(docs_dir, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(docs_dir), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def render_edition(docs_dir: Path) -> str:
    """State which version and commit this build is, and whether it is an edition."""
    version = _installed_version("spohnbook") or "unknown"
    commit = _git(docs_dir, "rev-parse", "HEAD") or os.environ.get(
        "READTHEDOCS_GIT_COMMIT_HASH", "unknown"
    )
    dirty = _git(docs_dir, "status", "--porcelain", "--untracked-files=no")
    tagged = re.fullmatch(r"\d+\.\d+\.\d+", version) is not None
    state = (
        f"This is edition {version}, built from commit `{commit}` (tag v{version})."
        if tagged
        else (
            f"This is a development build, version {version} from commit "
            f"`{commit}`. It is a draft and not a tagged edition."
        )
    )
    if dirty:
        state += " The source had uncommitted changes when it was built."
    return (
        state
        + " Its proposed conventions remain proposals unless the"
        + " {doc}`decision register </profiles/decisions>` records a decision.\n"
    )


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def render_evidence_page(catalog: dict, ledger, manifest) -> str:
    """The generated part of the evidence page: identities and case results."""
    rows = evidence_rows(catalog, ledger, manifest)
    if ledger is None or manifest is None:
        header = (
            "This build has no evidence ledger and build manifest, so every case is "
            "unassessed. The continuous-integration build records both; see "
            "{doc}`reproducing an edition </evidence/reproducing>`.\n\n"
        )
    else:
        book = manifest.get("book") or {}
        header = (
            f"Evidence recorded {ledger.get('recorded')} for commit "
            f"`{book.get('commit')}` with dependency resolution "
            f"`{manifest.get('resolution', 'unknown')}`, on "
            f"{(manifest.get('platform') or {}).get('system', 'unknown')} "
            f"{(manifest.get('platform') or {}).get('machine', '')} with Python "
            f"{(manifest.get('python') or {}).get('version', 'unknown')}.\n\n"
        )
    return header + render_evidence_table(rows)


def render_environment(docs_dir: Path, manifest) -> str:
    """Versions of the suite's libraries that executed the examples in this build."""
    names = [
        lib["name"]
        for lib in yaml.safe_load(
            (Path(docs_dir).parent / "libraries.yaml").read_text()
        )["libraries"]
    ]
    if manifest is not None:
        packages = manifest.get("packages") or {}
        versions = {n: (packages.get(n) or {}).get("version") for n in names}
        origin = "the build manifest recorded with this build"
    else:
        versions = {n: _installed_version(n) for n in names}
        origin = "the environment that built this page (no build manifest was supplied)"
    listed = ", ".join(f"{n} {v}" for n, v in versions.items() if v)
    return (
        f"Executed with {listed}, as read from {origin}. The full dependency "
        "identity and the lock are described in "
        "{doc}`reproducing an edition </evidence/reproducing>`.\n"
    )


def write_handbook_pages(
    docs_dir: Path, evidence_dir: Path | None = None
) -> list[Path]:
    """Validate the catalog, then write the generated fragments.

    Raises:
        ValueError: listing every structural error when the catalog is invalid.
    """
    docs_dir = Path(docs_dir)
    catalog = load_catalog(docs_dir)
    errors = validate_catalog(catalog, docs_dir)
    if errors:
        raise ValueError("handbook catalog is invalid:\n  " + "\n  ".join(errors))
    if evidence_dir is None:
        evidence_dir = Path(
            os.environ.get("SPOHNBOOK_EVIDENCE_DIR", docs_dir / "_evidence")
        )
    ledger = _read_json(Path(evidence_dir) / "ledger.json")
    manifest = _read_json(Path(evidence_dir) / "build-manifest.json")
    out_dir = docs_dir / "_generated"
    out_dir.mkdir(exist_ok=True)
    pages = {
        "references.md": render_references(catalog),
        "clause-bases.md": render_clause_bases(catalog),
        "profiles.md": render_profiles(catalog),
        "evidence.md": render_evidence_page(catalog, ledger, manifest),
        "edition.md": render_edition(docs_dir),
        "environment.md": render_environment(docs_dir, manifest),
    }
    written = []
    for name, text in pages.items():
        path = out_dir / name
        path.write_text(text)
        written.append(path)
    return written
