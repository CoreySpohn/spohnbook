# Releases, errata and migrations

An edition of the book is a tagged release of this repository. Its version number
identifies the text and is independent of the version of every convention
profile, library, data schema and calibration the text describes. Profile
changes and dependency changes are listed separately in each edition's notes, so
that a reader can tell whether a physical meaning changed or only the software
that executed the examples.

(releases-policy)=
## How an edition is made

The repository uses release-please, as every library in the suite does
({ref}`library conventions <library-release-and-ci>`). A library
releases only on `feat` and `fix` commits, and documentation commits do not
release. For this book the documentation is the product, so the same mechanism
is used with a book-specific classification: new substantive content, such as a
chapter, a clause, a profile, a reference case or an evidence page, is committed
as `feat(book)`; a correction of scientific content is committed as `fix(book)`;
copyediting stays `docs` and does not by itself produce an edition.
release-please collects those commits into a release pull request, and merging
it tags the edition and publishes its GitHub release. The workflow and its
configuration are unchanged; only the commit classification is specific to the
book.

A development build rendered from the default branch shows its version and commit
on the {ref}`citing page <citing-this-build>` and states that it is not an
edition. It is never cited as one.

(releases-bundle)=
(releases-archival)=
## What an edition consists of

An edition is its git tag and its ReadTheDocs version. The tag holds the source,
the source catalog, `CITATION.cff` and the dependency lock. The ReadTheDocs
version of the tag is built with that lock and records its own evidence ledger
and build manifest while it builds, so the case states it shows come from that
build. The artifacts of the tag's locked continuous-integration build (rendered
HTML, logs, ledger and manifest) are kept for the platform's retention period and
are not archived beyond it; a rebuild from the tag with its lock reproduces them,
provided the input data remain available. Editions have no DOI.

(releases-review-records)=
## Review records

A publication review record states the scope reviewed, the reviewer, the date,
the outcome and every unresolved concern. An edition without an independent
scientific review says so in its record.

(releases-editions)=
## Editions

(releases-0-0-1)=
### 0.0.1 (2026-09-24)

The first edition is a draft. Every convention in it that is marked pending or
proposed remains so.

**Content.** Scope statement and reading routes; stable clause labels on every
load-bearing section; the source catalog with claim-level citations and a
generated bibliography; the single decision register; the convention profile
page with the proposed profile `photon-electron-reference-v1`; the evidence
pages, with computed case states, the limitations page carrying all 81 findings
of the earlier audit and the procedure for reproducing an edition; the
photon-to-electron reference case; the contribution, citation and release
policies.

**Profile changes.** `photon-electron-reference-v1` is introduced as proposed.
No profile is adopted.

**Dependency changes.** The documentation build is locked for Python 3.12 on
Linux x86_64 in `requirements-docs.txt`. A separate compatibility build runs
against the newest releases. The test extra declares the packages the tests
import.

**Corrections before first publication.** These were corrected in the draft
text before any edition existed, so they are not errata:

- The AB zero point is $10^{-48.60/2.5}$ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ =
  3630.780547701013 Jy. The draft quoted 3630.7805477010028 Jy as exact; those
  digits are the float64 representation, which differs after the fourteenth
  significant figure.
- The draft stated that hwostyle resolves the `probability` role to YlOrRd in
  light mode; hwostyle 1.5.0 resolves it to the reversed map `YlOrRd_r`. The
  record now sits on the limitations page.
- The Savransky et al. (2019) mapping now notes that the paper's own $\alpha$ is
  the angular separation, not this book's illumination angle.

**Review record.**

| Field | Record |
|---|---|
| Scientific scope | Conventions chapters, decision register, profile, reference case, evidence and limitation pages of this edition |
| Tested platform | Linux x86_64 with Python 3.12.3 and the locked dependencies; the pages were also executed on macOS arm64 with Python 3.12.12 against unlocked PyPI releases during preparation |
| Reproducibility result | The locked build of the tag commit `e9b23cf` passed (continuous-integration run 36071397378): its environment matched the lock exactly, every page executed, the reference case passed and the input data were recorded with their SHA-256 |
| Resolved findings | None of the 81 audit findings is closed; five of them, and two further records, are reproduced at stated versions |
| Unresolved concerns | The source gaps listed on the {ref}`references page <references-source-gaps>`; every finding marked "historical report; reproduction unavailable"; all pending decisions |
| Author | Corey Spohn |
| Scientific reviewer | unassigned; no independent scientific review has been performed |
| Review date and outcome | none |

(releases-errata)=
## Scientific errata

No errata have been issued. An erratum names the editions that carried the
error, the affected clauses and results, and whether earlier published outputs
need reassessment.

(releases-migrations)=
## Migrations

No convention profile has changed meaning, so no migration is required.
