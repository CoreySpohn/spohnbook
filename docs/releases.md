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
## The release bundle

An edition's archival bundle is assembled by `tools/make_release_bundle.py` from a
clean checkout of its tag and the artifacts of its locked build. It contains:

- the source tree at the tag, as `git archive` produces it;
- the rendered HTML;
- the source catalog and `CITATION.cff`;
- the dependency lock;
- the evidence ledger and the build manifest;
- the build and execution logs;
- `inputs.json`, the identity of every input data file with its upstream
  address and SHA-256, stating that the bytes are not included;
- `SHA256SUMS`, the checksum of every file in the bundle.

The bundle is a directory and a `.tar.gz` of it, and it can be inspected with
ordinary tools and without the author's workspace.

(releases-archival)=
## Archival and DOI

No DOI has been minted for any edition. To archive an edition, a maintainer with
the necessary account access enables the repository in the Zenodo GitHub
integration before the release is published, so that Zenodo archives the tagged
source and reads `CITATION.cff`; uploads the release bundle to the resulting
record as an additional file; and adds the DOI to this page and to
`CITATION.cff` in a following commit. Minting a DOI is a separate, deliberate
act, and nothing in the build or the tests performs it.

(releases-review-records)=
## Review records

A publication review record states the scope reviewed, the reviewer, the date,
the outcome and every unresolved concern. An edition without an independent
scientific review says so in its record.

(releases-editions)=
## Editions

(releases-0-0-1)=
### 0.0.1 (release candidate, not yet published)

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
| Tested platform | The locked continuous-integration build of the release commit is the record. The locked builds of the release pull request passed on Linux x86_64 with Python 3.12.3 in an environment identical to the lock; the pages were also executed on macOS arm64 with Python 3.12.12 against unlocked PyPI releases during preparation |
| Reproducibility result | Recorded by the locked build of the release pull request; see its build manifest |
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
