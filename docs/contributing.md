# Contributing

This page sets the rules for adding to or changing the book: what evidence each
kind of statement needs, who is responsible for its scientific content, and how a
change of meaning is handled. The coding, testing and release conventions that
every library in the suite follows, including this repository's Python tooling,
are in the [library conventions](library-conventions.md). Figures follow
[the figure atlas](conventions/figures.md) and [figures and color](conventions/figures-and-color.md).

(contributing-grounding)=
## What a statement needs

A statement is grounded when a reader can reconstruct why it is justified and
where it applies, using only public material. A bibliography entry alone does not
supply that chain, because it cannot say which paper supports which normalization
or assumption. Each kind of statement therefore carries its own evidence.

| Statement | Evidence it carries |
|---|---|
| Identity derived in the book | Definitions, assumptions, the derivation and at least one independently checkable limiting case. A citation for every algebraic line is unnecessary. |
| Borrowed scientific model or external convention | A primary source with its section, equation, table or page, and an explicit mapping from the source's symbols, units and assumptions to the book's notation. |
| Convention chosen for this suite | A row in the [decision register](profiles/decisions.md) with its rationale, alternatives and state. A literature citation must not imply that an external author endorses a local choice. |
| Empirical value, instrument property or calibration | The dataset or document and its version, provenance, units, uncertainty, calibration conditions and permitted domain. |
| Claim about library behavior | The affected release or commit, a public source or test link, the inputs and the observed result, recorded on the [limitations page](evidence/limitations.md) rather than in the evergreen explanation. |
| Figure, benchmark or analysis result | Inputs and configuration, generating code, dependency identity, data hashes, numerical settings and the expected observable. |

Every borrowed source is entered once in the source catalog,
`docs/_data/handbook.yaml`, which generates the [references](references.md)
page. A chapter cites it at the statement it supports, with a locator, for
example {ref}`Leinert et al. 1998, Table 19 <source-leinert1998>`, and the
catalog entry for that chapter's clause repeats the locator. Do not add a
BibTeX file. The build rejects a catalog in which a cited source, a clause label
or a profile is missing, a borrowed-source citation has no locator, or an
identifier is duplicated. Those checks are structural: they confirm that a
citation and a locator exist, not that the source supports the prose. Checking
the support is the scientific reviewer's job.

(contributing-evidence-classes)=
## Kinds of evidence

Four kinds of evidence are kept apart in text, captions and tables. An analytic
fixture compares an implementation with an answer calculated independently from
primitives, and it verifies only the mathematics it exercises. Successful
execution of a tutorial shows that a composition runs with the recorded versions,
not that its result is scientifically correct. Agreement with an independently
developed code is a cross-code benchmark. Agreement with measured data over a
stated domain is validation, and nothing else is. A passing fixture does not adopt
a proposed profile, and a proposed profile does not become verified because one of
its reference cases passes. The [evidence pages](evidence/index.md) show each
state separately.

(contributing-roles)=
## Editorial roles

Three roles are responsible for a domain of the book. The **author** writes and
maintains the text and its evidence. The **responsible domain editor** decides
whether a change of scientific meaning is ready to publish and keeps the domain's
limitations current. The **scientific reviewer** is a person, other than the
author, who checks the sources, derivations and evidence of a publication and
records the outcome. An automated check, a passing build or an agreement between
two runs of the same code is never described as scientific review.

| Domain | Author | Responsible domain editor | Scientific reviewer |
|---|---|---|---|
| Geometry and time | Corey Spohn | unassigned | unassigned |
| Radiometry and detectors | Corey Spohn | unassigned | unassigned |
| Optics and images | Corey Spohn | unassigned | unassigned |
| Measurements, probability and records | Corey Spohn | unassigned | unassigned |
| Figures and color | Corey Spohn | unassigned | unassigned |
| Library map, examples and evidence pages | Corey Spohn | unassigned | unassigned |

An unassigned role stays marked unassigned until a named person accepts it. A
publication review record states the scope reviewed, the reviewer, the date, the
outcome and every unresolved concern. Records for editions are kept on the
[releases page](releases.md).

(contributing-changes)=
## Changing the book

Ordinary wording changes keep every clause label. A label such as
`radiometry-jy-photons` identifies a clause independently of its heading, so a
retitled section keeps its label, and a removed or split section leaves a
compatibility target behind so that published links still resolve.

A change of scientific meaning, such as a different sign, unit, measure or
normalization, needs a profile and migration assessment. Record the choice in the
[decision register](profiles/decisions.md), give the affected profile a new
version rather than editing the old one in place, and write a migration note on
the [releases page](releases.md) naming the clauses, tests and results that need
reassessment. A parameter rename alone does not migrate old physical assumptions.

A correction of scientific content that was wrong in a published edition needs an
erratum on the [releases page](releases.md) with an impact statement: which
editions carried the error, which clauses and results it affects, and whether
earlier published outputs need reassessment. Keep the historical meaning visible
rather than silently replacing it.

A finding about current library behavior goes on the
[limitations page](evidence/limitations.md) with a descriptive identifier, the
affected versions, a public reproduction or source link, its status and its
consequence, and the affected chapter links to it.

Commit messages follow the conventions in the
[library conventions](library-conventions.md). For this book, new substantive
content is a `feat(book)` commit and a scientific correction is a `fix(book)`
commit, so that both reach a release; copyediting stays `docs`. The
[releases page](releases.md) explains why.

(contributing-boundary)=
## Adding or repairing a boundary

When you repair or add a boundary between two libraries, ship the explanation of
the physical observable, its reference table, a worked example with an
independently calculated answer and a negative control, an executable example at
the public API, a cross-library fixture that does not take its expected value from
the implementation under test, and the migration and evidence records. The
{ref}`handbook's contribution list <conventions-contributing>`
gives the full checklist.
