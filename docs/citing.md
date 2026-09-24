# Citing the book

Cite the edition you read, the clause you rely on, and the upstream source of any
borrowed model. The book changes between editions, so a citation without an
edition cannot say which text a reader saw, and a citation of the book alone
cannot say whose model a clause adopts.

(citing-this-build)=
## This build

```{include} _generated/edition.md
```

(citing-edition)=
## An edition

An edition is a tagged release of this repository. Its version number identifies
the book text, and it is separate from the version of any convention profile,
library, dataset or calibration the text describes. Cite an edition as

> Spohn, C., *The Spohn Book*, version X.Y.Z, https://github.com/CoreySpohn/spohnbook
> (tag vX.Y.Z), rendered at https://spohnbook.readthedocs.io/en/vX.Y.Z/.

The repository's `CITATION.cff` carries the same author, title and license
metadata, and GitHub offers it in APA and BibTeX form through "Cite this
repository". Editions have no DOI; the tag and its rendered version identify
them. A development build, such as one rendered from the default branch, is not an
edition: cite its commit as shown above and say that it is a draft.

Every edition published so far is a draft edition. Its proposed conventions are
proposals: cite them as "proposed in version X.Y.Z", never as adopted, and check
the [decision register](profiles/decisions.md) for their state.

(citing-clause)=
## A clause

Every load-bearing section has a stable label, such as `radiometry-jy-photons`.
The label is the section's anchor in the rendered page and does not change when a
heading is reworded, so a clause citation is the edition plus the page and label:

> *The Spohn Book* version X.Y.Z, radiometry chapter, clause `radiometry-jy-photons`,
> https://spohnbook.readthedocs.io/en/vX.Y.Z/conventions/radiometry-detectors.html#radiometry-jy-photons

The [profiles page](profiles/index.md) lists the clauses each convention profile
includes, and the [references](references.md) page lists which clauses cite each
source.

(citing-upstream)=
## An upstream source

When a clause adopts a published model, convention or dataset, cite that primary
source as well, using the locator the clause gives. The book's selection or
notation for the model is its own choice, and a citation of the book should not
imply that the original authors endorse that choice.
