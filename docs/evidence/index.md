# Evidence

This section says what the book's evidence shows and what it does not. The
conventions chapters define meanings; the evidence here concerns particular
implementations at particular versions. Four questions are kept separate, and
no answer to one of them is inferred from another.

| Question | Where it is answered |
|---|---|
| Has a convention been adopted? | The profile state in the [profiles](../profiles/index.md) and the [decision register](../profiles/decisions.md). |
| Does an implementation reproduce an independent reference case? | The case results below. A result applies to its case only, at the recorded versions. |
| What is known to be wrong, restricted or worked around? | The [limitations page](limitations.md). |
| Does a model agree with measured data over a stated domain? | Measured-data validation. No case in this edition is a validation case, so the validation column below reads "not assessed". |

A successful build shows that every executed example ran with the recorded
dependency versions. It is not evidence that any example's result is
scientifically correct, and several examples run across recorded limitations.
Agreement between two codes is a cross-code benchmark, not validation. The book
makes no determination of compliance with NASA-STD-7009B or any other standard;
the suite records test evidence with
[ineedvalidation](https://ineedvalidation.readthedocs.io), whose documentation
describes the evidence kinds and the validation hierarchy it reports against.

(evidence-case-states)=
## Case states

A reference case binds a precise claim to a profile, the exact test identifiers
that exercise it, and the packages whose versions it depends on. Its state is
computed at each build from the evidence ledger written by the tests and the
build manifest written beside it ([reproducing an edition](reproducing.md)
describes both), and never typed by hand.

| State | Meaning |
|---|---|
| unassessed | No ledger and manifest were supplied to this build. |
| incomplete | A required test is absent, was skipped, was only collected, or the manifest lacks an identity the case needs, such as a committed source or a package version. |
| failed | A required test failed or errored. |
| stale | The evidence was recorded for other bytes: a different commit, catalog, profile, ledger or package version than the one this page describes. |
| passed | Every required test passed, and every recorded identity matches. The result covers this case only. |

A passed case does not adopt its profile, does not verify the boundary for
inputs outside the case, and does not validate anything against measured data.
Tests that happen to pass elsewhere in the suite never satisfy a case: only the
listed test identifiers, recorded under the case's own identifier, count.

(evidence-results)=
## Results for this build

```{include} ../_generated/evidence.md
```

```{toctree}
:maxdepth: 1

limitations
reproducing
```
