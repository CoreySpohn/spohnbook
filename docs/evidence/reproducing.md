# Reproducing an edition

Two builds answer two different questions. The **locked build** asks whether an
edition still executes with exactly the dependencies it was tested with. The
**compatibility build** asks whether the same pages and tests still run against
the newest releases on PyPI. A failure of the compatibility build does not
change an edition's lock, and a passing locked build says nothing about newer
releases.

(reproducing-locked)=
## The locked build

`requirements-docs.txt` pins every direct and transitive dependency of the
`docs` and `test` extras, with hashes, for Python 3.12 on Linux x86_64
(`manylinux_2_28`). It is compiled from PyPI with no workspace or local sources.
One pin is a nightly pre-release, `tfp-nightly`, which a
dependency of the inference library requires; PyPI has kept that project's
nightly releases since 2018, but this repository does not archive wheels, so its
availability is an external dependency of exact restoration. The lock does not
cover macOS or Windows; see
[other platforms](#reproducing-other-platforms). From a clean checkout of the
edition's tag, on that platform and with [uv](https://docs.astral.sh/uv/):

```console
$ uv venv .venv --python 3.12
$ uv pip sync --python .venv/bin/python requirements-docs.txt
$ uv pip install --python .venv/bin/python --no-deps -e .
$ mkdir -p docs/_evidence
$ .venv/bin/python -m pytest tests --vv-evidence docs/_evidence/ledger.json
$ .venv/bin/python tools/record_build.py --resolution locked --fetch-inputs \
      --ledger docs/_evidence/ledger.json --output docs/_evidence/build-manifest.json
$ .venv/bin/python -m sphinx -W --keep-going -D nb_execution_mode=force \
      -b html docs docs/_build/html
$ SPOHNBOOK_HTML_DIR=docs/_build/html .venv/bin/python -m pytest tests/test_handbook.py -k rendered
```

The tests write the evidence ledger, `record_build.py` downloads the declared
input data through its owning library and writes the build manifest beside it, and the documentation build executes every example page and renders
the {ref}`case results <evidence-results>` from the two files. The last
command checks that every page and anchor published by earlier editions still
resolves in the rendered site. Nothing in the procedure needs the author's
workspace, a personal cache, a reference manager or network access beyond PyPI
and the input data below.

(reproducing-manifest)=
## The build manifest

`record_build.py` records, for one build:

- the book commit and whether the checkout had uncommitted changes;
- the Python version and implementation, the operating system, machine and C
  library;
- every installed distribution with its version, installer and origin (package
  index, version-control URL and commit, or a local path, which is recorded only
  as "local");
- digests of the lock, of the source catalog and of each convention profile,
  whether the environment matches the lock, and every mismatch;
- the digest of the evidence ledger and the commit it names;
- the numerical precision the tests set;
- the identity of every input data file the examples read;
- the command that ran.

Paths under the home directory or the checkout are rewritten, and the tool
refuses to write a manifest in which one remains. It exits with an error when a
required identity is missing, when a declared input is not present, or when a
build that claims the locked resolution differs from the lock; the evidence page
then shows the affected cases as incomplete rather than passed. The evidence page joins the ledger
and the manifest only when their identities agree, so a manifest recorded for
different bytes marks a case stale rather than passed.

(reproducing-inputs)=
## Input data

The executed examples read one input file, found by building the pages with empty
download caches:

| Input | Owner | Upstream | Registry hash | Used by |
|---|---|---|---|---|
| `eac1_optimal_order_6_1d.zip`, a yield input package describing a coronagraph's response | yippy | `https://github.com/HabitableWorldsObservatory/yippy/releases/download/data-v2/eac1_optimal_order_6_1d.zip` | `md5:df52540008a0e85467720ec91c3a84b8` | the exposure-time and scene examples |

The manifest records the SHA-256 of the bytes actually read. This repository does
not archive the file: recording its address and hash identifies it but does not
preserve it, and its redistribution terms are not recorded here. Exact
restoration of an edition therefore depends on the owning library's data release
remaining available. The build manifest of an edition's locked build lists the
identity of every input.

(reproducing-compatibility)=
## The compatibility build

The compatibility job installs the current releases of every dependency with
`uv pip install -e ".[test,docs]"`, runs the same tests and pages, and records its
own manifest with resolution `latest`, which lists every difference from the
lock. It runs on a schedule as well as on changes, and its result is reported
separately from the locked build.

(reproducing-other-platforms)=
## Other platforms

On macOS or Windows, install the extras without the lock and record the build
with `--resolution latest`. The manifest then lists every package that differs
from the lock. Such a build reproduces the pages on that platform; it is not the
tested environment of the edition.

(reproducing-retention)=
## What is retained

Every continuous-integration run uploads the rendered HTML, the execution logs,
the ledger, the build manifest and the lock, including runs that fail. Those
artifacts expire. The durable record of an edition is its tag, which keeps the
source and the lock, and its ReadTheDocs version, which is rebuilt from them and
records its own evidence ({ref}`releases <releases-bundle>`).
