# Library conventions

Every library in the suite is built the same way, so that a reader who knows one
can open any other and a collaborator can start a conforming one from this page.
The scientific conventions the libraries share live in the
[handbook](conventions/index.md); the [map](map.md) lists the libraries and stages.

## Repository layout

A library is one public git repository in the `src` layout:

| Path | Holds |
|---|---|
| `src/<package>/` | The importable package. `_version.py` is generated at build time and is not tracked. |
| `tests/` | The pytest suite: `conftest.py` for shared fixtures, `test_<module>.py` per source module. |
| `docs/` | The Sphinx site: `conf.py`, `index.md`, and executed example pages. |
| `tools/` | Repository tooling outside the package, such as the internal-reference guard. |
| `.github/workflows/` | `tests.yml`, `release-please.yml`, `publish-to-pypi.yml`. The repository root also holds `.pre-commit-config.yaml`, `.readthedocs.yaml`, `LICENSE` (MIT), `README.md` and the generated `CHANGELOG.md`. |

The build backend is hatchling with hatch-vcs: the version comes from the git tag
and `[tool.hatch]` writes it to `src/<package>/_version.py`. The fields every
library carries:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "<package>"
dynamic = ["version"]
requires-python = ">=3.11"
license = { file = "LICENSE" }
classifiers = ["Development Status :: 4 - Beta", "License :: OSI Approved :: MIT License"]

[project.optional-dependencies]
dev = ["pre-commit"]
test = ["pytest", "hypothesis", "ineedvalidation"]
docs = ["myst-nb", "sphinx-autoapi", "sphinx-book-theme"]
```

The `Development Status` classifier carries the declared maturity stage in the
map's vocabulary: pre-alpha (`2 - Pre-Alpha`, a scaffold), alpha (`3 - Alpha`,
tagged releases and tests in continuous integration), beta (`4 - Beta`, a
documentation site, a changelog and a dependent library) and stable
(`5 - Production/Stable`, beta plus a stated API compatibility policy and
verification evidence at the required level). The classifier and the map agree.

## Naming

- Package names are lowercase everywhere, titles and sentence starts included: the
  name is the import name. A library name is short, evocative, says what the
  library does, is never named after a backend dependency (it must survive a
  backend swap), and is checked for availability on PyPI before it is settled.
- Modules are lowercase, single-word where possible (`scene`, `datasets`) and
  underscored otherwise (`optical_path`); private submodules start with an underscore.
- Types: the canonical (most complete) type takes the bare noun; a variant takes an
  adjective prefix (`FlatStar`), never a noun-noun compound; a loaded variant takes
  its data source as prefix (`ExovistaDisk`); a tabulated wrapper is
  `Precomputed<Type>`; an abstract base is `Abstract<Type>`. Drop a suffix the
  inheritance chain or submodule already carries and qualifiers such as `Only`.
- Functions are verb-first when they transform (`add_planet`) and noun-first when
  they return the named thing (`star_rate`); classmethod constructors are
  `from_<source>`.
- Every numeric field and argument carries its unit as a suffix; a dimensionless
  quantity (a fraction, an eccentricity, a count) carries none; when in doubt,
  include it. Composite rates use `_per_` (`dark_current_rate_e_per_s`).

| Quantity | Suffix | Quantity | Suffix |
|---|---|---|---|
| time in seconds, days | `_s`, `_d` | angle in arcseconds, degrees, radians | `_arcsec`, `_deg`, `_rad` |
| length in meters, parsecs, AU | `_m`, `_pc`, `_AU` | mass in kilograms | `_kg` |
| wavelength in nanometers | `_nm` | radius in Earth radii | `_rearth` |
| Julian date | `_jd` | flux density in janskys | `_jy` |
| electrons, photons | `_e`, `_phot` | plate scale in lambda/D per pixel | `_lod` |

## Code style

- Source is ASCII only: no em dashes, smart quotes, Greek letters, superscripts,
  arrows or math symbols. Write `--`, `->`, `**2`, `deg`, `um`. LaTeX notation
  belongs in Markdown documentation, not in `.py` files. Spelling is American.
- Comments explain why, not what: no numbered steps, no narration of the next
  line, no filler docstrings that restate the name, no hedging. Imports go at the
  top of the file.
- Physical constants, unit conversions and image transforms are imported from
  hwoutils, never retyped; a missing one is added to hwoutils. JAX code does not
  reach for `scipy.ndimage`; interpolation inside JIT uses interpax.
- ruff lints and formats: rule set `B, D, E, F, I, UP, RUF`, the Google docstring
  convention, `tests/**` exempt from `D`, and `ruff format` at the default
  88-column width; `N`, `ANN` and `SIM` are not selected.
- Pre-commit runs the internal-reference guard, whitespace and end-of-file fixes,
  `name-tests-test` (every file under `tests/` is `test_*.py`), `ruff check --fix`,
  `ruff format`, and conventional-pre-commit on the commit message (both stages).

## JAX and Equinox

- Every stateful object (model, data container, estimator) is an `eqx.Module` with
  declared typed fields and the generated `__init__`; a custom `__init__` with plain
  `self.x = value` only when derived state must be computed; no `__post_init__`.
- Validation runs at construction in `__check_init__`, never at trace time, where a
  `raise` fires deep in a JAX traceback far from the mistake. Unit and convention
  choices are resolved at construction so the traced path carries no intent branches.
- Abstract interfaces are `Abstract<Type>` bases with `AbstractVar` and
  `AbstractClassVar`. Equinox is nominal subtyping, so `typing.Protocol` is not used
  across packages; a foreign class is wrapped by composition, and composition is
  preferred to inheritance for has-a relationships.
- A field that controls computation structure (a count that sets a shape, a mode
  flag) is `eqx.field(static=True)`; array inputs use `converter=jnp.asarray`;
  shared constants are `ClassVar`; updates go through `eqx.tree_at`.
- Functions are pure: inputs through arguments, no global state, `jax.debug.print`
  for printing. JIT with `eqx.filter_jit`, not `jax.jit`; branch and loop with
  `lax.cond`, `lax.scan` and `lax.fori_loop`, never with Python loops over traced
  data; shapes are static; PRNG keys are split and never reused; prefer `vmap`.
- Precision follows the global `jax_enable_x64` flag uniformly: float32 when it is
  off, float64 when it is on, no pinned dtype, no mixed pipeline, and the library
  never sets the flag itself. Where a result needs float64 the docs say so, the
  library warns when the flag is off, and the tests measure the float32 floor.
- Every public method commits to one canonical shape contract, matched to what its
  upstream producer returns (for example `(K, T)` for K bodies at T epochs) and
  documented in the docstring; no rank-polymorphic inputs or `ndim` broadcast helpers.

## Testing and evidence

- pytest, with `[tool.pytest.ini_options] testpaths = ["tests"]`, one
  `tests/test_<module>.py` per source module, and shared fixtures in `conftest.py`,
  where `jax.config.update("jax_enable_x64", True)` is set once (never at module
  import, where collection order decides whether it leaks). The random seed is 0.
- hypothesis covers mathematical invariants (round trips, flux conservation,
  rotations), with `deadline=None` and `max_examples` of 50 to 100 under JIT.
  Every tolerance states its basis in the docstring (a floating-point floor, exact
  algebra, Monte Carlo sigma from a stated draw count, a published value), and the
  expected value comes from primitives in the test, never from the code under test.
- The tests workflow runs on every push and pull request to `main`: check out,
  install uv, `uv venv`, `uv pip install -e ".[test]"`, an import diagnostic, then
  `pytest tests/ -v --tb=short`, over a Python and platform matrix. The `test`
  extra carries everything the tests import; a shared venv hides a missing one.
- A silent zero-collection run is a failure to guard against: a test that fails to
  import, skips on a missing dependency or data file, or is never collected leaves
  the run green. A data-gated fixture fails, or skips with a stated reason that an
  environment flag turns into a failure; the CI log must say PASSED, not SKIPPED.
- Every test declares what evidence it provides with
  [ineedvalidation](https://ineedvalidation.readthedocs.io) markers:
  `@vv.case("<slug>", "<kind>")` with the kind `code-verification`,
  `solution-verification`, `cross-code-benchmark`, `validation` or
  `uncertainty-quantification`; `@vv.seam(producer, consumer)` for an
  absolute-scale anchor between two libraries; `@vv.regression` for a frozen
  output. A `[tool.ineedvalidation.defaults]` table in `pyproject.toml` assigns a
  case and kind by path prefix, and `pytest --vv-evidence PATH` writes the ledger.
  Agreement with another code is a benchmark, never validation.

## Documentation

- Sphinx with MyST-NB and sphinx-book-theme, built on ReadTheDocs from
  `.readthedocs.yaml`, which installs the package with the `docs` extra and points
  at `docs/conf.py`. That one extra installs everything the build needs, including
  every package an example page imports.
- `conf.py` reads the version with `importlib.metadata.version` (never by importing
  the package), enables `myst_nb`, `autoapi.extension`, `napoleon`, `viewcode` and
  `mathjax`, points `autoapi_dirs` at `../src` (ignoring `*version.py`), maps `.md`
  to `myst-nb`, and executes notebooks at build with errors raised.
- Example pages are MyST Markdown with `{code-cell}` blocks, executed at build, so
  they diff as text; `.ipynb` is reserved for interactive pages, outputs stripped.
- The API reference is generated by autoapi from Google-style docstrings. Code
  examples in docstrings use `::` literal blocks, never `>>>`.
- A library's own conventions page states its models, constants and known
  cross-code differences with sources; for the equations and sign conventions
  every library shares it links to the [handbook](conventions/index.md) instead.

## Data

- Test and example data are fetched with pooch from a public
  `src/<package>/datasets.py` module, part of the installed package so that users,
  tests and docs builds share one registry. The pooch instance is named `PIKACHU`,
  the registry maps each zipped file to its md5 hash, downloads land in
  `pooch.os_cache("<package>")`, and fetch functions are `fetch_<thing>()`, or a
  catalog (`fetch_<thing>(name, **filters)`, `list_<things>()`) for many variants.
- No data file ships in the wheel or sits beside the source. The registry points at
  the owning library's hosted copy (a `data/` directory served raw from its
  repository, or an asset on a tagged data release), and file names carry a version
  (`<name>-v<N>.zip`) so that an old release keeps finding its bytes.
- Each dataset has exactly one owning library; a downstream library imports the
  owner's fetch function rather than duplicating the registry entry. Tests fetch
  through session-scoped fixtures so that a file is downloaded once per run.

## Release and CI

```{figure} figures/release-flow-light.svg
:class: only-light
:name: fig-release-flow

How a commit becomes a release: the tests workflow and ReadTheDocs run on every push; release-please turns feat and fix commits into a release pull request; merging it tags the commit, and the tag publishes to PyPI through trusted publishing.
```

```{figure} figures/release-flow-dark.svg
:class: only-dark

How a commit becomes a release: the tests workflow and ReadTheDocs run on every push; release-please turns feat and fix commits into a release pull request; merging it tags the commit, and the tag publishes to PyPI through trusted publishing.
```

- Commit messages follow Conventional Commits on one line, `type(scope): message`:
  imperative mood, lowercase start, no trailing period, at most 72 characters, no
  body and no co-author trailer; the scope is the package or submodule. The one
  exception is a `BREAKING-CHANGE:` footer after a blank line, paired with `feat!:`,
  for a major bump. The commit-msg hook rejects anything else.
- release-please runs on every push to `main`, collects the `feat` and `fix`
  commits since the last tag, and opens or updates a release pull request with the
  version bump and `CHANGELOG.md`; merging it tags the commit and creates the GitHub
  release. `refactor`, `docs`, `test` and `chore` do not release, so a user-visible
  change is `feat` or `fix`. The action uses a personal access token, because a tag
  pushed with the default token does not trigger workflows.
- A tag triggers `publish-to-pypi.yml`: build the sdist and wheel with
  `python -m build`, store them as a workflow artifact, then publish from a `pypi`
  environment with `id-token: write` through `pypa/gh-action-pypi-publish`. This is
  trusted publishing; no API token is stored.
- The tests workflow installs with uv into a venv, as above; the package name, the
  version file and the platform matrix are the only per-library values.

## Keeping the library public-safe

- Every library ships `tools/check_internal_refs.py` and runs it as a pre-commit
  hook over every staged text file (excluding `CHANGELOG.md` and itself) and as a
  pytest, `tests/test_internal_refs_guard.py`, over every tracked text file. It
  scans line by line against a short pattern list (paths and names of private
  working documents, underscored all-caps Markdown names); a hit fails the commit.
- A line that must contain such a string for a functional reason (a
  development-only data fallback path in a fixture) is waived by appending the
  comment `internal-ref-ok` to that line. Everything else is rephrased
  self-contained: public physics, published citations, the library's own words.
- The README and the docs state current facts only: what the library owns, how it
  is installed, what the evidence shows. No dated fix notes, strikethroughs, "now"
  or "no longer" narration, roadmap language, internal jargon or design-document
  pointers. History lives in git and in the changelog, so commit messages follow
  the same rule.
