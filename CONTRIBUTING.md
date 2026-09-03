# Contributing to ZipperGen

ZipperGen is a small project with a formal core. Most of what you need to know
is in [`docs/architecture.md`](docs/architecture.md): the layers, the module
boundaries, the design patterns, and which published result covers which
construct. You may read that first. This file holds only what that guide does
not cover, namely how work is agreed and what a change has to pass.

## Working agreements

- **External contributions go through pull requests.** Open an issue before a
  substantial or compatibility-affecting change so the direction can be
  agreed before implementation.
- **No ad-hoc workarounds.** Name the rule, put it in one place, and add a
  completeness test that fails when a second copy appears. Most defects found
  in this repository have been one rule written down twice, with the copies
  drifting apart.
- **ZipperGen is pre-1.0.** Discuss compatibility-affecting changes before
  implementing them. Do not add migrations or deprecation machinery
  speculatively.

## The gate

These commands cover the same three gates as CI. Run them from the repository
root.

```bash
# Tests, on every supported interpreter: 3.11, 3.12, 3.13
uv run --python 3.11 --isolated --with pytest pytest -q
uv run --python 3.12 --isolated --with pytest pytest -q
uv run --python 3.13 --isolated --with pytest pytest -q

# Types
uv sync --extra google
uvx pyright

# Package. The Makefile clears the build tree before building.
make dist
uvx twine check dist/*
```

CI additionally installs the built wheel and runs `zg --help` and
`zg deploy --help` against it, which catches a package that imports but does
not ship a working entry point.

Run the full suite, not a subset. Several tests exist to catch a rule being
copied rather than shared, and they only fire from a whole-suite run.

## Generated and mirrored files

After editing `docs/*.tex`, run `make docs`. This rebuilds the committed PDFs.
`test_document_artifacts.py` checks that they match their sources.

The workflow-authoring skill is stored in both
`src/zippergen/skills/zippergen-workflows/` and
`.agents/skills/zippergen-workflows/`. Edit the packaged copy under `src/`
first, then mirror the change to `.agents/`. The test suite checks that the two
copies match.

## Adding to the language

Do not add a construct to the grammar before a published result covers it.
Every construct ZipperGen offers today is covered by the ISoLA paper or the
EXPRESS/SOS paper, and the deadlock-freedom claim has no exceptions to
remember. [`docs/architecture.md`](docs/architecture.md) states the scope
of each result.
