# Building the ZipperGen documentation

The documentation is deliberately split by reader and purpose:

- `docs/first-workflow.tex` — the short, linear first-workflow tutorial;
- `docs/workflow-development-deployment-guide.tex` — the comprehensive
  development and deployment manual;
- `examples/tutorial_review.py` — complete fallback source imported by the
  manual.

A TeX distribution is the only external requirement. On macOS, use MacTeX; on
Linux, use a TeX Live installation that includes `latexmk`, TikZ/PGF,
`listings`, `lmodern`, `microtype`, and the common LaTeX packages.

From the ZipperGen Git root, run:

```bash
make docs
```

The command first checks the compiler and required packages. It then builds each
document under `docs/_build/` and copies the finished PDF next to its source:

```text
docs/first-workflow.pdf
docs/workflow-development-deployment-guide.pdf
```

Those two files are committed and are the ones the project README links, so
rebuild them whenever you change a `.tex` source. All auxiliary files stay under
the ignored `docs/_build/` directory.

Build only one document with:

```bash
make docs-first-workflow
make docs-manual
```

To check the installation without compiling the document:

```bash
make docs-check
```

The direct equivalents, useful when configuring a TeX editor, are:

```bash
cd docs
mkdir -p _build
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -file-line-error -outdir=_build \
  first-workflow.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -file-line-error -outdir=_build \
  workflow-development-deployment-guide.tex
```

The source paths are intentionally relative to `docs/`, so this works whether
it is launched manually or by an editor whose working directory is the source
file's directory. These direct commands leave the result in `docs/_build/` only;
use `make docs` when you intend to update the committed PDFs.

If `latexmk` is not found after installing MacTeX, start a new terminal and
check:

```bash
/Library/TeX/texbin/latexmk --version
```

Add `/Library/TeX/texbin` to the shell's `PATH` if necessary. If the build
reports one missing `.sty` file, install the corresponding TeX package rather
than copying that generated or system-owned file into this repository.
