# Building the ZipperGen guide

The repository contains every source file used by the guide:

- `docs/workflow-development-deployment-guide.tex`
- `examples/tutorial_review.py`, imported as the complete tutorial source

A TeX distribution is the only external requirement. On macOS, use MacTeX; on
Linux, use a TeX Live installation that includes `latexmk`, TikZ/PGF,
`listings`, `lmodern`, `microtype`, and the common LaTeX packages.

From the ZipperGen Git root, run:

```bash
make docs
```

The command first checks the compiler and required packages. It then creates:

```text
docs/_build/workflow-development-deployment-guide.pdf
```

All auxiliary files stay under the ignored `docs/_build/` directory.

To check the installation without compiling the document:

```bash
make docs-check
```

The direct equivalent, useful when configuring a TeX editor, is:

```bash
cd docs
mkdir -p _build
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -file-line-error -outdir=_build \
  workflow-development-deployment-guide.tex
```

The source paths are intentionally relative to `docs/`, so this works whether
it is launched manually or by an editor whose working directory is the source
file's directory.

If `latexmk` is not found after installing MacTeX, start a new terminal and
check:

```bash
/Library/TeX/texbin/latexmk --version
```

Add `/Library/TeX/texbin` to the shell's `PATH` if necessary. If the build
reports one missing `.sty` file, install the corresponding TeX package rather
than copying that generated or system-owned file into this repository.
