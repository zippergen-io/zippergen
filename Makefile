DOC_DIR := docs
DOC_BUILD_DIR := $(DOC_DIR)/_build
FIRST_WORKFLOW_DOC := first-workflow
MANUAL_DOC := workflow-development-deployment-guide
DOC_NAMES := $(FIRST_WORKFLOW_DOC) $(MANUAL_DOC)
DOC_SOURCES := $(addprefix $(DOC_DIR)/,$(addsuffix .tex,$(DOC_NAMES)))

.PHONY: docs docs-check docs-first-workflow docs-manual

docs: docs-first-workflow docs-manual

docs-first-workflow: docs-check
	@mkdir -p "$(DOC_BUILD_DIR)"
	@cd "$(DOC_DIR)" && latexmk \
		-pdf \
		-interaction=nonstopmode \
		-halt-on-error \
		-file-line-error \
		-outdir=_build \
		"$(FIRST_WORKFLOW_DOC).tex"
	@printf 'Built %s/%s.pdf\n' "$(DOC_BUILD_DIR)" "$(FIRST_WORKFLOW_DOC)"

docs-manual: docs-check
	@mkdir -p "$(DOC_BUILD_DIR)"
	@cd "$(DOC_DIR)" && latexmk \
		-pdf \
		-interaction=nonstopmode \
		-halt-on-error \
		-file-line-error \
		-outdir=_build \
		"$(MANUAL_DOC).tex"
	@printf 'Built %s/%s.pdf\n' "$(DOC_BUILD_DIR)" "$(MANUAL_DOC)"

docs-check:
	@command -v latexmk >/dev/null 2>&1 || { \
		printf '%s\n' \
			'Error: latexmk is not installed or is not on PATH.' \
			'Install a TeX Live or MacTeX distribution, then run make docs again.'; \
		exit 127; \
	}
	@command -v kpsewhich >/dev/null 2>&1 || { \
		printf '%s\n' \
			'Error: kpsewhich is not installed or is not on PATH.' \
			'Install a TeX Live or MacTeX distribution, then run make docs again.'; \
		exit 127; \
	}
	@for package in article.cls amssymb.sty geometry.sty xcolor.sty \
		lmodern.sty microtype.sty enumitem.sty booktabs.sty tabularx.sty \
		longtable.sty listings.sty tikz.sty fancyhdr.sty hyperref.sty; do \
		kpsewhich "$$package" >/dev/null 2>&1 || { \
			printf 'Error: required TeX file is missing: %s\n' "$$package"; \
			printf '%s\n' \
				'Install the corresponding TeX Live/MacTeX package, then retry.'; \
			exit 1; \
		}; \
	done
	@for source in $(DOC_SOURCES); do \
		test -f "$$source" || { \
			printf 'Error: document source is missing: %s\n' "$$source"; \
			exit 1; \
		}; \
	done
	@test -f "examples/tutorial_review.py" || { \
		printf '%s\n' \
			'Error: imported tutorial source is missing: examples/tutorial_review.py'; \
		exit 1; \
	}
	@printf '%s\n' 'Documentation build prerequisites are available.'
