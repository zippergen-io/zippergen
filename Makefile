DOC_NAME := workflow-development-deployment-guide
DOC_DIR := docs
DOC_BUILD_DIR := $(DOC_DIR)/_build
DOC_SOURCE := $(DOC_NAME).tex

.PHONY: docs docs-check

docs: docs-check
	@mkdir -p "$(DOC_BUILD_DIR)"
	@cd "$(DOC_DIR)" && latexmk \
		-pdf \
		-interaction=nonstopmode \
		-halt-on-error \
		-file-line-error \
		-outdir=_build \
		"$(DOC_SOURCE)"
	@printf 'Built %s/%s.pdf\n' "$(DOC_BUILD_DIR)" "$(DOC_NAME)"

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
	@test -f "$(DOC_DIR)/$(DOC_SOURCE)" || { \
		printf 'Error: document source is missing: %s/%s\n' \
			"$(DOC_DIR)" "$(DOC_SOURCE)"; \
		exit 1; \
	}
	@test -f "examples/tutorial_review.py" || { \
		printf '%s\n' \
			'Error: imported tutorial source is missing: examples/tutorial_review.py'; \
		exit 1; \
	}
	@printf '%s\n' 'Documentation build prerequisites are available.'
