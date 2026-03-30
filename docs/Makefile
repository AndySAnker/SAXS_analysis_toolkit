# Sphinx documentation (run from this directory: cd docs && make docx).
# Requires Sphinx on PATH (e.g. activate the environment where you ran ``pip install -e ".[docs]"``).
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html docx clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

docx:
	@$(SPHINXBUILD) -M docx "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)"
