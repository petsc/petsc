# Initially generated with sphinx-quickstart

# Specify SPHINXOPTS via 'make' invocation to change/reset defaults, SPHINXOPTS_EXTRA to add extra options to defaults
SPHINXOPTS    = --fresh-env --write-all --show-traceback -j auto
SPHINXBUILD   = sphinx-build
export PETSC_DIR = $(abspath ..)
override PETSC_ARCH := $(or $(strip $(PETSC_ARCH)),arch-docs)
export PETSC_ARCH
DOCS_DIR      = $(PETSC_DIR)/$(PETSC_ARCH)/doc
SOURCEDIR     = $(DOCS_DIR)/source/doc
BUILDDIR      = $(DOCS_DIR)/_build
LINKCHECKDIR  = $(BUILDDIR)/linkcheck
DOCS_VENV     = $(PETSC_DIR)/petsc-doc-env
DOCS_IMAGES   = $(DOCS_DIR)/images

# First so that "make" without arguments is like "make help".
help:
	-@echo "   make docs - build the HTML website inside a created Python virtual environment (venv)"

# The following does not work because it triggers the loading of docutils that many people may not have in Python
#	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)

.PHONY: help makefile checklinks checklinks_rule clean docs-env docs-source docs docspdf images website-deploy

checklinks_rule: docs-source
	@cd "$(SOURCEDIR)" && $(SPHINXBUILD) --builder linkcheck . "$(abspath $(LINKCHECKDIR))" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)
	@echo
	@echo "Check finished. Report is in $(LINKCHECKDIR)."

docs-env:
	test -x "$(DOCS_VENV)/bin/python" || python3 -m venv "$(DOCS_VENV)"
	. "$(DOCS_VENV)/bin/activate" && python -m pip install -r requirements.txt

checklinks: docs-env
	. "$(DOCS_VENV)/bin/activate" && $(MAKE) checklinks_rule

docs-source: images
	python3 -B prepare_docs.py "$(PETSC_DIR)" "$(PETSC_ARCH)"

clean:
	${RM} -r "$(abspath $(BUILDDIR))" "$(DOCS_DIR)/source"

# Keep the downloaded images across documentation rebuilds.
images: $(DOCS_IMAGES)

$(DOCS_IMAGES):
	mkdir -p "$(DOCS_DIR)"
	git clone https://gitlab.com/petsc/images.git -b main-2 --depth=1 "$@"

website-deploy: docs-source
	cd "$(SOURCEDIR)" && $(SPHINXBUILD) --builder dirhtml . "$(abspath $(BUILDDIR))" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)

# use this rule to build the documentation on your machine when working on PETSc documentation
docs: docs-env clean
	. "$(DOCS_VENV)/bin/activate" && $(MAKE) SPHINXOPTS_EXTRA="--fail-on-warning --keep-going -j 8" html

# use this rule AFTER make docs to build the users manual as a PDF file
docspdf: docs-env
	${RM} -r "$(abspath $(BUILDDIR))/latex"
	. "$(DOCS_VENV)/bin/activate" && $(MAKE) SPHINXOPTS_EXTRA="--fail-on-warning --keep-going -j 8" latexpdf

# Catch-all target: route all unknown targets to Sphinx using the new "make mode" option.
%: makefile docs-source
	@cd "$(SOURCEDIR)" && $(SPHINXBUILD) -M $@ . "$(abspath $(BUILDDIR))" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)
