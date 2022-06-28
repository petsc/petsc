# Initially generated with sphinx-quickstart

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
LINKCHECKDIR  = _build/linkcheck

# First so that "make" without arguments is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help makefile checklinks

checklinks:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(LINKCHECKDIR)" $(SPHINXOPTS)
	@echo
	@echo "Check finished. Report is in $(LINKCHECKDIR)."

clean-classic-docs:
	./build_classic_docs.py --clean

clean: clean-classic-docs

images:
	git clone https://gitlab.com/petsc/images.git -b main-1 --depth=1 images

website-deploy: images
	$(SPHINXBUILD) -b dirhtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -T -E -j auto

sphinxhtml:
	cd .. ; python3 -m venv petsc-doc-env ; source petsc-doc-env/bin/activate ; python3 -m pip install -r doc/requirements.txt ; cd doc ; make html; deactivate

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.
%: makefile images
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -j auto

