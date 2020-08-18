# Minimal makefile for Sphinx documentation
#
# This was generated automatically with sphinx-quickstart

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
LINKCHECKDIR  = _build/linkcheck

# Additional variables for the manual-only LaTeX-to-PDF build
SOURCEDIR_MANUAL = $(SOURCEDIR)/manual
BUILDDIR_MANUAL = _build_manual_latex
LATEXMK = latexmk

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile checklinks

checklinks:
	@$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(LINKCHECKDIR)
	@echo
	@echo "Check finished. Report is in $(LINKCHECKDIR)."

rebuild:
	@$(SPHINXBUILD) -E "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Generate a PDF version of the users manual, by using the manual directory
# as the root, using the latex builder, and calling latexmk
manual_pdf:
	@$(SPHINXBUILD) -b latex -c "$(SOURCEDIR)" "$(SOURCEDIR_MANUAL)" "$(BUILDDIR_MANUAL)"
	cd "$(BUILDDIR_MANUAL)" && $(LATEXMK)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
