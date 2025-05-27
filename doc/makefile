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
	-@echo "   make sphinxhtml - build the HTML website inside a created Python virtual environment (venv)"

# The following does not work because it triggers the loading of docutils that many people may not have in Python
#	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help makefile checklinks

checklinks_rule:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(LINKCHECKDIR)" $(SPHINXOPTS)
	@echo
	@echo "Check finished. Report is in $(LINKCHECKDIR)."

checklinks:
	cd .. ; ${RM} -rf petsc-doc-env; python3 -m venv petsc-doc-env ; source petsc-doc-env/bin/activate ; python3 -m pip install -r doc/requirements.txt ; cd doc ; make checklinks_rule; deactivate

clean:
	-@${RM} -rf _build __pycache__

images:
	git clone https://gitlab.com/petsc/images.git -b main-2 --depth=1 images

website-deploy: images
	$(SPHINXBUILD) -b dirhtml "$(SOURCEDIR)" "$(BUILDDIR)"  -T -E -j auto $(SPHINXOPTS)

# use this rule to build the documentation on your machine when working on PETSc documentation
docs:
	cd .. ; if [ ! -z ${PETSC_ARCH} ]; then export PETSC_DOC=${PETSC_ARCH}-doc; ${RM} -rf $${PETSC_DOC} ;  cp -r  doc $${PETSC_DOC}; else export PETSC_DOC=doc ; fi;  python3 -m venv petsc-doc-env ; . petsc-doc-env/bin/activate ; python3 -m pip install --quiet -r doc/requirements.txt ; cd $${PETSC_DOC} ;  make  SPHINXOPTS="-T -E -W --keep-going -j 8" html  ;  deactivate

docspdf:
	cd .. ; if [ ! -z ${PETSC_ARCH} ]; then export PETSC_DOC=${PETSC_ARCH}-doc; ${RM} -rf $${PETSC_DOC} ;  cp -r  doc $${PETSC_DOC}; else export PETSC_DOC=doc ; fi;  python3 -m venv petsc-doc-env ; . petsc-doc-env/bin/activate ; python3 -m pip install --quiet -r doc/requirements.txt ; cd $${PETSC_DOC} ;  make  SPHINXOPTS="-T -E -W --keep-going -j 8" latexpdf  ;  deactivate

# Catch-all target: route all unknown targets to Sphinx using the new "make mode" option.
%: makefile images
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"  -j auto $(SPHINXOPTS)

