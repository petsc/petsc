-include ../../../petscdir.mk
.PHONY: default
default: build
default: opt=--inplace

package = petsc4py
MODULE  = PETSc

PYTHON  = python$(py)
MPIEXEC = mpiexec

# ----

.PHONY: config build test
config:
	${PYTHON} setup.py config ${opt}
build:
	${PYTHON} setup.py build ${opt}
test:
	${VALGRIND} ${PYTHON} ${PWD}/test/runtests.py ${opt}
test-%:
	${MPIEXEC} -n $* ${VALGRIND} ${PYTHON} ${PWD}/test/runtests.py ${opt}

.PHONY: srcbuild srcclean
srcbuild:
	${PYTHON} setup.py build_src ${opt}
srcclean:
	-${RM} src/${package}/${MODULE}.c
	-${RM} src/${package}/${MODULE}.h
	-${RM} src/${package}/${MODULE}_api.h

.PHONY: clean distclean fullclean
clean: srcclean
	${PYTHON} setup.py clean --all
distclean: clean
	-${RM} -r build  _configtest.* *.py[co]
	-${RM} -r MANIFEST dist ${package}.egg-info
	-${RM} -r `find . -name '__pycache__'`
	-${RM} `find . -name '*.py[co]'`
fullclean: distclean srcclean docsclean
	-find . -name '*~' -exec rm -f {} ';'

# ----

.PHONY: install uninstall
install: build
	${PYTHON} setup.py install --prefix='' --user ${opt}
uninstall:
	-${RM} -r $(shell ${PYTHON} -m site --user-site)/${package}
	-${RM} -r $(shell ${PYTHON} -m site --user-site)/${package}-*-py*.egg-info

# ----

.PHONY: lint ruff-lint cython-lint check-docs
CYTHONLINT_SRC = src demo
CYTHONLINT_OPTS =
RUFF_SRC = .
RUFF_OPTS = check

cython-lint:
	conf/cythonize.sh -Wextra -Werror
	cython-lint ${CYTHONLINT_OPTS} ${CYTHONLINT_SRC}


ruff-lint:
	ruff ${RUFF_OPTS} ${RUFF_SRC}

lint:
	${RM} -r petsc4py-lint-env
	${PYTHON} -m venv petsc4py-lint-env
	. petsc4py-lint-env/bin/activate && python -m pip install -r conf/requirements-lint.txt
	. petsc4py-lint-env/bin/activate && ${MAKE} cython-lint
	. petsc4py-lint-env/bin/activate && ${MAKE} ruff-lint

check-docs:
	${RM} -r petsc4py-docs-env
	${PYTHON} -m venv petsc4py-docs-env
	. petsc4py-docs-env/bin/activate && python -m pip install -r conf/requirements-docs.txt
	. petsc4py-docs-env/bin/activate && CFLAGS=-O0 python -m pip install .
	. petsc4py-docs-env/bin/activate && ${MAKE} sphinx-html SPHINXOPTS=-Wj1

# ----

.PHONY: docs docs-html docs-pdf docs-misc
docs: docs-html docs-pdf docs-misc
docs-html: rst2html sphinx-html
docs-pdf:  sphinx-pdf
docs-misc: sphinx-man sphinx-info

RST2HTML = $(shell command -v rst2html || command -v rst2html.py || false)
RST2HTMLOPTS  = --input-encoding=utf-8
RST2HTMLOPTS += --no-compact-lists
RST2HTMLOPTS += --cloak-email-addresses
.PHONY: rst2html
rst2html: checkdocutils
	${RST2HTML} ${RST2HTMLOPTS} ./LICENSE.rst  > docs/LICENSE.html
	${RST2HTML} ${RST2HTMLOPTS} ./CHANGES.rst  > docs/CHANGES.html
	${RST2HTML} ${RST2HTMLOPTS} docs/index.rst > docs/index.html

SPHINXBUILD = sphinx-build
SPHINXOPTS  =
.PHONY: sphinx sphinx-html sphinx-pdf sphinx-man sphinx-info
sphinx: sphinx-html sphinx-pdf sphinx-man sphinx-info
sphinx-html:
	mkdir -p build/doctrees docs/html
	${SPHINXBUILD} -a -b html -d build/doctrees ${SPHINXOPTS} \
	docs/source docs/html
	${RM} docs/html/.buildinfo
sphinx-pdf:
	mkdir -p build/doctrees build/latex
	${SPHINXBUILD} -b latex -d build/doctrees ${SPHINXOPTS} \
	docs/source build/latex
	${MAKE} -C build/latex all-pdf > /dev/null
	mv build/latex/*.pdf docs/
sphinx-man:
	mkdir -p build/doctrees build/man
	${SPHINXBUILD} -b man -d build/doctrees ${SPHINXOPTS} \
	docs/source build/man
	mv build/man/*.[137] docs/
sphinx-info:
	mkdir -p build/doctrees build/texinfo
	${SPHINXBUILD} -b texinfo -d build/doctrees ${SPHINXOPTS} \
	docs/source build/texinfo
	${MAKE} -C build/texinfo info > /dev/null
	mv build/texinfo/*.info docs/

.PHONY: docsclean
docsclean:
	-${RM} docs/*.html docs/*.pdf
	-${RM} docs/*.info docs/*.[137]
	-${RM} -r docs/html
	-${RM} -r docs/source/_build
	-${RM} -r docs/source/demo
	-${RM} -r docs/source/reference
	-${RM} docs/source/*.inv

website:
	CFLAGS=-O0 python -m pip install .
	${MAKE} sphinx-html SPHINXOPTS=-Wj2
	mkdir -p ${LOC}/petsc4py
	mv docs/html/* ${LOC}/petsc4py

# ----

.PHONY: sdist
sdist:
	${PYTHON} setup.py sdist ${opt}

# ----
