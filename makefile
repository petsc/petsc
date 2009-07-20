.PHONY: default \
	src cython \
	config build test install sdist \
	docs rst2html sphinx sphinx-html sphinx-pdf epydoc epydoc-html \
	clean distclean srcclean docsclean fullclean uninstall

PYTHON = python

default: build

config: 
	${PYTHON} setup.py config ${CONFIGOPT}

build: src
	${PYTHON} setup.py build ${BUILDOPT}

test:
	${MPIEXEC} ${VALGRIND} ${PYTHON} test/runtests.py < /dev/null

install: build
	${PYTHON} setup.py install ${INSTALLOPT} --home=${HOME}

sdist: src docs
	${PYTHON} setup.py sdist ${SDISTOPT}

clean:
	${PYTHON} setup.py clean --all

distclean: clean docsclean
	-${RM} -r build  _configtest.* *.py[co]
	-${RM} -r MANIFEST dist petsc4py.egg-info
	-${RM} `find . -name '*~'`
	-${RM} `find . -name '*.py[co]'`

fullclean: distclean srcclean docsclean

uninstall:
	-${RM} -r ${HOME}/lib/python/petsc4py
	-${RM} -r ${HOME}/lib/python/petsc4py-*-py*.egg-info

# ----

src: src/PETSc.c

srcclean:
	-${RM} src/petsc4py.PETSc.c
	-${RM} src/include/petsc4py/petsc4py.PETSc.h
	-${RM} src/include/petsc4py/petsc4py.PETSc_api.h

CY_SRC_PXD = $(wildcard src/include/petsc4py/*.pxd)
CY_SRC_PXI = $(wildcard src/PETSc/*.pxi)
CY_SRC_PYX = $(wildcard src/PETSc/*.pyx)
src/PETSc.c: src/petsc4py.PETSc.c
src/petsc4py.PETSc.c: ${CY_SRC_PXD} ${CY_SRC_PXI} ${CY_SRC_PYX}
	${PYTHON} ./conf/cythonize.py

cython:
	${PYTHON} ./conf/cythonize.py

# ----

docs: rst2html sphinx epydoc

docsclean:
	-${RM} docs/*.html docs/*.pdf
	-${RM} -r docs/usrman docs/apiref

RST2HTML = rst2html
RST2HTMLOPTS = --no-compact-lists --cloak-email-addresses
rst2html:
	${RST2HTML} ${RST2HTMLOPTS} docs/index.rst > docs/index.html
	${RST2HTML} ${RST2HTMLOPTS} LICENSE.txt    > docs/LICENSE.html
	${RST2HTML} ${RST2HTMLOPTS} HISTORY.txt    > docs/HISTORY.html

SPHINXBUILD = sphinx-build
SPHINXOPTS  =
sphinx: sphinx-html sphinx-pdf
sphinx-html:
	${PYTHON} -c 'import petsc4py.PETSc'
	mkdir -p build/doctrees docs/usrman
	${SPHINXBUILD} -b html -d build/doctrees ${SPHINXOPTS} \
	docs/source docs/usrman
sphinx-pdf:
	${PYTHON} -c 'import petsc4py.PETSc'
	mkdir -p build/doctrees build/latex
	${SPHINXBUILD} -b latex -d build/doctrees ${SPHINXOPTS} \
	docs/source build/latex
	${MAKE} -C build/latex all-pdf > /dev/null
	mv build/latex/*.pdf docs/

EPYDOCBUILD = ${PYTHON} ./conf/epydocify.py
EPYDOCOPTS  =
epydoc: epydoc-html
epydoc-html:
	${PYTHON} -c 'import petsc4py.PETSc'
	mkdir -p docs/apiref
	${EPYDOCBUILD} ${EPYDOCOPTS} -o docs/apiref

# ----
