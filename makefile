.PHONY: default \
	src cython \
	config build test install \
	docs sphinx sphinx-html sphinx-pdf epydoc \
	sdist \
	clean distclean srcclean docsclean fullclean uninstall

PYTHON = python

default: build

src: src/PETSc.c

config: 
	${PYTHON} setup.py config ${CONFIGOPT}

build: src
	${PYTHON} setup.py build ${BUILDOPT}

test:
	${MPIEXEC} ${VALGRIND} ${PYTHON} test/runtests.py < /dev/null

install: build
	${PYTHON} setup.py install ${INSTALLOPT} --home=${HOME}

docs: sphinx epydoc

clean:
	${PYTHON} setup.py clean --all

distclean: clean docsclean
	-${RM} -r build  _configtest.* *.py[co]
	-${RM} -r MANIFEST dist petsc4py.egg-info
	-${RM} `find . -name '*~'`
	-${RM} `find . -name '*.py[co]'`

srcclean:
	-${RM} src/petsc4py_PETSc.c
	-${RM} src/include/petsc4py/petsc4py_PETSc.h
	-${RM} src/include/petsc4py/petsc4py_PETSc_api.h

docsclean:
	-${RM} -r docs/html docs/*.pdf

fullclean: distclean srcclean docsclean

uninstall:
	-${RM} -r ${HOME}/lib/python/petsc4py
	-${RM} -r ${HOME}/lib/python/petsc4py-*-py*.egg-info

CY_SRC_PXD = $(wildcard src/include/petsc4py/*.pxd)
CY_SRC_PXI = $(wildcard src/PETSc/*.pxi)
CY_SRC_PYX = $(wildcard src/PETSc/*.pyx)
src/PETSc.c: src/petsc4py_PETSc.c
src/petsc4py_PETSc.c: ${CY_SRC_PXD} ${CY_SRC_PXI} ${CY_SRC_PYX}
	${PYTHON} ./conf/cythonize.py

cython:
	${PYTHON} ./conf/cythonize.py

SPHINXBUILD = sphinx-build
SPHINXOPTS  =
sphinx: sphinx-html sphinx-pdf
sphinx-html:
	${PYTHON} -c 'import petsc4py.PETSc'
	mkdir -p build/doctrees docs/html/man
	${SPHINXBUILD} -b html -d build/doctrees ${SPHINXOPTS} \
	docs/source docs/html/man
sphinx-pdf:
	${PYTHON} -c 'import petsc4py.PETSc'
	mkdir -p build/doctrees build/latex
	${SPHINXBUILD} -b latex -d build/doctrees ${SPHINXOPTS} \
	docs/source build/latex
	${MAKE} -C build/latex all-pdf > /dev/null
	mv build/latex/*.pdf docs/

EPYDOCBUILD = ${PYTHON} ./conf/epydocify.py
EPYDOCOPTS  =
epydoc:
	mkdir -p docs/html/api
	${EPYDOCBUILD} ${EPYDOCOPTS} -o docs/html/api 


sdist: src docs
	${PYTHON} setup.py sdist ${SDISTOPT}
