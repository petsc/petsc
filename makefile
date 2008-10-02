.PHONY: default config src build test install uninstall sdist clean distclean srcclean fullclean

PYTHON = python

default: build

config: 
	${PYTHON} setup.py config ${CONFIGOPT}

src: src/PETSc.c

build: src
	${PYTHON} setup.py build ${BUILDOPT}

test:
	${MPIEXEC} ${PYTHON} test/runtests.py < /dev/null

install: build
	${PYTHON} setup.py install ${INSTALLOPT} --home=${HOME}

uninstall:
	-${RM} -r ${HOME}/lib/python/petsc4py
	-${RM} -r ${HOME}/lib/python/petsc4py-*-py*.egg-info

sdist:
	${PYTHON} setup.py sdist ${SDISTOPT}


clean:
	${PYTHON} setup.py clean --all
	-${RM} _configtest.* *.py[co]

distclean: clean 
	-${RM} -r build  *.py[co]
	-${RM} -r MANIFEST dist petsc4py.egg-info
	-${RM} `find . -name '*~'`
	-${RM} `find . -name '*.py[co]'`

srcclean:
	-${RM} src/petsc4py_PETSc.c
	-${RM} src/include/petsc4py/petsc4py_PETSc.h
	-${RM} src/include/petsc4py/petsc4py_PETSc_api.h

fullclean: distclean srcclean

CYTHON = cython
CYTHON_FLAGS = --cleanup 9
CYTHON_INCLUDE = -I. -Iinclude
CY_SRC_PXD = $(wildcard src/include/petsc4py/*.pxd)
CY_SRC_PXI = $(wildcard src/PETSc/*.pxi)
CY_SRC_PYX = $(wildcard src/PETSc/*.pyx)
src/PETSc.c: src/petsc4py_PETSc.c
src/petsc4py_PETSc.c: ${CY_SRC_PXD} ${CY_SRC_PXI} ${CY_SRC_PYX}
	${CYTHON} ${CYTHON_FLAGS} ${CYTHON_INCLUDE} -w src petsc4py.PETSc.pyx -o petsc4py_PETSc.c
	mv src/petsc4py_PETSc_api.h src/petsc4py_PETSc.h src/include/petsc4py

EPYDOC = ./misc/epydoc-cython.py
EPYDOC_CONF = ./misc/epydoc.cfg
EPYDOC_FLAGS =
EPYDOC_CMD = ${EPYDOC} -v --config=${EPYDOC_CONF} ${EPYDOC_FLAGS}
EPYDOC_OUT = /tmp/petsc4py-api-doc
epydoc:
	${EPYDOC_CMD} -o ${EPYDOC_OUT}
