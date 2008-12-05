.PHONY: default src \
	config build test install \
	clean distclean srcclean fullclean uninstall \
	cython epydoc sdist

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

clean:
	${PYTHON} setup.py clean --all
	-${RM} _configtest.* *.py[co]
	-${MAKE} -C docs clean

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

epydoc:
	${PYTHON} ./conf/epydocify.py -o /tmp/petsc4py-api-doc

sdist:
	${PYTHON} setup.py sdist ${SDISTOPT}
