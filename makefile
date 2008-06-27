.PHONY: default config src build test install uninstall sdist clean distclean srcclean fullclean

PYTHON = python

default: build

config: 
	${PYTHON} setup.py config ${CONFIGOPT}

src: src/petsc4py_PETSc.c

build:
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
CY_SRC_DIR = src/PETSc
CY_SRC_PXD = $(wildcard ${CY_SRC_DIR}/*.pxd)
CY_SRC_PXI = $(wildcard ${CY_SRC_DIR}/*.pxi)
CY_SRC_PYX = $(wildcard ${CY_SRC_DIR}/*.pyx)
src/petsc4py_PETSc.c: ${CY_SRC_PXD} ${CY_SRC_PXI} ${CY_SRC_PYX}
	cd src && ${CYTHON} ${CYTHON_FLAGS} petsc4py.PETSc.pyx -o petsc4py_PETSc.c
	cd src && mv petsc4py_PETSc_api.h petsc4py_PETSc.h include/petsc4py
