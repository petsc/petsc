# -*- makefile -*-

MPIEXEC=
PYTHON=python

.PHONY:test
test: run clean

SCRIPT=run_demo
MODULE=Bratu3D

.PHONY:build
build: ${MODULE}.so

.PHONY:run
run: build
	${MPIEXEC} ${PYTHON} ${SCRIPT}.py

${MODULE}.so: ${MODULE}.pyx ${MODULE}impl.h ${MODULE}impl.c
	CC=${CC} F90=${FC} LDSHARED='${CLINKER} -shared' \
	${PYTHON} setup.py -q build_ext --inplace
	${RM} -r build ${MODULE}_wrap.c

.PHONY:clean
clean::
	${RM} ${MODULE}.c ${MODULE}*.so
	${RM} *.py[co]
	${RM} -r __pycache__

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
MPIEXEC=
