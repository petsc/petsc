# -*- makefile -*-

MPIEXEC=
PYTHON=python

.PHONY:test
test: run clean

SCRIPT=run_demo
MODULE=Bratu3D
.PHONY:run
run: ${MODULE}.py _${MODULE}.so
	${MPIEXEC} ${PYTHON} ${SCRIPT}.py

${MODULE}.py _${MODULE}.so: ${MODULE}.i ${MODULE}.c ${MODULE}.h
	CC=${CC} F90=${FC} LDSHARED='${CLINKER} -shared' \
	${PYTHON} setup.py -q build_ext --inplace
	${RM} -r build ${MODULE}_wrap.c

.PHONY:clean
clean::
	${RM} ${MODULE}.py _${MODULE}*.so
	${RM} *.py[co]
	${RM} -r __pycache__

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
MPIEXEC=
