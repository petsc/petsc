-include ../../../../../petsckdir.mk
# -*- makefile -*-

MPIEXEC=
PYTHON=python

F2PY = f2py --quiet
F2PY_FLAGS =
F2PY_FLAGS = -DF2PY_REPORT_ATEXIT -DF2PY_REPORT_ON_ARRAY_COPY=0
F2PY_FLAGS =--noarch --f90flags=''
F2PY_FLAGS +=-DF2PY_REPORT_ON_ARRAY_COPY=1

.PHONY:test
test: run clean

.PHONY:run
run: run_py run_f90

.PHONY:run_py
run_py:
	${MPIEXEC} ${PYTHON} bratu2d.py -impl python

MODULE=bratu2df90
.PHONY:${MODULE}
${MODULE}: ${MODULE}.so
${MODULE}.so: ${MODULE}.f90
	${F2PY} ${F2PY_FLAGS} -c $< -m ${MODULE}

.PHONY:run_f90
run_f90: ${MODULE}
	${MPIEXEC} ${PYTHON} bratu2d.py -impl fortran

.PHONY:clean
clean:
	${RM} *.py[co] ${MODULE}*.so
	${RM} -r __pycache__
