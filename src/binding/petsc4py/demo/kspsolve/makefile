-include ../../../../../petsckdir.mk
# -*- makefile -*-

MPIEXEC=
PYTHON=python

.PHONY:test
test: run clean

.PHONY:run
run: run_1 run_2

SCRIPT1=test_mat_cg
.PHONY:run_1
run_1:
	${MPIEXEC} ${PYTHON} ${SCRIPT1}.py

SCRIPT2= test_mat_ksp
.PHONY:run_2
run_2:
	${MPIEXEC} ${PYTHON} ${SCRIPT2}.py

.PHONY:clean
clean:
	${RM} *.py[co]
	${RM} -r __pycache__
