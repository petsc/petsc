# -*- makefile -*-

MPIEXEC=
PYTHON=python

-include ../../../../../petscdir.mk
-include ${PETSC_DIR}/lib/petsc/conf/variables

.PHONY:test
test: run clean

.PHONY:run
run: run_fd run_mf

.PHONY:run_fd
run_fd:
	${MPIEXEC} ${PYTHON} bratu3d.py -fd -nx 7 -ny 8 -nz 9

.PHONY:run_mf
run_mf:
	${MPIEXEC} ${PYTHON} bratu3d.py -mf -nx 9 -ny 8 -nz 7

.PHONY:clean
clean:
	${RM} *.py[co]
	${RM} -r __pycache__
