# -*- makefile -*-

MPIEXEC=
PYTHON=python

-include ../../../../../petscdir.mk
-include ${PETSC_DIR}/lib/petsc/conf/variables

.PHONY:test
test: run clean

.PHONY:run
run:
	${MPIEXEC} ${PYTHON} poisson2d.py -nx 15 -ny 16

.PHONY:clean
clean:
	${RM} *.py[co]
	${RM} -r __pycache__
