# -*- makefile -*-

MPIEXEC=
PYTHON=python

-include ../../../../../petscdir.mk
-include ${PETSC_DIR}/lib/petsc/conf/variables

.PHONY:test
test: run clean

.PHONY:run
run:
	${MPIEXEC} ${PYTHON} matvecio.py

.PHONY:clean
clean:
	${RM} matrix-*.dat* vector-*.dat*
	${RM} *.py[co]
	${RM} -r __pycache__
