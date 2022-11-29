-include ../../../../../petscdir.mk
# -*- makefile -*-

MPIEXEC=
PYTHON=python

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
