# -*- makefile -*-

MPIEXEC=
PYTHON=python

.PHONY:test
test: run clean

.PHONY:run
run:
	${MPIEXEC} ${PYTHON} poisson2d.py -fd -nx 15 -ny 16 -nz 17

.PHONY:clean
clean:
	${RM} *.py[co]
	${RM} -r __pycache__
