# -*- makefile -*-

MPIEXEC=mpiexec -n 2
PYTHON=python3

scripts = ex2 ex23

all: $(scripts)

$(scripts):
	${PYTHON} $@.py
	${MPIEXEC} ${PYTHON} $@.py
