# -*- makefile -*-

MPIEXEC=mpiexec -n 2
PYTHON=python3

-include ../../../../../../petscdir.mk
-include ${PETSC_DIR}/lib/petsc/conf/variables

scripts = ex2 ex23

all: $(scripts)

$(scripts):
	${PYTHON} $@.py
	${MPIEXEC} ${PYTHON} $@.py
