# -*- makefile -*-
.PHONY: all f2py petsc clean

all: f2py petsc

f2py:
	${MAKE} -f makefile.f2py
petsc:
	${MAKE} -f makefile.petsc

clean:
	${RM} *.py[co]
	${MAKE} -f makefile.f2py clean
	${MAKE} -f makefile.petsc clean lclean

.PHONY: test test_py test_cc
test: f2py petsc test_py test_cc
test_py: f2py
	${PYTHON} matfree.py
test_cc: petsc
	./matfree.exe

include makefile.conf
