.PHONY: f2py petsc

all: f2py petsc

f2py:
	${MAKE} -f makefile.f2py
petsc:
	${MAKE} -f makefile.petsc

clean:
	${MAKE} -f makefile.f2py clean
	${MAKE} -f makefile.petsc clean lclean
	${RM} *.py[co] *.o
