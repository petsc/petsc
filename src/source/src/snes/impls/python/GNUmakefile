# -*- makefile -*-

#PETSC_PYTHON_LIB = -L${PYTHON_LIBDIR} -l${PYTHON_LIB}
PETSC_PYTHON_LIB = ${PYTHON_LIBDIR}/${PYTHON_LIB}/config/lib${PYTHON_LIB}.a -lutil -lm -rdynamic

SOURCEC = 
OBJSC   = 
EXAMPLESC = ex1.c

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SNES_LIB} ${PETSC_PYTHON_LIB}
	${RM} ex1.o

distclean:
	${RM} *.o  ex1 *.py[co]

include makefile
