# $Id: makefile,v 1.10 1998/06/13 05:18:18 balay Exp bsmith $ 

ALL: ex1

CFLAGS	        =
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/viewer/examples/tutorials/

TESTEXAMPLES_2  = 
TESTEXAMPLES_4  = 
BUILDEXAMPLES_4 = 
EXAMPLESC       = ex1.c ex2.c

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o


