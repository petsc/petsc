# $Id: makefile,v 1.22 2001/08/22 18:01:35 balay Exp $ 

CFLAGS	        = 
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sles/pc/examples/tutorials/
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = PC

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_SLES_LIB} 
	${RM} ex1.o

TESTEXAMPLES_1  = ex1.PETSc ex1.rm

include ${PETSC_DIR}/bmake/common/test

