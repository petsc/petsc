# $Id: makefile,v 1.1 2000/01/24 17:11:48 bsmith Exp bsmith $ 

ALL: ex1

#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/pf/examples/tutorials/

TESTEXAMPLES_1	 = ex1.PETSc ex1.rm
TESTEXAMPLES_2	 = 
TESTEXAMPLES_3	 = 
TESTEXAMPLES_4	 = 
TESTEXAMPLES_9	 = 
TESTEXAMPLES_12	 = 
TESTEXAMPLES_13  = 
BUILDEXAMPLES_1	 = 
BUILDEXAMPLES_2	 = 
BUILDEXAMPLES_3	 = 
BUILDEXAMPLES_4	 = 
BUILDEXAMPLES_9	 = 
BUILDEXAMPLES_12 = 

EXAMPLESC	 = ex1.c
EXAMPLESF	 = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts 
	-${CLINKER} -o ex1 ex1.o ${PETSC_LIB}
	${RM} -f ex1.o

