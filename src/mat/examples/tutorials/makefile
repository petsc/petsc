# $Id: makefile,v 1.16 1999/04/01 17:10:02 balay Exp bsmith $ 

ALL: ex1

CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/

TESTEXAMPLES_1  = ex1.PETSc ex1.rm clean
TESTEXAMPLES_2  =
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
TESTEXAMPLES_6  = 
TESTEXAMPLES_7  =
TESTEXAMPLES_9  =
TESTEXAMPLES_13 = ex1.PETSc ex1.rm clean
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_3 =
BUILDEXAMPLES_4 =
BUILDEXAMPLES_6 =
BUILDEXAMPLES_7 =
BUILDEXAMPLES_9 =

EXAMPLESC       = ex1.c
EXAMPLESF       =

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_MAT_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_MAT_LIB} 
	${RM} ex2.o



