# $Id: makefile,v 1.17 2000/07/13 03:03:52 bsmith Exp bsmith $ 



CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/
EXAMPLESC       = ex1.c
EXAMPLESF       =

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_MAT_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_MAT_LIB} 
	${RM} ex2.o

#-------------------------------------------------------------
TESTEXAMPLES_1  = ex1.PETSc ex1.rm
TESTEXAMPLES_2  =
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
TESTEXAMPLES_6  = 
TESTEXAMPLES_7  =
TESTEXAMPLES_9  =
TESTEXAMPLES_13 = ex1.PETSc ex1.rm


include ${PETSC_DIR}/bmake/common_test

