# $Id: makefile,v 1.21 2001/09/07 20:10:34 bsmith Exp $ 



CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       =
MANSEC          = Mat

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_MAT_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_MAT_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER}  -o ex3 ex3.o ${PETSC_KSP_LIB}
	${RM} ex3.o

#-------------------------------------------------------------
TESTEXAMPLES_1  = ex1.PETSc ex1.rm
TESTEXAMPLES_2  = ex2.PETSc ex2.rm
TESTEXAMPLES_3  = ex3.PETSc ex3.rm
TESTEXAMPLES_4  =
TESTEXAMPLES_6  = 
TESTEXAMPLES_9  =
TESTEXAMPLES_13 = ex1.PETSc ex1.rm


include ${PETSC_DIR}/bmake/common/test

