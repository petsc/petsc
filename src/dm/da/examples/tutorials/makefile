# $Id: makefile,v 1.21 2001/08/22 18:02:42 balay Exp $ 

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = src/dm/da/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = 
MANSEC          = DA

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o
#-----------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 2 ex1
runex2:
	-@${MPIRUN} -np 3 ex2

TESTEXAMPLES_1  =
TESTEXAMPLES_2  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
TESTEXAMPLES_5  =
TESTEXAMPLES_13 = 

include ${PETSC_DIR}/bmake/common/test
