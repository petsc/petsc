# $Id: makefile,v 1.16 1999/07/16 14:06:54 balay Exp bsmith $ 

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = src/dm/da/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o
#-----------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 2 ex1 -nox > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
TESTEXAMPLES_1  =
TESTEXAMPLES_2  =
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
TESTEXAMPLES_5  =
TESTEXAMPLES_13 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm

include ${PETSC_DIR}/bmake/common_test