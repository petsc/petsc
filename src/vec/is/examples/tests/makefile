
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/is/examples/tests/
DIRS            =
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       =
MANSEC          = Vec
SUBMANSEC       = IS

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_VEC_LIB}
	${RM} -f ex2.o

#-------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2
	@${RM} testfile

runex2_2:
	-@${MPIEXEC} -n 2 ./ex2
	@${RM} testfile

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 runex2_2 ex2.rm
TESTEXAMPLES_C_X	  =
TESTEXAMPLES_FORTRAN	  =
TESTEXAMPLES_C_X_MPIUNI =
TESTEXAMPLES_C3		  =

include ${PETSC_DIR}/lib/petsc/conf/test
