
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/classes/random/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       =
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o

#----------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1

runex1_2:
	-@${MPIEXEC}  -n 2 ./ex1

runex1_3:
	-@${MPIEXEC} -n 1 ./ex1 -view_randomvalues 0 > ex1_3.tmp 2>&1;   \
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_3.tmp

runex2:
	-@${MPIEXEC} -n 2 ./ex2

TESTEXAMPLES_C		       = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm
TESTEXAMPLES_C_X	       =
TESTEXAMPLES_FORTRAN	       =
TESTEXAMPLES_FORTRAN_NOCOMPLEX =
TESTEXAMPLES_FORTRAN_COMPLEX   =
TESTEXAMPLES_FORTRAN_MPIUNI    =
TESTEXAMPLES_C_X_MPIUNI      =
TESTEXAMPLES_13		       =
TESTEXAMPLES_C_NOCOMPLEX       = ex2.PETSc runex2 ex2.rm

include ${PETSC_DIR}/lib/petsc/conf/test
