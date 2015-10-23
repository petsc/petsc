
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/dt/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o
ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o
ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

#-------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -degrees 1,2,3,4,5 -points 0,.2.-.5,.8,.9,1 -interval -.5,1 > ex1_1.tmp 2>&1;	  \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1_1, diffs above\n==========================================\n"; \
	   ${RM} -f ex1_1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -degrees 1,2,3 -target_points -0.3,0,.2 -src_points -1,-.3,0,.2,1 > ex2_1.tmp 2>&1;	  \
	   ${DIFF} output/ex2_1.out ex2_1.tmp || printf "${PWD}\nPossible problem with ex2_1, diffs above\n==========================================\n"; \
	   ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 > ex3_0.tmp 2>&1;	  \
	   ${DIFF} output/ex3_0.out ex3_0.tmp || printf "${PWD}\nPossible problem with ex3_0, diffs above\n==========================================\n"; \
	   ${RM} -f ex3_0.tmp

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm
TESTEXAMPLES_C_X	  =
TESTEXAMPLES_FORTRAN	  =
TESTEXAMPLES_C_X_MPIUNI =
TESTEXAMPLES_C_COMPLEX	  =
TESTEXAMPLES_C_NOCOMPLEX  =
TESTEXAMPLES_13		  =

include ${PETSC_DIR}/lib/petsc/conf/test
