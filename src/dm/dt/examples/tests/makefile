
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/dt/examples/tests/
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

#-------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -degrees 1,2,3,4,5 -points 0,.2.-.5,.8,.9,1 -interval -.5,1 > ex1_1.tmp 2>&1;	  \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || printf "Possible problem with with ex1_1, diffs above \n==========================================n"; \
	   ${RM} -f ex1_1.tmp

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm
TESTEXAMPLES_C_X	  =
TESTEXAMPLES_FORTRAN	  =
TESTEXAMPLES_C_X_MPIUNI =
TESTEXAMPLES_C_COMPLEX	  =
TESTEXAMPLES_C_NOCOMPLEX  =
TESTEXAMPLES_13		  =

include ${PETSC_DIR}/conf/test
