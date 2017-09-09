
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/classes/random/examples/tests/
EXAMPLESC       = ex3.c
EXAMPLESF       =
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

#----------------------------------------------------------------------------
runex3:
	-@${MPIEXEC} -n 1 ./ex3

TESTEXAMPLES_C		       =
TESTEXAMPLES_C_X	       =
TESTEXAMPLES_FORTRAN	       =
TESTEXAMPLES_FORTRAN_NOCOMPLEX =
TESTEXAMPLES_FORTRAN_COMPLEX   =
TESTEXAMPLES_FORTRAN_MPIUNI    =
TESTEXAMPLES_C_X_MPIUNI      =
TESTEXAMPLES_13		       =

include ${PETSC_DIR}/lib/petsc/conf/test
