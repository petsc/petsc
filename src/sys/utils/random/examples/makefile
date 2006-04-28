
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/utils/random/examples
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = 
MANSEC          = Sys

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o

#----------------------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 2 ./ex1

runex2:
	-@${MPIRUN} -np 2 ./ex2

TESTEXAMPLES_C		       = 
TESTEXAMPLES_C_X11	       =
TESTEXAMPLES_FORTRAN	       =
TESTEXAMPLES_FORTRAN_NOCOMPLEX =
TESTEXAMPLES_FORTRAN_COMPLEX   =
TESTEXAMPLES_FORTRAN_MPIUNI    =
TESTEXAMPLES_C_X11_MPIUNI      =
TESTEXAMPLES_13		       =

include ${PETSC_DIR}/bmake/common/test
