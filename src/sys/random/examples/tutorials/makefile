
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/random/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = 
MANSEC          = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

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

runex2:
	-@${MPIEXEC} -n 2 ./ex2

TESTEXAMPLES_C		       = ex1.PETSc runex1 runex1_2 ex1.rm 
TESTEXAMPLES_C_X	       =
TESTEXAMPLES_FORTRAN	       =
TESTEXAMPLES_FORTRAN_NOCOMPLEX =
TESTEXAMPLES_FORTRAN_COMPLEX   =
TESTEXAMPLES_FORTRAN_MPIUNI    =
TESTEXAMPLES_C_X_MPIUNI      =
TESTEXAMPLES_13		       =
TESTEXAMPLES_C_NOCOMPLEX       = ex2.PETSc runex2 ex2.rm

include ${PETSC_DIR}/conf/test
