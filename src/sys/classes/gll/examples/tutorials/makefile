
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/classes/gll/examples/tutorials/
EXAMPLESC       =
EXAMPLESF       =
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_LIB}
	${RM} -f ex1.o

#------------------------------------------------------------------
TESTEXAMPLES_C = ex1.PETSc runex1 ex1.rm

include ${PETSC_DIR}/lib/petsc/conf/test
