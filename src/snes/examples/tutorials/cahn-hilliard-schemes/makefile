
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/snes/examples/tutorials/cahn-hilliard-schemes/
EXAMPLESC       = ex1.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = SNES

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o


TESTEXAMPLES_C		  = ex1.PETSc ex1.rm
TESTEXAMPLES_C_X	  =

include ${PETSC_DIR}/conf/test
