
# This directory contains SemiLagrange example programs for
# advecting fields using the method of characteristics.

CFLAGS		   =
FFLAGS		   =
CPPFLAGS           =
FPPFLAGS           =
LOCDIR		   = src/ts/characteristic/examples/tutorials/
MANSEC             = SemiLagrange
EXAMPLESC	   = 
EXAMPLESF	   =
EXAMPLESCH	   =
EXAMPLESFH         =
EXAMPLESMATLAB     =
DIRS		   =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_TS_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TS_LIB}
	${RM} ex2.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_TS_LIB}
	${RM} ex4.o
#--------------------------------------------------------------------------

TESTEXAMPLES_C_NOCOMPLEX = 

include ${PETSC_DIR}/conf/test
