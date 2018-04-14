
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

include ${PETSC_DIR}/lib/petsc/conf/test
