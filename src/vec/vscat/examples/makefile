
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/vscat/examples/tests/
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = Vec

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_VEC_LIB}
	${RM} -f ex3.o

include ${PETSC_DIR}/lib/petsc/conf/test
