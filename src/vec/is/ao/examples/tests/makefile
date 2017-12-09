
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/is/ao/examples/tests/
DIRS            = ex3d
EXAMPLESC       = ex1.c ex2.c ex4.c ex7.c
EXAMPLESF       = ex4f.F
MANSEC          = Vec
SUBMANSEC       = AO

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_VEC_LIB}
	${RM} -f ex4.o
ex4f: ex4f.o  chkopts
	-${FLINKER} -o ex4f ex4f.o ${PETSC_VEC_LIB}
	${RM} -f ex4f.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_VEC_LIB}
	${RM} -f ex7.o

include ${PETSC_DIR}/lib/petsc/conf/test
