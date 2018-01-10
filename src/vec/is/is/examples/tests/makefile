
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/is/is/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
EXAMPLESF       = ex1f.F90 ex2f.F90

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex1f: ex1f.o chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_VEC_LIB}
	${RM} -f ex1f.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex2f: ex2f.o chkopts
	-${FLINKER} -o ex2f ex2f.o ${PETSC_VEC_LIB}
	${RM} -f ex2f.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_VEC_LIB}
	${RM} -f ex3.o

ex4: ex4.o chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_VEC_LIB}
	${RM} -f ex4.o

ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_VEC_LIB}
	${RM} -f ex5.o

ex6: ex6.o chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_VEC_LIB}
	${RM} -f ex6.o

ex7: ex7.o chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_VEC_LIB}
	${RM} -f ex7.o

ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_VEC_LIB}
	${RM} -f ex8.o

include ${PETSC_DIR}/lib/petsc/conf/test
