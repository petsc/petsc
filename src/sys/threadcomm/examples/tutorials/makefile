

CFLAGS           =
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/sys/threadcomm/examples/tutorials/
MANSEC           = PetscThreadComm
EXAMPLESC        = ex1.c ex2.c ex3.c ex4.c ex5.c

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_LIB}
	${RM} -f ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_VEC_LIB}
	${RM} -f ex3.o

ex4: ex4.o chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_VEC_LIB}
	${RM} -f ex4.o

ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_VEC_LIB}
	${RM} -f ex5.o
