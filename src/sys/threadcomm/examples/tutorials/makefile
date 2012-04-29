

CFLAGS           =
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/sys/threadcomm/examples/tutorials/
MANSEC           = PetscThreadComm
EXAMPLESC        = ex1.c ex2.c

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_VEC_LIB}
	${RM} -f ex2.o
