CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_MAT_LIB}
	${RM} -f ex1.o

include ${PETSC_DIR}/conf/base
