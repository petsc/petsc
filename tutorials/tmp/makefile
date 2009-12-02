CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_MAT_LIB}
	${RM} -f ex1.o

ex1f90: ex1f90.o chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_MAT_LIB}
	${RM} -f ex1f90.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
