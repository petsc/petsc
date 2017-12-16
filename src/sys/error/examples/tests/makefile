
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/sys/error/examples/tests/
EXAMPLESC        = ex1.c
EXAMPLESF        =
MANSEC           = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_LIB}
	${RM} ex1.o

include ${PETSC_DIR}/lib/petsc/conf/test
