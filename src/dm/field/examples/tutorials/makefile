#
CFLAGS		 =
FFLAGS		 =
CPPFLAGS   =
FPPFLAGS   =
MANSEC     = DM
LOCDIR		 = src/dm/field/examples/tutorials/
EXAMPLESC	 = ex1.c

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o chkopts

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

include ${PETSC_DIR}/lib/petsc/conf/test

