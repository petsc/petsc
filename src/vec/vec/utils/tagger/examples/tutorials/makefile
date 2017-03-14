#
CFLAGS		 =
FFLAGS		 =
CPPFLAGS   =
FPPFLAGS   =
MANSEC     = Vec
LOCDIR		 = src/vec/vec/utils/tagger/examples/tutorials/
EXAMPLESC	 = ex1.c

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o chkopts

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_VEC_LIB}
	${RM} -f ex1.o

include ${PETSC_DIR}/lib/petsc/conf/test
