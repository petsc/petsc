-include ../../../../../petscdir.mk
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = pipes1.c pipeInterface.c pipeImpls.c
LOCDIR           = src/ts/tutorials/network/wash/

OBJECTS_PIPE = pipeInterface.o pipeImpls.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

pipes1: pipes1.o $(OBJECTS_PIPE)
	-${CLINKER} -o pipes1 pipes1.o $(OBJECTS_PIPE) ${PETSC_TS_LIB}
	${RM} pipes1.o $(OBJECTS_PIPE)

include ${PETSC_DIR}/lib/petsc/conf/test
