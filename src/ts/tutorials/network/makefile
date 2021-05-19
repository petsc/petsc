-include ../../../../petscdir.mk
CFLAGS           =
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = pipes.c pipeInterface.c pipeImpls.c
LOCDIR           = src/ts/tutorials/network/

OBJECTS_PIPE = pipeInterface.o pipeImpls.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

pipes: pipes.o $(OBJECTS_PIPE)

include ${PETSC_DIR}/lib/petsc/conf/test
