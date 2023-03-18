-include ../../../../petscdir.mk

OBJECTS_PIPE = pipeInterface.o pipeImpls.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

pipes: pipes.o $(OBJECTS_PIPE)

