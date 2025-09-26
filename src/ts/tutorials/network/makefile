-include ../../../../petscdir.mk

OBJECTS_PIPE = pipeInterface.o pipeImpls.o
CLEANFILES   = pipes ${OBJECTS_PIPE}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

pipes: $(OBJECTS_PIPE)

