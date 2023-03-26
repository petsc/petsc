-include ../../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

OBJECTS_PF       = PFReadData.o pffunctions.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


power: power.o ${OBJECTS_PF}
power2: power2.o ${OBJECTS_PF}

