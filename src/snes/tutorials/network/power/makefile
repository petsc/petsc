-include ../../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

OBJECTS_PF       = PFReadData.o pffunctions.o
CLEANFILES       = power power2 ${OBJECTS_PF}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


power: ${OBJECTS_PF}
power2: ${OBJECTS_PF}

