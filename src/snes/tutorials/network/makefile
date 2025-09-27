-include ../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

OBJECTS_EX1 = power/PFReadData.o power/pffunctions.o water/waterreaddata.o water/waterfunctions.o
CLEANFILES  = ${OBJECTS_EX1}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: $(OBJECTS_EX1)
