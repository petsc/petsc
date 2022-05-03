-include ../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

EXAMPLESC        = ex1.c
LOCDIR           = src/snes/tutorials/network/
DIRS		 = power water
MANSEC           = SNES

OBJECTS_EX1 = ex1.o power/PFReadData.o power/pffunctions.o water/waterreaddata.o water/waterfunctions.o
OBJECTS_EX2 = ex2.o power/PFReadData.o power/pffunctions.o water/waterreaddata.o water/waterfunctions.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex1: $(OBJECTS_EX1)
ex2: $(OBJECTS_EX2)
