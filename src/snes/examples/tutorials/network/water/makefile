#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:water
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = water.c
LOCDIR           = src/snes/examples/tutorials/network/water/

OBJECTS_WATER = waterreaddata.o waterfunctions.o water.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


water: $(OBJECTS_WATER) chkopts
	-${CLINKER} -o water $(OBJECTS_WATER) ${PETSC_SNES_LIB}
	${RM} $(OBJECTS_WATER)

include ${PETSC_DIR}/lib/petsc/conf/test
