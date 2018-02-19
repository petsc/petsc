#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:pf
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = pfc.c
LOCDIR           = src/snes/examples/tutorials/network/pflow/

OBJECTS_PF2 = pf2.o PFReadData.o pffunctions.o
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


pfc: pfc.o PFReadData.o pffunctions.o chkopts
	-${CLINKER} -o pfc pfc.o pffunctions.o PFReadData.o ${PETSC_SNES_LIB}
#	${RM} pfc.o PFReadData.o pffunctions.o

pf2: $(OBJECTS_PF2) chkopts
	-${CLINKER} -o pf2 $(OBJECTS_PF2) ${PETSC_SNES_LIB}
	${RM} $(OBJECTS_PF2)

include ${PETSC_DIR}/lib/petsc/conf/test
