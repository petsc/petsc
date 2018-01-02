#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:pf
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = pfc.c
LOCDIR           = src/snes/examples/tutorials/network/pflow/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


pfc: pfc.o PFReadData.o chkopts
	-${CLINKER} -o pfc pfc.o PFReadData.o ${PETSC_SNES_LIB}
	${RM} pfc.o PFReadData.o

include ${PETSC_DIR}/lib/petsc/conf/test
