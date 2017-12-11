#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:pf
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = pf.c
LOCDIR           = src/snes/examples/tutorials/network/pflow/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


pf: pf.o PFReadData.o chkopts
	-${CLINKER} -o pf pf.o PFReadData.o ${PETSC_SNES_LIB}
	${RM} pf.o PFReadData.o

include ${PETSC_DIR}/lib/petsc/conf/test
