ALL:PF
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = pf.c
LOCDIR           = src/snes/examples/tutorials/network/pflow/

OBJECTS_PF = pf.o PFReadData.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

PF: $(OBJECTS_PF) chkopts
	-${CLINKER} -o PF $(OBJECTS_PF) ${PETSC_DM_LIB}
	${RM} $(OBJECTS_PF)

