ALL:PF
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =

OBJECTS_PF = pf.o PFReadData.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

PF: $(OBJECTS_PF) chkopts
	-${CLINKER} -o PF $(OBJECTS_PF) ${PETSC_DM_LIB}
	${RM} $(OBJECTS_PF)

