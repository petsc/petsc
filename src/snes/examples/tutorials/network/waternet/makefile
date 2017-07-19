#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:waternet
CFLAGS	         = 
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = waternet.c
LOCDIR           = src/snes/examples/tutorials/network/waternet/

OBJECTS_WATERNET = waternetreaddata.o waternet.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


waternet: $(OBJECTS_WATERNET) chkopts
	-${CLINKER} -o waternet $(OBJECTS_WATERNET) ${PETSC_SNES_LIB}
	${RM} $(OBJECTS_WATERNET)

