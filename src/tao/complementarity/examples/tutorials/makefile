ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = minsurf1.c blackscholes.c
EXAMPLESF        =
EXAMPLESCH       =
EXAMPLESFH       =
TUTORIALS        = minsurf1.c blackscholes.c
LOCDIR		 = src/tao/complementarity/examples/tutorials/
MANSEC		 =
DOCS		 =
DIRS		 =
CLEANFILES       = minsurf1 blackscholes

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

minsurf1: minsurf1.o chkopts
	-${CLINKER} -o minsurf1 minsurf1.o ${PETSC_TAO_LIB}
	${RM} minsurf1.o

blackscholes: blackscholes.o chkopts
	-${CLINKER} -o blackscholes blackscholes.o ${PETSC_TAO_LIB}
	${RM} blackscholes.o

include ${PETSC_DIR}/lib/petsc/conf/test

