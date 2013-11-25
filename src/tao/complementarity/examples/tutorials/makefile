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
MANSEC		 = Complementarity
DOCS		 = 
DIRS		 = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

minsurf1: minsurf1.o chkopts
	-${CLINKER} -o minsurf1 minsurf1.o ${PETSC_TAO_LIB}
	${RM} minsurf1.o

blackscholes: blackscholes.o chkopts
	-${CLINKER} -o blackscholes blackscholes.o ${PETSC_TAO_LIB}
	${RM} blackscholes.o

