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
LOCDIR		 = src/complementarity/examples/tutorials/
MANSEC		 = Complementarity
DOCS		 = 
DIRS		 = 

include ${TAO_DIR}/conf/tao_base

minsurf1: minsurf1.o tao_chkopts
	-${CLINKER} -o minsurf1 minsurf1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} minsurf1.o

blackscholes: blackscholes.o tao_chkopts
	-${CLINKER} -o blackscholes blackscholes.o ${TAO_LIB} ${PETSC_LIB}
	${RM} blackscholes.o

