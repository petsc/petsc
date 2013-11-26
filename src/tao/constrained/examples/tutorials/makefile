ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = maros.c toy.c
EXAMPLESF        = toyf.F
EXAMPLESCH       =
EXAMPLESFH       = 
TUTORIALS        = 
LOCDIR		 = src/tao/constrained/examples/tutorials/
MANSEC		 =
DOCS		 = 
DIRS		 = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

istest: istest.o chkopts
	-${CLINKER} -o istest istest.o ${PETSC_TAO_LIB}
	${RM} istest.o

maros: maros.o chkopts
	-${CLINKER} -o maros maros.o ${PETSC_TAO_LIB}
	${RM} maros.o

toy: toy.o chkopts
	-${CLINKER} -o toy toy.o ${PETSC_TAO_LIB}
	${RM} toy.o

toyf: toyf.o chkopts
	-${CLINKER} -o toyf toyf.o ${PETSC_TAO_LIB}
	${RM} toyf.o

polygon1: polygon1.o chkopts
	-${CLINKER} -o polygon1 polygon1.o ${PETSC_TAO_LIB}
	${RM} polygon1.o



