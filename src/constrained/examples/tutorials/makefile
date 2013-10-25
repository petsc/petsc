ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = maros1.c maros2.c toy.c
EXAMPLESF        = toyf.F
EXAMPLESCH       =
EXAMPLESFH       = 
TUTORIALS        = 
LOCDIR		 = src/constrained/examples/tutorials/
MANSEC		 = Constrained
DOCS		 = 
DIRS		 = 

include ${TAO_DIR}/conf/tao_base

istest: istest.o tao_chkopts
	-${CLINKER} -o istest istest.o ${TAO_LIB} ${PETSC_LIB}
	${RM} istest.o

maros: maros.o tao_chkopts
	-${CLINKER} -o maros maros.o ${TAO_LIB} ${PETSC_LIB}
	${RM} maros.o

toy: toy.o tao_chkopts
	-${CLINKER} -o toy toy.o ${TAO_LIB} ${PETSC_LIB}
	${RM} toy.o

toyf: toyf.o tao_chkopts
	-${CLINKER} -o toyf toyf.o ${TAO_LIB} ${PETSC_LIB}
	${RM} toyf.o

polygon1: polygon1.o tao_chkopts
	-${CLINKER} -o polygon1 polygon1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} polygon1.o



