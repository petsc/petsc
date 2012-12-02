ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = maros1.c maros2.c
EXAMPLESF        = 
EXAMPLESCH       =
EXAMPLESFH       = 
TUTORIALS        = 
LOCDIR		 = src/constrained/examples/tutorials/
MANSEC		 = Constrained
DOCS		 = 
DIRS		 = 

include ${TAO_DIR}/conf/tao_base


maros1: maros1.o tao_chkopts
	-${CLINKER} -o maros1 maros1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} maros1.o

maros2: maros2.o tao_chkopts
	-${CLINKER} -o maros2 maros2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} maros2.o




