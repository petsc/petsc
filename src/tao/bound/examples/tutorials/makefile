ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
TUTORIALS        = plate2.c jbearing2.c plate2f.F
EXAMPLESC        = plate2.c jbearing2.c
EXAMPLECH        =
EXAMPLESF        = plate2f.F
EXAMPLESFH       = plate2f.h
LOCDIR		 = src/tao/bound/examples/tutorials/
MANSEC		 =
DOCS		 = 
DIRS		 = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

plate2: plate2.o chkopts
	-${CLINKER} -o plate2 plate2.o ${PETSC_TAO_LIB}
	${RM} plate2.o

plate2f: plate2f.o chkopts
	-${FLINKER} -o plate2f plate2f.o ${PETSC_TAO_LIB}
	${RM} plate2f.o

jbearing2: jbearing2.o chkopts
	-${CLINKER} -o jbearing2 jbearing2.o ${PETSC_TAO_LIB}
	${RM} jbearing2.o
