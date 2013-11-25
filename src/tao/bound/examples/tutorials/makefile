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
MANSEC		 = Bound
DOCS		 = 
DIRS		 = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

plate2: plate2.o chkopts
	-${CLINKER} -o plate2 plate2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} plate2.o

plate2f: plate2f.o chkopts
	-${FLINKER} -o plate2f plate2f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} plate2f.o

jbearing2: jbearing2.o chkopts
	-${CLINKER} -o jbearing2 jbearing2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} jbearing2.o
