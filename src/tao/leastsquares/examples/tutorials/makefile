ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/leastsquares/examples/tutorials/
MANSEC		 =
EXAMPLESC        = chwirut1.c chwirut2.c
EXAMPLESCH       =
EXAMPLESF        = chwirut1f.F chwirut2f.F
EXAMPLESFH       = chwirut2f.h
TUTORIALS        = chwirut1.c chwirut2.c chwirut1f.F chwirut2f.F
DOCS		 = 
DIRS		 = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

chwirut1: chwirut1.o chkopts
	-${CLINKER} -o chwirut1 chwirut1.o ${PETSC_TAO_LIB}
	${RM} chwirut1.o

chwirut2: chwirut2.o chkopts
	-${CLINKER} -o chwirut2 chwirut2.o ${PETSC_TAO_LIB}
	${RM} chwirut2.o

chwirut1f: chwirut1f.o chkopts
	-${FLINKER} -o chwirut1f chwirut1f.o ${PETSC_TAO_LIB}
	${RM} chwirut1f.o

chwirut2f: chwirut2f.o chkopts
	-${FLINKER} -o chwirut2f chwirut2f.o ${PETSC_TAO_LIB}
	${RM} chwirut2f.o
