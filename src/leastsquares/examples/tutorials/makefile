ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/unconstrained/examples/tutorials/
MANSEC		 = Unconstrained
EXAMPLESC        = chwirut1.c chwirut2.c
EXAMPLESCH       =
EXAMPLESF        = chwirut1f.F chwirut2f.F
EXAMPLESFH       = chwirut2f.h
DOCS		 = 
DIRS		 = 

include ${TAO_DIR}/conf/tao_base

chwirut1: chwirut1.o tao_chkopts
	-${CLINKER} -o chwirut1 chwirut1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} chwirut1.o

chwirut2: chwirut2.o tao_chkopts
	-${CLINKER} -o chwirut2 chwirut2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} chwirut2.o

chwirut1f: chwirut1f.o tao_chkopts
	-${FLINKER} -o chwirut1f chwirut1f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} chwirut1f.o

chwirut2f: chwirut2f.o tao_chkopts
	-${FLINKER} -o chwirut2f chwirut2f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} chwirut2f.o
