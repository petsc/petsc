ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/leastsquares/examples/tutorials/
MANSEC		 =
EXAMPLESC        = chwirut1.c chwirut2.c cs1.c tomography.c
EXAMPLESCH       =
EXAMPLESF        = chwirut1f.F90 chwirut2f.F90
EXAMPLESFH       = chwirut2f.h
TUTORIALS        = chwirut1.c chwirut2.c chwirut1f.F90 chwirut2f.F90 cs1.c tomography.c
DOCS		 = 
DIRS		 = 
CLEANFILES       = chwirut1 chwirut2 chwirut1f chwirut2f cs1 tomography

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

chwirut1: chwirut1.o chkopts
	-${CLINKER} -o chwirut1 chwirut1.o ${PETSC_TAO_LIB}
	${RM} chwirut1.o

chwirut2: chwirut2.o chkopts
	-${CLINKER} -o chwirut2 chwirut2.o ${PETSC_TAO_LIB}
	${RM} chwirut2.o

cs1: cs1.o chkopts
	-${CLINKER} -o cs1 cs1.o ${PETSC_TAO_LIB}
	${RM} cs1.o

tomography: tomography.o chkopts
	-${CLINKER} -o tomography tomography.o ${PETSC_TAO_LIB}
	${RM} tomography.o


chwirut1f: chwirut1f.o chkopts
	-${FLINKER} -o chwirut1f chwirut1f.o ${PETSC_TAO_LIB}
	${RM} chwirut1f.o

chwirut2f: chwirut2f.o chkopts
	-${FLINKER} -o chwirut2f chwirut2f.o ${PETSC_TAO_LIB}
	${RM} chwirut2f.o

include ${PETSC_DIR}/lib/petsc/conf/test


