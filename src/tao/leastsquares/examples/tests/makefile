ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/leastsquares/examples/tests/
MANSEC		 =
EXAMPLESC        = chwirut2.c
DOCS		 =
DIRS		 =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


chwirut2: chwirut2.o chkopts
	-${CLINKER} -o chwirut2 chwirut2.o ${PETSC_TAO_LIB}
	${RM} chwirut2.o

include ${PETSC_DIR}/lib/petsc/conf/test


