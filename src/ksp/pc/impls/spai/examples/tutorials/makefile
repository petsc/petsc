#
#   SPAI_LIB should be defined in bmake/$PETSC_ARCH/packages
#
ALL: ex1

CFLAGS 	        =
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/contrib/pc/spai/examples/tutorials/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_LIB} ${SPAI_LIB} ${PETSC_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_LIB} ${SPAI_LIB} ${PETSC_LIB} 
	${RM} ex2.o


EXAMPLESC       =  ex2.c
EXAMPLESF       =


