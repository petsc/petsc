# $Id: makefile,v 1.8 1999/01/12 23:18:09 bsmith Exp balay $ 

ALL: lib

CFLAGS	 =
FFLAGS   =
CPPFLAGS =
FPPFLAGS =
LOCDIR   = src/contrib/sda/examples/tutorials/

EXAMPLES_2    =
EXAMPLES_3    =

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

PETSC_CONTRIB = -L${LDIR} -lpetsccontrib

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_CONTRIB}  ${PETSC_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${FLINKER} -o ex2 ex2.o  ${PETSC_FORTRAN_LIB} ${PETSC_CONTRIB} \
                    ${PETSC_LIB}
	${RM} -f ex2.o

include ${PETSC_DIR}/bmake/common_test
