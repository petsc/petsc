# $Id: makefile,v 1.3 1998/04/23 22:52:20 balay Exp balay $ 

ALL: sif

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/contrib/sif

EXAMPLESC       = 
EXAMPLESF       = sif.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

sif: sif.o chkopts
	-${FLINKER} -o sif sif.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} sif.o 

